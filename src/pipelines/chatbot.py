# -*- coding: utf-8 -*-
"""
Conversational RAG Chatbot

A multi-turn chatbot that uses retrieval-augmented generation to answer
questions grounded in uploaded PDF documents. It keeps a clean conversation
history for answer generation and rewrites follow-up questions into
standalone retrieval queries so document search works better for pronouns,
shorthand references, and comparisons.

Author: Tingyi Zhang
"""

from __future__ import annotations

import re

from src.config import TOP_K
from src.core.llm import call_llm, call_llm_chat
from src.core.retriever import format_context, retrieve_top_k
from src.core.usage import finalise_usage, usage_tracking_session


# ── System prompt ───────────────────────────────────────────────

CHATBOT_SYSTEM_PROMPT = (
    "You are a knowledgeable research assistant with access to a collection "
    "of technical documents. Your role is to help users understand, analyse, "
    "and extract insights from these documents through natural conversation.\n\n"
    "Guidelines:\n"
    "1. Answer using ONLY the provided document context. Do not use outside knowledge.\n"
    "2. If the context does not contain enough information to answer, say so clearly.\n"
    "3. Cite specific documents and page numbers when making claims.\n"
    "4. For follow-up questions, use both the new context and conversation history.\n"
    "5. Be conversational but precise. Avoid unnecessary hedging.\n"
    "6. When comparing documents, structure your response clearly.\n"
    "7. If the user asks something unrelated to the documents, politely redirect.\n\n"
    "You have access to the following documents:\n{doc_list}\n"
)

QUERY_REWRITE_SYSTEM_PROMPT = (
    "You rewrite user follow-up questions for a document retrieval system.\n\n"
    "Given recent conversation turns and the latest user message, produce ONE "
    "short standalone retrieval query that can be used to search the documents.\n\n"
    "Rules:\n"
    "1. Resolve pronouns and implicit references using the conversation history.\n"
    "2. Preserve important technical terms, acronyms, document names, entities, and compared methods.\n"
    "3. Do not answer the question.\n"
    "4. Do not add explanations, bullets, prefixes, or quotation marks.\n"
    "5. Return exactly one standalone search query.\n"
)

REWRITE_HISTORY_TURNS = 3
ANSWER_HISTORY_TURNS = 10
MAX_REWRITE_MESSAGE_CHARS = 400
MAX_RETRIEVAL_QUERY_CHARS = 300


# ── Chatbot class ───────────────────────────────────────────────

class RAGChatbot:
    """
    Conversational RAG chatbot with message history and history-aware retrieval.

    Usage:
        bot = RAGChatbot(df_chunks, embeddings, doc_names=["paper1.pdf", ...])
        response, sources = bot.chat("What methods were used?")
        response, sources = bot.chat("Can you elaborate on the PIV setup?")
    """

    def __init__(self, df_chunks, embeddings, doc_names=None, top_k=TOP_K):
        self.df_chunks = df_chunks
        self.embeddings = embeddings
        self.top_k = top_k
        self.history = []  # list of {"role": ..., "content": ...}

        if doc_names is None:
            doc_names = sorted(df_chunks["doc_name"].unique().tolist())
        self.doc_names = doc_names

        doc_list = "\n".join(f"  - {name}" for name in doc_names)
        self.system_prompt = CHATBOT_SYSTEM_PROMPT.format(doc_list=doc_list)

        # Debug / inspection state
        self.last_retrieval_query: str | None = None
        self.last_retrieval_rewritten: bool = False
        self.last_usage: dict | None = None

    @staticmethod
    def _normalise_text(text: str) -> str:
        """Collapse repeated whitespace for cleaner prompts and comparisons."""
        return re.sub(r"\s+", " ", text).strip()

    def _truncate_for_rewrite(self, text: str) -> str:
        """Shorten long turns so the rewrite prompt stays compact."""
        clean = self._normalise_text(text)
        if len(clean) <= MAX_REWRITE_MESSAGE_CHARS:
            return clean
        return clean[: MAX_REWRITE_MESSAGE_CHARS - 1].rstrip() + "…"

    def _trim_history(self, max_turns: int = ANSWER_HISTORY_TURNS) -> list[dict]:
        """Keep the last max_turns pairs of messages."""
        max_messages = max_turns * 2
        if len(self.history) <= max_messages:
            return self.history.copy()
        return self.history[-max_messages:]

    def _format_history_for_rewrite(self, max_turns: int = REWRITE_HISTORY_TURNS) -> str:
        """Build a compact history block for retrieval-query rewriting."""
        recent = self._trim_history(max_turns=max_turns)
        if not recent:
            return ""

        lines = []
        for message in recent:
            role = message["role"].upper()
            content = self._truncate_for_rewrite(message["content"])
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _clean_retrieval_query(query: str, fallback: str) -> str:
        """Normalise model output and fall back safely when needed."""
        cleaned = (query or "").strip()
        cleaned = cleaned.strip("` ")

        # Remove common prefixes if the model adds them.
        lower = cleaned.lower()
        prefixes = (
            "standalone retrieval query:",
            "standalone query:",
            "retrieval query:",
            "search query:",
            "query:",
        )
        for prefix in prefixes:
            if lower.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        # Keep the first non-empty line if the model returns multiple lines.
        if "\n" in cleaned:
            for line in cleaned.splitlines():
                line = line.strip(" -\t\"'")
                if line:
                    cleaned = line
                    break

        cleaned = cleaned.strip("\"' ")
        cleaned = re.sub(r"\s+", " ", cleaned)

        if not cleaned:
            return fallback

        if len(cleaned) > MAX_RETRIEVAL_QUERY_CHARS:
            clipped = cleaned[:MAX_RETRIEVAL_QUERY_CHARS].rstrip()
            if " " in clipped:
                clipped = clipped.rsplit(" ", 1)[0]
            cleaned = clipped.strip() or fallback

        return cleaned or fallback

    def _rewrite_retrieval_query(self, user_message: str) -> str:
        """Rewrite the latest user message into a standalone retrieval query."""
        history_block = self._format_history_for_rewrite(max_turns=REWRITE_HISTORY_TURNS)
        if not history_block:
            return user_message

        user_prompt = (
            f"Conversation history:\n{history_block}\n\n"
            f"Latest user message:\n{user_message}\n\n"
            "Standalone retrieval query:"
        )

        try:
            rewritten = call_llm(
                QUERY_REWRITE_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.0,
            )
        except Exception:
            return user_message

        return self._clean_retrieval_query(rewritten, fallback=user_message)

    def chat(self, user_message: str) -> tuple[str, list[dict]]:
        """
        Send a message and get a grounded response.

        Returns:
            tuple[str, list[dict]]: (response_text, retrieved_sources)
        """
        with usage_tracking_session() as usage:
            retrieval_query = self._rewrite_retrieval_query(user_message)
            self.last_retrieval_query = retrieval_query
            self.last_retrieval_rewritten = (
                self._normalise_text(retrieval_query) != self._normalise_text(user_message)
            )

            retrieved = retrieve_top_k(
                retrieval_query,
                self.df_chunks,
                self.embeddings,
                top_k=self.top_k,
            )
            context = format_context(retrieved)

            augmented_message = (
                f"{user_message}\n\n"
                f"--- Retrieved Document Context ---\n{context}"
            )

            # Temporarily store the current user turn with retrieved context so the
            # answer-generation step can see both the clean conversation and the
            # newly retrieved evidence.
            self.history.append({"role": "user", "content": augmented_message})

            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self._trim_history(max_turns=ANSWER_HISTORY_TURNS))

            response = call_llm_chat(messages)

            # Replace the augmented message with the clean user turn so the history
            # remains readable and compact for later rewrites.
            self.history[-1] = {"role": "user", "content": user_message}
            self.history.append({"role": "assistant", "content": response})

            sources = []
            for _, row in retrieved.iterrows():
                sources.append(
                    {
                        "file": row["doc_name"],
                        "page": row["page_number"],
                        "score": round(row["similarity"], 4),
                        "chunk_id": row["chunk_id"],
                    }
                )

            self.last_usage = finalise_usage(usage)

        return response, sources

    def clear_history(self):
        """Reset conversation history and debug state."""
        self.history = []
        self.last_retrieval_query = None
        self.last_retrieval_rewritten = False
        self.last_usage = None

    def get_history(self) -> list[dict]:
        """Return the conversation history (user/assistant messages only)."""
        return [
            msg for msg in self.history
            if msg["role"] in ("user", "assistant")
        ]

    def get_last_retrieval_debug(self) -> dict:
        """Expose retrieval rewrite info for debugging or UI display."""
        return {
            "retrieval_query": self.last_retrieval_query,
            "rewritten": self.last_retrieval_rewritten,
            "usage": self.last_usage,
        }

    @property
    def turn_count(self) -> int:
        """Number of complete conversation turns."""
        return len([m for m in self.history if m["role"] == "assistant"])
