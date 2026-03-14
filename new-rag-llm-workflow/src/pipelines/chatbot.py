# -*- coding: utf-8 -*-
"""
Conversational RAG Chatbot

A multi-turn chatbot that uses retrieval-augmented generation to answer
questions grounded in uploaded PDF documents. Maintains conversation
history for follow-up questions and contextual understanding.

Author: Tingyi Zhang
"""

from src.core.retriever import retrieve_top_k, format_context
from src.core.llm import call_llm_chat
from src.config import TOP_K

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
    "5. Be conversational but precise — avoid unnecessary hedging.\n"
    "6. When comparing documents, structure your response clearly.\n"
    "7. If the user asks something unrelated to the documents, politely redirect.\n\n"
    "You have access to the following documents:\n{doc_list}\n"
)


# ── Chatbot class ───────────────────────────────────────────────

class RAGChatbot:
    """
    Conversational RAG chatbot with message history and retrieval.

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

        # Build document list for system prompt
        if doc_names is None:
            doc_names = sorted(df_chunks["doc_name"].unique().tolist())
        self.doc_names = doc_names

        doc_list = "\n".join(f"  - {name}" for name in doc_names)
        self.system_prompt = CHATBOT_SYSTEM_PROMPT.format(doc_list=doc_list)

    def chat(self, user_message: str) -> tuple[str, list[dict]]:
        """
        Send a message and get a grounded response.
        Returns (response_text, retrieved_sources).
        """
        # Retrieve relevant chunks
        retrieved = retrieve_top_k(
            user_message, self.df_chunks, self.embeddings, top_k=self.top_k,
        )
        context = format_context(retrieved)

        # Build the augmented user message with context
        augmented_message = (
            f"{user_message}\n\n"
            f"--- Retrieved Document Context ---\n{context}"
        )

        # Add to history
        self.history.append({"role": "user", "content": augmented_message})

        # Build full messages list
        messages = [{"role": "system", "content": self.system_prompt}]
        # Keep last N turns to manage token budget
        recent_history = self._trim_history(max_turns=10)
        messages.extend(recent_history)

        # Call LLM
        response = call_llm_chat(messages)

        # Store clean version in history (without context block)
        self.history[-1] = {"role": "user", "content": user_message}
        self.history.append({"role": "assistant", "content": response})

        # Extract source info
        sources = []
        for _, row in retrieved.iterrows():
            sources.append({
                "file": row["doc_name"],
                "page": row["page_number"],
                "score": round(row["similarity"], 4),
                "chunk_id": row["chunk_id"],
            })

        return response, sources

    def _trim_history(self, max_turns: int = 10) -> list[dict]:
        """Keep the last max_turns pairs of messages."""
        max_messages = max_turns * 2
        if len(self.history) <= max_messages:
            return self.history.copy()
        return self.history[-max_messages:]

    def clear_history(self):
        """Reset conversation history."""
        self.history = []

    def get_history(self) -> list[dict]:
        """Return the conversation history (user/assistant messages only)."""
        return [
            msg for msg in self.history
            if msg["role"] in ("user", "assistant")
        ]

    @property
    def turn_count(self) -> int:
        """Number of complete conversation turns."""
        return len([m for m in self.history if m["role"] == "assistant"])
