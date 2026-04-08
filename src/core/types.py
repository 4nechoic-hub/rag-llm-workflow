# -*- coding: utf-8 -*-
"""Shared types for pipeline results — unified contract across all backends."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceChunk:
    """A single retrieved source chunk."""
    file: str
    page: int | str
    score: float | None = None
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "page": self.page,
            "score": self.score,
            "chunk_id": self.chunk_id,
        }


@dataclass
class PipelineResult:
    """
    Unified return type for all RAG pipelines.

    Every pipeline — manual, LangGraph, LlamaIndex — returns one of these.
    The Streamlit app and evaluation harness consume this contract instead
    of branching on pipeline name.
    """
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    latency: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def source_dicts(self) -> list[dict]:
        """Convenience: sources as a list of plain dicts (for Streamlit / JSON)."""
        return [s.to_dict() for s in self.sources]

    @property
    def num_chunks(self) -> int:
        return len(self.sources)

    @property
    def avg_similarity(self) -> float:
        scores = [s.score for s in self.sources if s.score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def max_similarity(self) -> float:
        scores = [s.score for s in self.sources if s.score is not None]
        return max(scores) if scores else 0.0


def sources_from_dataframe(df) -> list[SourceChunk]:
    """Convert a retrieved DataFrame (manual/LangGraph style) to SourceChunk list."""
    sources = []
    for _, row in df.iterrows():
        sources.append(SourceChunk(
            file=row["doc_name"],
            page=int(row["page_number"]),
            score=round(float(row["similarity"]), 4),
            chunk_id=row.get("chunk_id", ""),
        ))
    return sources


def sources_from_llamaindex(response) -> list[SourceChunk]:
    """Convert a LlamaIndex response's source_nodes to SourceChunk list."""
    sources = []
    for node in response.source_nodes:
        meta = node.metadata
        sources.append(SourceChunk(
            file=meta.get("file_name", "unknown"),
            page=meta.get("page_label", "?"),
            score=round(node.score, 4) if node.score else None,
        ))
    return sources
