from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON)
    abstract: Mapped[str] = mapped_column(Text)
    category: Mapped[list[str]] = mapped_column(JSON)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    url: Mapped[str] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    # Content-addressed — sha256(arxiv_id + chunk_text), not a positional
    # index (see A3b). A chunk keeps its identity across re-ingestion of
    # unrelated papers, which is what lets qrels labeled against a
    # chunk_id survive the corpus growing.
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.arxiv_id"))
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    # Provenance: which model/width produced the vector OpenSearch holds
    # for this chunk. Postgres is the source of truth OpenSearch must be
    # rebuildable from (A0) — this is what a rebuild checks against
    # models.lock.yaml to know whether a re-embed is needed.
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    paper: Mapped["Paper"] = relationship(back_populates="chunks")


class ChunkVariant(Base):
    """A7's toggleable chunking-strategy corpus. Deliberately a SEPARATE
    table from `Chunk`, not a column added there — the production
    `chunks` table (and its `rag_chunks` OpenSearch index) is never
    touched by A7 at all, so the 147 chunk-level qrels/gold questions
    already labeled this session stay valid regardless of what strategy
    is toggled live. `strategy` holds the toggle's role name directly
    ("winner"/"median"/"efficient" — see evals/run_chunking_eval.py),
    not the underlying technical strategy name, since only one variant
    ever plays each role.
    """

    __tablename__ = "chunk_variants"

    strategy: Mapped[str] = mapped_column(String(32), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.arxiv_id"))
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Run(Base):
    """One persisted `StageTrace`, so a past answer can be replayed
    exactly — see src/store/runs.py. Re-running the same query text is
    NOT the same thing: retrieval, reranking, and generation can all
    return different results on a second live call, so the *original*
    trace has to be stored, not reconstructed later.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    trace_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ingestion_state and run_metadata (see A0's store table) land with the
# ingest pipeline and the eval harness respectively — adding them here
# now would be schema for features that don't exist yet.
