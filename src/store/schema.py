from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Date, DateTime, ForeignKey, Index, Integer, JSON, String, Text, func
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    # Phase 2: despite the column name, this is a generic document id, not
    # necessarily a real arXiv id — kept unrenamed on purpose so the
    # default RETRIEVAL_BACKEND=opensearch path's existing code (every FK
    # reference, every query) needs zero changes. An uploaded PDF
    # (RETRIEVAL_BACKEND=postgres only, see src/app/routes/upload.py)
    # stores a synthetic id like "upload-<uuid>" here instead.
    arxiv_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(JSON)
    abstract: Mapped[str] = mapped_column(Text)
    category: Mapped[list[str]] = mapped_column(JSON)
    published_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    url: Mapped[str] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Phase 2, both nullable and always-NULL on the OpenSearch path (only
    # the new upload route ever sets them): `source` distinguishes an
    # arXiv paper from a visitor's uploaded PDF; `owner_session_id` is the
    # private-per-uploader isolation key — NULL means "part of the shared
    # corpus," non-null means "only visible to that one browser session."
    # See hybrid_postgres.py for where this is actually enforced.
    source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    owner_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

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

    # Phase 2, RETRIEVAL_BACKEND=postgres only — the OpenSearch path never
    # reads or writes these; they stay NULL for every row ingested through
    # the default arXiv pipeline.
    #
    # Denormalized from the parent Paper rather than joined at query time
    # — every retrieval query (hybrid_postgres.py's lexical AND dense arm)
    # needs this filter, and a denormalized column keeps that filter a
    # plain WHERE clause instead of a join on the hottest code path in
    # the app.
    owner_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # 1024 dims to match the existing Jina/Mistral dimension lock in
    # src/config/models.lock.yaml. Nullable until the embed step of
    # ingestion actually runs (mirrors embedding_model/embedding_dim
    # above being nullable for the same reason on the OpenSearch path).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    # Real bug this project already found and fixed once for the
    # OpenSearch path (embed-provider toggle poisoning a shared index
    # with mixed-provider vectors, see project-docs/result.md) — tagging
    # each row with the provider that actually produced its vector, and
    # filtering on it at query time, is what stops the same bug class
    # from being reintroduced here.
    embedding_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Postgres full-text search equivalent of the OpenSearch path's
    # english-analyzed `text` field — a generated column, computed once
    # at write time, not recomputed per query.
    text_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True), nullable=True
    )

    __table_args__ = (Index("ix_chunks_text_tsv", "text_tsv", postgresql_using="gin"),)

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
    # Phase 2: NULL for every run made against the shared corpus (the
    # default, and every run made before this column existed). A
    # /pipeline?run_id=... replay must check this against the requesting
    # visitor's own session before rendering — otherwise a guessed or
    # shared run_id from a private-upload session becomes a real
    # cross-visitor data-exposure vector. See src/app/routes/pipeline.py.
    owner_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


# ingestion_state and run_metadata (see A0's store table) land with the
# ingest pipeline and the eval harness respectively — adding them here
# now would be schema for features that don't exist yet.
