import hashlib

import pytest

from src.store.relational import get_session
from src.store.schema import Chunk, Paper

pytestmark = pytest.mark.integration

_TEST_ARXIV_ID = "test.0000"


@pytest.fixture
def session():
    s = get_session()
    yield s
    # Clean up regardless of test outcome — never leave synthetic rows
    # mixed in with the real ingested corpus.
    existing_chunks = s.query(Chunk).filter_by(paper_id=_TEST_ARXIV_ID).all()
    for c in existing_chunks:
        s.delete(c)
    existing_paper = s.get(Paper, _TEST_ARXIV_ID)
    if existing_paper is not None:
        s.delete(existing_paper)
    s.commit()
    s.close()


def test_paper_and_chunk_round_trip_via_relationship(session):
    chunk_text = "this is a test chunk for the relational integration test"
    chunk_id = hashlib.sha256((_TEST_ARXIV_ID + chunk_text).encode()).hexdigest()

    session.add(
        Paper(
            arxiv_id=_TEST_ARXIV_ID,
            title="Test Paper",
            authors=["A. Author"],
            abstract="abstract",
            category=["cs.IR"],
            url="https://arxiv.org/abs/test.0000",
        )
    )
    session.add(
        Chunk(
            chunk_id=chunk_id,
            paper_id=_TEST_ARXIV_ID,
            section="intro",
            text=chunk_text,
            char_start=0,
            char_end=len(chunk_text),
            embedding_model="jina:jina-embeddings-v3",
            embedding_dim=1024,
        )
    )
    session.commit()

    fetched = session.get(Chunk, chunk_id)
    assert fetched is not None
    assert fetched.text == chunk_text
    assert fetched.paper.title == "Test Paper"  # relationship traversal, not just a raw column


def test_chunk_id_primary_key_prevents_duplicate_insert(session):
    text = "duplicate test text"
    chunk_id = hashlib.sha256((_TEST_ARXIV_ID + text).encode()).hexdigest()
    session.add(
        Paper(
            arxiv_id=_TEST_ARXIV_ID,
            title="T",
            authors=[],
            abstract="",
            category=[],
            url="",
        )
    )
    session.add(Chunk(chunk_id=chunk_id, paper_id=_TEST_ARXIV_ID, text=text, char_start=0, char_end=len(text)))
    session.commit()

    session.add(Chunk(chunk_id=chunk_id, paper_id=_TEST_ARXIV_ID, text=text, char_start=0, char_end=len(text)))
    with pytest.raises(Exception):  # IntegrityError — driver-specific, asserted broadly on purpose
        session.commit()
    session.rollback()
