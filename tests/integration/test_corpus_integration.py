import uuid

import pytest
from fastapi.testclient import TestClient

from src.app.main import app
from src.store.relational import get_session
from src.store.schema import Paper

pytestmark = pytest.mark.integration

# Real bug found in review (Phase 2, stage 6): /corpus has no auth gate —
# every anonymous visitor sees it — so it must only ever render the
# SHARED corpus (owner_session_id IS NULL). Before this fix, corpus_page's
# query had no owner_session_id filter at all, so any visitor's private
# uploaded PDF (title, category, chunk count) rendered here for every
# other visitor. This is a real Postgres round trip against the actual
# ORM query, not a mock, since the bug was in the query construction
# itself.


@pytest.fixture
def session():
    s = get_session()
    yield s
    s.close()


def test_corpus_page_never_shows_a_privately_owned_paper(session):
    shared_id = f"corpus-test-shared-{uuid.uuid4().hex}"
    private_id = f"corpus-test-private-{uuid.uuid4().hex}"
    session.add(
        Paper(
            arxiv_id=shared_id,
            title="A Shared Corpus Test Paper",
            authors=["A. Author"],
            abstract="abstract",
            category=["cs.IR"],
            url="https://example.com",
        )
    )
    session.add(
        Paper(
            arxiv_id=private_id,
            title="Someone's Private Uploaded Paper",
            authors=[],
            abstract="",
            category=[],
            url="",
            source="upload",
            owner_session_id="a-different-visitor-session",
        )
    )
    session.commit()

    try:
        client = TestClient(app)
        response = client.get("/corpus")
        assert response.status_code == 200
        assert b"A Shared Corpus Test Paper" in response.content
        assert b"Someone&#39;s Private Uploaded Paper" not in response.content
        assert b"Someone's Private Uploaded Paper" not in response.content
    finally:
        session.delete(session.get(Paper, shared_id))
        session.delete(session.get(Paper, private_id))
        session.commit()
