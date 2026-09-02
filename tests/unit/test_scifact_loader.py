import pytest

from evals.scifact_loader import _RAW_DIR, load_corpus, load_qrels, load_queries

# Reads the locally cached download (evals/corpora/scifact_raw/) rather
# than a live service — no docker compose, no network at test time, which
# is what makes this a unit test rather than an integration one. Skips
# cleanly if the one-time download hasn't been run yet.
_downloaded = (_RAW_DIR / "corpus.jsonl").exists()

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _downloaded,
        reason="SciFact benchmark not downloaded — see evals/validate_harness.py for the fetch step",
    ),
]


def test_load_corpus_real_document_count():
    docs = load_corpus()
    assert len(docs) == 5183


def test_load_corpus_documents_have_text():
    docs = load_corpus()
    assert all(doc.text for doc in docs)
    assert all(doc.doc_id for doc in docs)


def test_load_queries_real_count():
    queries = load_queries()
    assert len(queries) > 0
    assert all(q.text for q in queries)


def test_load_qrels_test_split_matches_known_shape():
    qrels = load_qrels("test")
    assert len(qrels) == 300  # verified against the real downloaded file
    # SciFact's test qrels are binary — every judgment is relevance=1
    all_scores = {score for judgments in qrels.values() for score in judgments.values()}
    assert all_scores == {1}


def test_qrels_reference_real_corpus_doc_ids():
    docs = {doc.doc_id for doc in load_corpus()}
    qrels = load_qrels("test")
    referenced = {doc_id for judgments in qrels.values() for doc_id in judgments}
    assert referenced.issubset(docs), "qrels must only reference documents that actually exist in the corpus"
