from unittest.mock import patch

import pytest

import src.platform.queue as queue

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_cached_singletons():
    queue._conn = None
    queue._queue = None
    yield
    queue._conn = None
    queue._queue = None


# get_redis_conn — Phase 3 stage 3: a SEPARATE connection from
# src/platform/cache.py::get_client(), deliberately, because
# decode_responses=True (get_client()'s setting) corrupts RQ's pickled
# job payloads on read.


def test_get_redis_conn_defaults_to_no_password_and_no_tls(monkeypatch):
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.delenv("REDIS_SSL", raising=False)
    with patch("src.platform.queue.redis.Redis") as mock_redis:
        queue.get_redis_conn()
    mock_redis.assert_called_once_with(host="localhost", port=6379, password=None, ssl=False, decode_responses=False)


def test_get_redis_conn_uses_password_and_tls_when_configured(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "usw1-example.upstash.io")
    monkeypatch.setenv("REDIS_PASSWORD", "real-token")
    monkeypatch.setenv("REDIS_SSL", "true")
    with patch("src.platform.queue.redis.Redis") as mock_redis:
        queue.get_redis_conn()
    mock_redis.assert_called_once_with(
        host="usw1-example.upstash.io", port=6379, password="real-token", ssl=True, decode_responses=False
    )


def test_get_redis_conn_is_cached_across_calls():
    conn_a = queue.get_redis_conn()
    conn_b = queue.get_redis_conn()
    assert conn_a is conn_b


def test_get_queue_returns_the_uploads_queue():
    q = queue.get_queue()
    assert q.name == "uploads"


def test_get_queue_is_cached_across_calls():
    assert queue.get_queue() is queue.get_queue()


# ---------------------------------------------------------------------
# The real gotcha, tested for real (not just asserted in a comment):
# a decode_responses=True connection (cache.get_client()'s setting)
# genuinely corrupts an RQ job enqueued/fetched through it, while
# get_redis_conn()'s decode_responses=False round-trips correctly.
# Needs real local Redis — same convention as this project's other
# real-infra-backed tests.
# ---------------------------------------------------------------------


@pytest.mark.integration
def test_a_real_job_round_trips_correctly_through_get_redis_conn():
    from rq import Queue
    from rq.job import Job

    def _sample_job(x):
        return x * 2

    q = Queue("test-queue-decode-responses-check", connection=queue.get_redis_conn())
    job = q.enqueue(_sample_job, 21)
    try:
        fetched = Job.fetch(job.id, connection=queue.get_redis_conn())
        assert fetched.args == (21,)
    finally:
        job.delete()


@pytest.mark.integration
def test_the_same_job_is_corrupted_through_a_decode_responses_true_connection():
    # The negative case, proven rather than assumed: fetching the exact
    # same job data through cache.get_client()'s decode_responses=True
    # connection either raises or returns garbage instead of the real
    # pickled payload — confirming get_redis_conn()'s own separate
    # connection is load-bearing, not redundant caution.
    import src.platform.cache as cache
    from rq import Queue
    from rq.job import Job

    def _sample_job(x):
        return x * 2

    cache._client = None
    q = Queue("test-queue-decode-responses-check", connection=queue.get_redis_conn())
    job = q.enqueue(_sample_job, 21)
    try:
        with pytest.raises(Exception):
            Job.fetch(job.id, connection=cache.get_client())
    finally:
        job.delete()
        cache._client = None
