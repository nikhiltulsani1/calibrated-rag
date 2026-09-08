from __future__ import annotations

import os

import redis
from rq import Queue, SimpleWorker

_QUEUE_NAME = "uploads"


class ThreadSafeWorker(SimpleWorker):
    """Real bug found live (Phase 3 stage 3 verification): RQ's default
    `Worker` forks a child OS PROCESS per job — `os.fork()` doesn't exist
    on Windows at all, which this project is developed on locally, so
    the default Worker class isn't usable here regardless of the signal
    issue below. `SimpleWorker` runs jobs in the same process instead,
    which is also exactly what's needed for correctness: nothing else
    changes about how jobs run.

    Separately, `BaseWorker.work()` unconditionally calls
    `_install_signal_handlers()`, which calls `signal.signal(...)` —
    Python only allows registering signal handlers from the main thread
    of the main interpreter, but this worker runs in a background daemon
    thread (see src/app/main.py), not the main thread. Confirmed live:
    the worker process crashed immediately with
    "ValueError: signal only works in main thread of the main
    interpreter" the moment `.work()` was called, before processing a
    single job. Overridden to a no-op — nothing in this deployment
    relies on RQ's own SIGINT/SIGTERM graceful-shutdown handling; the
    worker thread is a daemon and simply dies when the process exits.
    """

    def _install_signal_handlers(self):
        pass

_conn: redis.Redis | None = None
_queue: Queue | None = None


def get_redis_conn() -> redis.Redis:
    """Phase 3 stage 3: a SEPARATE connection from
    src/platform/cache.py::get_client() — not a refactor to share one,
    deliberately. cache.get_client() sets decode_responses=True, which
    is exactly wrong for RQ: RQ pickles job payloads and stores them as
    raw bytes in Redis, and a decode_responses=True connection tries to
    UTF-8-decode those bytes on read, corrupting them. Same host/port/
    password/ssl env-var logic as get_client() (Upstash on live, plain
    local Redis in dev), just with decode_responses=False, which is
    RQ's own real requirement, not a style choice.
    """
    global _conn
    if _conn is None:
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        password = os.environ.get("REDIS_PASSWORD") or None
        ssl = os.environ.get("REDIS_SSL", "false").lower() == "true"
        _conn = redis.Redis(host=host, port=port, password=password, ssl=ssl, decode_responses=False)
    return _conn


def get_queue() -> Queue:
    global _queue
    if _queue is None:
        _queue = Queue(_QUEUE_NAME, connection=get_redis_conn())
    return _queue
