from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# R3: Postgres is authoritative (A0); OpenSearch is explicitly a derived,
# rebuildable artifact (src/index/reindex.py) and is NOT backed up here —
# only the authoritative store needs one. Shells out to the running
# `postgres` compose service's own pg_dump (verified live: no host-side
# pg_dump/pg_restore exists on this machine, only inside the container —
# confirmed via `which pg_dump` before writing this, not assumed) rather
# than requiring a host-side Postgres client install.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKUP_DIR = _REPO_ROOT / "backups"


def backup(*, service: str = "postgres") -> Path:
    load_dotenv(_REPO_ROOT / ".env")
    db = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]

    _BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = _BACKUP_DIR / f"rag_backup_{timestamp}.dump"

    # -F c: pg_restore-compatible custom format, not plain SQL — smaller,
    # supports selective/parallel restore, the standard choice for a real
    # restore drill rather than a human-readable dump.
    cmd = ["docker", "compose", "exec", "-T", service, "pg_dump", "-U", user, "-d", db, "-F", "c"]
    with open(out_path, "wb") as f:
        result = subprocess.run(cmd, cwd=_REPO_ROOT, stdout=f, stderr=subprocess.PIPE)

    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dump failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"backed up to {out_path} ({size_mb:.2f} MB)")
    return out_path


if __name__ == "__main__":
    backup()
