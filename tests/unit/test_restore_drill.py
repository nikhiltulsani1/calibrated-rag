from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


def test_run_drill_raises_clearly_when_no_backup_exists(tmp_path, monkeypatch):
    import scripts.restore_drill as restore_drill

    monkeypatch.setattr(restore_drill, "_REPO_ROOT", tmp_path)
    (tmp_path / "backups").mkdir()
    with patch("scripts.restore_drill.load_dotenv"):
        with pytest.raises(RuntimeError, match="no backup file found"):
            restore_drill.run_drill()


def test_run_drill_picks_the_most_recent_backup_by_name(tmp_path, monkeypatch):
    import scripts.restore_drill as restore_drill

    monkeypatch.setattr(restore_drill, "_REPO_ROOT", tmp_path)
    backups_dir = tmp_path / "backups"
    backups_dir.mkdir()
    older = backups_dir / "rag_backup_20260101T000000Z.dump"
    newer = backups_dir / "rag_backup_20260819T000000Z.dump"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")

    with patch("scripts.restore_drill.load_dotenv"), patch(
        "scripts.restore_drill._create_drill_database"
    ), patch("scripts.restore_drill._restore_into_drill_database") as mock_restore, patch(
        "scripts.restore_drill._rebuild_opensearch_index", return_value={"chunks_restored": 0}
    ), patch(
        "scripts.restore_drill._validate_retrieval", return_value={"n_questions": 0, "recall@10": 0.0}
    ), patch(
        "scripts.restore_drill._cleanup_opensearch"
    ), patch("scripts.restore_drill._drop_drill_database"):
        restore_drill.run_drill()

    mock_restore.assert_called_once_with(newer)


def test_restore_reports_fatal_failures_but_not_benign_warnings():
    import scripts.restore_drill as restore_drill

    fatal = MagicMock(returncode=1, stderr=b"pg_restore: error: FATAL:  role does not exist")
    with patch("scripts.restore_drill.subprocess.run", return_value=fatal), patch(
        "builtins.open", MagicMock()
    ), patch("scripts.restore_drill.os.environ", {"POSTGRES_USER": "u"}):
        with pytest.raises(RuntimeError, match="pg_restore failed"):
            restore_drill._restore_into_drill_database(restore_drill._REPO_ROOT / "x.dump")


def test_restore_tolerates_nonfatal_nonzero_exit():
    import scripts.restore_drill as restore_drill

    benign = MagicMock(returncode=1, stderr=b"pg_restore: warning: errors ignored on restore: 1")
    with patch("scripts.restore_drill.subprocess.run", return_value=benign), patch(
        "builtins.open", MagicMock()
    ), patch("scripts.restore_drill.os.environ", {"POSTGRES_USER": "u"}):
        restore_drill._restore_into_drill_database(restore_drill._REPO_ROOT / "x.dump")  # must not raise
