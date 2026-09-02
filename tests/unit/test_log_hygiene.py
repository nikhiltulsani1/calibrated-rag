import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# R6's "log hygiene" gap, turned from a one-time manual grep (README's R6
# audit) into a real regression guard: fails pytest -m unit today, before
# R1's CI exists, if a future change ever logs or traces a secret-shaped
# value. Static, line-level, deliberately simple — it isn't a full
# taint-tracker, it catches the actual leak shape this project's own
# secrets take: a variable or literal whose name/content looks like a
# credential, passed directly as an argument to logger.*()/
# span.set_attribute().

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

_LOG_CALL = re.compile(r"\b(logger\.\w+|span\.set_attribute)\s*\(")

# Case-insensitive substrings that suggest a credential is present in the
# call's arguments. Deliberately narrow (not "key" alone — too many false
# positives: chunk_id keys, dict keys, S3 keys) to keep this a real signal.
_SUSPECT_SUBSTRINGS = ("api_key", "_password", "_secret", "authorization", "bearer ")


def _iter_src_files():
    return sorted(_SRC_ROOT.rglob("*.py"))


def test_no_source_file_logs_or_traces_a_credential_shaped_value():
    violations: list[str] = []
    for path in _iter_src_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _LOG_CALL.search(line):
                continue
            lowered = line.lower()
            for substring in _SUSPECT_SUBSTRINGS:
                if substring in lowered:
                    violations.append(f"{path.relative_to(_SRC_ROOT.parent)}:{lineno}: {line.strip()}")

    assert not violations, (
        "possible credential passed to a logger/span call — move it out of "
        "the log line or redact it before merging:\n" + "\n".join(violations)
    )


def test_the_check_itself_actually_catches_a_real_leak(tmp_path, monkeypatch):
    # A test with no way to fail is not a real regression guard — this
    # proves the regex/substring logic genuinely flags a planted leak,
    # not just that the real (currently clean) src/ tree passes by luck.
    leaky_file = tmp_path / "leaky.py"
    leaky_file.write_text('logger.info(f"using api_key={api_key}")\n', encoding="utf-8")

    monkeypatch.setattr(sys.modules[__name__], "_SRC_ROOT", tmp_path)
    violations = []
    for path in _iter_src_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not _LOG_CALL.search(line):
                continue
            lowered = line.lower()
            if any(s in lowered for s in _SUSPECT_SUBSTRINGS):
                violations.append(f"{path}:{lineno}")
    assert violations, "the planted leak should have been caught"
