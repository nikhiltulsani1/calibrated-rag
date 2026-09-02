import hashlib

import pytest

from src.platform.cache import build_cache_key

pytestmark = pytest.mark.unit

# build_cache_key — promoted 2026-08-25 (full-codebase review) out of two
# independent hand-rolled copies (query_planner.py's plan cache,
# reason/answer_cache.py's answer cache) doing the identical
# "join parts with |, sha256, prefix" pattern.


def test_single_part_matches_the_original_query_planner_scheme():
    # query_planner.py's old scheme: prefix + sha256(query).hexdigest()
    # — must produce byte-identical output so any already-cached plan
    # entries in Redis stay reachable after this refactor.
    expected = "query_plan:" + hashlib.sha256(b"what is rlhf").hexdigest()
    assert build_cache_key("query_plan:", "what is rlhf") == expected


def test_multi_part_matches_the_original_answer_cache_scheme():
    # answer_cache.py's old scheme: prefix + sha256(f"{q}|{strategy}|{provider}")
    # — same byte-identical requirement.
    raw = "what is rlhf|default|jina"
    expected = "answer_cache:" + hashlib.sha256(raw.encode()).hexdigest()
    assert build_cache_key("answer_cache:", "what is rlhf", "default", "jina") == expected


def test_different_parts_produce_different_keys():
    assert build_cache_key("p:", "a") != build_cache_key("p:", "b")
    assert build_cache_key("p:", "a", "x") != build_cache_key("p:", "a", "y")


def test_different_prefix_produces_a_different_key_for_the_same_parts():
    assert build_cache_key("p1:", "a") != build_cache_key("p2:", "a")
