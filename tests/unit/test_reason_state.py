from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def test_fetch_metadata_empty_candidates_short_circuits():
    from src.reason.state import fetch_metadata

    with patch("src.reason.state.get_client") as mock_client:
        result = fetch_metadata([])
    assert result == {}
    assert not mock_client.called
