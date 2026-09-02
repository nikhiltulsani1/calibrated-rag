from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.app.main import app

pytestmark = pytest.mark.unit


def test_post_chunking_strategy_sets_it_and_redirects():
    client = TestClient(app, follow_redirects=False)
    with patch("src.app.routes.corpus.set_active_strategy") as mock_set:
        response = client.post("/corpus/chunking-strategy", data={"strategy": "winner"})
    mock_set.assert_called_once_with("winner")
    assert response.status_code == 303
    assert response.headers["location"] == "/corpus"


def test_corpus_page_shows_not_run_yet_without_results_file():
    client = TestClient(app)
    with patch("src.app.routes.corpus._CHUNKING_RESULTS_PATH") as mock_path:
        mock_path.exists.return_value = False
        response = client.get("/corpus")
    assert response.status_code == 200
    assert b"hasn&#39;t been run yet" in response.content or b"hasn't been run yet" in response.content
