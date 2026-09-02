from __future__ import annotations

import os

# Single source of truth for "which retrieval backend is this process
# running" — real duplication found in review: this exact one-line check
# had drifted into 8 independent copies across the app/routes and reason/
# layers before this module existed. src/app/deps.py re-exports this same
# function (and registers it as a Jinja global) rather than defining its
# own copy, so template-layer and route-layer/reason-layer code all read
# one flag. Lives in src/platform/ (not src/app/) since reason/*.py must
# not import from the web layer — this is the layer-neutral home shared
# by both sides.


def is_postgres_backend() -> bool:
    return os.environ.get("RETRIEVAL_BACKEND", "opensearch") == "postgres"
