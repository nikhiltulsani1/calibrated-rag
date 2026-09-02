from __future__ import annotations

import os

from opensearchpy import OpenSearch

from src.index.mapping import INDEX_NAME, build_index_body


def get_client() -> OpenSearch:
    host = os.environ.get("OPENSEARCH_HOST", "localhost")
    port = int(os.environ.get("OPENSEARCH_PORT", "9200"))
    password = os.environ["OPENSEARCH_ADMIN_PASSWORD"]
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=("admin", password),
        use_ssl=True,
        # Dev-only self-signed cert (compose.yml sets no custom CA). Do not
        # carry verify_certs=False into any config reachable from outside
        # localhost.
        verify_certs=False,
        ssl_show_warn=False,
        timeout=30,
    )


def create_index(client: OpenSearch | None = None, index_name: str = INDEX_NAME, dimension: int | None = None) -> bool:
    """Idempotent: returns False if the index already existed.

    `dimension` lets a caller building a NON-default-embed-provider index
    (see src/index/embed_toggle.py) pass that provider's real vector width
    instead of the locked default's — added for the Mistral-embed
    alternative index, whose vectors (1024-d) don't match Jina's locked
    dimension. Every existing caller omits it and gets today's behavior
    unchanged (build_index_body() falls back to locked_embed_dimension()).
    """
    client = client or get_client()
    if client.indices.exists(index=index_name):
        return False
    client.indices.create(index=index_name, body=build_index_body(dimension=dimension))
    return True


if __name__ == "__main__":
    created = create_index()
    print(f"index {'created' if created else 'already existed'}: {INDEX_NAME}")
