"""
utils.py — Ollama embeddings + direct PostgREST via httpx (no supabase-py client).
"""

import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

# ─── Embedding via Ollama ─────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


def get_embedding(text: str) -> list[float]:
    """Generate a single embedding using local Ollama."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts in a single Ollama request."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=300.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> list[str]:
    """
    Split text by markdown headers first, then by size with overlap.
    Returns a list of text chunks.
    """
    header_pattern = re.compile(r"(?=^#{1,4}\s)", re.MULTILINE)
    sections = header_pattern.split(text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            start = 0
            while start < len(section):
                end = start + max_chars
                chunks.append(section[start:end])
                start += max_chars - overlap

    return chunks


# ─── Source ID extraction ─────────────────────────────────────────────────────

def extract_source_id(url: str) -> str:
    """Extract a clean source identifier from a URL (domain + first path segment)."""
    parsed = urlparse(url)
    host = parsed.netloc.replace("www.", "")
    path_parts = [p for p in parsed.path.split("/") if p]
    if path_parts:
        return f"{host}/{path_parts[0]}"
    return host


# ─── Direct PostgREST client (replaces supabase-py) ──────────────────────────

class PostgRESTClient:
    """Minimal PostgREST HTTP client using httpx."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> list[dict]:
        r = httpx.get(
            f"{self.base_url}{path}",
            headers=self._headers,
            params=params,
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict, extra_headers: dict | None = None) -> list[dict]:
        headers = {**self._headers, **(extra_headers or {})}
        r = httpx.post(
            f"{self.base_url}{path}",
            headers=headers,
            json=body,
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []


# ─── PostgREST helpers ────────────────────────────────────────────────────────

def store_page_chunks(
    client: PostgRESTClient,
    url: str,
    chunks: list[str],
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> int:
    """Embed and store all chunks of a page into the database. Returns stored count."""
    if metadata is None:
        metadata = {}

    source_id = extract_source_id(url)

    # Filter empty chunks while preserving original index for chunk_number
    indexed_chunks = [(i, c) for i, c in enumerate(chunks) if c.strip()]
    if not indexed_chunks:
        return 0

    # Batch embed all chunks in a single Ollama request
    texts = [c for _, c in indexed_chunks]
    embeddings = get_embeddings_batch(texts)

    # Batch upsert all records in a single PostgREST request
    records = [
        {
            "url": url,
            "chunk_number": i,
            "title": title,
            "content": chunk,
            "metadata": metadata,
            "source_id": source_id,
            "embedding": embedding,
        }
        for (i, chunk), embedding in zip(indexed_chunks, embeddings)
    ]

    httpx.post(
        f"{client.base_url}/crawled_pages",
        headers={
            **client._headers,
            "Prefer": "resolution=merge-duplicates",
        },
        json=records,
        params={"on_conflict": "url,chunk_number"},
        timeout=120.0,
    ).raise_for_status()

    return len(records)


def perform_rag_query_util(
    client: PostgRESTClient,
    query: str,
    source_filter: str | None = None,
    match_count: int = 10,
    use_hybrid: bool = False,
) -> list[dict[str, Any]]:
    """Run a semantic (or hybrid) search against crawled pages."""
    embedding = get_embedding(query)

    if use_hybrid:
        body = {
            "query_embedding": embedding,
            "query_text": query,
            "match_count": match_count,
            "filter_source": source_filter,
        }
        return client._post("/rpc/hybrid_search_crawled_pages", body)
    else:
        body = {
            "query_embedding": embedding,
            "match_count": match_count,
            "filter_source": source_filter,
        }
        return client._post("/rpc/match_crawled_pages", body)


def get_available_sources_util(client: PostgRESTClient) -> list[str]:
    """Return distinct source_ids currently in the database."""
    rows = client._get("/crawled_pages", params={"select": "source_id"})
    seen: set[str] = set()
    sources = []
    for row in rows:
        s = row.get("source_id")
        if s and s not in seen:
            seen.add(s)
            sources.append(s)
    return sorted(sources)
