"""
crawl4ai_mcp.py — patched for 100% local operation:
  - Embeddings via Ollama
  - Storage via PostgREST → PostgreSQL+pgvector (Supabase-compatible)
  - No OpenAI dependency
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from urllib.parse import urlparse, urljoin

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from mcp.server.fastmcp import FastMCP, Context
from utils import (
    PostgRESTClient,
    chunk_text,
    store_page_chunks,
    perform_rag_query_util,
    get_available_sources_util,
    extract_source_id,
)

# ─── Config ───────────────────────────────────────────────────────────────────

USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
USE_RERANKING = os.getenv("USE_RERANKING", "false").lower() == "true"

SUPABASE_URL = os.getenv("SUPABASE_URL", "http://localhost:3000")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

MAX_CRAWL_DEPTH = int(os.getenv("MAX_CRAWL_DEPTH", "3"))
MAX_CONCURRENT_CRAWLS = int(os.getenv("MAX_CONCURRENT_CRAWLS", "5"))

# ─── Reranking (local model, no API needed) ───────────────────────────────────

reranker = None
if USE_RERANKING:
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Reranker loaded")
    except Exception as e:
        print(f"⚠️  Reranker not available: {e}")


def rerank_results(query: str, results: list[dict]) -> list[dict]:
    if not reranker or not results:
        return results
    pairs = [(query, r["content"]) for r in results]
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)


# ─── MCP lifespan ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[dict]:
    supabase = PostgRESTClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    browser_config = BrowserConfig(headless=True, verbose=False)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        yield {"supabase": supabase, "crawler": crawler}


mcp = FastMCP(
    "verse-rag",
    description="Local RAG over Verse/UEFN documentation — powered by Ollama + pgvector",
    lifespan=app_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051")),
)

# Keep references to background tasks so they aren't garbage-collected
_background_tasks: set[asyncio.Task] = set()
# Limit concurrent Ollama embedding requests to 1 — it can only handle one at a time
_embed_sem = asyncio.Semaphore(1)


async def _store_page_async(supabase, url, chunks, title):
    """Serialize embedding requests through a semaphore, then store in a thread."""
    async with _embed_sem:
        try:
            n = await asyncio.to_thread(store_page_chunks, supabase, url, chunks, title)
            print(f"✅ Background: stored {n} chunks for {url}")
        except Exception as e:
            print(f"❌ Background: failed to store chunks for {url}: {e}")

# ─── Tools ────────────────────────────────────────────────────────────────────


@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """Crawl a single page and store it in the RAG database."""
    crawler: AsyncWebCrawler = ctx.request_context.lifespan_context["crawler"]
    supabase: PostgRESTClient = ctx.request_context.lifespan_context["supabase"]

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        wait_until="domcontentloaded",
        delay_before_return_html=3.0,
        page_timeout=30000,
    )
    result = await crawler.arun(url=url, config=run_config)

    if not result.success or not result.markdown:
        return f"❌ Failed to crawl {url}: {getattr(result, 'error_message', 'unknown error')}"

    content = result.markdown.raw_markdown
    title = result.metadata.get("title", "") if result.metadata else ""
    chunks = chunk_text(content)

    # Fire-and-forget: embedding + DB insert run in background so MCP call returns fast
    task = asyncio.create_task(
        _store_page_async(supabase, url, chunks, title)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return (
        f"⏳ Crawled {url} — {len(chunks)} chunks queued for embedding in background\n"
        f"   Source: {extract_source_id(url)}\n"
        f"   Check server logs for completion status."
    )


async def _crawl_site_background(
    crawler: "AsyncWebCrawler",
    supabase: "PostgRESTClient",
    url: str,
    max_depth: int,
    max_pages: int,
) -> None:
    """Background coroutine: crawl a site recursively and store all pages."""
    visited: set[str] = set()
    to_visit: list[tuple[str, int]] = [(url, 0)]
    total_chunks = 0
    pages_crawled = 0
    base_domain = urlparse(url).netloc
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False,
        wait_until="domcontentloaded",
        delay_before_return_html=3.0,
        page_timeout=30000,
    )
    sem = asyncio.Semaphore(MAX_CONCURRENT_CRAWLS)

    async def crawl_page(page_url: str, depth: int) -> list[str]:
        nonlocal total_chunks, pages_crawled
        async with sem:
            result = await crawler.arun(url=page_url, config=run_config)
        if not result.success or not result.markdown:
            return []

        content = result.markdown.raw_markdown
        title = result.metadata.get("title", "") if result.metadata else ""
        chunks = chunk_text(content)

        embed_task = asyncio.create_task(
            _store_page_async(supabase, page_url, chunks, title)
        )
        _background_tasks.add(embed_task)
        embed_task.add_done_callback(_background_tasks.discard)

        total_chunks += len(chunks)
        pages_crawled += 1
        print(f"📄 [{pages_crawled}/{max_pages}] Crawled {page_url} ({len(chunks)} chunks)")

        if depth < max_depth and result.links:
            internal = []
            for link in result.links.get("internal", []):
                href = link.get("href", "")
                if not href:
                    continue
                absolute = urljoin(page_url, href).split("#")[0]
                if urlparse(absolute).netloc == base_domain and absolute not in visited:
                    internal.append(absolute)
            return internal
        return []

    while to_visit and pages_crawled < max_pages:
        batch = []
        while to_visit and len(batch) < MAX_CONCURRENT_CRAWLS:
            page_url, depth = to_visit.pop(0)
            if page_url in visited:
                continue
            visited.add(page_url)
            batch.append((page_url, depth))
        if not batch:
            break
        results = await asyncio.gather(*[crawl_page(u, d) for u, d in batch], return_exceptions=True)
        for (_, depth), new_links in zip(batch, results):
            if isinstance(new_links, list) and depth + 1 <= max_depth:
                for link in new_links:
                    if link not in visited and pages_crawled < max_pages:
                        to_visit.append((link, depth + 1))

    source = extract_source_id(url)
    print(f"✅ Crawl complete: {pages_crawled} pages, ~{total_chunks} chunks queued — source: {source}")


@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 2, max_pages: int = 50) -> str:
    """
    Intelligently crawl a URL recursively, following internal links.
    Perfect for indexing an entire documentation section.
    """
    crawler: AsyncWebCrawler = ctx.request_context.lifespan_context["crawler"]
    supabase: PostgRESTClient = ctx.request_context.lifespan_context["supabase"]

    task = asyncio.create_task(
        _crawl_site_background(crawler, supabase, url, max_depth, max_pages)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    source = extract_source_id(url)
    return (
        f"⏳ Started crawling {url} in background (max_depth={max_depth}, max_pages={max_pages})\n"
        f"   Source: {source}\n"
        f"   Follow progress with: docker logs verse-rag-mcp -f"
    )


@mcp.tool()
async def perform_rag_query(
    ctx: Context,
    query: str,
    source_filter: str | None = None,
    match_count: int = 10,
) -> str:
    """
    Search the RAG database with a natural language query.
    Optionally filter by source_id (e.g. 'dev.epicgames.com/documentation').
    """
    supabase: PostgRESTClient = ctx.request_context.lifespan_context["supabase"]

    results = await asyncio.to_thread(
        perform_rag_query_util,
        supabase,
        query,
        source_filter,
        match_count,
        USE_HYBRID_SEARCH,
    )

    if not results:
        return "No results found. Have you crawled any documentation yet? Use smart_crawl_url first."

    if USE_RERANKING:
        results = rerank_results(query, results)

    output_parts = [f"## RAG Results for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("similarity", 0))
        output_parts.append(
            f"### [{i}] {r.get('title', 'Untitled')} (score: {score:.3f})\n"
            f"**URL**: {r['url']} (chunk {r['chunk_number']})\n"
            f"**Source**: {r.get('source_id', 'unknown')}\n\n"
            f"{r['content'][:800]}{'...' if len(r['content']) > 800 else ''}\n"
        )

    return "\n---\n".join(output_parts)


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """List all documentation sources currently indexed in the RAG database."""
    supabase: PostgRESTClient = ctx.request_context.lifespan_context["supabase"]
    sources = await asyncio.to_thread(get_available_sources_util, supabase)

    if not sources:
        return "No sources indexed yet. Use smart_crawl_url to start crawling documentation."

    lines = ["## Indexed Sources\n"]
    for s in sources:
        lines.append(f"- `{s}`")
    return "\n".join(lines)


@mcp.tool()
async def crawl_verse_docs(ctx: Context) -> str:
    """
    Shortcut: crawl the official Verse/UEFN documentation from Epic Games.
    Equivalent to calling smart_crawl_url on the Verse reference pages.
    """
    urls = [
        "https://dev.epicgames.com/documentation/en-US/uefn/verse-language-reference",
        "https://dev.epicgames.com/documentation/en-US/uefn/verse-api",
    ]

    results = []
    for url in urls:
        result = await smart_crawl_url(ctx, url, max_depth=2, max_pages=100)
        results.append(result)

    return "\n\n".join(results)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.getenv("TRANSPORT", "sse")
    mcp.run(transport=transport)
