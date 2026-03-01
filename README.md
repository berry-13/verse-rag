<h1 align="center">verse-rag</h1>

<p align="center">
  <strong>Local-first RAG MCP Server for Verse / UEFN Documentation</strong><br>
  <em>Fully air-gapped · Ollama embeddings · pgvector · Model Context Protocol</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MCP-SSE-blue" />
  <img src="https://img.shields.io/badge/embeddings-Ollama-green" />
  <img src="https://img.shields.io/badge/storage-pgvector-purple" />
  <img src="https://img.shields.io/badge/python-3.12+-yellow" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" />
</p>

---

## Overview

**verse-rag** is a production-ready [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that gives AI coding assistants the ability to crawl, index, and semantically search documentation — entirely on local infrastructure.

Originally forked from [coleam00/mcp-crawl4ai-rag](https://github.com/coleam00/mcp-crawl4ai-rag), this fork is a ground-up rewrite of the storage and embedding layers:

| Upstream | This fork |
|---|---|
| OpenAI embeddings | Ollama (`qwen3-embedding:8b`, 4096 dims) |
| Supabase cloud | Self-hosted pgvector + PostgREST |
| `supabase-py` client | Direct `httpx` calls to PostgREST |
| Blocking embed + store | Fire-and-forget background tasks |

The result is a stack that runs entirely offline with no external API calls, no cloud dependencies, and no API costs.

---

## Architecture

```
Claude Code / AI Client
        │  SSE (port 8051)
        ▼
  ┌─────────────┐
  │  MCP Server │  crawl4ai_mcp.py — FastMCP + Crawl4AI
  └──────┬──────┘
         │ httpx
         ▼
  ┌──────────────┐       ┌─────────────────┐
  │  PostgREST   │──────▶│  PostgreSQL 16  │
  │  (REST API)  │       │  + pgvector     │
  └──────────────┘       └─────────────────┘
         │
         │ /api/embed
         ▼
  ┌─────────────┐
  │   Ollama    │  qwen3-embedding:8b (4096 dims)
  └─────────────┘
```

**Services (Docker Compose):**

| Container | Image | Role |
|---|---|---|
| `verse-rag-db` | `pgvector/pgvector:pg16` | Vector store |
| `verse-rag-postgrest` | `postgrest/postgrest:v12.2.0` | REST API over PostgreSQL |
| `verse-rag-mcp` | `verse-rag:latest` (local build) | MCP server |

Ollama runs as a separate service on the host (or in another container) — this stack connects to it over the `ollama` Docker network.

---

## MCP Tools

| Tool | Description |
|---|---|
| `crawl_single_page` | Crawl one URL and queue it for indexing |
| `smart_crawl_url` | Recursively crawl a site, following internal links |
| `crawl_verse_docs` | Shortcut to crawl the official Verse/UEFN documentation |
| `perform_rag_query` | Semantic (or hybrid) search with optional source filtering |
| `get_available_sources` | List all indexed sources in the database |

All crawl tools return immediately — embedding and storage run in background tasks so the MCP call never blocks waiting for Ollama.

---

## Prerequisites

- **Docker** and **Docker Compose**
- **Ollama** running with `qwen3-embedding:8b` pulled:
  ```bash
  ollama pull qwen3-embedding:8b
  ```
- Three external Docker networks: `mcps`, `nginx`, `ollama`
  ```bash
  docker network create mcps
  docker network create nginx
  docker network create ollama
  ```

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/berry-13/verse-rag.git
cd verse-rag
```

### 2. Build the MCP image

```bash
docker build -f docker/Dockerfile -t verse-rag:latest .
```

### 3. Start the stack

```bash
docker compose -f docker/compose.yml up -d
```

This starts:
- PostgreSQL with pgvector (`verse-rag-db`)
- PostgREST auto-REST layer (`verse-rag-postgrest`)
- The MCP server on port `8051` (`verse-rag-mcp`)

### 4. Connect your MCP client

**Claude Code:**
```bash
claude mcp add-json verse-rag '{"type":"sse","url":"http://localhost:8051/sse"}' --scope user
```

**Any SSE-compatible client:**
```json
{
  "mcpServers": {
    "verse-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

---

## Configuration

All configuration is via environment variables. The defaults in `docker/compose.yml` are ready to use for a standard local setup.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama endpoint |
| `EMBEDDING_MODEL` | `qwen3-embedding:8b` | Ollama embedding model |
| `SUPABASE_URL` | `http://postgrest:3000` | PostgREST endpoint |
| `SUPABASE_SERVICE_KEY` | *(JWT in compose.yml)* | PostgREST JWT token |
| `USE_HYBRID_SEARCH` | `true` | Combine vector + full-text search |
| `USE_RERANKING` | `true` | Cross-encoder reranking of results |
| `HOST` | `0.0.0.0` | MCP server bind address |
| `PORT` | `8051` | MCP server port |
| `TRANSPORT` | `sse` | MCP transport (`sse` or `stdio`) |
| `MAX_CRAWL_DEPTH` | `3` | Maximum recursive crawl depth |
| `MAX_CONCURRENT_CRAWLS` | `5` | Parallel crawl workers |

---

## RAG Strategies

### Hybrid Search (`USE_HYBRID_SEARCH=true`)

Combines pgvector cosine similarity with PostgreSQL full-text search (`tsvector`). Results are scored as a weighted combination (`0.7` vector + `0.3` text) using a `FULL OUTER JOIN` in the `hybrid_search_crawled_pages` SQL function.

Best for technical documentation where exact term matches (function names, keywords) matter alongside semantic similarity.

### Reranking (`USE_RERANKING=true`)

After initial retrieval, applies a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score and reorder results against the original query. Runs locally on CPU with no API cost. Adds ~100–200 ms to query latency in exchange for meaningfully better result ordering.

---

## Database Schema

The `init.sql` creates:

- **`crawled_pages`** — main table with `vector(4096)` embeddings (matched to `qwen3-embedding:8b`), source labelling, and a `UNIQUE(url, chunk_number)` constraint for idempotent upserts
- **`match_crawled_pages()`** — vector similarity search function
- **`hybrid_search_crawled_pages()`** — combined vector + full-text search function
- GIN index on `content` for full-text search
- B-tree index on `source_id` for source filtering

> **Note:** `ivfflat`/`hnsw` indexes are limited to 2000 dimensions. At 4096 dims, queries use a sequential scan — acceptable for documentation-scale datasets.

---

## Performance Notes

- **Batch embedding:** all chunks from a page are embedded in a single `/api/embed` call to Ollama
- **Batch upsert:** all records are written in a single `POST` to PostgREST with `Prefer: resolution=merge-duplicates`
- **Semaphore:** embedding requests are serialized through `asyncio.Semaphore(1)` since Ollama processes one embedding job at a time
- **Fire-and-forget:** crawl tools return immediately; background `asyncio.Task` handles embedding + storage. Monitor progress with:
  ```bash
  docker logs verse-rag-mcp -f
  ```

---

## Development

### Rebuild after code changes

Only the last Docker layer (`COPY src/`) is invalidated on source changes, so rebuilds are fast:

```bash
docker build -f docker/Dockerfile -t verse-rag:latest . && \
docker compose -f docker/compose.yml up -d mcp
```

### Project structure

```
verse-rag/
├── src/
│   ├── crawl4ai_mcp.py   # MCP server, tool definitions, lifespan
│   └── utils.py          # Embeddings, chunking, PostgREST client
├── docker/
│   ├── Dockerfile        # Build definition
│   ├── compose.yml       # Full stack definition
│   └── init.sql          # PostgreSQL schema + functions
├── knowledge_graphs/     # Optional Neo4j hallucination detection
└── pyproject.toml
```

---

## Troubleshooting

**CUDA unavailable / Ollama SIGABRT**

The GPU may be in `Exclusive_Process` mode. Reset it:
```bash
sudo nvidia-smi -c 0
```
This resets on reboot and must be re-applied after driver reloads.

**MCP tools hang after container restart**

Claude Code caches SSE session IDs. After restarting the MCP container, restart Claude Code to clear the stale session before using any tools.

**Embeddings failing silently**

Check background task output:
```bash
docker logs verse-rag-mcp -f | grep -E "✅|❌"
```

---

## License

MIT
