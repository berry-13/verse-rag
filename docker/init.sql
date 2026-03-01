-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Roles for PostgREST ──────────────────────────────────────────────────────
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'anon') THEN
    CREATE ROLE anon NOLOGIN;
  END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'authenticated') THEN
    CREATE ROLE authenticated NOLOGIN;
  END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO anon, authenticated;

-- ─── Main crawled pages table ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS crawled_pages (
    id          BIGSERIAL PRIMARY KEY,
    url         TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    title       TEXT,
    summary     TEXT,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    source_id   TEXT,                         -- domain / source label
    embedding   vector(4096),                  -- qwen3-embedding:8b = 4096 dims
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (url, chunk_number)
);

-- Note: ivfflat/hnsw indexes are limited to 2000 dims; vector(4096) uses sequential scan
-- Index for source filtering
CREATE INDEX IF NOT EXISTS crawled_pages_source_idx ON crawled_pages (source_id);

-- Full-text search index for hybrid search
CREATE INDEX IF NOT EXISTS crawled_pages_fts_idx
    ON crawled_pages USING gin(to_tsvector('english', content));

-- ─── Code examples table (for agentic RAG, kept for future use) ──────────────
CREATE TABLE IF NOT EXISTS code_examples (
    id          BIGSERIAL PRIMARY KEY,
    url         TEXT NOT NULL,
    chunk_number INTEGER NOT NULL,
    summary     TEXT,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    source_id   TEXT,
    embedding   vector(768),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (url, chunk_number)
);

CREATE INDEX IF NOT EXISTS code_examples_embedding_idx
    ON code_examples USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- ─── Vector similarity search function ───────────────────────────────────────
CREATE OR REPLACE FUNCTION match_crawled_pages(
    query_embedding vector(4096),
    match_count     INTEGER DEFAULT 10,
    filter_source   TEXT DEFAULT NULL
)
RETURNS TABLE (
    id          BIGINT,
    url         TEXT,
    chunk_number INTEGER,
    title       TEXT,
    summary     TEXT,
    content     TEXT,
    metadata    JSONB,
    source_id   TEXT,
    similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        cp.id,
        cp.url,
        cp.chunk_number,
        cp.title,
        cp.summary,
        cp.content,
        cp.metadata,
        cp.source_id,
        1 - (cp.embedding <=> query_embedding) AS similarity
    FROM crawled_pages cp
    WHERE
        cp.embedding IS NOT NULL
        AND (filter_source IS NULL OR cp.source_id = filter_source)
    ORDER BY cp.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ─── Hybrid search function (vector + full-text) ──────────────────────────────
CREATE OR REPLACE FUNCTION hybrid_search_crawled_pages(
    query_embedding vector(4096),
    query_text      TEXT,
    match_count     INTEGER DEFAULT 10,
    filter_source   TEXT DEFAULT NULL,
    vector_weight   FLOAT DEFAULT 0.7,
    text_weight     FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    id          BIGINT,
    url         TEXT,
    chunk_number INTEGER,
    title       TEXT,
    summary     TEXT,
    content     TEXT,
    metadata    JSONB,
    source_id   TEXT,
    similarity  FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT
            cp.id,
            1 - (cp.embedding <=> query_embedding) AS vec_score
        FROM crawled_pages cp
        WHERE
            cp.embedding IS NOT NULL
            AND (filter_source IS NULL OR cp.source_id = filter_source)
        ORDER BY cp.embedding <=> query_embedding
        LIMIT match_count * 2
    ),
    text_results AS (
        SELECT
            cp.id,
            ts_rank(to_tsvector('english', cp.content), plainto_tsquery('english', query_text)) AS txt_score
        FROM crawled_pages cp
        WHERE
            to_tsvector('english', cp.content) @@ plainto_tsquery('english', query_text)
            AND (filter_source IS NULL OR cp.source_id = filter_source)
        LIMIT match_count * 2
    ),
    combined AS (
        SELECT
            COALESCE(v.id, t.id) AS id,
            COALESCE(v.vec_score, 0) * vector_weight + COALESCE(t.txt_score, 0) * text_weight AS combined_score
        FROM vector_results v
        FULL OUTER JOIN text_results t ON v.id = t.id
    )
    SELECT
        cp.id,
        cp.url,
        cp.chunk_number,
        cp.title,
        cp.summary,
        cp.content,
        cp.metadata,
        cp.source_id,
        c.combined_score AS similarity
    FROM combined c
    JOIN crawled_pages cp ON cp.id = c.id
    ORDER BY c.combined_score DESC
    LIMIT match_count;
END;
$$;

-- Grants
GRANT SELECT, INSERT, UPDATE, DELETE ON crawled_pages TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON code_examples TO anon, authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT EXECUTE ON FUNCTION match_crawled_pages TO anon, authenticated;
GRANT EXECUTE ON FUNCTION hybrid_search_crawled_pages TO anon, authenticated;
