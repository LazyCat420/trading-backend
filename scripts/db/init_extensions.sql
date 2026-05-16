-- PostgreSQL extensions for vllm-trading-bot
-- Loaded automatically on first DB creation via docker-entrypoint-initdb.d

-- pgvector: vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_trgm: trigram-based fuzzy text matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- btree_gist: GiST index support for scalar types (enables exclusion constraints)
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Allow connections from the local network (10.0.0.0/24)
-- This is handled by pg_hba.conf but we document the intent here.
-- The actual pg_hba.conf entry is set via Docker environment.
