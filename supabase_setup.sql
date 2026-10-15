-- ============================================================================
-- Supabase SQL Setup for SEC-Analyzer RAG Project
-- ============================================================================
-- Enable pgvector extension for vector operations
create extension if not exists vector;

-- ============================================================================
-- Main chunks table for SEC 10-K filings with embeddings
-- ============================================================================
create table if not exists public.chunks (
  id bigserial primary key,
  doc_id text not null,                    -- e.g., "AAPL_10-K_2023"
  chunk_index text not null,               -- e.g., "section_1.2_row0"
  content text not null,                   -- actual text/table row content
  metadata jsonb default '{}'::jsonb,      -- {company, year, item, is_table}
  embedding vector(768),                   -- MPNet-base-v2 produces 768 dims
  created_at timestamp default now(),
  updated_at timestamp default now()
);

-- Create unique index on doc_id + chunk_index to prevent duplicates
create unique index if not exists idx_chunks_doc_chunk 
  on public.chunks(doc_id, chunk_index);

-- Create index on metadata fields for filtering (useful for company/year searches)
create index if not exists idx_chunks_metadata_company
  on public.chunks using gin(metadata);

-- ============================================================================
-- IVFFlat Index for Fast ANN (Approximate Nearest Neighbor) Search
-- ============================================================================
-- This index enables sub-second similarity searches on large datasets
-- 'lists = 100' is good for datasets up to 1M vectors. Adjust if needed:
--   - Small datasets (< 100k): lists = 10-50
--   - Medium datasets (100k-1M): lists = 50-100
--   - Large datasets (> 1M): lists = 100-200
create index if not exists idx_chunks_embedding_ivfflat
  on public.chunks using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- ============================================================================
-- Match Documents Function - Similarity Search with Cosine Distance
-- ============================================================================
-- Purpose: Find the most relevant chunks to a query embedding
-- Usage: select * from match_documents(your_query_embedding, 5)
create or replace function public.match_documents(
  query_embedding vector(768),
  match_count int default 5,
  company_filter text default null,
  year_filter int default null,
  min_similarity float default 0.0
)
returns table (
  id bigint,
  doc_id text,
  chunk_index text,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
stable
as $$
begin
  return query
  select
    c.id,
    c.doc_id,
    c.chunk_index,
    c.content,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.chunks c
  where
    -- Optional company filter
    (company_filter is null or c.metadata->>'company' = company_filter)
    -- Optional year filter
    and (year_filter is null or (c.metadata->>'year')::int = year_filter)
    -- Min similarity threshold (0 = no threshold, 1 = perfect match only)
    and (1 - (c.embedding <=> query_embedding)) >= min_similarity
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- ============================================================================
-- Advanced Search - Find Similar Chunks Plus Metadata Info
-- ============================================================================
-- Returns additional useful context for your RAG pipeline
create or replace function public.match_documents_detailed(
  query_embedding vector(768),
  match_count int default 5,
  company_filter text default null
)
returns table (
  id bigint,
  doc_id text,
  chunk_index text,
  content text,
  company text,
  year int,
  item text,
  is_table boolean,
  similarity float
)
language plpgsql
stable
as $$
begin
  return query
  select
    c.id,
    c.doc_id,
    c.chunk_index,
    c.content,
    c.metadata->>'company' as company,
    (c.metadata->>'year')::int as year,
    c.metadata->>'item' as item,
    (c.metadata->>'is_table')::boolean as is_table,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.chunks c
  where (company_filter is null or c.metadata->>'company' = company_filter)
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- ============================================================================
-- Metadata Statistics - Useful for UI/Analytics
-- ============================================================================
create or replace function public.get_filing_stats()
returns table (
  company text,
  total_chunks bigint,
  years text
)
language plpgsql
stable
as $$
begin
  return query
  select
    c.metadata->>'company' as company,
    count(*)::bigint as total_chunks,
    string_agg(distinct c.metadata->>'year', ', ' order by c.metadata->>'year') as years
  from public.chunks c
  group by c.metadata->>'company'
  order by company;
end;
$$;

-- ============================================================================
-- Enable RLS (Row Level Security) - Optional but Recommended
-- ============================================================================
-- Uncomment if you want to restrict access by user/role
-- alter table public.chunks enable row level security;
-- create policy "Enable read access for all users" on public.chunks
--   for select using (true);
-- create policy "Enable insert for authenticated users" on public.chunks
--   for insert with check (true);

-- ============================================================================
-- Grant Permissions (if using service role)
-- ============================================================================
grant select, insert, update on public.chunks to anon, authenticated;
grant execute on function public.match_documents to anon, authenticated;
grant execute on function public.match_documents_detailed to anon, authenticated;
grant execute on function public.get_filing_stats to anon, authenticated;
