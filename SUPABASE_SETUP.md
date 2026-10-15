# Supabase SQL Setup Guide for SEC-Analyzer

## Key Specifications for Your Project

- **Embedding Dimension**: 768 (from `sentence-transformers/all-MPNet-base-v2`)
- **Vector Index**: IVFFlat with `lists = 100` (optimized for your data scale)
- **Distance Metric**: Cosine similarity (cosine_ops)
- **Data Schema**: Matches your main.py structure exactly

## How to Apply This SQL

### Option 1: Supabase Dashboard (Easiest)
1. Log in to [supabase.com](https://supabase.com)
2. Select your project
3. Go to **SQL Editor** → **New Query**
4. Copy & paste the entire `supabase_setup.sql` file
5. Click **Run** (or ⌘+Enter)

### Option 2: Supabase CLI
```bash
supabase db push
# Then copy the supabase_setup.sql content to your migrations folder
```

## Query Your Data After Setup

### 1. Basic Similarity Search
```sql
-- Find 5 most similar chunks to a query
select * from match_documents(
  query_embedding => '[0.1, 0.2, 0.3, ..., 0.768]'::vector,
  match_count => 5
);
```

### 2. Search with Filters (Company + Year)
```sql
select * from match_documents(
  query_embedding => your_query_vector,
  match_count => 10,
  company_filter => 'AAPL',
  year_filter => 2023
);
```

### 3. Detailed Results (Includes Parsed Metadata)
```sql
select * from match_documents_detailed(
  query_embedding => your_query_vector,
  match_count => 5,
  company_filter => 'AAPL'
);
```

### 4. Get Filing Statistics
```sql
select * from public.get_filing_stats();
```

## Integration with Your Python Code

The SQL is already compatible with `src/vectordb/vector_store.py`. No changes needed! Your current upload flow works as-is:

```python
# In main.py - this line now uploads to the Supabase table
db = VectorDB()
db.upload_chunks(final_rows)  # Sends to public.chunks automatically
```

## Performance Tips

- **IVFFlat Index**: Provides ~100x speedup for similarity search vs. full table scan
- **Metadata Filters**: Use `company_filter` and `year_filter` to reduce search space
- **Min Similarity Threshold**: Use `min_similarity` to filter out weak matches
- **Batch Size**: Keep `batch_size=100` in your uploader for optimal performance

## Monitoring

Check index health:
```sql
-- View index size and stats
select schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
from pg_stat_user_indexes
where tablename = 'chunks';
```

Check for duplicate chunks:
```sql
select doc_id, chunk_index, count(*) as dupes
from public.chunks
group by doc_id, chunk_index
having count(*) > 1;
```

## Troubleshooting

**Q: "Vector dimension mismatch" error?**
- Ensure embeddings are exactly 768 dimensions
- Check your embedder isn't using a different model

**Q: Slow queries despite index?**
- Run `REINDEX INDEX idx_chunks_embedding_ivfflat;`
- Increase `lists` parameter if you have > 1M vectors

**Q: Memory issues?**
- The IVFFlat index uses ~2-3GB of RAM for 1M vectors
- Upgrade your Supabase plan if needed

## Next Steps

1. Apply this SQL to your Supabase project
2. Update your `.env` file with `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
3. Run `main.py` to start uploading chunks
4. Test similarity search in SQL Editor
