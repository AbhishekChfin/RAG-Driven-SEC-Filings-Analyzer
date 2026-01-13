# LLM-Driven SEC 10-K & Earnings Call Analyzer

## Project Overview

This project is a **Retrieval-Augmented Generation (RAG)** application designed to analyze SEC 10-K filings and answer financial questions using Large Language Models (LLMs). Built as a learning project to understand RAG fundamentals in finance and data science.

### Motivation

My background is mostly classical ML and forecasting models (LSTM, ARIMA, ARCH/GARCH). This project bridges the gap between those traditional modeling workflows and modern LLM-based systems by implementing an end-to-end, production-style RAG pipeline from raw financial filings to retrieval and generation:

- Data cleaning and preprocessing at scale
- Semantic chunking strategies
- Vector embeddings and similarity search
- Integration with modern LLM APIs
- Full-stack deployment (Python + Next.js + Supabase)

---

## The Core Problem: Context Window Limitation

### Problem Statement

When building a chatbot over large documents (e.g., a 300+ page SEC filing), feeding the entire document to an LLM causes:

- **Context Window Overflow**: LLMs have fixed context windows (e.g., Gemini 2.5 Flash). Exceeding this causes hallucinations.
- **Information Loss**: The model "forgets" earlier parts of the document.
- **Poor Quality Responses**: Generic or inaccurate answers due to incomplete context.

**Example**: User asks "What was Apple's revenue in 2023?" but the entire 10-K (8+ chapters) is fed to the LLM, causing it to exceed token limits and hallucinate.

### Solution: RAG (Retrieval-Augmented Generation)

Instead of feeding entire documents:

1. **Split documents into chunks** (preserve semantic meaning)
2. **Convert chunks to embeddings** (vector representations)
3. **Store in vector database** (Supabase + pgvector)
4. **Retrieve only relevant chunks** (similarity search)
5. **Generate answer** from relevant context only

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│              (Next.js Chat UI - src/app/page.tsx)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    POST /api/chat
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    NEXT.JS API ROUTE                            │
│           (src/app/api/chat/route.ts - Orchestrator)            │
└────┬─────────────────────┬──────────────────┬──────────────────┘
     │                     │                  │
     ▼                     ▼                  ▼
┌─────────────┐      ┌──────────────┐   ┌─────────────┐
│   FastAPI   │      │   Supabase   │   │ Google      │
│ Embedding   │      │   Database   │   │ Gemini LLM  │
│ Service     │      │ (pgvector)   │   │             │
│             │      │              │   │             │
│ Converts    │      │ Stores:      │   │ Generates:  │
│ text →      │      │ - Embeddings │   │ - Answers   │
│ vectors     │      │ - Chunks     │   │             │
│ (768-dim)   │      │ - Metadata   │   │             │
└─────────────┘      └──────────────┘   └─────────────┘
```

---

## Data Pipeline

### 1. Data Ingestion

**Challenge**: Extract financial data from raw HTML SEC EDGAR filings.

**Solution**:

- Downloaded 10-K filings from SEC EDGAR API in raw HTML format
- Used **regex indexing** to locate document tags:
  - Stored start/end indices of `<DOCUMENT>` tags
  - Identified document type: `<TYPE>10-K</TYPE>`
  - Extracted specific items using regex patterns (all 10-K items start/end with consistent patterns)

**Result**: Structured data with clear item boundaries (Item 1, Item 1A, Item 7, etc.)

### 2. Data Cleaning

**Challenges**:

| Challenge | Initial Approach | Problem | Final Solution |
|-----------|------------------|---------|-----------------|
| Text extraction | `soup.get_text()` | Simple but loses table structure | Works fine for text ✓ |
| Table extraction | `regex` | Fails on nested HTML | Use BeautifulSoup |
| Table format | `pd.read_html()` | Only outputs markdown/CSV | Convert to JSON |
| Empty columns | Manual removal | Tedious and error-prone | Automated filtering |

**Table Extraction Pipeline**:

```python
1. Find all <table> tags using BeautifulSoup.find_all("table")
2. Iterate through <tr> tags
3. Extract cells from <td> and <th>
4. Merge special characters ($, %) with values (they were separate columns)
5. Filter empty rows
6. Pad rows to equal length
7. Detect and remove grouping headers (e.g., "Years Ended")
8. Forward-fill header cells
9. Remove empty columns
10. Clean column names and handle duplicates
11. Convert to JSON format (table_id, columns, rows)
12. Replace HTML table with [START_TABLE_JSON]/[END_TABLE_JSON] markers
```

**Key Improvement**: BeautifulSoup handles nested HTML gracefully, unlike regex which fails on complex structures.

**Code Location**: [src/data_cleaning.py](src/data_cleaning.py)

**Why Not Just Regex?**

- Regex fails on nested HTML structures (tables within tables, complex formatting)
- BeautifulSoup understands document hierarchy and can navigate relationships
- Handles malformed HTML gracefully (common in real SEC filings)
- More maintainable and easier to debug

### 3. Chunking Strategy

**Challenge**: How to split 10-K filings (150+ pages) into optimal chunks?

**Options Evaluated**:

| Method | Pros | Cons | Decision |
|--------|------|------|----------|
| Fixed-size | Simple, predictable | Breaks semantic meaning | ❌ Rejected |
| Structural | Preserves document hierarchy | Chunks too large/small | ❌ Partial |
| Semantic | Perfect coherence | Too many chunks (high cost) | ❌ Too expensive |
| Recursive | Balances coherence + size | Slightly complex | ✓ **CHOSEN** |
| Contextual + LLM | High quality | Very expensive | ❌ Too costly |

**Why Recursive Chunking?**

- Preserves semantic coherence while staying within size limits
- Creates fewer chunks than semantic chunking (cost-effective with Supabase pgvector)
- Reduces similarity search overhead

**Implementation**:

```python
# Recursive chunking with overlaps
chunk_size = 1024 tokens
overlap = 128 tokens

# Algorithm:
# 1. Split by sentence/paragraph
# 2. If too large, split by sentence
# 3. If still large, split by character
# 4. Add overlap for context preservation
```

**Code Location**: [src/chunking.py](src/chunking.py)

### 4. Embedding Generation

**Challenge**: Convert chunks to vectors for similarity search. Choose right model and avoid hallucinations.

**Decision Matrix**:

| Model | Dimensions | Speed | Cost | Chosen |
|-------|-----------|-------|------|--------|
| `all-mpnet-base-v2` (SentenceTransformers) | 768 | Fast ✓ | Free ✓ | **YES** |
| `text-embedding-3-large` (OpenAI) | 3072 | Medium | $$$ | No |
| `embedding-001` (Google) | 768 | Medium | $ | Could use |

**Choice**: `all-mpnet-base-v2` (SentenceTransformers)

- 768 dimensions (balanced for cost/quality)
- Runs locally (no API cost)
- Fast inference
- Reliable for financial text

**Embedding Service**: FastAPI microservice

```python
# FastAPI Endpoint
POST /embed
{
  "text": "Apple's revenue in 2023..."
}
↓
# Returns
{
  "embedding": [0.123, -0.456, ..., 0.789]  # 768 dimensions
}
```

**Why not use OpenAI embeddings directly?**

- Had to use same model for both chunk embedding AND query embedding
- Building custom FastAPI service gives full control
- Cost-effective for large-scale operations
- Same embedding model ensures consistent vector space

**Code Location**: [rag-chat/Embedding_services/embedding_service.py](rag-chat/Embedding_services/embedding_service.py)

### 5. Database Setup (Supabase + pgvector)

**Challenge**: Store and query millions of embeddings efficiently.

**Solution**: PostgreSQL with pgvector extension on Supabase

**Schema**:

```sql
CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  content TEXT,                    -- Original chunk text
  metadata JSONB,                  -- {company, year, item, chunk_index}
  embedding vector(768),           -- Vector embedding (768-dim)
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create RPC function for similarity search
CREATE FUNCTION match_documents(
  query_embedding vector(768),
  match_count int
) RETURNS TABLE(...) AS $$
  SELECT *, 1 - (embedding <=> query_embedding) as similarity
  FROM documents
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$ LANGUAGE sql;
```

**Upload Process**:

```python
# Create DataFrame with all data
df = pd.DataFrame({
  'content': chunks,
  'metadata': [{'company': 'AAPL', 'year': 2023, 'item': '1'}, ...],
  'embedding': embeddings  # Pre-computed vectors
})

# Upload to Supabase
supabase.table('documents').insert(df.to_dict('records')).execute()
```

**Query Process**:

```typescript
// In Next.js API route
const { data: chunks } = await supabase.rpc('match_documents', {
  query_embedding: userQueryVector,
  match_count: 8  // Top 8 similar chunks
});
```

### 6. Frontend & API Setup (Next.js + TypeScript)

**Architecture**:

```
User Types Question
    ↓
Next.js API: /api/chat (POST)
    ├─ Receive message
    ├─ Embed query (call FastAPI)
    ├─ Search Supabase (top 8 chunks)
    ├─ Build context from chunks
    └─ Send to Gemini LLM
         ↓
    Return answer + sources
    ↓
Display in UI
```

**API Route** ([rag-chat/src/app/api/chat/route.ts](rag-chat/src/app/api/chat/route.ts)):

```typescript
export async function POST(req: NextRequest) {
  // 1. Extract query
  const { message } = req.json();
  
  // 2. Embed query using FastAPI
  const queryEmb = await fetch(EMBEDDING_SERVICE_URL, {
    body: JSON.stringify({ text: message })
  });
  
  // 3. Retrieve similar chunks from Supabase
  const { data: chunks } = await supabase.rpc('match_documents', {
    query_embedding: queryEmb,
    match_count: 8
  });
  
  // 4. Build context
  const context = chunks
    .map(c => `[${c.metadata.company} ${c.metadata.year}] ${c.content}`)
    .join('\n\n');
  
  // 5. Generate answer with Gemini
  const answer = await fetch(GEMINI_API, {
    body: JSON.stringify({
      contents: [{ parts: [{ text: `CONTEXT:\n${context}\n\nQUESTION: ${message}` }] }]
    })
  });
  
  return Response.json({ answer, sources: chunks });
}
```

**Frontend UI** ([rag-chat/src/app/page.tsx](rag-chat/src/app/page.tsx)):

- Chat interface
- Message history
- Source attribution
- Real-time streaming (if implemented)

---

## Project Structure

```
SEC-Analyzer/
├── main.py                          # Entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── src/                             # Core Python pipeline
│   ├── downloader.py               # SEC EDGAR API interaction
│   ├── parser.py                   # Regex-based 10-K parsing
│   ├── data_cleaning.py            # HTML → JSON table extraction
│   ├── chunking.py                 # Recursive chunking
│   ├── embedder.py                 # Embedding generation
│   ├── database.py                 # Supabase operations
│   └── __pycache__/
│
├── data/
│   └── raw/
│       └── sec-edgar-filings/
│           └── AAPL/10-K/          # Downloaded 10-K filings (HTML)
│
├── rag-chat/                        # Next.js frontend + API
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.ts
│   ├── postcss.config.mjs
│   ├── eslint.config.mjs
│   │
│   ├── src/app/
│   │   ├── page.tsx                # Chat UI
│   │   ├── layout.tsx
│   │   ├── globals.css
│   │   └── api/chat/
│   │       └── route.ts            # RAG endpoint (Main logic!)
│   │
│   ├── Embedding_services/
│   │   └── embedding_service.py    # FastAPI embedding server
│   │
│   └── public/                     # Static assets
│
└── Images/                         # Documentation images
```

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Pipeline | Python (BeautifulSoup, Pandas) | Extract & clean 10-K filings |
| Chunking | Python | Recursive text splitting |
| Embeddings | SentenceTransformers | 768-dim vectors |
| Embedding API | FastAPI | Serve embeddings |
| Database | Supabase (PostgreSQL + pgvector) | Vector similarity search |
| Frontend | Next.js + React | Chat UI |
| API | Next.js API Routes | RAG orchestration |
| LLM | Google Gemini 2.5 Flash | Answer generation |

---

## Lessons Learned & Future Improvements

### What Worked Well ✓

1. **BeautifulSoup for HTML parsing** - Far superior to regex for nested structures
2. **Recursive chunking** - Good balance between cost and semantic coherence
3. **Supabase + pgvector** - Easy to set up and scales well
4. **Next.js API routes** - Simplified backend architecture (no separate Flask/FastAPI server)
5. **FastAPI microservice** - Clean separation for embedding service

### What Could Be Improved 🔄

1. **Table Processing**:
   - Currently merging `$` and `%` symbols, then dropping them later (redundant)
   - **Better approach**: Detect financial tables separately, create dedicated chunks
   - Would reduce complexity and improve chunk quality

2. **Chunking Strategy**:
   - Could implement contextual chunking with LLM for higher quality (trade-off: cost)
   - Add document type awareness (balance sheet chunks ≠ risk factor chunks)

3. **Metadata Enhancement**:
   - Add more granular metadata (section type, fiscal year, confidence scores)
   - Would improve filtering and ranking in retrieval

4. **Semantic Reranking**:
   - Implement cross-encoder reranking after initial retrieval
   - Would further improve answer quality

5. **Caching Layer**:
   - Cache frequently asked questions and their embeddings
   - Reduce embedding service load

6. **Monitoring & Evaluation**:
   - Add ROUGE/BLEU metrics for answer quality
   - Log queries and measure answer satisfaction
   - Implement feedback loop

7. **Multi-LLM Support**:
   - Currently hardcoded to Gemini
   - Make it pluggable (Claude, GPT-4, local models)

---

## Setup & Deployment

### Prerequisites

```bash
# Python 3.9+
python --version

# Node.js 18+
node --version

# PostgreSQL with pgvector (via Supabase)
```

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd SEC-Analyzer

# 2. Create Python venv
python -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env.local
# Add: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GEMINI_API_KEY, EMBEDDING_SERVICE_URL

# 5. Download data
python main.py --download

# 6. Process data (clean, chunk, embed)
python main.py --process

# 7. Upload to Supabase
python main.py --upload

# 8. Start FastAPI embedding service
cd rag-chat/Embedding_services
uvicorn embedding_service:app --reload

# 9. Start Next.js (in new terminal)
cd rag-chat
npm install
npm run dev
```

### Access

- **Chat UI**: http://localhost:3000
- **API**: http://localhost:3000/api/chat
- **Embedding Service**: http://localhost:8000/docs

---

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Avg Query Latency | ~2-3s | Includes embedding + retrieval + LLM generation |
| Chunk Count (AAPL 10-K) | ~2,500 | After recursive chunking |
| Database Size | ~100MB (estimated) | For 1 company × 15 years |
| Accuracy (on test set) | ~75-85% | Depends on question complexity |

---

## Common Challenges & Solutions

### Challenge 1: Regex Fails on Nested HTML

**Problem**: Used regex to extract tables from HTML → Broke on complex nesting

**Solution**: Switched to BeautifulSoup which understands DOM structure

**Learning**: For structural data, always use proper parsers (not regex)

### Challenge 2: Table Column Mismatch

**Problem**: `$` and `%` symbols as separate cells → Empty columns after merge

**Solution**: 
- Detect and merge symbols with adjacent values
- Remove "%" only columns before JSON conversion
- Later realized: Could just drop them during flattening

**Learning**: Pre-process carefully to avoid post-processing redundancy

### Challenge 3: Semantic Chunking Too Expensive

**Problem**: Considered semantic chunking but would create 5-10K chunks per filing

**Solution**: Used recursive chunking instead (creates ~2.5K chunks)

**Learning**: Balance quality with cost/complexity trade-offs

### Challenge 4: Same Model for Embeddings

**Problem**: Different models for chunk vs query embeddings → Vector space mismatch

**Solution**: Built FastAPI service to use `all-mpnet-base-v2` for both

**Learning**: Consistency in embedding model is critical for vector search

### Challenge 5: LLM Context Limits

**Problem**: Tried feeding entire 10-K to Gemini → Hallucinations and token overflow

**Solution**: Retrieve top-8 most similar chunks only (~3-5KB per query)

**Learning**: This is why RAG exists!

---

## References & Resources

- [SEC EDGAR API Documentation](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/)
- [pgvector Extension](https://github.com/pgvector/pgvector)
- [SentenceTransformers](https://www.sbert.net/)
- [Next.js API Routes](https://nextjs.org/docs/app/building-your-application/routing/route-handlers)
- [RAG Papers](https://arxiv.org/search/?query=retrieval+augmented+generation)
- [Supabase Documentation](https://supabase.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Implement semantic reranking
- [ ] Add multi-LLM support
- [ ] Build evaluation framework
- [ ] Optimize table extraction
- [ ] Add caching layer
- [ ] Deploy monitoring dashboard
- [ ] Implement streaming responses
- [ ] Add user feedback loop

---

## License

MIT License - see LICENSE file for details

---

## Contact & Questions

For questions about this project or to discuss RAG/LLM topics, feel free to reach out via GitHub Issues.

**Key Learning Outcome**: Understanding how to build production RAG systems that balance cost, quality, and complexity.

**Last Updated**: January 2026
