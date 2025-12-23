# SEC 10-K RAG-Based Summarizer (Apple Case Study)

## Overview

This project builds a Retrieval-Augmented Generation (RAG) pipeline for querying and summarizing SEC 10-K filings.
It focuses on **structural accuracy**, **traceable retrieval**, and **low-latency semantic search** using Apple’s 10-K filings as the initial dataset.

The pipeline parses raw SEC EDGAR HTML filings, performs item-wise structural chunking, embeds the chunks using a Sentence Transformer model, and stores them in Supabase with pgvector for cosine similarity search.

## Architecture Overview




## Data Processing Pipeline

### 1. Data Ingestion
- Downloaded Apple 10-K filings from SEC EDGAR in raw text/HTML format.

### 2. Item-wise Parsing
- Extracted SEC items (e.g., Item 1, Item 7, Item 8) using regex-based indexing.
- This enables **structural chunking** aligned with SEC filing semantics.

### 3. HTML Cleaning & Table Preservation
- Cleaned noisy HTML while preserving `<table>` tags.
- Tables remain in their original item context to avoid semantic drift.

### 4. Chunking Strategy
- Applied recursive chunking on item-level sections.
- **Semantic chunking was intentionally avoided** due to increased compute cost and added system complexity.

> Trade-off: lower semantic granularity in exchange for predictable structure and faster ingestion.


## Embedding & Retrieval

- Embedding model: `sentence-transformers/all-mpnet-base-v2`
- Both documents and queries use the **same embedding model** to ensure vector space consistency.
- Vectors are stored in Supabase PostgreSQL using `pgvector`.
- Cosine similarity is used to retrieve top-K relevant chunks for RAG context injection.


## RAG Chat Application

- Built using Next.js (App Router).
- Supabase client integrated via `@supabase/supabase-js`.
- Query flow:
  1. User query → embedding
  2. Top-K chunks retrieved from pgvector
  3. Chunks injected into the LLM prompt


## 🛠️ Prerequisites

1.  **Python 3.8+**
2.  **Supabase Account**: You need a Supabase project with a vector store enabled (pgvector).
3.  **wkhtmltopdf**: Required for PDF conversion (used by `pdfkit`).
    * **Mac**: `brew install wkhtmltopdf`
    * **Debian/Ubuntu**: `sudo apt-get install wkhtmltopdf`
    * **Windows**: Download installer from [wkhtmltopdf.org](https://wkhtmltopdf.org/) and add to PATH.
  
## Setup & Usage

1. Install dependencies:

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/sec-10k-analyzer.git](https://github.com/yourusername/sec-10k-analyzer.git)
    cd sec-10k-analyzer
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment**
    Copy `.env.example` to `.env` and fill in your credentials:
    ```bash
    cp .env.example .env
    ```
    *Update `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `SEC_USER_AGENT` in the `.env` file.*

## 🏃 Usage

The project is orchestrated via `main.py`.

### Step 1: Download Data
Open `main.py` and uncomment the download line:
```python
# main.py
download_filings(TICKERS, EMAIL, data_dir=DATA_DIR)
