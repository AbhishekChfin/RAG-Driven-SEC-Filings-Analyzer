# 10-K RAG Pipeline: SEC Filing Analyzer

A robust ETL pipeline that downloads SEC 10-K filings, parses complex HTML structures (including financial tables), chunks text intelligently, generates vector embeddings using `all-MPNet-base-v2`, and uploads them to a Supabase vector database.

## 🚀 Features

* **Automated Downloader**: Fetches 10-K filings directly from the SEC Edgar database.
* **Smart Parsing**: Custom BeautifulSoup logic to clean HTML, preserving financial table structures as JSON blocks.
* **Intelligent Chunking**: Recursive character splitting that respects semantic boundaries (JSON tables, paragraphs, sentences).
* **Table flattening**: Converts 10-K financial tables into flattened string representations for better embedding context.
* **Vector Embedding**: Uses HuggingFace `sentence-transformers` for high-quality semantic embeddings.

## 🛠️ Prerequisites

1.  **Python 3.8+**
2.  **Supabase Account**: You need a Supabase project with a vector store enabled (pgvector).
3.  **wkhtmltopdf**: Required for PDF conversion (used by `pdfkit`).
    * **Mac**: `brew install wkhtmltopdf`
    * **Debian/Ubuntu**: `sudo apt-get install wkhtmltopdf`
    * **Windows**: Download installer from [wkhtmltopdf.org](https://wkhtmltopdf.org/) and add to PATH.

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