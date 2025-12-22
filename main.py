import json
import copy
from pathlib import Path

# Local imports
from src.downloader import download_filings
from src.chunking import process_and_chunk_files
from src.data_cleaning import flatten_table_row_financial
from src.embedder import Embedder
from src.database import VectorDB


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"/ "raw"

TICKERS = ["AAPL"]
EMAIL = "your.email@example.com" # CHANGE THIS
BATCH_SIZE = 4


def main():
    # 1. Download Data
    print("Downloading data from SEC...")
    download_filings(["AAPL"], DATA_DIR)

    # 2. Parse & Chunk
    print("Starting Parser Pipeline...")
    all_rows = process_and_chunk_files(DATA_DIR)

    if not all_rows:
        print("No data found. Did you run the downloader?")
        return
    
    print(f"Created {len(all_rows)} chunks.")

    # 3. Prepare for Embedding
    # We map "text" to what actually gets embedded.
    # Tables get flattened into strings like "Revenue: 100 | Cost: 50"
    texts_for_embedding = []
    row_mapping = []

    for row in all_rows:
        if row["is_table"]:
            try:
                table_json = json.loads(row["text"])
                flattened_rows = [flatten_table_row_financial(r) for r in table_json.get("rows", [])]
                
                # Each row in the table becomes a vector
                for flat_idx, flat in enumerate(flattened_rows):
                    texts_for_embedding.append(flat)
                    new_row = copy.deepcopy(row)
                    new_row["chunk_idx"] = f"{row['chunk_idx']}_row{flat_idx}"
                    new_row["text"] = flat # Store flat text for context
                    row_mapping.append(new_row)
            except:
                # Fallback if table parsing fails
                texts_for_embedding.append(row["text"])
                row_mapping.append(row)
        else:
            texts_for_embedding.append(row["text"])
            row_mapping.append(row)

    # 4. Generate Embeddings
    embedder = Embedder()
    embeddings = embedder.generate_embeddings(texts_for_embedding, batch_size=BATCH_SIZE)

    if len(embeddings) != len(row_mapping):
        print(f"Mismatch: {len(embeddings)} embeddings vs {len(row_mapping)} rows")
        return

    # 5. Prepare Payload for DB
    final_rows = []
    for emb, meta in zip(embeddings, row_mapping):
        final_rows.append({
            "doc_id": meta["doc_id"],
            "chunk_index": meta["chunk_idx"],
            "content": meta["text"],
            "metadata": {
                "company": meta["company"],
                "year": meta["year"],
                "item": meta["item"],
                "is_table": meta["is_table"]
            },
            "embedding": emb.tolist()
        })

    # 6. Upload
    db = VectorDB()
    db.upload_chunks(final_rows)

if __name__ == "__main__":
    main()





