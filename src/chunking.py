import re
import json
from typing import List, Dict
import os
import nltk
from tqdm.auto import tqdm
from src.parser import parse_10k

# Regex for detecting our custom JSON table blocks
JSON_TABLE_PATTERN = re.compile(
    r'\[START_TABLE_JSON\s+\d+\](.*?)\[END_TABLE_JSON\s+\d+\]',
    re.DOTALL
)

def clean_chunk(chunk: str) -> str:
    """
    Cleans text chunks by removing artifacts, repeated punctuation, 
    and extra whitespace. Preserves JSON table blocks.
    """
    if "[START_TABLE_JSON" in chunk or "[END_TABLE_JSON" in chunk:
        return chunk.strip()
    
    chunk = re.sub(r"(\.\s?){3,}", " ", chunk)  # Remove dotted leaders (...)
    chunk = re.sub(r"[-=_]{3,}", " ", chunk)    # Remove long dashes
    chunk = re.sub(r"\s{2,}", " ", chunk)       # Collapse multiple spaces
    chunk = re.sub(r"\s*([.,:;])\s*", r"\1 ", chunk) # Fix punctuation spacing
    return chunk.strip()


# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def extract_json_from_chunk(chunk: str) -> str:
    """Extracts valid JSON string from inside the custom tags."""
    match = JSON_TABLE_PATTERN.search(chunk)
    if not match:
        return ""
    
    content = match.group(1).strip()
    json_start = content.find('{')
    json_end = content.rfind('}')
    
    if json_start != -1 and json_end != -1 and json_start < json_end:
        return content[json_start:json_end + 1]
    return ""

def split_json_table_by_rows(json_text: str, rows_per_chunk: int = 10) -> List[str]:
    """Splits a large JSON table into smaller JSON blocks by row count."""
    try:
        data = json.loads(json_text)
        table_id = data.get("table_id", 0)
        columns = data.get("columns", [])
        rows = data.get("rows", [])

        if not rows:
            return [json_text]

        chunks = []
        for i in range(0, len(rows), rows_per_chunk):
            payload = {
                "table_id": table_id,
                "columns": columns,
                "rows": rows[i:i + rows_per_chunk],
                "is_split": True,
                "part": (i // rows_per_chunk) + 1
            }
            chunks.append(json.dumps(payload, indent=2))
        return chunks

    except Exception as e:
        print(f"Error splitting JSON table: {e}")
        return [json_text]

def recursive_chunk_text(text: str, max_chunk_size: int = 1000, min_chunk_size: int = 50) -> List[str]:
    """
    Recursively splits text into chunks.
    Prioritizes keeping JSON tables intact (or splitting them logically),
    then splits by paragraph, then by sentence.
    """
    # 1. Split by JSON blocks
    segments = JSON_TABLE_PATTERN.split(text)
    
    final_chunks = []
    table_id = 1 # Fallback counter

    for i, segment in enumerate(segments):
        if not segment or not segment.strip():
            continue

        # Odd indices are the content INSIDE the split pattern capture group
        if i % 2 == 1:
            json_part = extract_json_from_chunk(f"[START_TABLE_JSON {table_id}]{segment}[END_TABLE_JSON {table_id}]")
            if json_part:
                # Reconstruct tags for downstream processing
                json_block = f"[START_TABLE_JSON {table_id}]\n{json_part}\n[END_TABLE_JSON {table_id}]"
                final_chunks.append(json_block)
                table_id += 1
            else:
                final_chunks.append(segment.strip())
            continue

        # Even indices are regular text
        paragraphs = segment.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para: continue

            if len(para) <= max_chunk_size:
                cleaned = clean_chunk(para)
                if cleaned: final_chunks.append(cleaned)
            else:
                # Split large paragraphs by sentence
                sentences = nltk.sent_tokenize(para)
                current_chunk = []
                current_size = 0
                
                for sent in sentences:
                    sent_len = len(sent)
                    if current_size + sent_len > max_chunk_size and current_chunk:
                        final_chunks.append(clean_chunk(" ".join(current_chunk)))
                        current_chunk = [sent]
                        current_size = sent_len
                    else:
                        current_chunk.append(sent)
                        current_size += sent_len + 1
                
                if current_chunk:
                    final_chunks.append(clean_chunk(" ".join(current_chunk)))

    return [c for c in final_chunks 
            if len(c) >= min_chunk_size or "[START_TABLE_JSON" in c]

def process_and_chunk_files(base_dir: str) -> List[Dict]:
    """
    Iterates through the directory structure, parses 10-Ks, and creates chunks.
    Specific to the folder structure created by sec_edgar_downloader.
    """
    all_rows = []
    base_dir = base_dir/ "sec-edgar-filings"

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return []
    
    for ticker in os.listdir(base_dir): # e.g., AAPL
        ticker_path = os.path.join(base_dir, ticker, "10-K")
        if not os.path.exists(ticker_path): continue

        for folder in os.listdir(ticker_path):
            folder_path = os.path.join(ticker_path, folder)
            if not os.path.isdir(folder_path): continue

            # Extract Year from folder name (e.g., ...-22-...)
            match = re.search(r"-(\d{2})-", folder)
            if not match: continue
            
            year_2digit = int(match.group(1))
            year = 1900 + year_2digit if year_2digit >= 90 else 2000 + year_2digit

            # Filter logic (optional)
            if year <= 2016: 
                continue

            # Locate the text file
            files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
            if not files: continue
            file_path = os.path.join(folder_path, files[0]) # usually full-submission.txt

            print(f"Parsing: {ticker} ({year})")

            try:
                cleaned_items = parse_10k(file_path)
            except Exception as e:
                print(f"Parse error for {folder}: {e}")
                continue

            # Process parsed sections
            for item_name, clean_text in cleaned_items.items():
                doc_id = f"{ticker}_{year}_{item_name.replace('raw_item', 'item')}"
                
                # Chunking
                chunks = recursive_chunk_text(clean_text)

                for chunk_idx, chunk in enumerate(chunks):
                    # Check if this chunk contains a JSON table
                    parts = JSON_TABLE_PATTERN.split(chunk)

                    for part_idx, part in enumerate(parts):
                        part = part.strip()
                        if not part: continue

                        # Odd indices in the split are the captured group (JSON content)
                        if part_idx % 2 == 1:
                            # Wrap it back in tags for extraction
                            json_text = extract_json_from_chunk(f"[START_TABLE_JSON 1]{part}[END_TABLE_JSON 1]")
                            
                            if json_text:
                                try:
                                    # Split large tables
                                    table_chunks = split_json_table_by_rows(json_text, rows_per_chunk=3) # Small chunk for embedding
                                    for j, table_chunk in enumerate(table_chunks):
                                        all_rows.append({
                                            "doc_id": doc_id,
                                            "company": ticker,
                                            "year": year,
                                            "item": item_name.replace("raw_item", "item"),
                                            "chunk_idx": f'chunk{chunk_idx}_table{j}',
                                            "is_table": True,
                                            "text": table_chunk
                                        })
                                except json.JSONDecodeError:
                                    pass
                        else:
                            # Regular text
                            all_rows.append({
                                "doc_id": doc_id,
                                "company": ticker,
                                "year": year,
                                "item": item_name.replace("raw_item", "item"),
                                "chunk_idx": f'chunk{chunk_idx}_part{part_idx}',
                                "is_table": False,
                                "text": part
                            })
    return all_rows
