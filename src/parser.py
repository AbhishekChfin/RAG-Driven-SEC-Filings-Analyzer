import re
from typing import Dict
import os
import pandas as pd

from src.data_cleaning import BSManualCleanerTextJSON

def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def extract_item_positions(text: str) -> pd.DataFrame:
    """Finds start/end indices of Items 1A-14."""
    regex = re.compile(
        r'(>Item(\s|&#160;|nbsp;)(1A|1B|2|3|4|5|6|7|7A|8|9A|9B|10|11|12|13|14)\.{0,1})'
        r'|(ITEM\s(1A|1B|2|3|4|5|6|7|7A|8|9A|9B|10|11|12|13|14))',
        re.IGNORECASE
    )
    matches = regex.finditer(text)
    rows = []
    for x in matches:
        rows.append((x.group(), x.start(), x.end()))
    
    df = pd.DataFrame(rows, columns=['item','start','end'])
    df['item'] = (
        df['item'].astype(str)
        .str.replace(r'\s+', '', regex=True)
        .str.replace(r'[^0-9a-zA-Z]', '', regex=True)
        .str.lower()
    )
    # Deduplicate, keep last occurrence (standard SEC format often puts TOC first, actual content later)
    df = df.sort_values('start').drop_duplicates(subset=['item'], keep='last')
    df.set_index('item', inplace=True)
    return df

def extract_sections(raw_10k: str) -> Dict[str, str]:
    """Extracts relevant sections from raw 10-K text."""
    # Find <DOCUMENT> blocks
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    
    doc_starts = [m.end() for m in doc_start_pattern.finditer(raw_10k)]
    doc_ends = [m.start() for m in doc_end_pattern.finditer(raw_10k)]
    doc_types = [t[len('<TYPE>'):] for t in type_pattern.findall(raw_10k)]
    
    # Isolate the specific '10-K' document part
    ten_k_text = ""
    for doc_type, start, end in zip(doc_types, doc_starts, doc_ends):
        if doc_type == "10-K":
            ten_k_text = raw_10k[start:end]
            break
    
    if not ten_k_text:
        return {}

    pos_df = extract_item_positions(ten_k_text)
    cleaned_items = {}
    
    # Iterate through identified sections
    for i in range(len(pos_df)-1):
        key = "raw_" + pos_df.index[i]
        start = pos_df['start'].iloc[i]
        end = pos_df['start'].iloc[i+1]
        
        # Clean the HTML within this section
        cleaner = BSManualCleanerTextJSON(ten_k_text[start:end])
        cleaned_items[key] = cleaner.clean_and_replace_tables()
    
    return cleaned_items

def parse_10k(file_path: str) -> Dict[str, str]:
    raw_text = read_file(file_path)
    cleaned_items = extract_sections(raw_text)
    return cleaned_items




