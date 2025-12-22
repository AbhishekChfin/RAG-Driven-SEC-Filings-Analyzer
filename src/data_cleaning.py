import re
import json
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from bs4 import BeautifulSoup, Tag

class BSManualCleanerTextJSON:
    """
    Parses HTML text, converts tables to JSON, and cleans content.
    """

    def __init__(self, html_text: str):
        self.soup = BeautifulSoup(html_text, "lxml")

    def _process_table(self, table: Tag) -> Optional[pd.DataFrame]:
        """Convert a single HTML table into a clean DataFrame."""
        rows = self._extract_rows(table)
        # Filter empty rows
        rows = [r for r in rows if any(cell.strip() for cell in r)]
        if not rows:
            return None

        rows = self._pad_rows(rows)
        df = pd.DataFrame(rows).fillna("")

        # Remove grouping headers (e.g., "Years Ended")
        df = self._remove_grouping_header(df)
        if df.empty:
            return None

        # Special case: Header only
        if len(df) == 1:
            return self._set_headers(df.copy())

        # Forward-fill header row
        header = df.iloc[0].replace(r"^\s*$", np.nan, regex=True).ffill().fillna("")
        df.iloc[0] = header.tolist()

        # Remove empty columns (keep first)
        body = df.iloc[1:]
        has_data = body.apply(lambda col: col.str.strip().astype(bool).any())
        has_data.iloc[0] = True # Always keep Item column
        df = df.iloc[:, has_data.values].copy()

        return self._set_headers(df)

    def _extract_rows(self, table: Tag) -> List[List[str]]:
        """Parses HTML table rows and handles colspans/formatting."""
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True).replace("\xa0", " ").strip() for cell in tr.find_all(["td", "th"])]
            if not cells:
                continue
            
            # Logic to merge currency symbols ($) and percents (%) with values
            merged_row = []
            i = 0
            while i < len(cells):
                value = cells[i]
                # Attach '$' to next cell
                if value == "$" and i + 1 < len(cells):
                    merged_row.append("$" + cells[i + 1])
                    i += 2
                    continue
                # Attach '%' to previous value
                if re.fullmatch(r'%+', value) and merged_row:
                    merged_row[-1] += value
                    i += 1
                    continue
                # Attach ')%' to previous value
                if value == ")%" and merged_row:
                    merged_row[-1] += ")%"
                    i += 1
                    continue
                merged_row.append(value)
                i += 1
            rows.append(merged_row)
        return rows

    def _pad_rows(self, rows: List[List[str]]) -> List[List[str]]:
        max_columns = max(len(row) for row in rows)
        return [row + [""] * (max_columns - len(row)) for row in rows]

    def _remove_grouping_header(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 2: return df
        first_row_count = sum(x.strip() != "" for x in df.iloc[0, 1:])
        second_row_count = sum(x.strip() != "" for x in df.iloc[1, 1:])
        
        # Heuristic: if second row has more data than first, first is likely a grouping header
        if second_row_count > first_row_count and first_row_count < 2:
            return df.iloc[1:].reset_index(drop=True)
        return df

    def _set_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: 
            return df
        header = df.iloc[0].tolist()
        header[0] = "Item" # Normalize first column

        # Remove columns where header is just '%'
        drop_idxs = [i for i, h in enumerate(header) 
                     if i != 0 and isinstance(h, str) and re.fullmatch(r'%+', h.strip())]
        if drop_idxs:
            cols_to_drop = [df.columns[i] for i in drop_idxs if i < len(df.columns)]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop).reset_index(drop=True)
                header = df.iloc[0].tolist()
                header[0] = "Item"

        # Handle duplicates
        seen = {}
        clean_header = []
        for col in header:
            if col in seen:
                seen[col] += 1
                clean_header.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                clean_header.append(col)
        df.columns = clean_header
        return df.iloc[1:].reset_index(drop=True)

    def _table_to_json(self, df: pd.DataFrame, table_id: int) -> str:
        df = df.fillna("")
        payload = {"table_id": table_id, "columns": list(df.columns), "rows": df.to_dict(orient="records")}
        return json.dumps(payload, indent=2)

    def clean_and_replace_tables(self) -> str:
        """Main method to process the HTML soup."""
        table_id = 1
        for table in self.soup.find_all("table"):
            if table.parent is None: continue
            
            df = self._process_table(table)
            if df is None or df.empty: continue

            json_block = self._table_to_json(df, table_id)
            
            # Create a replacement tag
            replacement = self.soup.new_tag("div")
            replacement.string = (f"\n\n[START_TABLE_JSON {table_id}]\n{json_block}\n[END_TABLE_JSON {table_id}]\n\n")
            
            target = table.find_parent("div") or table
            try:
                target.replace_with(replacement)
                table_id += 1
            except Exception as e:
                print(f"Table {table_id} replacement failed: {e}")

        # Clean non-text tags
        for tag in self.soup(["script", "style", "head", "meta", "noscript"]):
            tag.decompose()
        
        return self.soup.get_text(" ", strip=True)
    
def flatten_table_row_financial(row: dict) -> str:
    """
    Flattens a JSON dictionary row into a string suitable for embedding.
    Optimized for financial data (checks for numbers).
    """
    item = row.get("Item", "Metric")
    parts = []
    
    for col, val in row.items():
        if col != "Item":
            try:
                # Simple heuristic to check if it's a number/financial value
                # Removes $, %, commas to test float conversion
                float(str(val).replace('$', '').replace(',', '').replace('%', ''))
                parts.append(f"{col}: {val}")
            except ValueError:
                # Still add it if it's not a number (e.g. status)
                parts.append(f"{col}: {val}")
    
    return f"[Financial Metric] {item}: " + " | ".join(parts)