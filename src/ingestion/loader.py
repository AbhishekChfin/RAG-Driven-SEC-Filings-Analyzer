from sec_edgar_downloader import Downloader  # type: ignore
import os
import shutil
from typing import List
import traceback

from pathlib import Path


def download_filings(tickers, base_dir, form_type="10-K"):

    for ticker in tickers:
        dl = Downloader(ticker, "my.email@domain.com", str(base_dir))
        print(f"Downloading {form_type} for {ticker}...")
        dl.get(form_type, ticker)

    print("Done.")