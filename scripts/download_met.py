"""Download Met Éireann hourly station files into data/raw/.

Usage: python scripts/download_met.py 532 [more station numbers ...]
Station numbers are listed on https://www.met.ie/climate/available-data/historical-data
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

URL = "https://clidata.met.ie/cli/climate_data/webdata/hly{station}.csv"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(station: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"hly{station}.csv"
    with requests.get(URL.format(station=station), stream=True, timeout=120) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for station in sys.argv[1:]:
        path = download(station)
        print(f"{station}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
