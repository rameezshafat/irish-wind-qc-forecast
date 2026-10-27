"""Download archived NWP wind forecasts from the Open-Meteo Previous Runs API into data/raw/.

For each station this saves hourly 10 m wind speed (knots) and direction at three lead times:
  _previous_day0  the current run (close to an analysis; used only for QC consistency checks)
  _previous_day1  predicted 24 h before the valid time
  _previous_day2  predicted 48 h before the valid time (used for the day-ahead forecast)

Usage: python scripts/download_nwp.py [start_date] [end_date]   (defaults 2024-01-01 .. 2026-09-30)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from windqc.stations import STATIONS

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
VARS = [f"{v}_previous_day{d}" for d in (0, 1, 2) for v in ("wind_speed_10m", "wind_direction_10m")]


def download(station: str, start: str, end: str) -> Path:
    meta = STATIONS[station]
    params = {
        "latitude": meta["lat"], "longitude": meta["lon"],
        "start_date": start, "end_date": end,
        "hourly": ",".join(VARS), "wind_speed_unit": "kn", "timezone": "GMT",
    }
    r = requests.get(URL, params=params, timeout=300)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"time": "date"})
    out = RAW / f"nwp{station}.csv"
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2024-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-09-30"
    for station in STATIONS:
        path = download(station, start, end)
        print(f"{station} {STATIONS[station]['name']}: {path.name}")
