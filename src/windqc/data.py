"""Load Met Éireann hourly station files and Open-Meteo NWP archives."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

# itabspeed codes 3-6 mark defective or partially defective wind readings. They are rare:
# a few hundred hours across all stations, almost all before 2000. Code 111 ("investigation
# required") covers whole decades at some stations, so it is a station status, not a fault label.
DEFECTIVE_WIND_CODES = {3, 4, 5, 6}


def _find_header_line(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.startswith("date,"):
            return i
    raise ValueError("No 'date,' header row found; not a Met Éireann hourly file?")


def _name_columns(raw: list[str]) -> list[str]:
    """Each 'ind' column qualifies the element after it, so name it ind_<element>."""
    names = []
    for i, col in enumerate(raw):
        col = col.strip()
        if col == "ind":
            nxt = raw[i + 1].strip() if i + 1 < len(raw) else "unknown"
            col = f"ind_{nxt}"
        names.append(col)
    return names


def load_met_hourly(source: str | Path | io.StringIO) -> pd.DataFrame:
    """Parse a Met Éireann hourly CSV into a DataFrame indexed by UTC timestamp.

    Stations publish different column sets, so names come from the file's own header.
    Blank values become NaN. Adds ``wind_defective`` from the wind speed indicator.
    """
    text = source.getvalue() if isinstance(source, io.StringIO) else Path(source).read_text()
    lines = text.splitlines()
    header = _find_header_line(lines)
    names = _name_columns(lines[header].split(","))
    if "wdsp" not in names:
        raise ValueError(f"No wind speed column in header: {names}")

    df = pd.read_csv(
        io.StringIO("\n".join(lines[header + 1:])),
        names=names,
        na_values=["", " "],
        skipinitialspace=True,
        low_memory=False,
    )
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M", utc=True)
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["wind_defective"] = df["ind_wdsp"].isin(DEFECTIVE_WIND_CODES)
    return df


def load_nwp(source: str | Path) -> pd.DataFrame:
    """Load an Open-Meteo Previous Runs CSV written by scripts/download_nwp.py.

    Returns columns nwp0/nwp1/nwp2 (speed, knots) and dir0/dir1/dir2 (degrees), where the
    number is the lead in days: 1 = predicted 24 h before the valid time, 2 = 48 h before.
    """
    df = pd.read_csv(source)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    rename = {}
    for col in df.columns:
        lead = col.rsplit("previous_day", 1)[1] if "previous_day" in col else "0"
        rename[col] = ("nwp" if col.startswith("wind_speed") else "dir") + lead
    return df.rename(columns=rename)
