"""QC evaluation against Met Éireann's wind indicator labels."""

from __future__ import annotations

import pandas as pd

from windqc.data import DEFECTIVE_WIND_CODES


def qc_metrics(flags: pd.Series, indicators: pd.Series, readings: pd.Series) -> dict[str, float | int | None]:
    """Score only observed hours with a known indicator code.

    Labels are imperfect: an unmarked fault counts as a false alarm here.
    """
    if not flags.index.equals(indicators.index) or not flags.index.equals(readings.index):
        raise ValueError("flags, indicators, and readings must have identical timestamps")
    eligible = indicators.notna() & readings.notna()
    predicted = flags.fillna(False).astype(bool)[eligible]
    actual = indicators[eligible].isin(DEFECTIVE_WIND_CODES)
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "evaluated": int(eligible.sum()), "excluded": int((~eligible).sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
    }
