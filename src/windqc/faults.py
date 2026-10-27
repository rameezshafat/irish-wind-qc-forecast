"""Inject known faults into a clean wind series so detectors can be scored.

Met Éireann's own defect codes are too rare to use as labels (see data.py), so the QC
evaluation plants faults with known positions instead. Each type mimics a real failure:

  stuck    the logger repeats the last value (frozen anemometer, icing)
  spike    one implausible reading (electrical noise)
  zero     the sensor reads 0 kn while the wind blows (failed cup or cable)
  degraded readings fall to a fraction of the truth for days (worn bearings)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FAULT_TYPES = ("stuck", "spike", "zero", "degraded")


def inject_faults(
    s: pd.Series,
    n_per_type: int = 20,
    seed: int = 0,
    degraded_factor: float = 0.6,
) -> tuple[pd.Series, pd.Series]:
    """Return (corrupted series, label series). Labels hold the fault type or "" for clean hours.

    Faults never overlap each other or missing hours, and a stuck fault only starts where the
    wind is not calm, so every planted fault actually changes the data.
    """
    rng = np.random.default_rng(seed)
    y = s.copy().astype(float)
    labels = pd.Series("", index=s.index, dtype=object)
    values = y.to_numpy(copy=True)
    taken = np.isnan(values).copy()
    lengths = {"stuck": (6, 36), "spike": (1, 1), "zero": (6, 24), "degraded": (48, 168)}

    for kind in FAULT_TYPES:
        placed, attempts = 0, 0
        while placed < n_per_type and attempts < 10_000:
            attempts += 1
            lo, hi = lengths[kind]
            length = int(rng.integers(lo, hi + 1))
            start = int(rng.integers(1, len(values) - length - 1))
            span = slice(start, start + length)
            # Keep one clean hour either side so faults stay separable.
            if taken[start - 1:start + length + 1].any():
                continue
            seg = values[span]
            if kind == "stuck":
                frozen = values[start - 1]
                if frozen < 3 or np.all(seg == frozen):
                    continue
                values[span] = frozen
            elif kind == "spike":
                values[span] = seg + rng.uniform(15, 35)
            elif kind == "zero":
                if np.nanmean(seg) < 8:
                    continue
                values[span] = 0.0
            elif kind == "degraded":
                values[span] = np.round(seg * degraded_factor)
            taken[start - 1:start + length + 1] = True
            labels.iloc[span] = kind
            placed += 1

    return pd.Series(values, index=s.index, name=s.name), labels


def detection_metrics(flags: pd.Series, labels: pd.Series, observed: pd.Series) -> pd.DataFrame:
    """Hour-level recall per fault type, event-level recall, and false-alarm rate on clean hours.

    An event counts as detected if any of its hours is flagged.
    """
    ok = observed.notna()
    flags = flags.fillna(False).astype(bool)
    rows = []
    for kind in FAULT_TYPES:
        is_kind = (labels == kind) & ok
        if not is_kind.any():
            continue
        run_id = (is_kind != is_kind.shift()).cumsum()
        events = flags[is_kind].groupby(run_id[is_kind]).any()
        rows.append({
            "fault": kind,
            "hours": int(is_kind.sum()),
            "hour_recall": float(flags[is_kind].mean()),
            "events": len(events),
            "event_recall": float(events.mean()),
        })
    clean = (labels == "") & ok
    faulty = (labels != "") & ok
    tp = int((flags & faulty).sum())
    fp = int((flags & clean).sum())
    rows.append({
        "fault": "all",
        "hours": int(faulty.sum()),
        "hour_recall": tp / faulty.sum() if faulty.any() else np.nan,
        "events": np.nan,
        "event_recall": np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "false_alarm_rate": fp / clean.sum() if clean.any() else np.nan,
    })
    return pd.DataFrame(rows)
