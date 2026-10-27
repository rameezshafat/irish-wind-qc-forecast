"""QC experiment: tune thresholds on 2024 with injected faults, score on 2025-26.

Writes results/qc_params.json, results/qc_detection.csv, results/qc_real_flag_rates.csv,
results/qc_malin_labels.json, results/cleaning_logs/*.csv and results/figures/qc_example.png.

Usage: python scripts/run_qc_eval.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from windqc.data import load_met_hourly, load_nwp
from windqc.evaluation import qc_metrics
from windqc.faults import FAULT_TYPES, detection_metrics, inject_faults
from windqc.qc import (
    flag_flatline,
    flag_model_inconsistent,
    flag_out_of_range,
    flag_spike,
    flag_step,
    run_checks,
    station_scale,
    to_cleaning_log,
)
from windqc.stations import STATIONS

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "results"
DEV = ("2024-01-03", "2024-12-31 23:00")
TEST = ("2025-01-01", "2026-09-30 23:00")
TEST_SEEDS = range(1, 6)

GRID = {
    "flatline": [{"min_run": m} for m in (6, 8, 10, 12)],
    "spike": [{"k": k} for k in (4.0, 5.0, 6.0, 8.0)],
    "step": [{"max_change": c} for c in (15.0, 20.0, 25.0, 30.0)],
    "model": [{"low": lo, "high": hi} for lo in (0.45, 0.55, 0.65) for hi in (1.8, 2.5)],
}


def load_station(st: str) -> tuple[pd.Series, pd.Series]:
    met = load_met_hourly(RAW / f"hly{st}.csv")
    nwp = load_nwp(RAW / f"nwp{st}.csv")
    idx = pd.date_range("2024-01-01", "2026-09-30 23:00", freq="h", tz="UTC")
    return met["wdsp"].reindex(idx), nwp["nwp0"].reindex(idx)


def detector_flags(name: str, y: pd.Series, ref: pd.Series, scale: float, p: dict) -> pd.Series:
    if name == "flatline":
        return flag_flatline(y, **p)
    if name == "spike":
        return flag_spike(y, **p)
    if name == "step":
        return flag_step(y, **p)
    return flag_model_inconsistent(y, ref, scale, **p)


def f1(flags: pd.Series, labels: pd.Series, observed: pd.Series) -> float:
    ok = observed.notna()
    faulty, flags = (labels != "") & ok, flags.fillna(False).astype(bool) & ok
    tp = (flags & faulty).sum()
    fp = (flags & ~faulty).sum()
    fn = (~flags & faulty).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def tune(data: dict) -> dict:
    """Pick the threshold combination with the best pooled hour-level F1 on 2024."""
    cache = {}
    for st, d in data.items():
        y, labels = inject_faults(d["obs_dev"], seed=0)
        cache[st] = {"labels": labels, "y": y, "base": flag_out_of_range(y)}
        for name, options in GRID.items():
            for i, p in enumerate(options):
                cache[st][(name, i)] = detector_flags(name, y, d["ref_dev"], d["scale"], p)
    best, best_score = None, -1.0
    for combo in itertools.product(*[range(len(v)) for v in GRID.values()]):
        scores = []
        for st in data:
            c = cache[st]
            flags = c["base"].copy()
            for name, i in zip(GRID, combo):
                flags |= c[(name, i)]
            scores.append(f1(flags, c["labels"], c["y"]))
        if np.mean(scores) > best_score:
            best_score, best = float(np.mean(scores)), combo
    params = {name: GRID[name][i] for name, i in zip(GRID, best)}
    return {"params": params, "dev_f1": best_score}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "cleaning_logs").mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    data = {}
    for st in STATIONS:
        obs, ref = load_station(st)
        dev, test = slice(*DEV), slice(*TEST)
        data[st] = {
            "obs_dev": obs.loc[dev], "ref_dev": ref.loc[dev],
            "obs_test": obs.loc[test], "ref_test": ref.loc[test],
        }
        data[st]["scale"] = station_scale(data[st]["obs_dev"], data[st]["ref_dev"])

    tuned = tune(data)
    params = tuned["params"]
    (OUT / "qc_params.json").write_text(json.dumps(
        {**tuned, "scales": {st: round(d["scale"], 4) for st, d in data.items()}}, indent=2) + "\n")
    print("tuned on 2024:", tuned)

    # Score on 2025-26 with fresh fault placements.
    det_rows, rate_rows = [], []
    for st, d in data.items():
        for seed in TEST_SEEDS:
            y, labels = inject_faults(d["obs_test"], seed=seed)
            point = run_checks(y, params=params)
            full = run_checks(y, d["ref_test"], d["scale"], params=params)
            for variant, flags in (("point checks", point["any_flag"]), ("point + model", full["any_flag"])):
                m = detection_metrics(flags, labels, y)
                m["station"], m["seed"], m["variant"] = st, seed, variant
                det_rows.append(m)
        # Flag rates on the real, uncorrupted test data, and a brightwind cleaning log.
        real = run_checks(d["obs_test"], d["ref_test"], d["scale"], params=params)
        observed = d["obs_test"].notna()
        rates = real[observed].mean().to_dict()
        rate_rows.append({"station": st, "name": STATIONS[st]["name"], **rates})
        logs = [
            to_cleaning_log(real[c], "wdsp", reason) for c, reason in
            (("flatline", "Stuck sensor"), ("spike", "Spike"), ("step", "Step change"),
             ("model", "Disagrees with NWP"), ("out_of_range", "Out of range"))
        ]
        pd.concat(logs).sort_values("Start").to_csv(OUT / "cleaning_logs" / f"cleaning_{st}.csv", index=False)

    det = pd.concat(det_rows, ignore_index=True)
    summary = (
        det.groupby(["variant", "fault"])[["hour_recall", "event_recall", "precision", "false_alarm_rate"]]
        .mean().round(4)
    )
    summary.to_csv(OUT / "qc_detection.csv")
    pd.DataFrame(rate_rows).round(5).to_csv(OUT / "qc_real_flag_rates.csv", index=False)
    print(summary)

    # Real labels: Malin Head 1980-1999 carries Met Éireann's own defect codes (point checks only;
    # the NWP archive does not reach back that far).
    malin = load_met_hourly(RAW / "hly1575.csv").loc["1980":"1999"]
    flags = run_checks(malin["wdsp"], params=params)["any_flag"]
    real_labels = qc_metrics(flags, malin["ind_wdsp"], malin["wdsp"])
    (OUT / "qc_malin_labels.json").write_text(json.dumps(real_labels, indent=2) + "\n")
    print("Malin Head 1980-99 vs defect codes:", real_labels)

    # Example figure: Belmullet, one injected seed, a window containing several fault types.
    d = data["2375"]
    y, labels = inject_faults(d["obs_test"], seed=1)
    flags = run_checks(y, d["ref_test"], d["scale"], params=params)["any_flag"]
    first = {k: labels.index[labels == k][0] for k in FAULT_TYPES}
    centre = sorted(first.values())[1]
    win = slice(centre - pd.Timedelta(days=5), centre + pd.Timedelta(days=5))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d["obs_test"].loc[win], color="0.75", lw=1, label="original")
    ax.plot(y.loc[win], color="tab:blue", lw=1, label="with injected faults")
    ax.plot(d["ref_test"].loc[win], color="tab:green", lw=0.8, ls="--", label="NWP (same-day run)")
    hit = flags.loc[win] & y.loc[win].notna()
    ax.scatter(y.loc[win][hit].index, y.loc[win][hit], color="tab:red", s=10, zorder=3, label="flagged")
    ax.set_ylabel("wind speed (kn)")
    ax.set_title("Belmullet: injected faults and QC flags")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "qc_example.png", dpi=130)


if __name__ == "__main__":
    main()
