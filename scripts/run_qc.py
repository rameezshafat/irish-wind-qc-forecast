"""Run QC on a downloaded Met Éireann hourly CSV and write flags, log, and metrics.

Usage: python scripts/run_qc.py data/raw/hly532.csv data/processed/dublin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from windqc.data import load_met_hourly
from windqc.evaluation import qc_metrics
from windqc.qc import run_checks, to_cleaning_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    df = load_met_hourly(args.input)
    flags = run_checks(df["wdsp"])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    flags.to_csv(args.output_prefix.with_suffix(".flags.csv"), index_label="timestamp")
    to_cleaning_log(flags["any_flag"], "wdsp", "Wind QC rule").to_csv(
        args.output_prefix.with_suffix(".cleaning.csv"), index=False
    )
    metrics = qc_metrics(flags["any_flag"], df["ind_wdsp"], df["wdsp"])
    args.output_prefix.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
