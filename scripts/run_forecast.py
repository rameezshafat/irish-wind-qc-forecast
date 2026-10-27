"""Day-ahead backtest: monthly-retrained NWP correction against four baselines.

Test period: forecasts issued 2025-01-01 to 2026-09-28 at 8 stations. Each month's model is
trained only on target hours at or before that month's first issue time.

QC is used for targets only. Features come from raw observations, because the QC rules look at
a centred window that would reach past the issue time.

Writes results/forecast_*.csv, results/figures/*.png and models/lgbm_dayahead.txt.

Usage: python scripts/run_qc_eval.py && python scripts/run_forecast.py
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from windqc.data import load_met_hourly, load_nwp
from windqc.dayahead import FEATURES, build_table, climatology_predict, climatology_table
from windqc.qc import run_checks
from windqc.stations import STATIONS

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, MODELS = ROOT / "data" / "raw", ROOT / "results", ROOT / "models"
TEST_START, TEST_END = "2025-01-01", "2026-09-28"
LGB_PARAMS = {
    "n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 100,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.9, "random_state": 0,
    "verbose": -1,
}
MODEL_NAMES = {
    "persistence": "Persistence",
    "seasonal": "Seasonal naive (t - 48 h)",
    "climatology": "Climatology",
    "nwp_raw": "Raw NWP (day-ahead)",
    "mos_linear": "Linear MOS",
    "lgbm_raw": "LightGBM (raw targets)",
    "lgbm_qc": "LightGBM (QC-cleaned targets)",
}


def load_all() -> pd.DataFrame:
    qc = json.loads((OUT / "qc_params.json").read_text())
    idx = pd.date_range("2024-01-01", "2026-09-30 23:00", freq="h", tz="UTC")
    tables = []
    for code, st in enumerate(STATIONS):
        obs = load_met_hourly(RAW / f"hly{st}.csv")["wdsp"].reindex(idx)
        nwp = load_nwp(RAW / f"nwp{st}.csv").reindex(idx)
        flags = run_checks(obs, nwp["nwp0"], qc["scales"][st], params=qc["params"])["any_flag"]
        t = build_table(obs, nwp, code)
        t["station"] = st
        t["qc_flag"] = flags.reindex(t["target"]).to_numpy().astype(bool)
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def fit_lgbm(train: pd.DataFrame) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(train[FEATURES], train["y"], categorical_feature=["station_code"])
    return model


def backtest(table: pd.DataFrame) -> pd.DataFrame:
    test = table[(table["origin"] >= TEST_START) & (table["origin"] <= TEST_END)].copy()
    months = pd.period_range(TEST_START, TEST_END, freq="M")
    preds = []
    for month in months:
        rows = test[test["origin"].dt.tz_localize(None).dt.to_period("M") == month].copy()
        if rows.empty:
            continue
        cutoff = rows["origin"].min()
        train = table[(table["target"] <= cutoff) & table["y"].notna()]
        train_qc = train[~train["qc_flag"]]

        rows["persistence"] = rows["obs_last"]
        rows["climatology"] = climatology_predict(climatology_table(train_qc), rows)
        rows["nwp_raw"] = rows["nwp2"]
        rows["mos_linear"] = np.nan
        for code, grp in train_qc.dropna(subset=["nwp2"]).groupby("station_code"):
            lr = LinearRegression().fit(grp[["nwp2"]], grp["y"])
            mask = rows["station_code"] == code
            rows.loc[mask, "mos_linear"] = lr.predict(rows.loc[mask, ["nwp2"]])
        rows["lgbm_raw"] = fit_lgbm(train).predict(rows[FEATURES])
        rows["lgbm_qc"] = fit_lgbm(train_qc).predict(rows[FEATURES])
        preds.append(rows)
        print(f"{month}: trained on {len(train_qc):,} rows, predicted {len(rows):,}")
    return pd.concat(preds, ignore_index=True)


def scored_rows(preds: pd.DataFrame) -> pd.DataFrame:
    """Common rows: observed, not QC-flagged, and every model has a prediction."""
    ok = preds["y"].notna() & ~preds["qc_flag"] & preds[list(MODEL_NAMES)].notna().all(axis=1)
    return preds[ok]


def metrics(rows: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for m in MODEL_NAMES:
        err = rows[m] - rows["y"]
        out[m] = {"mae": err.abs().mean(), "rmse": float(np.sqrt((err**2).mean())), "bias": err.mean()}
    df = pd.DataFrame(out).T
    for ref in ("persistence", "climatology", "nwp_raw"):
        df[f"skill_vs_{ref}"] = 1 - df["rmse"] / df.loc[ref, "rmse"]
    df["n"] = len(rows)
    return df


def bootstrap_skill(rows: pd.DataFrame, model: str, ref: str, n_boot: int = 1000, seed: int = 0) -> tuple:
    """95% CI for RMSE skill, resampling whole forecast days so hourly errors stay together."""
    rng = np.random.default_rng(seed)
    se_m = ((rows[model] - rows["y"]) ** 2).groupby(rows["origin"]).agg(["sum", "size"])
    se_r = ((rows[ref] - rows["y"]) ** 2).groupby(rows["origin"]).sum()
    days = se_m.index.to_numpy()
    skills = []
    for _ in range(n_boot):
        pick = rng.choice(len(days), len(days), replace=True)
        m = se_m["sum"].to_numpy()[pick].sum()
        r = se_r.to_numpy()[pick].sum()
        skills.append(1 - np.sqrt(m) / np.sqrt(r))
    return float(np.percentile(skills, 2.5)), float(np.percentile(skills, 97.5))


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    table = load_all()
    preds = backtest(table)
    rows = scored_rows(preds)

    overall = metrics(rows)
    overall.insert(0, "model_name", [MODEL_NAMES[m] for m in overall.index])
    overall.round(4).to_csv(OUT / "forecast_overall.csv", index_label="model")

    ci = {f"{m}_vs_{r}": bootstrap_skill(rows, m, r)
          for m in ("lgbm_qc", "lgbm_raw", "mos_linear") for r in ("nwp_raw", "persistence")}
    ci["lgbm_qc_vs_lgbm_raw"] = bootstrap_skill(rows, "lgbm_qc", "lgbm_raw")
    (OUT / "forecast_skill_ci.json").write_text(json.dumps(
        {k: [round(v[0], 4), round(v[1], 4)] for k, v in ci.items()}, indent=2) + "\n")

    by_station = []
    for st, grp in rows.groupby("station"):
        m = metrics(grp)
        by_station.append({
            "station": st, "name": STATIONS[st]["name"], "n": len(grp),
            "mae_nwp_raw": m.loc["nwp_raw", "mae"], "mae_lgbm_qc": m.loc["lgbm_qc", "mae"],
            "skill_lgbm_qc_vs_nwp_raw": m.loc["lgbm_qc", "skill_vs_nwp_raw"],
            "skill_lgbm_qc_vs_persistence": m.loc["lgbm_qc", "skill_vs_persistence"],
        })
    by_station = pd.DataFrame(by_station).round(4)
    by_station.to_csv(OUT / "forecast_by_station.csv", index=False)

    by_hour = rows.groupby(rows["target"].dt.hour).apply(
        lambda g: pd.Series({m: (g[m] - g["y"]).abs().mean() for m in MODEL_NAMES}))
    by_hour.round(4).to_csv(OUT / "forecast_mae_by_hour.csv", index_label="hour_utc")
    print(overall.round(3).to_string())
    print(json.dumps(ci, indent=2))
    print(by_station.to_string())

    # Figures.
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m, style in (("persistence", ":"), ("climatology", "-."), ("nwp_raw", "--"),
                     ("mos_linear", "-"), ("lgbm_qc", "-")):
        ax.plot(by_hour.index, by_hour[m], ls=style, lw=2 if m == "lgbm_qc" else 1.2, label=MODEL_NAMES[m])
    ax.set_xlabel("target hour (UTC)")
    ax.set_ylabel("MAE (kn)")
    ax.set_title("Day-ahead MAE by hour of day, 8 stations, 2025-26")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "mae_by_hour.png", dpi=130)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    order = by_station.sort_values("skill_lgbm_qc_vs_nwp_raw")
    ax.barh(order["name"], 100 * order["skill_lgbm_qc_vs_nwp_raw"], color="tab:blue")
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("RMSE skill vs raw NWP (%)")
    ax.set_title("Day-ahead correction gain by station")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "skill_by_station.png", dpi=130)

    # Final model on all clean data, for the API.
    final = fit_lgbm(table[table["y"].notna() & ~table["qc_flag"]])
    final.booster_.save_model(str(MODELS / "lgbm_dayahead.txt"))
    (MODELS / "lgbm_dayahead.json").write_text(json.dumps({
        "features": FEATURES,
        "station_codes": {st: code for code, st in enumerate(STATIONS)},
        "trained_through": str(table["target"].max()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
