"""Rolling-origin backtest for day-ahead forecasts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Forecaster = Callable[[pd.Series, int], pd.Series]


def rolling_origin(
    y: pd.Series,
    forecasters: dict[str, Forecaster],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    horizon: int = 24,
    issue_hour: int = 23,
) -> pd.DataFrame:
    """Issue one forecast per day at ``issue_hour`` and score the next ``horizon`` hours.

    The default 23:00 UTC issue forecasts the following UTC day. Each forecaster sees
    only ``y`` up to and including the issue time. Returns one row
    per (origin, forecaster, target hour) with the forecast and the observation.
    """
    if not isinstance(y.index, pd.DatetimeIndex) or y.empty:
        raise ValueError("y must have a nonempty DatetimeIndex")
    if not y.index.is_monotonic_increasing or not y.index.is_unique:
        raise ValueError("y timestamps must be sorted and unique")
    if horizon < 1 or not 0 <= issue_hour <= 23:
        raise ValueError("horizon must be positive and issue_hour must be 0..23")
    tz = y.index.tz
    start = _timestamp(start, tz)
    last = y.index[-1] - pd.Timedelta(hours=horizon)
    end = last if end is None else min(_timestamp(end, tz), last)
    origins = pd.date_range(start.normalize() + pd.Timedelta(hours=issue_hour), end, freq="D")

    rows = []
    for origin in origins:
        history = y.loc[:origin]
        target = pd.date_range(origin + pd.Timedelta(hours=1), periods=horizon, freq="h")
        truth = y.reindex(target)
        for name, fc in forecasters.items():
            pred = fc(history.copy(), horizon).reindex(target)
            rows.append(
                pd.DataFrame(
                    {"origin": origin, "model": name, "target": target,
                     "y_pred": pred.to_numpy(), "y_true": truth.to_numpy()}
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _timestamp(value: str | pd.Timestamp, tz: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(tz) if ts.tzinfo is None and tz is not None else ts


def score(results: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    """Score common valid targets when comparing models to a reference."""
    if results.empty:
        return pd.DataFrame(columns=["mae", "rmse", "n"])
    keys = ["origin", "target"]
    if results.duplicated(keys + ["model"]).any():
        raise ValueError("duplicate origin, target, model rows")
    r = results.dropna(subset=["y_pred", "y_true"])
    if reference is not None:
        if reference not in set(results["model"]):
            raise ValueError(f"unknown reference model: {reference}")
        valid = r.pivot(index=keys, columns="model", values="y_pred").dropna()
        r = r.set_index(keys).loc[valid.index].reset_index()
        if r.empty:
            return pd.DataFrame(columns=["mae", "rmse", "n", "skill_vs_" + reference])
    err = r["y_pred"] - r["y_true"]
    out = (
        r.assign(abs_err=err.abs(), sq_err=err**2)
        .groupby("model")
        .agg(mae=("abs_err", "mean"), rmse=("sq_err", lambda x: float(np.sqrt(x.mean()))), n=("abs_err", "size"))
    )
    if reference is not None:
        ref_rmse = out.loc[reference, "rmse"]
        out["skill_vs_" + reference] = 1 - out["rmse"] / ref_rmse if ref_rmse else np.nan
    return out
