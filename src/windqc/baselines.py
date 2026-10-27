"""Reference forecasts for day-ahead hourly wind speed.

Each function takes the observed history up to the issue time and returns a forecast
for the next ``horizon`` hours. None of them may look past ``history.index[-1]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _future_index(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")


def persistence(history: pd.Series, horizon: int = 24) -> pd.Series:
    """Repeat the last observed value."""
    valid = history.dropna()
    last = valid.iloc[-1] if not valid.empty else np.nan
    return pd.Series(np.full(horizon, last), index=_future_index(history, horizon))


def seasonal_naive(history: pd.Series, horizon: int = 24, season: int = 24) -> pd.Series:
    """Repeat the most recent full season (by default, the last 24 hours)."""
    if horizon > season:
        raise ValueError("horizon must not exceed season length")
    idx = _future_index(history, horizon)
    # Select by timestamp: missing source hours must not shift the daily pattern.
    source = idx - pd.Timedelta(hours=season)
    return pd.Series(history.reindex(source).to_numpy(), index=idx)


def climatology(history: pd.Series, horizon: int = 24) -> pd.Series:
    """Mean of the training history for the same month and hour of day."""
    idx = _future_index(history, horizon)
    table = history.groupby([history.index.month, history.index.hour]).mean()
    overall = history.mean()
    values = [table.get((ts.month, ts.hour), overall) for ts in idx]
    return pd.Series(values, index=idx)
