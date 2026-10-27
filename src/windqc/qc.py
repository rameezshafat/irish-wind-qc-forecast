"""Rule-based quality checks for an hourly wind speed series."""

from __future__ import annotations

import pandas as pd


def flag_out_of_range(s: pd.Series, low: float = 0.0, high: float = 100.0) -> pd.Series:
    """True where a value lies outside [low, high]. Missing values are not flagged here."""
    return (s < low) | (s > high)


def flag_flatline(s: pd.Series, min_run: int = 6, ignore_values: tuple[float, ...] = (0.0,)) -> pd.Series:
    """True for every point in a run of at least ``min_run`` identical consecutive values.

    Calm conditions legitimately repeat 0 knots, so values in ``ignore_values`` are never flagged.
    """
    if min_run < 2:
        raise ValueError("min_run must be at least 2")
    continuity = s.index.to_series().diff().eq(pd.Timedelta(hours=1)).to_numpy()
    run_id = ((s != s.shift()) | ~continuity).cumsum()
    run_len = s.groupby(run_id).transform("size")
    return (run_len >= min_run) & s.notna() & ~s.isin(ignore_values)


def flag_spike(s: pd.Series, window: int = 7, k: float = 5.0, min_mad: float = 1.0) -> pd.Series:
    """True where a value sits more than ``k`` median absolute deviations from its rolling median.

    ``min_mad`` stops near-constant windows from flagging small, normal changes.
    """
    med = s.rolling(window, center=True, min_periods=window // 2 + 1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=window // 2 + 1).median()
    return ((s - med).abs() > k * mad.clip(lower=min_mad)).fillna(False)


def flag_step(s: pd.Series, max_change: float = 30.0) -> pd.Series:
    """Flag the later reading after a large one-hour change in knots.

    The threshold is a screening parameter, not a physical impossibility bound.
    """
    adjacent = s.index.to_series().diff().eq(pd.Timedelta(hours=1)).to_numpy()
    return ((s - s.shift()).abs() > max_change) & adjacent


def missing_hours(s: pd.Series) -> pd.DatetimeIndex:
    """Hourly timestamps absent from the index or holding NaN, between the first and last reading."""
    if s.empty:
        return pd.DatetimeIndex([])
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    present = s.dropna().index
    return full.difference(present)


def model_ratio(s: pd.Series, ref: pd.Series, window: int = 12) -> pd.DataFrame:
    """Centred rolling ratio of observed to reference wind, and the reference mean in the window.

    ``ref`` is a model value for the same hours (here the same-day NWP run). Hours where
    either side is missing drop out of both sums, so the ratio compares like with like.
    """
    ref = ref.reindex(s.index)
    both = s.notna() & ref.notna()
    obs_sum = s.where(both).rolling(window, center=True, min_periods=window // 2).sum()
    ref_sum = ref.where(both).rolling(window, center=True, min_periods=window // 2).sum()
    n = both.astype(float).rolling(window, center=True, min_periods=window // 2).sum()
    return pd.DataFrame({"ratio": obs_sum / ref_sum, "ref_mean": ref_sum / n}, index=s.index)


def flag_model_inconsistent(
    s: pd.Series,
    ref: pd.Series,
    scale: float,
    low: float = 0.55,
    high: float = 1.8,
    window: int = 12,
    min_ref: float = 8.0,
) -> pd.Series:
    """Flag hours where observed wind disagrees with a model reference for a sustained window.

    The observed/model ratio is divided by ``scale``, the station's typical ratio (fitted on
    training data), and flagged outside [low, high]. Only judged when the model expects at
    least ``min_ref`` knots, because ratios of light winds are noisy. This is the check that
    catches a degraded anemometer or a sensor stuck at zero, which point checks cannot see.
    """
    r = model_ratio(s, ref, window)
    norm = r["ratio"] / scale
    judged = (r["ref_mean"] >= min_ref) & s.notna()
    return (judged & ((norm < low) | (norm > high))).fillna(False)


def station_scale(s: pd.Series, ref: pd.Series, window: int = 12, min_ref: float = 8.0) -> float:
    """Median observed/model ratio for a station, used to normalise ``flag_model_inconsistent``."""
    r = model_ratio(s, ref, window)
    return float(r.loc[r["ref_mean"] >= min_ref, "ratio"].median())


def run_checks(
    s: pd.Series,
    ref: pd.Series | None = None,
    scale: float | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """All checks as boolean columns, plus ``any_flag``.

    Point checks always run. The model-consistency check runs when ``ref`` and ``scale`` are given.
    ``params`` overrides thresholds, e.g. {"flatline": {"min_run": 8}, "spike": {"k": 6}}.
    """
    if not isinstance(s.index, pd.DatetimeIndex) or not s.index.is_monotonic_increasing:
        raise ValueError("readings must have sorted datetime timestamps")
    if not s.index.is_unique:
        raise ValueError("readings must have unique timestamps")
    params = params or {}
    names = ["out_of_range", "flatline", "spike", "step"] + (["model"] if ref is not None else [])
    if s.empty:
        return pd.DataFrame(columns=names + ["any_flag"], index=s.index)
    flags = pd.DataFrame(
        {
            "out_of_range": flag_out_of_range(s, **params.get("out_of_range", {})),
            "flatline": flag_flatline(s, **params.get("flatline", {})),
            "spike": flag_spike(s, **params.get("spike", {})),
            "step": flag_step(s, **params.get("step", {})),
        },
        index=s.index,
    )
    if ref is not None:
        if scale is None:
            raise ValueError("scale is required with ref; fit it on training data with station_scale")
        flags["model"] = flag_model_inconsistent(s, ref, scale, **params.get("model", {}))
    flags["any_flag"] = flags.any(axis=1)
    return flags


def to_cleaning_log(flags: pd.Series, sensor: str, reason: str) -> pd.DataFrame:
    """Collapse a boolean flag series into brightwind cleaning-log rows (Sensor, Start, Stop, Reason)."""
    flags = flags.fillna(False).astype(bool)
    if not flags.any():
        return pd.DataFrame(columns=["Sensor", "Start", "Stop", "Reason"])
    contiguous = flags.index.to_series().diff().eq(pd.Timedelta(hours=1)).to_numpy()
    run_id = ((flags != flags.shift()) | ~contiguous).cumsum()
    rows = [
        {"Sensor": sensor, "Start": grp.index.min(), "Stop": grp.index.max(), "Reason": reason}
        for _, grp in flags[flags].groupby(run_id[flags])
    ]
    return pd.DataFrame(rows, columns=["Sensor", "Start", "Stop", "Reason"])
