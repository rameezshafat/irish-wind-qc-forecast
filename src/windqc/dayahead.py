"""Day-ahead forecast table: one row per (station, target hour), features known at issue time.

Setup. A forecast is issued at 00:00 UTC on day D for every hour of day D+1, so leads run from
25 to 48 hours. For target hour t the issue time is

    origin(t) = floor_day(t - 1 h) - 1 day

Every feature uses only information available at the origin:
  nwp2        NWP wind predicted 48 h before t (Open-Meteo `_previous_day2`), issued before the origin
  obs_last    last observation at the origin
  obs_mean24  mean of the 24 observations ending at the origin
  bias7d      mean(obs - nwp2) over the 7 days ending at the origin
  seasonal    observation at t - 48 h, which is at or before the origin
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "nwp2", "dir_sin", "dir_cos", "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "obs_last", "obs_mean24", "bias7d", "lead", "station_code",
]


def origin_of(target: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (target - pd.Timedelta(hours=1)).normalize() - pd.Timedelta(days=1)


def build_table(obs: pd.Series, nwp: pd.DataFrame, station_code: int) -> pd.DataFrame:
    """Rows for every hour where the NWP day-ahead value exists. ``obs`` may contain NaN.

    ``obs`` and ``nwp`` must share an hourly UTC index.
    """
    idx = obs.index
    target = idx[nwp["nwp2"].reindex(idx).notna().to_numpy()]
    origin = origin_of(target)

    mean24 = obs.rolling(24, min_periods=18).mean()
    resid = (obs - nwp["nwp2"].reindex(idx))
    bias7d = resid.rolling(24 * 7, min_periods=24 * 3).mean()

    t = pd.DataFrame(index=pd.RangeIndex(len(target)))
    t["target"] = target
    t["origin"] = origin
    t["station_code"] = station_code
    t["y"] = obs.reindex(target).to_numpy()
    t["nwp2"] = nwp["nwp2"].reindex(target).to_numpy()
    direction = np.deg2rad(nwp["dir2"].reindex(target).to_numpy())
    t["dir_sin"], t["dir_cos"] = np.sin(direction), np.cos(direction)
    t["hour_sin"] = np.sin(2 * np.pi * target.hour / 24)
    t["hour_cos"] = np.cos(2 * np.pi * target.hour / 24)
    t["doy_sin"] = np.sin(2 * np.pi * target.day_of_year / 365.25)
    t["doy_cos"] = np.cos(2 * np.pi * target.day_of_year / 365.25)
    t["lead"] = ((target - origin) / pd.Timedelta(hours=1)).astype(int)
    t["obs_last"] = obs.reindex(origin).to_numpy()
    t["obs_mean24"] = mean24.reindex(origin).to_numpy()
    t["bias7d"] = bias7d.reindex(origin).to_numpy()
    t["seasonal"] = obs.reindex(target - pd.Timedelta(hours=48)).to_numpy()
    return t


def climatology_table(train: pd.DataFrame) -> pd.Series:
    """Mean observation by (station, month, hour) from training rows."""
    month = train["target"].dt.month
    hour = train["target"].dt.hour
    return train.groupby([train["station_code"], month, hour])["y"].mean()


def climatology_predict(table: pd.Series, rows: pd.DataFrame) -> np.ndarray:
    keys = pd.MultiIndex.from_arrays(
        [rows["station_code"], rows["target"].dt.month, rows["target"].dt.hour]
    )
    return table.reindex(keys).to_numpy()
