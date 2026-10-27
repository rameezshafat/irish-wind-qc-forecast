import numpy as np
import pandas as pd

from windqc.backtest import rolling_origin, score
from windqc.baselines import climatology, persistence, seasonal_naive


def daily_cycle(days=10):
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    return pd.Series(10 + 5 * np.sin(2 * np.pi * idx.hour / 24), index=idx)


def test_persistence_repeats_last_value():
    y = daily_cycle(2)
    fc = persistence(y, 24)
    assert (fc == y.iloc[-1]).all()
    assert fc.index[0] == y.index[-1] + pd.Timedelta(hours=1)


def test_seasonal_naive_is_exact_on_pure_cycle():
    y = daily_cycle(3)
    fc = seasonal_naive(y.iloc[:-24], 24)
    assert np.allclose(fc.to_numpy(), y.iloc[-24:].to_numpy())


def test_climatology_uses_hour_means():
    y = daily_cycle(5)
    fc = climatology(y.iloc[:-24], 24)
    assert np.allclose(fc.to_numpy(), y.iloc[-24:].to_numpy())


def test_backtest_never_sees_the_future():
    y = daily_cycle(6)
    seen_last = []

    def spy(history, horizon):
        seen_last.append(history.index[-1])
        return persistence(history, horizon)

    res = rolling_origin(y, {"spy": spy}, start="2024-01-03")
    assert set(res["origin"]) == set(seen_last)
    for origin, grp in res.groupby("origin"):
        assert (grp["target"] > origin).all()


def test_score_skill_is_zero_for_reference():
    y = daily_cycle(6)
    res = rolling_origin(y, {"persistence": persistence, "seasonal": seasonal_naive}, start="2024-01-03")
    table = score(res, reference="persistence")
    assert table.loc["persistence", "skill_vs_persistence"] == 0
    assert table.loc["seasonal", "rmse"] < table.loc["persistence", "rmse"]
