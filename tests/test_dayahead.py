import numpy as np
import pandas as pd

from windqc.dayahead import build_table, origin_of
from windqc.faults import detection_metrics, inject_faults


def frames(days=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    obs = pd.Series(10 + rng.normal(0, 2, len(idx)), index=idx)
    nwp = pd.DataFrame({"nwp2": obs + rng.normal(0, 1, len(idx)), "dir2": 270.0}, index=idx)
    return obs, nwp


def test_origin_is_midnight_the_day_before_the_target_day():
    t = pd.DatetimeIndex(["2024-01-05 01:00", "2024-01-05 23:00", "2024-01-06 00:00"], tz="UTC")
    expected = pd.Timestamp("2024-01-04", tz="UTC")
    assert (origin_of(t) == expected).all()
    assert ((t - origin_of(t)) / pd.Timedelta(hours=1)).tolist() == [25, 47, 48]


def test_features_do_not_change_when_the_future_is_poisoned():
    obs, nwp = frames()
    clean = build_table(obs, nwp, 0)
    row = clean.iloc[200]
    poisoned = obs.copy()
    poisoned[poisoned.index > row["origin"]] = 999.0
    after = build_table(poisoned, nwp, 0).iloc[200]
    for col in ["obs_last", "obs_mean24", "bias7d", "seasonal", "nwp2", "lead"]:
        assert after[col] == row[col] or (np.isnan(after[col]) and np.isnan(row[col])), col


def test_injected_faults_are_labelled_and_change_the_data():
    obs, _ = frames(days=60)
    y, labels = inject_faults(obs.round(), n_per_type=3, seed=1)
    assert set(labels.unique()) == {"", "stuck", "spike", "zero", "degraded"}
    changed = (y != obs.round()) & (labels != "")
    assert changed.groupby(labels).any().drop("").all()
    m = detection_metrics(labels != "", labels, y)
    assert m.loc[m.fault == "all", "hour_recall"].iloc[0] == 1.0
    assert m.loc[m.fault == "all", "false_alarm_rate"].iloc[0] == 0.0
