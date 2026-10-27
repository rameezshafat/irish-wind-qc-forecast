import numpy as np
import pandas as pd

from windqc.qc import flag_flatline, flag_out_of_range, flag_spike, missing_hours, to_cleaning_log


def hourly(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h", tz="UTC"), dtype="float64")


def test_out_of_range():
    s = hourly([5, -1, 120, np.nan])
    assert flag_out_of_range(s).tolist() == [False, True, True, False]


def test_flatline_flags_long_run_only():
    s = hourly([3, 7, 7, 7, 7, 7, 7, 4, 4, 5])
    flags = flag_flatline(s, min_run=6)
    assert flags.tolist() == [False] + [True] * 6 + [False] * 3


def test_flatline_ignores_calm():
    s = hourly([0.0] * 10)
    assert not flag_flatline(s, min_run=6).any()


def test_spike_flags_isolated_jump():
    values = [10, 11, 10, 12, 11, 60, 11, 10, 12, 11, 10]
    flags = flag_spike(hourly(values))
    assert flags.iloc[5]
    assert flags.sum() == 1


def test_missing_hours_counts_gaps_and_nans():
    s = hourly([1, 2, np.nan, 4])
    s = s.drop(s.index[1])
    assert len(missing_hours(s)) == 2


def test_cleaning_log_merges_consecutive_flags():
    flags = hourly([0, 1, 1, 0, 1]).astype(bool)
    log = to_cleaning_log(flags, sensor="Spd", reason="Flatline")
    assert list(log.columns) == ["Sensor", "Start", "Stop", "Reason"]
    assert len(log) == 2
    assert log.iloc[0]["Start"] == flags.index[1] and log.iloc[0]["Stop"] == flags.index[2]
