import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from windqc.api import app
from windqc.backtest import rolling_origin, score
from windqc.baselines import seasonal_naive
from windqc.evaluation import qc_metrics
from windqc.qc import flag_flatline, flag_step, run_checks, to_cleaning_log


def test_missing_source_hour_does_not_shift_seasonal_forecast():
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    y = pd.Series(np.arange(48.0), index=idx).drop(idx[25])
    result = seasonal_naive(y, 24)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 26


def test_backtest_preserves_missing_target_hour():
    idx = pd.date_range("2024-01-01", periods=96, freq="h", tz="UTC")
    y = pd.Series(3.0, index=idx).drop(idx[49])

    def constant(history, horizon):
        return pd.Series(3.0, index=pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"))

    result = rolling_origin(y, {"constant": constant}, start="2024-01-02", end="2024-01-02 23:00")
    assert len(result) == 24
    assert pd.isna(result.loc[result.target == idx[49], "y_true"]).all()
    assert result.loc[result.target == idx[50], "y_true"].iloc[0] == 3


def test_score_uses_common_targets_for_skill():
    origin = pd.Timestamp("2024-01-01", tz="UTC")
    targets = [origin + pd.Timedelta(hours=i) for i in (1, 2)]
    rows = [
        (origin, "reference", targets[0], 0.0, 1.0),
        (origin, "reference", targets[1], 100.0, 1.0),
        (origin, "candidate", targets[0], 1.0, 1.0),
        (origin, "candidate", targets[1], np.nan, 1.0),
    ]
    result = pd.DataFrame(rows, columns=["origin", "model", "target", "y_pred", "y_true"])
    table = score(result, reference="reference")
    assert (table.n == 1).all()
    assert table.loc["reference", "rmse"] == 1
    assert table.loc["candidate", "skill_vs_reference"] == 1


def test_qc_does_not_bridge_missing_hours_and_flags_step():
    idx = pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T01:00Z", "2024-01-01T03:00Z", "2024-01-01T04:00Z"])
    y = pd.Series([8.0, 8.0, 8.0, 45.0], index=idx)
    assert not flag_flatline(y, min_run=3).any()
    assert flag_step(y).tolist() == [False, False, False, True]
    gap_flags = pd.Series([True, True, True, False], index=idx)
    assert len(to_cleaning_log(gap_flags, "wdsp", "test")) == 2
    with pytest.raises(ValueError, match="unique"):
        run_checks(pd.concat([y, y.iloc[-1:]]))


def test_qc_metrics_exclude_unknown_labels_and_missing_readings():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    result = qc_metrics(
        pd.Series([True, True, False, True], index=idx),
        pd.Series([3, 0, np.nan, 3], index=idx),
        pd.Series([2.0, 2.0, 2.0, np.nan], index=idx),
    )
    assert result == {"evaluated": 2, "excluded": 2, "tp": 1, "fp": 1, "fn": 0, "tn": 0, "precision": 0.5, "recall": 1.0}


def test_api_rejects_duplicate_or_naive_timestamps():
    client = TestClient(app)
    reading = {"timestamp": "2024-01-01T00:00:00Z", "value": 5}
    assert client.post("/qc", json={"readings": [reading, reading]}).status_code == 422
    assert client.post("/qc", json={"readings": [{"timestamp": "2024-01-01T00:00:00", "value": 5}]}).status_code == 422
