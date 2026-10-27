"""HTTP service: QC checks on a posted series, and day-ahead forecasts from the trained model."""

from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from windqc import __version__
from windqc.dayahead import build_table
from windqc.qc import run_checks

app = FastAPI(title="windqc", version=__version__)
MODEL_DIR = Path(os.environ.get("WINDQC_MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))


class Reading(BaseModel):
    timestamp: datetime
    value: float | None


class QCRequest(BaseModel):
    readings: list[Reading]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/qc")
def qc(req: QCRequest) -> list[dict]:
    timestamps = [r.timestamp for r in req.readings]
    if len(timestamps) != len(set(timestamps)):
        raise HTTPException(status_code=422, detail="timestamps must be unique")
    if any(ts.tzinfo is None for ts in timestamps):
        raise HTTPException(status_code=422, detail="timestamps must include a timezone")
    s = pd.Series(
        [r.value for r in req.readings],
        index=pd.to_datetime(timestamps, utc=True),
        dtype="float64",
    ).sort_index()
    flags = run_checks(s)
    return [
        {"timestamp": ts.isoformat(), "value": None if pd.isna(v) else float(v), **flags.loc[ts].to_dict()}
        for ts, v in s.items()
    ]


class NWPValue(BaseModel):
    timestamp: datetime
    speed: float | None
    direction: float | None = None


class ForecastRequest(BaseModel):
    station: str
    origin: datetime
    observations: list[Reading]
    nwp_day_ahead: list[NWPValue]


@lru_cache(maxsize=1)
def _model():
    import lightgbm as lgb

    meta = json.loads((MODEL_DIR / "lgbm_dayahead.json").read_text())
    return lgb.Booster(model_file=str(MODEL_DIR / "lgbm_dayahead.txt")), meta


@app.post("/forecast")
def forecast(req: ForecastRequest) -> list[dict]:
    """Corrected wind forecast for every hour of the day after ``origin`` (00:00 UTC).

    ``observations`` should cover the 7 days up to ``origin``. ``nwp_day_ahead`` holds the NWP
    values predicted 48 h before each valid time (Open-Meteo ``_previous_day2``), for the same
    7 days and the 48 hours after ``origin``.
    """
    if req.origin.tzinfo is None:
        raise HTTPException(status_code=422, detail="origin must include a timezone")
    origin = pd.Timestamp(req.origin).tz_convert("UTC")
    if origin != origin.normalize():
        raise HTTPException(status_code=422, detail="origin must be 00:00 UTC")
    try:
        booster, meta = _model()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="model not trained; run scripts/run_forecast.py")
    if req.station not in meta["station_codes"]:
        raise HTTPException(status_code=404, detail=f"unknown station {req.station}")

    idx = pd.date_range(origin - pd.Timedelta(days=7), origin + pd.Timedelta(hours=48), freq="h")
    obs = pd.Series({pd.Timestamp(r.timestamp).tz_convert("UTC"): r.value for r in req.observations},
                    dtype="float64").reindex(idx)
    obs[obs.index > origin] = float("nan")
    nwp = pd.DataFrame(
        {"nwp2": {pd.Timestamp(v.timestamp).tz_convert("UTC"): v.speed for v in req.nwp_day_ahead},
         "dir2": {pd.Timestamp(v.timestamp).tz_convert("UTC"): v.direction for v in req.nwp_day_ahead}},
        dtype="float64",
    ).reindex(idx)
    table = build_table(obs, nwp, meta["station_codes"][req.station])
    rows = table[table["origin"] == origin]
    if len(rows) < 24:
        raise HTTPException(status_code=422, detail="need NWP values for all 24 hours of the target day")
    pred = booster.predict(rows[meta["features"]])
    return [
        {"timestamp": t.isoformat(), "nwp": float(n), "forecast": round(float(p), 2)}
        for t, n, p in zip(rows["target"], rows["nwp2"], pred)
    ]
