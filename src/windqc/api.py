"""HTTP service: run the QC checks on a posted series."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from windqc import __version__
from windqc.qc import run_checks

app = FastAPI(title="windqc", version=__version__)


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
