# Irish Wind Data QC and Day-Ahead Forecast Correction

This project checks hourly wind observations at Irish weather stations and evaluates corrections to day-ahead wind forecasts. It uses Met Éireann station records and Open-Meteo archived numerical weather predictions (NWP).

## Data and methods

| Source | Use |
|---|---|
| [Met Éireann hourly records](https://www.met.ie/climate/available-data/historical-data) | Observed wind speed, direction, and wind-speed quality indicators |
| [Open-Meteo Previous Runs](https://open-meteo.com/en/docs/previous-runs-api) | Archived 10 m wind forecasts at multiple lead times |

The QC checks cover out-of-range values, repeated values, spikes, one-hour jumps, and sustained disagreement with a station-calibrated NWP reference. `scripts/run_qc_eval.py` calibrates thresholds on injected faults in 2024 and evaluates fresh injected faults in 2025–26 across eight stations. It exports rule-level results, flag rates, and cleaning logs in `results/`.

The forecast experiment in `scripts/run_forecast.py` compares persistence, seasonal naive, climatology, raw NWP, linear model output statistics, and LightGBM corrections. Models are retrained monthly. Forecast features use observations available at the 00:00 UTC issue time for the following UTC day. Models are scored on common observed target hours, with skill intervals resampled by forecast day.

Met Éireann wind indicators are incomplete fault labels. The loader treats codes 3–6 as defective or partially defective; code 111 is not counted as a confirmed defect. Agreement with these indicators should not be read as a verified fault-detection rate. The injected-fault evaluation measures performance on the simulated fault types specified in `src/windqc/faults.py`.

## Setup and use

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,model]"
pytest -q
ruff check .
```

Download station and NWP files into `data/raw/` before running the experiments:

```bash
python scripts/download_met.py 532 1575 2375 2275 518 3904 1075 275
python scripts/download_nwp.py 2024-01-01 2026-09-30
python scripts/run_qc_eval.py
python scripts/run_forecast.py
```

To check one downloaded station file, run:

```bash
python scripts/run_qc.py data/raw/hly532.csv data/processed/dublin
```

This writes `.flags.csv`, `.cleaning.csv`, and `.metrics.json` files using the supplied output prefix. The API exposes `GET /health` and `POST /qc`:

```bash
uvicorn windqc.api:app --reload
```

The Docker image runs the API:

```bash
docker build -t windqc .
docker run -p 8000:8000 windqc
```

## Results and limitations

The committed [QC results](results/) include detection metrics for injected faults, flag rates on unmodified observations, and a comparison with historical Met Éireann indicators. The real-label comparison has substantially lower agreement than the injected-fault experiment. Forecast evaluation code is present, but forecast result files are not part of this committed version.

The QC rules use centred windows and are intended for retrospective cleaning. They should not be used as issue-time forecast features without a causal implementation. The NWP lead-time assumption must also be checked against the archive's forecast availability before operational use.

Background sources and methodology are listed in [docs/READING.md](docs/READING.md).

## Attribution

Data sources: Met Éireann and Open-Meteo. Check each provider's current licence and attribution terms before redistributing their data.
