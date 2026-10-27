# Methods and references

## Wind data quality control

- [Montana Mesonet quality-control procedures](https://climate.umt.edu/mesonet/qaqc/) describe range, step, persistence, and comparison checks.
- [RMI Belgium: quality control of automated weather-station wind data](https://www.adv-sci-res.net/13/13/2016/asr-13-13-2016.pdf) discusses temporal and spatial consistency checks.
- [brightwind documentation](https://brightwind-dev.github.io/brightwind-docs/tutorials/tutorial_welcome.html) describes wind-data coverage and cleaning workflows. This project exports a cleaning log with `Sensor`, `Start`, `Stop`, and `Reason` columns.

## Forecasting and evaluation

- [Forecasting: Principles and Practice, the Pythonic Way](https://otexts.com/fpppy/) covers benchmark forecasts, accuracy measures, and time-series evaluation.
- [scikit-learn: time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) illustrates cyclical calendar features and time-aware validation.
- [scikit-learn `TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) documents chronological splits and the `gap` parameter.
- [Probabilistic day-ahead forecasting of system-level renewable energy and electricity demand](https://www.nature.com/articles/s41467-026-69015-w) discusses day-ahead evaluation and skill scores.
- [Approaches for multi-step density forecasts with application to aggregated wind power](https://arxiv.org/pdf/1003.0996) provides an Irish wind forecasting benchmark.
- [Ensemble model output statistics for wind forecasts](https://arxiv.org/pdf/2108.12174) describes statistical calibration of numerical weather forecasts.

## MIT OpenCourseWare

- [6.005 Software Construction](https://ocw.mit.edu/courses/6-005-software-construction-spring-2016/) covers specifications, invariants, and testing. The project's timestamp validation and missing-hour tests follow these principles.
- [6.170 Software Studio](https://ocw.mit.edu/courses/6-170-software-studio-spring-2013/) covers modular design and external behavior. Data loading, QC, evaluation, backtesting, and the API are separate modules.
- [14.384 Time Series Analysis](https://ocw.mit.edu/courses/14-384-time-series-analysis-fall-2013/pages/lecture-notes/) covers serial dependence and inference. Forecast origins are chronological, comparisons use common target hours, and skill intervals resample forecast days.
- [18.650 Statistics for Applications](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/) covers estimation and statistical testing relevant to QC threshold calibration.
