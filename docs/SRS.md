# Software Requirements Specification – PipeRisk

- SR-1 `panel.load_breaks`: break records to 0.02-degree grid cells; `count_panel`: yearly break counts per cell that ever had a break (1992–2025; 2026 is partial and excluded).
- SR-2 `panel.neighbour_sum`, `panel.features`: for year t only years before t are used (counts at lags 1–3, 5-year sum, cumulative count, neighbour counts); a unit test checks that later years do not alter the features.
- SR-3 `calib.ece`, `calib.top_share` (share of all breaks falling in the top x % of cells by risk), `calib.fit_isotonic`.
- SR-4 `research/run_study.py`: logistic regression and gradient boosting; train 1997–2012, calibration 2013–2016, test 2017–2025; baselines: last-year, 5-year and cumulative break counts.
- SR-5 Two unit tests.
