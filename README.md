# articles-PipeRisk

Calibrated spatial risk of water-main breaks from break history only, on real municipal open data
(City of Calgary "Water Main Breaks", 37 558 breaks 1956–2026). TRL 4 (public real data, laboratory validation only).

```
pip install -e ".[dev]" && python -m pytest
# data: https://data.calgary.ca/resource/dpcu-jr23 (fields break_date, break_type, status, point); save as CSV with
# columns date,type,status,lon,lat (not committed)
python research/run_study.py <calgary_breaks.csv>
```
