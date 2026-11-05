# TRL assessment – PipeRisk

**Claimed level: TRL 4** (laboratory validation on public real data). Not TRL 5: no pipe attributes, coarse grid, one city.

Data: City of Calgary open data, 11739 breaks in the 1992–2025 panel, 152 grid cells (0.02 degrees) with at least one break. Share of cell-years with a break: train 0.61, calibration 0.54, test 0.54. Breaks per year fell from 531 (1992) to 160 (2025).

Test 2017–2025 (cell-years):

| Method | AUC | Brier | ECE | Top 5 % cells: share of breaks | Top 10 % cells: share of breaks |
|---|---|---|---|---|---|
| baseline: last_year_count | 0.771 | | | 0.158 | 0.271 |
| baseline: last_5y_count | 0.857 | | | 0.170 | 0.288 |
| baseline: cumulative_count | 0.865 | | | 0.153 | 0.266 |
| logistic (raw) | 0.869 | 0.145 | 0.036 | 0.175 | 0.296 |
| logistic (isotonic) | 0.866 | 0.146 | 0.034 | 0.177 | 0.299 |
| boosting (raw) | 0.829 | 0.186 | 0.128 | 0.156 | 0.279 |
| boosting (isotonic) | 0.826 | 0.170 | 0.048 | 0.152 | 0.269 |

## Findings and caveats
- Break history alone ranks cells well (AUC about 0.87), but **a model adds almost nothing over a simple count**: logistic regression has AUC 0.869 against 0.865 for the cumulative count and 0.857 for the 5-year count; top-5 % capture is 0.175 against 0.170.
- Gradient boosting is worse than the baselines (AUC 0.829) and badly calibrated (ECE 0.128); isotonic recalibration on 2013–2016 reduces ECE to 0.048 but not the ranking loss.
- Logistic regression is already well calibrated (ECE 0.036); isotonic recalibration changes little.
- The grid is coarse: with 152 cells and an event rate above 50 %, the top 5 % is only about 7 cells, and they hold 17 % of the breaks, so the ranking is far from a sharp targeting tool.
- The break rate declines over the whole period; the models were trained when the event rate was 0.61 and tested at 0.54, which explains the small over-prediction (mean prediction 0.56 against 0.536 observed).
- No pipe age, material, diameter or soil data; the `break_type` codes were not used because their meaning is not documented in the data. A finer grid was not tried.
