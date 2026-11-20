# Environment and Cancer Research NYC

**Research question:** how do environmental factors affect cancer incidence in New York City?

## Status
- [x] Data preparation
- [x] Lung and bronchus cancer incidence maps by PUMA, 2010–2023
- [x] Overview of four machine learning models (slides)
- [x] Interactive explorer (maps, factor layers, age-adjusted rates)
- [x] Factor-selection analysis: screening, multicollinearity, theory, eight statistical and machine-learning methods (preliminary, for group review)
- [ ] Group review of the theory table and the selected factors
- [ ] Next models: Decision Tree, MLP, spatial models

## Repository layout
| Folder | Contents |
|---|---|
| `outputs/maps/` | 14 yearly maps (2010–2023), a 14-year average map, and a one-page overview of all years |
| `outputs/puma_crosswalk.csv` | Links the table's 0–54 neighborhood codes to Census 2010 PUMA codes |
| `scripts/` | `make_puma_maps.py` and `puma_geo.py`, which regenerate every map |
| `slides/` | `ML_Models_Overview.pptx`: definitions and how each model could be used in this project |
| `outputs/NYC_Environment_Cancer_Explorer.html` | The offline explorer page (five tabs, see below) |
| `outputs/analysis/` | Tidy analysis tables (`puma_year_panel_2010_2023.csv` etc.), `analysis_results.json` with every number of the factor-selection analysis, and CSV summaries |
| `scripts/analysis/` | The factor-selection analysis: `build_tables.py`, `run_analysis.py`, `export_summaries.py`, `stats_lib.py`, `ml_lib.py`, `theory.py` |
| `scripts/web/` | `template.html` and `assemble.py` (and `build_data.py`) that build the explorer page |
| `docs/` | `Factor_Selection_Methods.md`: methods, results and cautions of the factor-selection analysis |
| `data/` | Raw data files. Kept out of git (see `.gitignore`) |

## Maps
Crude lung and bronchus cancer incidence per 100,000 residents for each of the 55 NYC PUMAs. All maps share one 7-shade color scale, so years are directly comparable.

Main patterns:
- Staten Island is highest (14-year rate 77 per 100,000, vs 55 Manhattan, 50 Queens, 46 Brooklyn, 45 Bronx).
- High rates sit on the outer edge of the city (south Brooklyn, northeast Queens); the lowest are Bushwick, Williamsburg & Greenpoint, and Jackson Heights.
- The spatial pattern is stable over time (rank correlation 0.82 between 2010–12 and 2021–23).
- 2020 shows a clear dip (3,760 new cases vs 4,558 in 2019), most likely due to pandemic disruption of screening and diagnosis.

Caveats: rates are crude (not age-adjusted), and a typical PUMA has only about 70 cases a year, so one-year changes between shades are often noise.

## Re-running the maps
Needs Python with `numpy`, `pandas`, `openpyxl` and `matplotlib`, plus a clone of [nycehs/NYC_geography](https://github.com/nycehs/NYC_geography) for the boundaries.

```
python scripts/make_puma_maps.py \
  --xlsx data/NYC_Lung_and_Bronchus_Cancer_Integrated_Table_2010_2023.xlsx \
  --repo path/to/NYC_geography \
  --out outputs/maps
```

The table's "PUMA Code" (0–54) is a running index, not the Census PUMA code; use `outputs/puma_crosswalk.csv` to join to other PUMA data. The two Midtown neighborhoods are defined differently in the table than in the Census file, so the script re-cuts one PUMA along the Community District 4|5 line.

## Data sources
- NYC Spatial-Temporal Cancer Dataset (lung and bronchus incidence, 2010–2023), provided by the research group. Not included in this repository.
- PUMA boundaries: [nycehs/NYC_geography](https://github.com/nycehs/NYC_geography).

## Interactive explorer

`outputs/NYC_Environment_Cancer_Explorer.html` is a single offline page (open it in any browser) with five tabs:

1. **Data browser** – every sheet of the Excel workbooks, the smoking / insurance / NDVI / air-quality CSVs, the food-environment & climate PUMA file, and computed SEER*Stat summaries (searchable, sortable, CSV download).
2. **Cancer incidence maps** – the 14 yearly lung maps (plus pooled average) with click-for-details; crude or age-adjusted rates, lung or any of 19 sites.
3. **Factor layers** – 16 layers (smoking, poverty, uninsured, NDVI, six air pollutants, SNAP retailers, supermarkets, convenience stores, three climate variables) that can be switched on and off; one layer shows its raw value, several show a weighted combined burden index. Correlation panel compares crude and age-adjusted lung incidence.
4. **Age-adjusted incidence** – direct age standardisation (2000 US standard population, both sexes) of the SEER*Stat PUMA export with 95% CIs (Fay-Feuer), crude vs adjusted ranks and age-specific curves.
5. **Important factors** – the result of the factor-selection analysis (see below): a verdict table with the evidence from eight methods, screening scatter plots and a 16 × 19 cancer-site heat map, the multicollinearity diagnostics (correlation matrix, VIF, what collinearity does to a coefficient), the theory table, model performance, regression and machine-learning results, the annual panel and sensitivity checks. One button shows the selected factors on the map in tab 3.

Rebuild with `scripts/web/build_data.py` then `scripts/web/assemble.py` (paths at the top of `build_data.py` point at the source files and need adjusting on your machine). If you only changed the analysis, `python3 scripts/web/assemble.py` alone re-embeds `outputs/analysis/analysis_results.json` into the existing page.

Caveats: smoking, insurance and NDVI are UHF34-level and air quality is community-district-level; both are transferred to the 55 neighborhoods by area overlap (approximate). The Duc workbooks' cancer sheet is all cancers, not lung. The SEER*Stat export has no sex split, so sex-specific sites are not shown.

## Factor-selection analysis (preliminary, for group review)

Which environmental factors matter most for lung cancer incidence across the 55 neighborhoods? The full explanation (plain-language multicollinearity primer, theory table, every method and why, results, cautions) is in [`docs/Factor_Selection_Methods.md`](docs/Factor_Selection_Methods.md); the results are also browsable in tab 5 of the explorer.

**What was done**
- 16 candidate factors (smoking, poverty, no insurance, NDVI, six air pollutants, three food-environment, three climate) against age-adjusted lung cancer incidence, in two designs: between the 55 neighborhoods (2010–2023 averages) and an annual panel (55 × 11 years, 2012–2022).
- Multicollinearity: correlation matrix, VIF and condition number; the 16 factors were cut to a 7-factor core set by rules fixed in advance (highest VIF 4.8, condition number 5.0).
- Eight methods: Spearman screen, OLS adjusted for smoking, OLS on the core set, Poisson with age strata, panel two-way fixed effects, Elastic Net with stability selection, Random Forest and Gradient Boosting (permutation importance, random 5-fold and leave-one-borough-out cross-validation).
- A factor is **selected** only if most methods agree, the direction matches theory, the support survives holding out a whole borough, and there is a scientific case. Benchmarks, Moran's I and eight sensitivity checks are included.

**Headline results (lung, age-adjusted)**
- Selected: **smoking** only (+6.8 rate points per 100,000 per SD, 95% CI +5.2 to +8.5; supported by 6 of 7 methods).
- Flagged: precipitation (no-mechanism negative control), no insurance (most likely detection), SO₂ (location fingerprint). Hypothesis only (good scientific case, no support in these data): PM₂.₅, NO₂, black carbon, greenness, poverty. Seven factors not supported.
- Environmental factors alone predict nothing across boroughs (cross-validated R² 0.15 random hold-out, 0.00 whole borough held out); smoking + poverty + insurance reach 0.54 on both tests.
- Absence of evidence is not evidence of absence: 55 neighborhoods, blended UHF / community-district data and an ecological design leave wide intervals. Results are associations, not causes.

**Files**
- `outputs/analysis/analysis_results.json`: every number; `factor_verdicts_*.csv`, `collinearity_vif.csv`, `model_performance.csv`, `regression_coefficients_lung_aa.csv`, `panel_variation_shares.csv`: spreadsheet-friendly summaries; `puma_year_panel_2010_2023.csv`, `puma_period_means_2010_2023.csv`: tidy tables for your own analyses.
- `scripts/analysis/`: `build_tables.py`, `stats_lib.py`, `ml_lib.py`, `run_analysis.py`, `export_summaries.py`, `theory.py`, `test_stats_lib.py`.

**Re-running** (Python with NumPy, pandas, SciPy and scikit-learn; seeds are fixed, so numbers repeat):

```
python3 scripts/analysis/build_tables.py [path to SEER*Stat export]   # tidy tables
python3 scripts/analysis/test_stats_lib.py                             # checks the hand-written statistics
python3 scripts/analysis/run_analysis.py                               # about 25 minutes (QUICK=1 for a 1-minute dry run)
python3 scripts/analysis/export_summaries.py                           # flat CSV summaries
python3 scripts/web/assemble.py                                        # re-embed results in the explorer page
```

statsmodels and SHAP were not available when this was written, so the regression, Poisson and fixed-effects code is plain NumPy (checked against scikit-learn and SciPy) and importance is permutation importance.
