# Environment and Cancer Research NYC

**Research question:** how do environmental factors affect cancer incidence in New York City?

## Status
- [x] Data preparation
- [x] Lung and bronchus cancer incidence maps by PUMA, 2010–2023
- [x] Overview of four machine learning models (slides)
- [ ] Models: Linear Regression, Decision Tree, Random Forest, MLP

## Repository layout
| Folder | Contents |
|---|---|
| `outputs/maps/` | 14 yearly maps (2010–2023), a 14-year average map, and a one-page overview of all years |
| `outputs/puma_crosswalk.csv` | Links the table's 0–54 neighborhood codes to Census 2010 PUMA codes |
| `scripts/` | `make_puma_maps.py` and `puma_geo.py`, which regenerate every map |
| `slides/` | `ML_Models_Overview.pptx`: definitions and how each model could be used in this project |
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

`outputs/NYC_Environment_Cancer_Explorer.html` is a single offline page (open it in any browser) with four tabs:

1. **Data browser** – every sheet of the Excel workbooks, the smoking / insurance / NDVI / air-quality CSVs, the food-environment & climate PUMA file, and computed SEER*Stat summaries (searchable, sortable, CSV download).
2. **Cancer incidence maps** – the 14 yearly lung maps (plus pooled average) with click-for-details; crude or age-adjusted rates, lung or any of 19 sites.
3. **Factor layers** – 15 layers (smoking, poverty, uninsured, NDVI, six air pollutants, SNAP retailers, supermarkets, convenience stores, three climate variables) that can be switched on and off; one layer shows its raw value, several show a weighted combined burden index. Correlation panel compares crude and age-adjusted lung incidence.
4. **Age-adjusted incidence** – direct age standardisation (2000 US standard population, both sexes) of the SEER*Stat PUMA export with 95% CIs (Fay-Feuer), crude vs adjusted ranks and age-specific curves.

Rebuild with `scripts/web/build_data.py` then `scripts/web/assemble.py` (paths at the top of `build_data.py` point at the source files and need adjusting on your machine).

Caveats: smoking, insurance and NDVI are UHF34-level and air quality is community-district-level; both are transferred to the 55 neighborhoods by area overlap (approximate). The Duc workbooks' cancer sheet is all cancers, not lung. The SEER*Stat export has no sex split, so sex-specific sites are not shown.
