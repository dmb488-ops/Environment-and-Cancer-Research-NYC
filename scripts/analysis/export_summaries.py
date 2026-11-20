#!/usr/bin/env python3
"""Flat CSV summaries of outputs/analysis/analysis_results.json, for teammates who prefer a spreadsheet.

    python3 export_summaries.py [results.json]

Writes next to the JSON:
  factor_verdicts_lung_aa.csv / factor_verdicts_all_aa.csv   one row per factor: evidence from each method, verdict
  collinearity_vif.csv                                        VIF with all 16 factors and in the core set, and the pruning decision
  model_performance.csv                                       cross-validated R2 (random 5-fold and leave-one-borough-out) by feature set and model
  regression_coefficients_lung_aa.csv                         effect of one SD of each factor: OLS, adjusted for smoking, Poisson, panel fixed effects
  panel_variation_shares.csv                                  how much of each factor's 2012-2022 variation is between neighborhoods, citywide by year,
                                                              or neighborhood-specific change (the only part a fixed-effects test can use)
"""
import json
import os
import sys

import pandas as pd

import common as c

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(c.OUT, "analysis_results.json")
out = os.path.dirname(os.path.abspath(path))
R = json.load(open(path))
M = R["meta"]
KEYS, NAMES, CORE = M["keys"], M["names"], M["core"]
METHODS = M["methods"]

for o in ("lung_aa", "all_aa"):
    C = R["consensus"][o]
    rows = []
    for k in KEYS:
        s = C[k]["summary"]
        r = {"factor": NAMES[k], "group": R["screening"][k]["group"]}
        for m in METHODS:
            e = C[k][m["key"]]
            r[m["name"]] = ("supports" if e["support"] else "tested, not supported") if e["tested"] else "not tested"
        r.update(methods_supporting=s["n_support"], methods_tested=s["n_tested"], direction={1: "higher factor, higher incidence", -1: "higher factor, lower incidence", 0: "none"}[s["direction"]],
                 scientific_case=s["rating"], theory_expects={"+": "more cancer", "-": "less cancer", "?": "unclear"}[s["expected"]],
                 contradicts_theory=s["contradicts_theory"], location_fingerprint=s["location_proxy"], verdict=s["recommendation"])
        rows.append(r)
    order = {"selected": 0, "flag": 1, "hypothesis": 2, "dropped": 3}
    pd.DataFrame(rows).sort_values("verdict", key=lambda s: s.map(order), kind="stable").to_csv(os.path.join(out, "factor_verdicts_%s.csv" % o), index=False)

Cc = R["collinearity"]
pd.DataFrame([{"factor": NAMES[k], "vif_all_16": round(Cc["vif_all"][k], 2), "kept_in_core_set": k in CORE,
               "vif_core_set": round(Cc["core"]["vif"][k], 2) if k in CORE else None, "reason_dropped": Cc["core"]["dropped"].get(k, "")} for k in KEYS]
             ).sort_values("vif_all_16", ascending=False).to_csv(os.path.join(out, "collinearity_vif.csv"), index=False)

rows = []
for o, label in (("lung_aa", "lung, age-adjusted"), ("all_aa", "all sites, age-adjusted")):
    for b in R["cross_bench"][o]:
        rows.append({"outcome": label, "features": b["features"], "model": b["model"], "cv_r2_random_5fold": round(b["cv_r2"], 3), "cv_r2_leave_one_borough_out": round(b["lobo_r2"], 3)})
pd.DataFrame(rows).to_csv(os.path.join(out, "model_performance.csv"), index=False)

X, P = R["cross"]["lung_aa"], R["panel"]["lung_aa"]
rows = []
for k in KEYS:
    ols = X["ols"]["coef"].get(k)
    adj = X["smoke_alone"] if k == "smoke" else X["adj_smoke"][k]
    po = X["poisson"]["coef"].get(k)
    fe = P["fe"]["coef"].get(k)
    rows.append({"factor": NAMES[k],
                 "ols_core_beta_rate_points_per_SD": ols and round(ols["beta"], 2), "ols_core_p": ols and round(ols["p"], 4),
                 "ols_adjusted_for_smoking_beta": round(adj["beta"], 2), "ols_adjusted_for_smoking_p": round(adj["p"], 4),
                 "poisson_core_rate_ratio_per_SD": po and round(po["irr"], 3), "poisson_core_p": po and round(po["p"], 4),
                 "panel_fixed_effects_beta": fe and round(fe["beta"], 2), "panel_fixed_effects_p": fe and round(fe["p"], 4)})
pd.DataFrame(rows).to_csv(os.path.join(out, "regression_coefficients_lung_aa.csv"), index=False)

# share of the annual-panel variation that a within-neighborhood (fixed-effects) test can use
pan = pd.read_csv(os.path.join(out, "puma_year_panel_2010_2023.csv"))
pan = pan[pan["year"].between(min(M["panel_years"]), max(M["panel_years"]))]
rows = []
for k in KEYS + ["lung_aa"]:
    if pan[k].notna().sum() < len(pan):
        continue  # SO2 stops in 2016: not in the annual panel
    x = pan[k]
    nb = x.groupby(pan["puma"]).transform("mean")
    yr = x.groupby(pan["year"]).transform("mean")
    rows.append({"variable": NAMES.get(k, "Lung cancer incidence (age-adjusted)"), "between_neighborhoods": round(nb.var() / x.var(), 3),
                 "citywide_by_year": round(yr.var() / x.var(), 3), "neighborhood_specific_change": round((x - nb - yr + x.mean()).var() / x.var(), 3)})
pd.DataFrame(rows).to_csv(os.path.join(out, "panel_variation_shares.csv"), index=False)
print("wrote summaries in", out)
