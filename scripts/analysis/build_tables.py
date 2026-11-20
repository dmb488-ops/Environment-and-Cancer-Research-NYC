#!/usr/bin/env python3
"""Writes the tidy tables that every analysis (ours and the teammates') can load.

  outputs/analysis/puma_year_panel_2010_2023.csv      55 neighborhoods x 14 years (770 rows): outcomes + 16 factors
  outputs/analysis/puma_period_means_2010_2023.csv    55 rows: pooled 2010-2023 outcomes + factor period means
  outputs/analysis/age_strata_lung_all.csv            cases / person-years by neighborhood x year x age group (lung, all sites)

The first two come from the data embedded in the explorer page (same numbers as the maps). The age strata need the
SEER*Stat export (pass its path):   python3 build_tables.py /path/to/Lungcancerandyearofdiagnosis.txt
"""
import os
import sys

import numpy as np
import pandas as pd

import common as c

sys.path.insert(0, os.path.join(c.REPO, "scripts", "web"))
import seer_build as sb  # noqa: E402


def main(seer_txt=None):
    os.makedirs(c.OUT, exist_ok=True)
    D = c.load_D()
    keys = [L["key"] for L in D["layers"]]
    panel = c.build_panel(D)
    panel.round(4).to_csv(os.path.join(c.OUT, "puma_year_panel_2010_2023.csv"), index=False)

    # ---- cross-section: pooled outcomes + period means of the factors (mean over the years each factor exists)
    cs = panel.groupby("unit")[keys].mean()
    nyr = panel.groupby("unit")[keys].count().iloc[0]
    meta = pd.DataFrame(dict(unit=range(55), puma_name=[u["name"] for u in D["units"]], borough=[u["borough"] for u in D["units"]],
                             puma=[u["puma"] for u in D["units"]], area_km2=[u["area"] for u in D["units"]],
                             cx=[u["cx"] for u in D["units"]], cy=[u["cy"] for u in D["units"]])).set_index("unit")
    meta["pop_mean"] = [np.mean(D["seer"]["pop"][u]) for u in range(55)]
    meta["pop_density"] = meta["pop_mean"] / meta["area_km2"]
    out = meta.join(cs)
    for s in D["seer"]["sites"]:
        out["aa_" + s["key"]] = s["aaP"]
        out["cr_" + s["key"]] = s["crP"]
    out.round(4).to_csv(os.path.join(c.OUT, "puma_period_means_2010_2023.csv"))
    pd.Series(nyr, name="years_with_data_2010_2023").to_csv(os.path.join(c.OUT, "factor_years_available.csv"))

    # ---- age strata
    if seer_txt:
        d = sb.load_export(seer_txt)
        rows = []
        for code, key in ((43, "lung"), (0, "all")):
            sub = d[(d.site == code) & (d.yr > 0)]
            cnt = np.zeros((14, 55, 20))
            pop = np.zeros((14, 55, 20))
            for (yr, u), g in sub.groupby(["yr", "puma"]):
                cnt[yr - 1, u, g.age.values] = g.cnt.values
                pop[yr - 1, u, g.age.values] = g["pop"].values
            cnt, pop = sb.collapse(cnt), sb.collapse(pop)
            for yi in range(14):
                for u in range(55):
                    for a in range(19):
                        rows.append((key, u, 2010 + yi, a, int(cnt[yi, u, a]), int(pop[yi, u, a])))
        pd.DataFrame(rows, columns=["site", "unit", "year", "age_group", "cases", "person_years"]).to_csv(
            os.path.join(c.OUT, "age_strata_lung_all.csv"), index=False)
    print("wrote tables to", c.OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
