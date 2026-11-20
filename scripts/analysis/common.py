"""Shared helpers for the factor-selection analysis.

Reads the data that is embedded in the explorer HTML (so the analysis uses exactly the same numbers as the
maps) and rebuilds the neighborhood x year table of outcomes and candidate factors.
"""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "outputs", "analysis")
HTML = os.path.join(REPO, "outputs", "NYC_Environment_Cancer_Explorer.html")
YEARS = list(range(2010, 2024))

# factor metadata: key -> (short name, group, source geography)
GROUPS = ["Behaviour & socioeconomic", "Green space", "Air quality", "Food environment", "Climate"]
SHORT = {
    "smoke": "Smoking", "pov": "Poverty", "ins": "No insurance", "ndvi": "Greenness (NDVI)",
    "so2": "SO₂", "o3": "Ozone (O₃)", "no2": "NO₂", "no": "NO", "pm25": "PM₂.₅", "bc": "Black carbon",
    "snap": "SNAP retailers", "super": "Supermarkets", "conv": "Convenience stores",
    "tmean": "Mean temperature", "tmax": "Summer max temp.", "precip": "Precipitation",
}


def load_D(html_path=HTML):
    """Parse the JSON blob embedded in the explorer page."""
    t = open(html_path, encoding="utf-8").read()
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', t, re.S)
    return json.loads(m.group(1))


def _geo_based(D, L, yi_series):
    """Port of geoBased() in the explorer: area-weighted blend of source areas. Returns list of 55 floats / nan."""
    S = D["series"][L["key"]]
    if L["geo"] == "puma":
        return [np.nan if S["v"][u][yi_series] is None else float(S["v"][u][yi_series]) for u in range(55)]
    W = D["weights"] if L["geo"] == "uhf" else D["cdW"]
    codes = D["uhfCodes"] if L["geo"] == "uhf" else [str(c["code"]) for c in D["cd"]]
    out = []
    for u in range(55):
        sw = sv = 0.0
        for j, w in W[u]:
            cell = S["v"][codes[j]][yi_series]
            if not cell or cell[0] is None:
                continue
            v = 100 - cell[0] if L["key"] == "ins" else cell[0]
            sw += w
            sv += w * v
        out.append(sv / sw if sw else np.nan)
    return out


def _pov_at(D, y):
    wins = D["pov"]["wins"]
    mid = lambda w: int(w[:4]) + 2
    best, bd = None, 1e9
    for w in wins:
        d = abs(mid(w) - y)
        if d < bd or (d == bd and mid(w) < mid(best)):
            bd, best = d, w
    return [np.nan if D["pov"]["v"][best][u] is None else float(D["pov"]["v"][best][u]) for u in range(55)], best, bd == 0


def build_panel(D=None):
    """Long table: one row per neighborhood x year (770 rows)."""
    D = D or load_D()
    layers = {L["key"]: L for L in D["layers"]}
    rows = []
    sites = {s["key"]: s for s in D["seer"]["sites"]}
    fac = {}
    for k, L in layers.items():
        arr = np.full((55, 14), np.nan)
        for yi, y in enumerate(YEARS):
            if k == "pov":
                arr[:, yi] = _pov_at(D, y)[0]
            else:
                ys = D["series"][k]["years"]
                if y in ys:
                    arr[:, yi] = _geo_based(D, L, ys.index(y))
        fac[k] = arr
    povwin = {y: _pov_at(D, y)[1:] for y in YEARS}
    for u in range(55):
        un = D["units"][u]
        for yi, y in enumerate(YEARS):
            r = dict(unit=u, puma_name=un["name"], borough=un["borough"], puma=un["puma"], area_km2=un["area"], year=y,
                     pop=D["seer"]["pop"][u][yi])
            for sk, pre in (("lung", "lung"), ("all", "all")):
                S = sites[sk]
                r[pre + "_aa"], r[pre + "_aa_lo"], r[pre + "_aa_hi"] = S["aa"][u][yi], S["lo"][u][yi], S["hi"][u][yi]
                r[pre + "_cr"], r[pre + "_cnt"] = S["cr"][u][yi], S["cnt"][u][yi]
            for k in layers:
                r[k] = fac[k][u, yi]
            r["pov_window"], r["pov_window_exact"] = povwin[y][0], int(povwin[y][1])
            rows.append(r)
    df = pd.DataFrame(rows)
    df["pop_density"] = df["pop"] / df["area_km2"]
    return df
