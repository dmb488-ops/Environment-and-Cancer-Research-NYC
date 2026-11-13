#!/usr/bin/env python3
"""Builds data.json for the NYC cancer / environment explorer (geometry + all tables + UHF->PUMA weights)."""
import hashlib, json, os, re, sys
import numpy as np
import pandas as pd

SCR = "/tmp/claude-0/-home-claude/f5a04314-3772-58d8-a22c-6ae2ba318cf0/scratchpad"
sys.path.insert(0, SCR)
import make_puma_maps as m          # noqa: E402
import puma_geo as g                # noqa: E402

ATT = "/mnt/user-data/attachments/f5a04314-3772-58d8-a22c-6ae2ba318cf0/"
REPO = "/home/claude/nycehs/nyc_geography"
F_LUNG = ATT + "2b7541db-NYC_Lung_and_Bronchus_Cancer_Integrated_Table_2010_2023_1.xlsx"
F_CLEAN = ATT + "4b017d8f-Duc_NYC_Cancer_Smoking_Socioeconomic_Cleaned_1.xlsx"
F_FULL = ATT + "fd513eda-Duc_Fully_Integrated_Cancer_Socioeconomic_Dataset_2.xlsx"
F_SMOKE = ATT + "cb9fbb83-Smoking_Data_Neighborhood_Level_2002_-2022_2.csv"
F_POV = ATT + "333397fe-Poverty_Neighborhood_Level_2007_-2019_2.csv"
F_INS = ATT + "8805b28f-Health_insurance_Neighborhood_Level_2012_-2022_2.csv"
F_INI = ATT + "0b33c553-Cancer_incidence_read_by_SEERstat.ini"
F_NDVI_UHF = ATT + "ac6410dd-NYC_NDVI_UHF34_2009_2025.csv"
F_NDVI_TRACT = ATT + "8e83ac30-NYC_NDVI_2009_2025.csv"
F_PUMAX = ATT + "197e6a34-NYC_PUMA_2006_2025.xlsx"
F_PUMACSV = ATT + "589c4b18-NYC_PUMA_2006_2025.csv"
AIR = [  # key, file, column, name, unit label, suffix, dec, step
    ("so2", "e28e0551-NYC_EH_Data_Portal_-_Sulfur_dioxide_SO2_full_table.csv", "Winter mean ppb", "Sulfur dioxide (SO\u2082), winter mean", "ppb", " ppb", 1, 0.1),
    ("o3", "a2c47a36-NYC_EH_Data_Portal_-_Ozone_O3_full_table.csv", "Summer mean ppb", "Ozone (O\u2083), summer mean", "ppb", " ppb", 1, 0.5),
    ("no2", "4523df39-NYC_EH_Data_Portal_-_Nitrogen_dioxide_NO2_full_table.csv", "Annual mean ppb", "Nitrogen dioxide (NO\u2082), annual mean", "ppb", " ppb", 1, 0.5),
    ("no", "064b926d-NYC_EH_Data_Portal_-_Nitric_oxide_NO_full_table.csv", "Annual mean ppb", "Nitric oxide (NO), annual mean", "ppb", " ppb", 1, 0.5),
    ("pm25", "17e4c797-NYC_EH_Data_Portal_-_Fine_particles_PM_2.5_full_table.csv", "Annual mean mcg/m3", "Fine particles (PM\u2082.\u2085), annual mean", "\u00b5g/m\u00b3", " \u00b5g/m\u00b3", 1, 0.1),
    ("bc", "9bb0b555-NYC_EH_Data_Portal_-_Black_carbon_full_table.csv", "Annual mean absorbance units", "Black carbon, annual mean", "absorbance units", "", 2, 0.05),
]
F_SEER = "/mnt/user-data/uploads/Lungcancerandyearofdiagnosis.txt"
F_DIC = ATT + "37c6dec5-Lungcancerandyearofdiagnosis.dic"
sys.path.insert(0, os.path.join(SCR, "web"))
import seer_build as sb          # noqa: E402
OUT = os.path.join(SCR, "web", "data.json")

YEARS = list(range(2010, 2024))
S = 40  # svg units per km
X0, Y0 = 22.9, 25.2   # shift so extent is positive; y flipped
W_, H_ = int((26.6 + X0) * S) + 1, int((25.2 + 23.9) * S) + 1


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def px(p):
    return int(round((p[0] + X0) * S)), int(round((Y0 - p[1]) * S))


def ring_d(r):
    pts, last = [], None
    for p in r[:-1]:
        q = px(p)
        if q != last:
            pts.append(q)
            last = q
    if len(pts) < 3:
        return ""
    s = "M%d %d" % pts[0]
    prev = pts[0]
    for q in pts[1:]:
        s += "l%d %d" % (q[0] - prev[0], q[1] - prev[1])
        prev = q
    return s + "z"


def polys_d(parts):
    return "".join(ring_d(r) for polys in parts for poly in polys for r in poly)


def num(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None


# ------------------------------------------------------------------ geometry
units, puma, bnames = m.build_units(REPO)
rates, cases, nyc = m.load_table(F_LUNG)
names = {u: str(rates.loc[u, "PUMA Neighborhood Name"]) for u in range(55)}


def borough(u):
    if u in (1, 9):
        return "Manhattan"
    pc = m.PUMA_OF[u]
    return {37: "Bronx", 38: "Manhattan", 39: "Staten Island", 40: "Brooklyn", 41: "Queens"}[pc // 100]


unit_out = []
for u in range(55):
    parts = units[u]
    area = sum(g.area_km2(p) for p in parts)
    # label anchor: interior point of largest part
    big = max(parts, key=g.area_km2)
    ip = g.interior_point(big)
    cx, cy = px(ip)
    unit_out.append(dict(id=u, name=names[u], borough=borough(u), puma=m.PUMA_OF.get(u),
                         area=round(area, 2), d=polys_d(parts), cx=cx, cy=cy,
                         note=m.SPECIAL.get(u, "")))

internal, coast = m.classify_edges(units)


def seg_d(segs):
    out = []
    for a, b in segs:
        (x1, y1), (x2, y2) = px(a), px(b)
        if (x1, y1) != (x2, y2):
            out.append("M%d %dL%d %d" % (x1, y1, x2, y2))
    return "".join(out)


internal_d, coast_d = seg_d(internal), seg_d(coast)

# ------------------------------------------------------------------ UHF34 geometry + overlap weights
uf = json.load(open(os.path.join(REPO, "UHF34.geo.json")))["features"]
uhf = []
for f in uf:
    pr = f["properties"]
    if not pr["GEOCODE"]:
        continue
    polys = g.polys_from_geometry(f["geometry"])
    uhf.append(dict(code=str(pr["GEOCODE"]), name=pr["GEONAME"], polys=polys))
uhf.sort(key=lambda x: int(x["code"]))
print("UHF34 polygons:", len(uhf))

step = 0.05
xs = np.arange(-22.9 + step / 2, 26.6, step)
ys = np.arange(-23.9 + step / 2, 25.2, step)
GX, GY = np.meshgrid(xs, ys)
pts = np.column_stack([GX.ravel(), GY.ravel()])


def label_grid(polylist):
    """polylist: list of polys. returns boolean mask (len pts)"""
    hit = np.zeros(len(pts), bool)
    for polys in polylist:
        x0, y0, x1, y1 = g.bbox(polys)
        sel = np.where((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))[0]
        if len(sel):
            hit[sel] |= g.inside(polys, pts[sel])
    return hit


unit_mask = [label_grid(units[u]) for u in range(55)]
uhf_mask = [label_grid([x["polys"]]) for x in uhf]
M = np.zeros((55, len(uhf)))
for i, um in enumerate(unit_mask):
    for j, hm in enumerate(uhf_mask):
        M[i, j] = (um & hm).sum()
tot = np.array([um.sum() for um in unit_mask], float)
cov = M.sum(1) / tot
print("unit coverage by UHF34: min %.3f median %.3f" % (cov.min(), np.median(cov)))
# for each unit: weights over UHFs, normalised over covered area; drop tiny slivers (<2%)
weights = []
for i in range(55):
    w = M[i] / M[i].sum()
    w[w < 0.02] = 0
    w = w / w.sum()
    weights.append([[int(j), round(float(w[j]), 4)] for j in np.nonzero(w)[0]])
dom = [max(wl, key=lambda t: t[1]) for wl in weights]
print("pure (>=95% one UHF):", sum(1 for d in dom if d[1] >= .95), "of 55")

uhf_out = [dict(code=x["code"], name=x["name"], d=polys_d([x["polys"]])) for x in uhf]

# ------------------------------------------------------------------ tabular series
# lung (workbook)
lung = dict(rate=[[float(rates.loc[u, y]) for y in YEARS] for u in range(55)],
            cases=[[float(cases.loc[u, y]) for y in YEARS] for u in range(55)],
            avg=[float(rates.loc[u, "Overall Rate (14-Yr Avg)"]) for u in range(55)],
            total=[float(cases.loc[u, "Total 14-Yr Cases"]) for u in range(55)],
            nyc_rate=[float(nyc[y]) for y in YEARS], nyc_avg=float(nyc["Overall Rate (14-Yr Avg)"]))

# all-cancer (Duc workbook)
dc = pd.read_excel(F_FULL, sheet_name="Cancer Incidence (PUMA)")
allc = dict(rate=[[None] * 14 for _ in range(55)], count=[[None] * 14 for _ in range(55)], pop=[[None] * 14 for _ in range(55)])
mism = 0
for _, r in dc.iterrows():
    u, yi = int(r["PUMA_Code"]), int(r["Year"]) - 2010
    if norm(r["PUMA_Name"]) != norm(names[u]):
        mism += 1
    allc["rate"][u][yi] = float(r["Cancer_Crude_Rate"])
    allc["count"][u][yi] = float(r["Cancer_Count"])
    allc["pop"][u][yi] = float(r["Population"])
print("Duc cancer name mismatches vs lung workbook:", mism)


# UHF CSVs: value / flag / ci
def parse_pct(s):
    mm = re.match(r"\s*([\d.]+)(\*?)\s*(?:\(([\d.]+),\s*([\d.]+)\))?", str(s))
    return (float(mm.group(1)), bool(mm.group(2)), num(mm.group(3)), num(mm.group(4))) if mm else (None, False, None, None)


def uhf_series(path, years):
    d = pd.read_csv(path, dtype=str)
    d = d[d.GeoType == "UHF34"]
    codes = [x["code"] for x in uhf]
    out = {c: {} for c in codes}
    for _, r in d.iterrows():
        v, fl, lo, hi = parse_pct(r["Percent"])
        out[str(r["GeoID"])][int(r["TimePeriod"])] = [v, int(fl), lo, hi]
    return {c: [out[c].get(y) for y in years] for c in codes}


SY = list(range(2003, 2023))
IY = list(range(2012, 2023))
smoke = uhf_series(F_SMOKE, SY)
ins = uhf_series(F_INS, IY)
for nm, d in (("smoke", smoke), ("ins", ins)):
    print(nm, "missing cells:", sum(1 for v in d.values() for x in v if x is None))

# poverty per unit per ACS window
pov = pd.read_csv(F_POV, dtype=str)
pov["P"] = pov.Percent.map(num)
xw = pd.read_csv(os.path.join(REPO, "Subborough-PUMA-NeighborhoodName Crosswalk (1).csv"))
sub2puma = dict(zip(xw.Subborough_ID.astype(str), xw.PUMA_ID))
puma2unit = {pc: u for u, pc in m.PUMA_OF.items()}
WIN_OLD = ["%d-%02d" % (y, (y + 4) % 100) for y in range(2007, 2016)]
WIN_NEW = ["2018-22", "2019-23"]
WINS = WIN_OLD + WIN_NEW
povv = {w: [None] * 55 for w in WINS}
povapprox = {w: [] for w in WINS}
for w in WIN_OLD:
    sub = pov[(pov.GeoType == "Subboro") & (pov.TimePeriod == w)]
    tmp = {}
    for _, r in sub.iterrows():
        tmp[int(sub2puma[str(r.GeoID)])] = r.P
    for pc, u in puma2unit.items():
        povv[w][u] = tmp[pc]
    povv[w][1] = tmp[3807]
    povv[w][9] = round((tmp[3807] + tmp[3808]) / 2, 1)
    povapprox[w] = [1, 9]
unit_norm = {norm(names[u]): u for u in range(55)}
for w in WIN_NEW:
    sub = pov[(pov.GeoType == "PUMA2020") & (pov.TimePeriod == w)]
    for _, r in sub.iterrows():
        k = norm(r.Geography)
        if k not in unit_norm:
            print("  unmatched PUMA2020:", w, r.GeoID, r.Geography)
            continue
        povv[w][unit_norm[k]] = r.P
for w in WINS:
    miss = [u for u in range(55) if povv[w][u] is None]
    print("poverty", w, "missing units:", miss)


# ------------------------------------------------------------------ community-district overlap weights (air quality)
cdf = json.load(open(os.path.join(REPO, "CD.geo.json")))["features"]
air0 = pd.read_csv(ATT + AIR[2][1])
cd_rows = air0[air0.GeoType == "CD"][["GeoID", "Geography"]].drop_duplicates()
cd_codes = sorted(int(x) for x in cd_rows.GeoID)
cd_names = {int(r.GeoID): r.Geography for r in cd_rows.itertuples()}
cd_poly = {}
for f in cdf:
    c = f["properties"]["GEOCODE"]
    if c in cd_names:
        cd_poly[c] = g.polys_from_geometry(f["geometry"])
assert sorted(cd_poly) == cd_codes, (len(cd_poly), len(cd_codes))
cd_mask = [label_grid([cd_poly[c]]) for c in cd_codes]
MC = np.zeros((55, len(cd_codes)))
for i, um in enumerate(unit_mask):
    for j, hm in enumerate(cd_mask):
        MC[i, j] = (um & hm).sum()
print("CD coverage of units: min %.3f" % (MC.sum(1) / tot).min())
cdW = []
for i in range(55):
    w = MC[i] / MC[i].sum(); w[w < 0.02] = 0; w = w / w.sum()
    cdW.append([[int(j), round(float(w[j]), 4)] for j in np.nonzero(w)[0]])
print("units that are >=95% one CD:", sum(1 for wl in cdW if max(t[1] for t in wl) >= .95), "of 55")

# ------------------------------------------------------------------ new series
series, layers = {}, []
series["smoke"] = dict(geo="uhf", years=SY, v=smoke)
series["ins"] = dict(geo="uhf", years=IY, v=ins)

nd = pd.read_csv(F_NDVI_UHF, dtype={"GeoID": str})
NY = sorted(nd.year.unique().tolist())
series["ndvi"] = dict(geo="uhf", years=NY, v={c: [[float(nd[(nd.GeoID == c) & (nd.year == y)].NDVI.iloc[0])] for y in NY] for c in [x["code"] for x in uhf]})

for key, fn, col, name, unit, suf, dec, step in AIR:
    d = pd.read_csv(ATT + fn, dtype={"GeoID": str})
    d = d[d.GeoType == "CD"]
    yrs = sorted(d.TimePeriod.unique().tolist())
    v = {}
    for c in cd_codes:
        sub = d[d.GeoID == str(c)].set_index("TimePeriod")[col]
        v[str(c)] = [[None if pd.isna(sub.get(y)) else float(sub.get(y))] if y in sub.index else None for y in yrs]
    miss = sum(1 for k in v.values() for x in k if x is None or x[0] is None)
    print("air", key, yrs[0], yrs[-1], "missing", miss)
    series[key] = dict(geo="cd", years=yrs, v=v)

# PUMA-level food + climate
px_ = pd.read_excel(F_PUMAX, sheet_name="PUMA-year", dtype={"puma2020": str, "geoid": str})
cs = pd.read_csv(F_PUMACSV, dtype={"puma2020": str, "geoid": str})
print("PUMA xlsx == csv:", px_.shape == cs.shape and np.allclose(px_.select_dtypes("number").values, cs[px_.select_dtypes("number").columns].values, equal_nan=True))
pgeo = pov[(pov.GeoType == "PUMA2020") & (pov.TimePeriod == "2019-23")]
p2u = {int(r.GeoID): unit_norm[norm(r.Geography)] for r in pgeo.itertuples()}
assert len(set(p2u.values())) == 55
PY = sorted(px_.year.unique().tolist())
px_["unit"] = px_.puma2020.astype(int).map(p2u)
assert px_.unit.notna().all()
px_["super_km2"] = px_.snap_supermarkets_superstores / px_.puma_area_km2
px_["conv_km2"] = px_.snap_convenience_stores / px_.puma_area_km2


def pser(col):
    return dict(geo="puma", years=PY, v=[[round(float(px_[(px_.unit == u) & (px_.year == y)][col].iloc[0]), 4) for y in PY] for u in range(55)])


for key, col in (("snap", "snap_retailers_per_km2"), ("super", "super_km2"), ("conv", "conv_km2"),
                 ("tmean", "tmean_c"), ("tmax", "summer_tmax_c"), ("precip", "precip_mm")):
    series[key] = pser(col)
print("climate coverage min %:", px_.climate_coverage_pct.min())

SES, ENV, AIRG, FOOD, CLIM = "Behaviour & socioeconomic", "Green space", "Air quality", "Food environment", "Climate"
layers = [
    dict(key="smoke", name="Smoking", group=SES, unit="% of adults who smoke", suffix="%", dec=1, step=0.5, geo="uhf", src="NYC Community Health Survey (UHF34)", dir=1, color="#8a5a2b"),
    dict(key="pov", name="Poverty", group=SES, unit="% of residents below poverty line", suffix="%", dec=1, step=0.5, geo="pov", src="ACS 5-year estimates (PUMA / sub-borough)", dir=1, color="#c0392b", years=[2010, 2023]),
    dict(key="ins", name="No health insurance", group=SES, unit="% uninsured (100 − insured %)", suffix="%", dec=1, step=0.5, geo="uhf", src="NYC Community Health Survey (UHF34)", dir=1, color="#2f6f8f"),
    dict(key="ndvi", name="Greenness (NDVI)", group=ENV, unit="mean NDVI, 0–1 (higher = greener)", suffix="", dec=2, step=0.01, geo="uhf", src="satellite NDVI, UHF34 summary", dir=-1, color="#2e7d32", hint="Assumed protective: greener = lower burden in the combined index."),
]
cols = {"so2": "#6d4c41", "o3": "#5e35b1", "no2": "#455a64", "no": "#607d8b", "pm25": "#8e24aa", "bc": "#212121"}
for key, fn, col, name, unit, suf, dec, step in AIR:
    layers.append(dict(key=key, name=name, group=AIRG, unit=unit, suffix=suf, dec=dec, step=step, geo="cd", src="NYC EH Data Portal (community districts)", dir=1, color=cols[key],
                       hint="SO\u2082 data end in 2016." if key == "so2" else ""))
layers += [
    dict(key="snap", name="SNAP-authorised retailers", group=FOOD, unit="retailers per km²", suffix=" /km²", dec=1, step=1, geo="puma", src="USDA SNAP retailer locator, year-end", dir=-1, color="#00897b", hint="Assumed: more outlets = better food access. Direction is a hypothesis, toggle as needed."),
    dict(key="super", name="Supermarkets & superstores", group=FOOD, unit="per km²", suffix=" /km²", dec=2, step=0.05, geo="puma", src="USDA SNAP retailer locator, year-end", dir=-1, color="#26a69a", hint="Assumed protective (food access)."),
    dict(key="conv", name="Convenience stores", group=FOOD, unit="per km²", suffix=" /km²", dec=1, step=0.5, geo="puma", src="USDA SNAP retailer locator, year-end", dir=1, color="#ef6c00", hint="Assumed unfavourable (limited fresh food)."),
    dict(key="tmean", name="Mean air temperature", group=CLIM, unit="°C, annual mean", suffix=" °C", dec=1, step=0.1, geo="puma", src="PRISM 800 m, area-weighted to PUMA", dir=1, color="#d84315", hint="Direction not established."),
    dict(key="tmax", name="Summer max temperature", group=CLIM, unit="°C, Jun–Aug mean daily max", suffix=" °C", dec=1, step=0.1, geo="puma", src="PRISM 800 m, area-weighted to PUMA", dir=1, color="#bf360c", hint="Heat exposure proxy."),
    dict(key="precip", name="Precipitation", group=CLIM, unit="mm per year", suffix=" mm", dec=0, step=10, geo="puma", src="PRISM 800 m, area-weighted to PUMA", dir=1, color="#1565c0", hint="Direction not established."),
]
for L in layers:
    if "years" not in L:
        L["years"] = [series[L["key"]]["years"][0], series[L["key"]]["years"][-1]]
cd_out = [dict(code=c, name=cd_names[c], d=polys_d([cd_poly[c]])) for c in cd_codes]

# ------------------------------------------------------------------ raw sheets for the data browser
def sheet_rows(df):
    out = []
    for row in df.itertuples(index=False):
        out.append([None if (isinstance(v, float) and np.isnan(v)) else (v.isoformat() if hasattr(v, "isoformat") else (int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, (np.floating,)) else v))) for v in row])
    return out


sheets, seen = [], {}


def add_sheet(file_label, fname, sheet, df, header_rows=1, title=None):
    rows = sheet_rows(df)
    h = hashlib.md5(json.dumps(rows, default=str).encode()).hexdigest()
    entry = dict(file=file_label, sheet=sheet, title=title or sheet, nrows=len(rows), ncols=df.shape[1])
    if h in seen:
        entry["same_as"] = seen[h]
        entry["rows"] = []
    else:
        seen[h] = "%s › %s" % (file_label, sheet)
        entry["rows"] = rows
    sheets.append(entry)


for label, path in (("Lung & Bronchus Integrated Table", F_LUNG),
                    ("Duc – Cleaned", F_CLEAN),
                    ("Duc – Fully Integrated", F_FULL)):
    x = pd.ExcelFile(path)
    for sn in x.sheet_names:
        df = x.parse(sn, header=None)
        add_sheet(label, os.path.basename(path), sn, df)
for label, path in (("Smoking CSV (NYC DOHMH)", F_SMOKE), ("Health insurance CSV (NYC DOHMH)", F_INS)):
    df = pd.read_csv(path, dtype=str, header=None)
    add_sheet(label, os.path.basename(path), re.sub(r'^[0-9a-f]{8}-','',os.path.basename(path)).replace('.csv',''), df)
def add_csv(label, path, **kw):
    df = pd.read_csv(path, **kw)
    # keep header as first row so the browser shows it like the Excel sheets
    df = pd.concat([pd.DataFrame([list(df.columns)], columns=range(df.shape[1])), df.set_axis(range(df.shape[1]), axis=1)], ignore_index=True)
    add_sheet(label, os.path.basename(path), re.sub(r'^[0-9a-f]{8}-', '', os.path.basename(path)).replace('.csv', ''), df)


add_csv("NDVI by UHF34 (CSV)", F_NDVI_UHF, dtype={"GeoID": str})
add_csv("NDVI by census tract (CSV)", F_NDVI_TRACT, dtype={"GEOID": str})
for key, fn, *_ in AIR:
    add_csv("Air quality – " + fn.split("Portal_-_")[1].split("_full")[0].replace("_", " ") + " (CSV)", ATT + fn, dtype={"GeoID": str})
xx = pd.ExcelFile(F_PUMAX)
for sn in xx.sheet_names:
    add_sheet("Food environment & climate by PUMA (xlsx)", os.path.basename(F_PUMAX), sn, xx.parse(sn, header=None, dtype=str if sn == "PUMA-year" else None))
ini_text = open(F_INI, errors="replace").read()

# ------------------------------------------------------------------ SEER*Stat PUMA export -> age-adjusted rates
seer_df = sb.load_export(F_SEER)
seer, aa_rows = sb.build(seer_df)
add_sheet("SEER*Stat export (computed here)", "Lungcancerandyearofdiagnosis.txt", "Age-adjusted rates (site × PUMA × year)",
          pd.DataFrame([["Site", "PUMA code", "PUMA neighborhood", "Year", "Age-adjusted rate /100k (2000 US std)", "95% CI low", "95% CI high", "Crude rate /100k", "Cases", "Population"]] +
                       [[r[0], r[1], names[r[1]], r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in aa_rows]))
lg = seer_df[(seer_df.site == 43) & (seer_df.yr > 0)].sort_values(["puma", "yr", "age"])
add_sheet("SEER*Stat export (computed here)", "Lungcancerandyearofdiagnosis.txt", "Lung & bronchus by PUMA × year × age (export)",
          pd.DataFrame([["PUMA code", "PUMA neighborhood", "Year", "Age group", "Cases", "Population", "Crude rate /100k"]] +
                       [[int(r.puma), names[int(r.puma)], 2009 + int(r.yr), sb.AGE_LABELS[min(int(r.age), 18)] if int(r.age) < 18 else ("85–89" if int(r.age) == 18 else "90+"), int(r.cnt), int(r["pop"]), float(r.rate)]
                        for _, r in lg.iterrows()]))
texts = [dict(name="Cancer_incidence_read_by_SEERstat.ini", text=ini_text),
         dict(name="Lungcancerandyearofdiagnosis.dic (SEER*Stat export layout)", text=open(F_DIC, errors="replace").read())]

data = dict(W=W_, H=H_, S=S, X0=X0, Y0=Y0, YEARS=YEARS, units=unit_out, internal=internal_d, coast=coast_d,
            boroughLabels=[[n, *px((x, y)), a] for n, x, y, a in m.BOROUGH_LABELS], uhf=uhf_out, weights=weights,
            coverage=[round(float(c), 3) for c in cov], lung=lung, allc=allc,
            series=series, layers=layers, cdW=cdW, cd=cd_out,
            pov=dict(wins=WINS, v=povv), uhfNames=[x["name"] for x in uhf], uhfCodes=[x["code"] for x in uhf],
            sheets=sheets, texts=texts, seer=seer)
json.dump(data, open(OUT, "w"), separators=(",", ":"), default=str)
print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB; viewBox", W_, H_)
