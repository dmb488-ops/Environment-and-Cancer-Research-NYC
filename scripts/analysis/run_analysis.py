#!/usr/bin/env python3
"""Which environmental factors are associated with cancer incidence in NYC neighborhoods?

Reads the tables written by build_tables.py and writes outputs/analysis/analysis_results.json (used by the explorer's
Section 5); export_summaries.py then turns that file into flat CSV summaries.  Pipeline:
  1  screening        coverage, variance structure, univariate Spearman, 19-site screen
  2  multicollinearity correlation clusters, VIF, condition number, pruning to a core set
  3  between-neighborhood models (n = 55, pooled 2010-2023)
        OLS, OLS adjusted for smoking, Poisson with age strata, Elastic Net + stability selection,
        Random Forest, Gradient Boosting (permutation importance; random and leave-one-borough-out CV)
  4  annual panel (55 x 11 years, 2012-2022)
        pooled OLS with year effects, two-way fixed effects, Poisson (year / neighborhood effects), Random Forest
  5  sensitivity checks (core-set OLS, eight variations)
  6  consensus across the eight methods, with the scientific-case rating from theory.py
Run:  python3 run_analysis.py        (about 25 minutes on 2 cores)
"""
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import common as c
import ml_lib as ml
import stats_lib as sl
from theory import EXCLUDED_BY_DESIGN, THEORY

warnings.filterwarnings("ignore")
T0 = time.time()
SEED = 20251120


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


# Optional environment switches (defaults reproduce the published run):
#   QUICK=1       one repeat, one shuffle, small forests: a ~4-minute dry run with the same output structure (not for reporting)
#   CACHE_DIR=... keep the slow stages (tree models, benchmarks, panel) on disk so a re-run after a late error skips them
#   OUTFILE=...   name of the results file inside outputs/analysis (default analysis_results.json)
QUICK = os.environ.get("QUICK") == "1"
NREP, NPERM, NLOBO, NSTAB, PREP, PPERM = (1, 1, 1, 40, 1, 1) if QUICK else (8, 6, 3, 400, 4, 4)
if QUICK:
    _make_rf = ml.make_rf
    ml.make_rf = lambda seed=0, leaf=3, n=300, mf=1 / 3: _make_rf(seed, leaf, min(n, 50), mf)


def cached(name, build):
    d = os.environ.get("CACHE_DIR")
    p = os.path.join(d, name + (".quick" if QUICK else "") + ".pkl") if d else None
    if p and os.path.exists(p):
        log("   (cached stage: %s)" % name)
        return pickle.load(open(p, "rb"))
    v = build()
    if p:
        os.makedirs(d, exist_ok=True)
        pickle.dump(v, open(p, "wb"))
    return v


# ----------------------------------------------------------------------------- inputs
D = c.load_D()
LAYERS = {L["key"]: L for L in D["layers"]}
KEYS = [L["key"] for L in D["layers"]]
P = len(KEYS)
cs = pd.read_csv(os.path.join(c.OUT, "puma_period_means_2010_2023.csv"))
panel = pd.read_csv(os.path.join(c.OUT, "puma_year_panel_2010_2023.csv"))
strata = pd.read_csv(os.path.join(c.OUT, "age_strata_lung_all.csv"))
SITES = [(s["key"], s["name"]) for s in D["seer"]["sites"]]
BORO = cs["borough"].values
GSETS = {"Smoking": ["smoke"], "SES & access": ["pov", "ins"], "Greenness": ["ndvi"],
         "Air quality": ["so2", "o3", "no2", "no", "pm25", "bc"], "Food environment": ["snap", "super", "conv"],
         "Climate": ["tmean", "tmax", "precip"]}
GIDX = {g: [KEYS.index(k) for k in v] for g, v in GSETS.items()}
OUTCOMES = {"lung_aa": dict(label="Lung & bronchus, age-adjusted", site="lung", col="aa_lung", pcol="lung_aa"),
            "all_aa": dict(label="All sites, age-adjusted", site="all", col="aa_all", pcol="all_aa"),
            "lung_cr": dict(label="Lung & bronchus, crude", site="lung", col="cr_lung", pcol="lung_cr")}


def R(x, nd=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return R(o, 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- feature preparation
def prep(X, keys, transforms=None):
    """log-transform strongly right-skewed positive variables (skew > 1), then z-score."""
    Xt = X.astype(float).copy()
    tr = {} if transforms is None else transforms
    for j, k in enumerate(keys):
        if transforms is None:
            tr[k] = "log" if (stats.skew(X[:, j]) > 1 and np.all(X[:, j] > 0)) else "none"
        if tr[k] == "log":
            Xt[:, j] = np.log(X[:, j])
    mu, sd = Xt.mean(0), Xt.std(0)
    return Xt, (Xt - mu) / sd, tr, sd


Xraw = cs[KEYS].values.astype(float)
Xt, Z, TRANS, _ = prep(Xraw, KEYS)
Y = {k: cs[v["col"]].values.astype(float) for k, v in OUTCOMES.items()}
SMOKE = KEYS.index("smoke")


def ols_rows(f, names, offset=1):
    rows = {}
    for i, k in enumerate(names):
        t, df = f["t"][i + offset], f["df"]
        rows[k] = dict(beta=f["beta"][i + offset], se=f["se"][i + offset], lo=f["lo"][i + offset], hi=f["hi"][i + offset],
                       t=t, p=f["p"][i + offset], partial_r2=t ** 2 / (t ** 2 + df))
    return rows


# ============================================================================= 1  SCREENING
log("1 screening")
screen = {}
panel_ok = panel.copy()
for j, k in enumerate(KEYS):
    L = LAYERS[k]
    p = panel_ok[["unit", "year", k]].dropna()
    gm = p[k].mean()
    tot = ((p[k] - gm) ** 2).sum()
    bu = ((p.groupby("unit")[k].transform("mean") - gm) ** 2).sum() / tot
    by = ((p.groupby("year")[k].transform("mean") - gm) ** 2).sum() / tot
    x = Xraw[:, j]
    eta = 1 - sum(((x[BORO == b] - x[BORO == b].mean()) ** 2).sum() for b in np.unique(BORO)) / ((x - x.mean()) ** 2).sum()
    yrs = sorted(p.year.unique().tolist())
    row = dict(name=c.SHORT[k], full_name=L["name"], group=L["group"], unit=L["unit"], geo=L["geo"], source=L["src"],
               n_years=len(yrs), year_first=yrs[0], year_last=yrs[-1], mean=x.mean(), sd=x.std(), skew=stats.skew(x),
               transform=TRANS[k], var_between_neighborhoods=bu, var_between_years=by, var_within=1 - bu - by,
               borough_eta2=eta, spearman={})
    for o, spec in OUTCOMES.items():
        r, pv, n = sl.spearman(x, Y[o])
        row["spearman"][o] = dict(rho=r, p=pv)
    lo, hi = sl.boot_spearman_ci(x, Y["lung_aa"], B=2000, seed=SEED)
    row["spearman"]["lung_aa"].update(lo=lo, hi=hi)
    row["rho_density"] = sl.spearman(x, cs["pop_density"].values)[0]
    screen[k] = row
row_out = {}
# 19-site screen with Benjamini-Hochberg FDR
sitemat = np.zeros((P, len(SITES)))
sitep = np.zeros((P, len(SITES)))
for si, (sk, sn) in enumerate(SITES):
    for j in range(P):
        r, pv, _ = sl.spearman(Xraw[:, j], cs["aa_" + sk].values)
        sitemat[j, si], sitep[j, si] = r, pv
flat = sitep.ravel()
order = np.argsort(flat)
m = len(flat)
q = np.empty(m)
q[order] = np.minimum.accumulate((flat[order] * m / (np.arange(m) + 1))[::-1])[::-1]
siteq = np.minimum(q, 1).reshape(sitep.shape)
sites_screen = dict(sites=[dict(key=k, name=n) for k, n in SITES], factors=KEYS, rho=sitemat, p=sitep, q=siteq)
log("   screening done; FDR-significant cells:", int((siteq < 0.05).sum()), "of", m)

# ============================================================================= 2  MULTICOLLINEARITY
log("2 multicollinearity")
Csp = sl.corr_matrix(Xraw, "spearman")                  # shown in the heat-map (robust to skew)
Cpe = np.corrcoef(Z, rowvar=False)                      # what enters the regressions (transformed, standardized)
vif_all = sl.vif_from_corr(Cpe)
ev = np.linalg.eigvalsh(Cpe)[::-1]
order_leaf, lab07, _ = sl.cluster_order(Csp, 0.3)
coll = dict(order=[KEYS[i] for i in order_leaf], spearman=Csp, cluster07=dict(zip(KEYS, lab07)),
            vif_all=dict(zip(KEYS, vif_all)), condition_all=sl.condition_number(Cpe), eigenvalues=ev,
            pairs=[dict(a=KEYS[i], b=KEYS[j], rho=Csp[i, j]) for i in range(P) for j in range(i + 1, P) if abs(Csp[i, j]) >= 0.8])
# theory-blind pruning from all 16
keep_blind, path_blind, vif_blind = sl.vif_prune_path(Cpe, KEYS, cutoff=5.0)
coll["blind_prune"] = dict(path=[dict(drop=a, vif=b) for a, b in path_blind], kept=keep_blind, vif=vif_blind)
# theory-guided: remove design exclusions, collapse near-duplicates (|rho| >= 0.85), keep the best-justified of each
pool = [k for k in KEYS if k not in EXCLUDED_BY_DESIGN]
par = {k: k for k in pool}


def find(a):
    while par[a] != a:
        par[a] = par[par[a]]
        a = par[a]
    return a


for a in pool:
    for b in pool:
        if a < b and abs(Csp[KEYS.index(a), KEYS.index(b)]) >= 0.85:
            par[find(a)] = find(b)
clusters = {}
for k in pool:
    clusters.setdefault(find(k), []).append(k)
between = {k: screen[k]["var_between_neighborhoods"] for k in KEYS}
reps, dropped = [], {}
for members in clusters.values():
    best = min(members, key=lambda k: (THEORY[k]["priority"], -between[k]))
    reps.append(best)
    for k in members:
        if k != best:
            dropped[k] = "near-duplicate of %s (|rho| >= 0.85); %s has the stronger scientific case" % (c.SHORT[best], c.SHORT[best])
CORE = [k for k in KEYS if k in reps]
ci = [KEYS.index(k) for k in CORE]
core_keep, core_path, core_vif = sl.vif_prune_path(Cpe[np.ix_(ci, ci)], CORE, protect=("smoke", "pm25", "ndvi"), cutoff=5.0)
for a, v in core_path:
    dropped[a] = "variance inflation factor %.1f (>= 5) within the core set" % v
CORE = core_keep
for k, why in EXCLUDED_BY_DESIGN.items():
    dropped[k] = why
CI = [KEYS.index(k) for k in CORE]
Zc = Z[:, CI]
coll["core"] = dict(keys=CORE, vif=dict(zip(CORE, sl.vif_from_corr(Cpe[np.ix_(CI, CI)]))),
                    condition=sl.condition_number(Cpe[np.ix_(CI, CI)]), dropped=dropped,
                    clusters=[m_ for m_ in clusters.values() if len(m_) > 1])
log("   core set:", CORE, " VIF max %.2f  condition %.1f" % (max(coll["core"]["vif"].values()), coll["core"]["condition"]))
# demonstration: what collinearity does to a coefficient
demo = {}
yl = Y["lung_aa"]
for label, cols in (("PM2.5 alone (with smoking)", ["smoke", "pm25"]),
                    ("PM2.5 with NO2, NO and black carbon added", ["smoke", "pm25", "no2", "no", "bc"])):
    f = sl.ols(Z[:, [KEYS.index(k) for k in cols]], yl)
    i = cols.index("pm25") + 1
    demo[label] = dict(beta=f["beta"][i], se=f["se"][i], lo=f["lo"][i], hi=f["hi"][i], p=f["p"][i])
coll["demo"] = demo
f16 = sl.ols(Z, yl)
coll["ols_all16"] = ols_rows(f16, KEYS)
coll["ols_all16_adj_r2"] = f16["adj_r2"]

# ============================================================================= helpers for models
def dummies(v, drop_first=True):
    u = np.unique(v)
    u = u[1:] if drop_first else u
    return np.column_stack([(v == a).astype(float) for a in u]) if len(u) else np.zeros((len(v), 0))


AGE_MAP = {"lung": np.array([0] * 8 + list(range(1, 12))), "all": np.arange(19)}


def strata_rows(site, years=None, by_year=False):
    s = strata[strata.site == site]
    if years is not None:
        s = s[s.year.isin(years)]
    s = s.assign(ag=AGE_MAP[site][s.age_group.values])
    keys = ["unit", "year", "ag"] if by_year else ["unit", "ag"]
    return s.groupby(keys)[["cases", "person_years"]].sum().reset_index()


def pois_rows(res, names, offset):
    return {k: dict(irr=res["irr"][offset + i], lo=res["irr_lo"][offset + i], hi=res["irr_hi"][offset + i], p=res["p"][offset + i],
                    beta=res["beta"][offset + i], se=res["se"][offset + i]) for i, k in enumerate(names)}


def knn_w(xy, k=5):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    for i in range(len(xy)):
        W[i, np.argsort(d[i])[:k]] = 1
    return W / W.sum(1, keepdims=True)


def morans_i(e, W, nperm=999, seed=0):
    e = e - e.mean()
    I = len(e) / W.sum() * (e @ W @ e) / (e @ e)
    rng = np.random.default_rng(seed)
    sims = np.array([(lambda z: len(z) / W.sum() * (z @ W @ z) / (z @ z))(rng.permutation(e)) for _ in range(nperm)])
    return float(I), float((1 + (sims >= I).sum()) / (nperm + 1))


WNN = knn_w(cs[["cx", "cy"]].values.astype(float))
FOLDS_RAND = ml.repeated_group_folds(np.arange(55), k=5, repeats=NREP, seed=SEED)
FOLDS_LOBO = ml.borough_folds(BORO)


def imp_summary(res, names):
    imp = res["imp"]
    return {k: dict(mean=imp[:, i].mean(), sd=imp[:, i].std(), pos=float((imp[:, i] > 0).mean())) for i, k in enumerate(names)}


def group_summary(res):
    return {g: dict(mean=res["gimp"][:, i].mean(), sd=res["gimp"][:, i].std(), pos=float((res["gimp"][:, i] > 0).mean()))
            for i, g in enumerate(res["group_names"])}


# ============================================================================= 3  BETWEEN-NEIGHBORHOOD MODELS
CROSS = {}
for oname, spec in OUTCOMES.items():
    y = Y[oname]
    res = dict(label=spec["label"], n=len(y), mean=y.mean(), sd=y.std())
    # --- OLS (core set) and OLS + smoking, one factor at a time
    f = sl.ols(Zc, y)
    res["ols"] = dict(coef=ols_rows(f, CORE), r2=f["r2"], adj_r2=f["adj_r2"])
    res["moran_ols_resid"] = dict(zip(("I", "p"), morans_i(f["resid"], WNN, seed=SEED)))
    fs = sl.ols(Z[:, [SMOKE]], y)
    res["smoke_alone"] = dict(beta=fs["beta"][1], lo=fs["lo"][1], hi=fs["hi"][1], t=fs["t"][1], p=fs["p"][1], r2=fs["r2"])
    res["moran_smoke_resid"] = dict(zip(("I", "p"), morans_i(fs["resid"], WNN, seed=SEED)))
    res["adj_smoke"] = {}
    for j, k in enumerate(KEYS):
        if k == "smoke":
            continue
        g = sl.ols(Z[:, [SMOKE, j]], y)
        t = g["t"][2]
        res["adj_smoke"][k] = dict(beta=g["beta"][2], lo=g["lo"][2], hi=g["hi"][2], t=t, p=g["p"][2], partial_r2=t ** 2 / (t ** 2 + g["df"]))
    # --- Poisson with age strata (core set)
    if oname != "lung_cr":
        st = strata_rows(spec["site"])
        A = dummies(st.ag.values)
        Xp = np.column_stack([A, Zc[st.unit.values]])
        pr = sl.poisson(Xp, st.cases.values.astype(float), np.log(st.person_years.values), cluster=st.unit.values)
        res["poisson"] = dict(coef=pois_rows(pr, CORE, 1 + A.shape[1]), dispersion=pr["dispersion"])
    # --- Elastic Net with stability selection (all 16)
    en = ml.ElasticNetCV(l1_ratio=[0.3, 0.5, 0.7, 0.9, 1.0], alphas=50, cv=5, random_state=SEED, max_iter=20000).fit(Z, y)
    freq = ml.stability_selection(Z, y, np.arange(55), en.alpha_, en.l1_ratio_, B=NSTAB, seed=SEED)
    res["enet"] = dict(alpha=en.alpha_, l1_ratio=en.l1_ratio_, coef=dict(zip(KEYS, en.coef_)), stability=dict(zip(KEYS, freq)))
    log("3 %s: OLS adj R2 %.2f; smoking beta %.2f (p=%.4f); EN keeps %s" % (oname, f["adj_r2"], f["beta"][1 + CORE.index("smoke")],
                                                                     f["p"][1 + CORE.index("smoke")], [k for k, v in zip(KEYS, en.coef_) if v != 0]))
    CROSS[oname] = res

# --- tree models on the two main outcomes (+ crude lung as a contrast for RF only)
def _trees():
    out = {}
    for oname in ("lung_aa", "all_aa"):
        y = Y[oname]
        for mname, mk in (("rf", ml.make_rf), ("gbm", ml.make_gbm)):
            rnd = ml.cv_run(mk(seed=SEED, leaf=3), Xraw, y, FOLDS_RAND, n_perm=NPERM, seed=SEED, group_sets=GIDX)
            lob = [ml.cv_run(mk(seed=SEED + r, leaf=3), Xraw, y, FOLDS_LOBO, n_perm=NPERM, seed=SEED + r, group_sets=GIDX) for r in range(NLOBO)]
            lob_all = dict(imp=np.vstack([x["imp"] for x in lob]), gimp=np.vstack([x["gimp"] for x in lob]), group_names=lob[0]["group_names"])
            full = mk(seed=SEED, leaf=3)().fit(Xraw, y)
            out[(oname, mname)] = dict(
                cv_r2=rnd["r2"], cv_r2_sd=rnd["r2_sd"], cv_rmse=rnd["rmse"], lobo_r2=float(np.mean([x["r2"] for x in lob])),
                lobo_rmse=float(np.mean([x["rmse"] for x in lob])),
                imp=imp_summary(rnd, KEYS), imp_lobo=imp_summary(lob_all, KEYS), group=group_summary(rnd), group_lobo=group_summary(lob_all),
                pdp=ml.pdp_curves(full, Xraw, KEYS))
            log("3 %s %s: CV R2 %.3f  LOBO R2 %.3f; top importance %s" % (oname, mname, rnd["r2"], np.mean([x["r2"] for x in lob]),
                  sorted(out[(oname, mname)]["imp"].items(), key=lambda kv: -kv[1]["mean"])[0][0]))
    return out


for (oname, mname), v_ in cached("trees", _trees).items():
    CROSS[oname][mname] = v_

# --- benchmark / incremental-value table
coords = cs[["cx", "cy"]].values.astype(float)
ix = lambda ks: [KEYS.index(k) for k in ks]
ENV = [k for k in KEYS if LAYERS[k]["group"] != "Behaviour & socioeconomic"]
bench_spec = [
    ("Neighborhood location only (x, y coordinates)", "Random Forest", ml.make_rf(SEED), coords),
    ("Smoking only", "OLS", ml.make_ols(), Xt[:, ix(["smoke"])]),
    ("Poverty + no insurance", "OLS", ml.make_ols(), Xt[:, ix(["pov", "ins"])]),
    ("Smoking + poverty + no insurance", "OLS", ml.make_ols(), Xt[:, ix(["smoke", "pov", "ins"])]),
    ("Environment only (13 greenness, air, food, climate factors)", "Random Forest", ml.make_rf(SEED), Xraw[:, ix(ENV)]),
    ("Environment only (13 greenness, air, food, climate factors)", "Ridge", ml.make_ridge(), Xt[:, ix(ENV)]),
    ("Core set (7 factors)", "OLS", ml.make_ols(), Xt[:, ix(CORE)]),
    ("All 16 factors", "OLS", ml.make_ols(), Xt),
    ("All 16 factors", "Ridge", ml.make_ridge(), Xt),
    ("All 16 factors", "Elastic Net", ml.make_enet(SEED), Xt),
    ("All 16 factors", "Random Forest", ml.make_rf(SEED), Xraw),
    ("All 16 factors", "Gradient Boosting", ml.make_gbm(SEED), Xraw),
    ("All 16 factors without smoking", "Random Forest", ml.make_rf(SEED), Xraw[:, [i for i in range(P) if i != SMOKE]]),
]
def _bench():
    out = {}
    for oname in ("lung_aa", "all_aa"):
        y = Y[oname]
        bench = []
        for name, mdl, mk, Xm in bench_spec:
            a_ = ml.cv_run(mk, Xm, y, FOLDS_RAND)
            b_ = ml.cv_run(mk, Xm, y, FOLDS_LOBO)
            bench.append(dict(features=name, model=mdl, cv_r2=a_["r2"], cv_rmse=a_["rmse"], lobo_r2=b_["r2"], lobo_rmse=b_["rmse"]))
        out[oname] = bench
    return out


CROSS_BENCH = cached("bench", _bench)
log("   benchmarks done")

# ============================================================================= 4  ANNUAL PANEL (2012-2022)
log("4 panel")
PY = list(range(2012, 2023))
PKEYS = [k for k in KEYS if k != "so2"]
pan = panel[panel.year.isin(PY)].sort_values(["unit", "year"]).reset_index(drop=True)
assert pan[PKEYS].notna().all().all() and len(pan) == 55 * len(PY)
Xp_raw = pan[PKEYS].values.astype(float)
Xp_t, Zp, _, _ = prep(Xp_raw, PKEYS, transforms={k: TRANS[k] for k in PKEYS})
PCORE = [k for k in CORE]
pci = [PKEYS.index(k) for k in PCORE]
Zpc = Zp[:, pci]
unit, year, pop = pan.unit.values, pan.year.values, pan["pop"].values.astype(float)
pvif = sl.vif_from_corr(np.corrcoef(Zpc, rowvar=False))
PANEL = dict(years=PY, n=len(pan), core_vif=dict(zip(PCORE, pvif)))
YD = dummies(year)
for oname in ("lung_aa", "all_aa"):
    spec = OUTCOMES[oname]
    y = pan[spec["pcol"]].values.astype(float)
    res = {}
    f = sl.ols(np.column_stack([Zpc, YD]), y, w=pop, cluster=unit)
    res["pooled"] = dict(coef=ols_rows(f, PCORE), r2=f["r2"])
    fe = sl.fe_ols(Zpc, y, unit, year, w=pop)
    res["fe"] = dict(coef=ols_rows(fe, PCORE, offset=0), r2_within=fe["r2"])
    # Poisson with age strata: year effects (between + within) and neighborhood + year effects (within only)
    st = strata_rows(spec["site"], PY, by_year=True)
    zi = {(u, yv): i for i, (u, yv) in enumerate(zip(unit, year))}
    row_ix = np.array([zi[(u, yv)] for u, yv in zip(st.unit.values, st.year.values)])
    A = dummies(st.ag.values)
    Yd = dummies(st.year.values)
    Ud = dummies(st.unit.values)
    cases, off, cl = st.cases.values.astype(float), np.log(st.person_years.values), st.unit.values
    p1 = sl.poisson(np.column_stack([A, Yd, Zpc[row_ix]]), cases, off, cluster=cl)
    res["poisson_year"] = dict(coef=pois_rows(p1, PCORE, 1 + A.shape[1] + Yd.shape[1]), dispersion=p1["dispersion"])
    p2 = sl.poisson(np.column_stack([A, Yd, Ud, Zpc[row_ix]]), cases, off, cluster=cl)
    res["poisson_fe"] = dict(coef=pois_rows(p2, PCORE, 1 + A.shape[1] + Yd.shape[1] + Ud.shape[1]), dispersion=p2["dispersion"])
    # Random Forest on year-centred data (removes the city-wide trend), neighborhoods never split between train and test
    def _panel_rf(y=y):
        yc = y - pd.Series(y).groupby(year).transform("mean").values
        Xc_ = Xp_raw - pd.DataFrame(Xp_raw).groupby(year).transform("mean").values
        fr = ml.repeated_group_folds(unit, k=5, repeats=PREP, seed=SEED)
        fl = ml.borough_folds(pan.borough.values)
        gidx = {g: [PKEYS.index(k) for k in v if k in PKEYS] for g, v in GSETS.items()}
        rnd = ml.cv_run(ml.make_rf(SEED, leaf=5, n=200), Xc_, yc, fr, n_perm=PPERM, seed=SEED, group_sets=gidx)
        lob = ml.cv_run(ml.make_rf(SEED, leaf=5, n=200), Xc_, yc, fl, n_perm=PPERM, seed=SEED, group_sets=gidx)
        rf_ = dict(cv_r2=rnd["r2"], lobo_r2=lob["r2"], imp=imp_summary(rnd, PKEYS), imp_lobo=imp_summary(lob, PKEYS))
        # baselines on the same folds
        bench_ = [dict(model=m_, features=f_, cv_r2=ml.cv_run(mk, Xm, yc, fr)["r2"], lobo_r2=ml.cv_run(mk, Xm, yc, fl)["r2"]) for m_, f_, mk, Xm in (
            ("OLS", "Smoking only", ml.make_ols(), Xc_[:, [PKEYS.index("smoke")]]),
            ("OLS", "Core set", ml.make_ols(), Xc_[:, pci]),
            ("Random Forest", "All 15 factors", ml.make_rf(SEED, leaf=5, n=200), Xc_))]
        return rf_, bench_, rnd["r2"], lob["r2"]

    res["rf"], res["bench"], rnd_r2, lob_r2 = cached("panel_" + oname, _panel_rf)
    PANEL[oname] = res
    log("4 %s: pooled smoking beta %.2f p=%.3f | FE smoking beta %.2f p=%.3f | RF CV R2 %.3f LOBO %.3f" % (
        oname, res["pooled"]["coef"]["smoke"]["beta"], res["pooled"]["coef"]["smoke"]["p"], res["fe"]["coef"]["smoke"]["beta"],
        res["fe"]["coef"]["smoke"]["p"], rnd_r2, lob_r2))

# ============================================================================= 5  SENSITIVITY (OLS, core set, lung age-adjusted)
log("5 sensitivity")
sens = {}


def sens_fit(label, keep=None, y=None, cols=None, note=""):
    keep = np.arange(55) if keep is None else keep
    y = Y["lung_aa"] if y is None else y
    cc = CORE if cols is None else cols
    Xm = Xt[:, [KEYS.index(k) for k in cc]][keep]
    Xm = (Xm - Xm.mean(0)) / Xm.std(0)
    f = sl.ols(Xm, y[keep])
    sens[label] = dict(note=note, n=len(keep), r2=f["adj_r2"], coef=ols_rows(f, cc))


sens_fit("Baseline: core set, lung age-adjusted, all 55 neighborhoods")
sens_fit("Without Staten Island (3 neighborhoods)", keep=np.where(BORO != "Staten Island")[0], note="Staten Island has the highest lung rates and a distinct climate and geography.")
sens_fit("PM2.5 replaced by NO2", cols=[("no2" if k == "pm25" else k) for k in CORE], note="Tests whether the choice of air-pollution representative matters.")
sens_fit("PM2.5 replaced by black carbon", cols=[("bc" if k == "pm25" else k) for k in CORE])
sens_fit("Without no-insurance (a detection variable)", cols=[k for k in CORE if k != "ins"])
sens_fit("Crude (not age-adjusted) lung rate as outcome", y=Y["lung_cr"], note="Shows how age structure distorts the picture.")
sens_fit("All sites age-adjusted as outcome", y=Y["all_aa"])
_p = panel.copy()
y_no2020 = _p[_p.year != 2020].groupby("unit")["lung_aa"].mean().values
Xm = Xt[:, CI]
Xm = (Xm - Xm.mean(0)) / Xm.std(0)
fq = sl.ols(Xm, y_no2020)
sens["Outcome without 2020 (pandemic diagnosis dip)"] = dict(note="Neighborhood mean of the yearly age-adjusted rates, 2020 left out.", n=55, r2=fq["adj_r2"], coef=ols_rows(fq, CORE))

# ============================================================================= 6  CONSENSUS
log("6 consensus")
METHODS = [("screen", "Spearman screen", "all"), ("adj", "OLS + smoking", "nonsmoke"), ("ols", "OLS (core set)", "core"),
           ("pois", "Poisson (core set)", "core"), ("enet", "Elastic Net", "all"), ("rf", "Random Forest", "all"),
           ("gbm", "Gradient Boosting", "all"), ("fe", "Panel two-way FE", "core")]
sign = lambda v: 0 if v is None or v == 0 else (1 if v > 0 else -1)


def consensus_for(oname):
    cr = CROSS[oname]
    pn = PANEL[oname]
    cons = {}
    for k in KEYS:
        d = {}
        sp = screen[k]["spearman"][oname]
        d["screen"] = dict(tested=True, support=sp["p"] < 0.05, dir=sign(sp["rho"]), stat=sp["rho"], p=sp["p"])
        if k != "smoke":
            a = cr["adj_smoke"][k]
            d["adj"] = dict(tested=True, support=a["p"] < 0.05, dir=sign(a["beta"]), stat=a["t"], p=a["p"])
        else:
            d["adj"] = dict(tested=False)
        if k in CORE:
            o = cr["ols"]["coef"][k]
            d["ols"] = dict(tested=True, support=o["p"] < 0.05, dir=sign(o["beta"]), stat=o["t"], p=o["p"])
            if "poisson" in cr:
                q_ = cr["poisson"]["coef"][k]
                d["pois"] = dict(tested=True, support=q_["p"] < 0.05, dir=sign(q_["beta"]), stat=q_["beta"] / q_["se"], p=q_["p"])
            fe_ = pn["fe"]["coef"][k]
            d["fe"] = dict(tested=True, support=fe_["p"] < 0.05, dir=sign(fe_["beta"]), stat=fe_["t"], p=fe_["p"])
        else:
            d["ols"] = d["pois"] = d["fe"] = dict(tested=False)
        e = cr["enet"]
        d["enet"] = dict(tested=True, support=e["stability"][k] >= 0.6, dir=sign(e["coef"][k]), stat=e["stability"][k], p=None)
        for m_ in ("rf", "gbm"):
            im, lo_ = cr[m_]["imp"][k], cr[m_]["imp_lobo"][k]
            pdp = cr[m_]["pdp"][k]
            d[m_] = dict(tested=True, support=(im["mean"] >= 0.01 and im["pos"] >= 0.8), dir=sign(pdp["rho"]) if abs(pdp["rho"]) > 0.3 else 0,
                         stat=im["mean"], p=None, lobo_ok=bool(lo_["mean"] > 0 and lo_["pos"] >= 0.6), lobo=lo_["mean"], pos=im["pos"])
        cons[k] = d
    return cons


def rank_methods(cons):
    for m_, _, _ in METHODS:
        tested = [k for k in KEYS if cons[k][m_].get("tested")]
        vals = {k: abs(cons[k][m_]["stat"]) for k in tested}
        for r_, k in enumerate(sorted(tested, key=lambda k: -vals[k]), 1):
            cons[k][m_]["rank"] = r_
        for k in tested:
            cons[k][m_]["n_tested"] = len(tested)
    return cons


def verdict(cons, k, check_theory):
    d = cons[k]
    tested = [m_ for m_, _, _ in METHODS if d[m_].get("tested")]
    sup = [m_ for m_ in tested if d[m_]["support"]]
    share = len(sup) / len(tested)
    dirs = {d[m_]["dir"] for m_ in sup if d[m_]["dir"] != 0}
    mixed_dir = len(dirs) > 1
    v = "consistent" if share >= 0.6 and not mixed_dir else ("mixed" if share >= 0.25 or mixed_dir and share > 0 else "not supported")
    sgn = int(np.sign(sum(d[m_]["dir"] for m_ in sup))) if sup else 0
    proxy = bool(any(d[m_].get("support") and not d[m_].get("lobo_ok", True) for m_ in ("rf", "gbm")) and screen[k]["borough_eta2"] > 0.5)
    out = dict(n_tested=len(tested), n_support=len(sup), methods_supporting=sup, share=share, verdict=v, direction=sgn, location_proxy=proxy)
    if check_theory:
        th = THEORY[k]
        contradicts = sgn != 0 and th["expected"] in ("+", "-") and (th["expected"] == "+") != (sgn > 0)
        if v == "consistent" and th["rating"] in ("strong", "moderate") and not contradicts and not proxy:
            rec = "selected"
        elif v in ("consistent", "mixed") and (th["rating"] in ("weak", "none") or contradicts or proxy):
            rec = "flag"
        elif th["rating"] in ("strong", "moderate") and v == "not supported":
            rec = "hypothesis"
        else:
            rec = "dropped" if v == "not supported" else "flag"
        out.update(rating=th["rating"], expected=th["expected"], contradicts_theory=bool(contradicts), recommendation=rec)
    return out


CONSENSUS = {}
for oname in ("lung_aa", "all_aa"):
    cons = rank_methods(consensus_for(oname))
    for k in KEYS:
        cons[k]["summary"] = verdict(cons, k, check_theory=True)
    CONSENSUS[oname] = cons
for k in KEYS:
    log("   %-7s lung: %s (%d/%d)  rec=%s  proxy=%s | all: %s (%d/%d)" % (
        k, CONSENSUS["lung_aa"][k]["summary"]["verdict"], CONSENSUS["lung_aa"][k]["summary"]["n_support"], CONSENSUS["lung_aa"][k]["summary"]["n_tested"],
        CONSENSUS["lung_aa"][k]["summary"]["recommendation"], CONSENSUS["lung_aa"][k]["summary"]["location_proxy"],
        CONSENSUS["all_aa"][k]["summary"]["verdict"], CONSENSUS["all_aa"][k]["summary"]["n_support"], CONSENSUS["all_aa"][k]["summary"]["n_tested"]))

# ============================================================================= write
results = dict(
    meta=dict(generated="2026-11-20", seed=SEED, n_neighborhoods=55, years=c.YEARS, panel_years=PY, outcomes={k: v["label"] for k, v in OUTCOMES.items()},
              methods=[dict(key=a, name=b, scope=s) for a, b, s in METHODS], keys=KEYS, groups=GSETS,
              names={k: c.SHORT[k] for k in KEYS}, full_names={k: LAYERS[k]["name"] for k in KEYS}, core=CORE, panel_keys=PKEYS,
              rf=dict(trees=300, max_features="1/3", min_leaf=3), gbm=dict(trees=200, depth=2, lr=0.05, subsample=0.7),
              cv=dict(random="5-fold on whole neighborhoods, repeated %d times" % NREP, lobo="leave one borough out", n_perm=NPERM), quick_mode=QUICK),
    screening=screen, site_screen=sites_screen, collinearity=coll, cross=CROSS, cross_bench=CROSS_BENCH, panel=PANEL,
    sensitivity=sens, consensus=CONSENSUS,
    points=[dict(unit=int(u), name=cs.puma_name[u], borough=cs.borough[u], lung_aa=cs.aa_lung[u], lung_cr=cs.cr_lung[u], all_aa=cs.aa_all[u],
                 all_cr=cs.cr_all[u], density=cs.pop_density[u], **{k: cs[k][u] for k in KEYS}) for u in range(55)],
    theory={k: dict(rating=v["rating"], expected=v["expected"], role=v["role"], why=v["why"]) for k, v in THEORY.items()})
OUTFILE = os.environ.get("OUTFILE", "analysis_results.json")
json.dump(clean(results), open(os.path.join(c.OUT, OUTFILE), "w"), separators=(",", ":"))
log("wrote", OUTFILE, os.path.getsize(os.path.join(c.OUT, OUTFILE)) // 1024, "KB")
