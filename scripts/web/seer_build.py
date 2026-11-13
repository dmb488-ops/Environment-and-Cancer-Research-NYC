"""Age-standardise the SEER*Stat PUMA export (direct method, 2000 US standard million, 19 age groups, both sexes)."""
import numpy as np, pandas as pd
from scipy.stats import chi2

STD19 = np.array([13818, 55317, 72533, 73032, 72169, 66478, 64529, 71044, 80762, 81851, 72118,
                  62716, 48454, 38793, 34264, 31773, 26999, 17842, 15508], float)
assert STD19.sum() == 1_000_000
AGE_LABELS = ["<1", "1–4", "5–9", "10–14", "15–19", "20–24", "25–29", "30–34", "35–39", "40–44", "45–49",
              "50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "80–84", "85+"]
# (site code in the export, key, display name, tobacco/other note)
SITES = [(0, "all", "All sites combined"), (43, "lung", "Lung & bronchus"), (40, "resp", "Respiratory system (all)"),
         (42, "larynx", "Larynx"), (99, "meso", "Mesothelioma"), (1, "oral", "Oral cavity & pharynx"),
         (13, "eso", "Esophagus"), (14, "stomach", "Stomach"), (16, "colorectal", "Colon & rectum"),
         (31, "liver", "Liver & intrahepatic bile duct"), (36, "pancreas", "Pancreas"),
         (67, "bladder", "Urinary bladder"), (68, "kidney", "Kidney & renal pelvis"),
         (76, "thyroid", "Thyroid"), (49, "melanoma", "Melanoma of the skin"), (82, "nhl", "Non-Hodgkin lymphoma"),
         (85, "myeloma", "Myeloma"), (86, "leukemia", "Leukemia"), (72, "brain", "Brain & other nervous system")]


def fay_feuer(R, v, wm, alpha=0.05):
    """Gamma-method CI for a directly standardised rate (Fay & Feuer 1997). Rates per person (not per 100k)."""
    if R <= 0:
        lo = 0.0
    else:
        lo = v / (2 * R) * chi2.ppf(alpha / 2, 2 * R * R / v) if v > 0 else R
    up = (v + wm * wm) / (2 * (R + wm)) * chi2.ppf(1 - alpha / 2, 2 * (R + wm) ** 2 / (v + wm * wm))
    return lo, up


def standardise(c19, p19):
    """c19, p19: (19,) counts and person-years. returns rate, lo, hi per 100k."""
    w = STD19 / STD19.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(p19 > 0, c19 / p19, 0.0)
        R = float((w * r).sum())
        v = float((w ** 2 * np.where(p19 > 0, c19 / p19 ** 2, 0.0)).sum())
        wm = float(np.max(np.where(p19 > 0, w / p19, 0.0)))
    lo, hi = fay_feuer(R, v, wm)
    return R * 1e5, lo * 1e5, hi * 1e5


def collapse(a20):
    """(…,20) -> (…,19) by merging 85-89 and 90+."""
    a = np.asarray(a20, float)
    return np.concatenate([a[..., :18], a[..., 18:].sum(-1, keepdims=True)], axis=-1)


def load_export(path):
    d = pd.read_csv(path, sep="\t", header=None, names=["yr", "site", "puma", "age", "rate", "cnt", "pop"], dtype=str)
    for c in ("yr", "site", "puma", "age"):
        d[c] = d[c].astype(int)
    d["cnt"] = d.cnt.str.replace(",", "").astype(int)
    d["pop"] = d["pop"].str.replace(",", "").astype(int)
    d["rate"] = d.rate.str.replace(",", "").astype(float)
    return d


def build(d):
    YEARS = list(range(2010, 2024))
    out = {"std": STD19.tolist(), "ageLabels": AGE_LABELS, "sites": []}
    # population by PUMA x year x age (identical for every site; take All Sites page)
    pp = d[d.site == 0].pivot_table(index=["yr", "puma"], columns="age", values="pop", aggfunc="first")
    pop = np.zeros((15, 55, 20))
    for (yr, u), row in pp.iterrows():
        pop[yr, u, :] = row.values
    out["pop"] = [[int(pop[y + 1, u].sum()) for y in range(14)] for u in range(55)]
    out["popAge"] = [[int(x) for x in collapse(pop[0, u])] for u in range(55)]   # pooled person-years by age (14 yr)
    rows = []
    for code, key, name in SITES:
        sub = d[(d.site == code)]
        cnt = np.zeros((15, 55, 20))
        for (yr, u), g in sub.groupby(["yr", "puma"]):
            cnt[yr, u, g.age.values] = g.cnt.values
        S = dict(key=key, name=name, code=code, aa=[], lo=[], hi=[], cr=[], cnt=[], aaP=[], loP=[], hiP=[], crP=[], cntP=[],
                 ageCnt=[], nycAA=[], nycLo=[], nycHi=[], nycCr=[], nycAAP=None)
        for u in range(55):
            a, l, h, c, n = [], [], [], [], []
            for y in range(1, 15):
                R, lo, hi = standardise(collapse(cnt[y, u]), collapse(pop[y, u]))
                tc, tp = cnt[y, u].sum(), pop[y, u].sum()
                a.append(round(R, 1)); l.append(round(lo, 1)); h.append(round(hi, 1)); c.append(round(tc / tp * 1e5, 1)); n.append(int(tc))
                rows.append([name, u, 2009 + y, round(R, 1), round(lo, 1), round(hi, 1), round(tc / tp * 1e5, 1), int(tc), int(tp)])
            S["aa"].append(a); S["lo"].append(l); S["hi"].append(h); S["cr"].append(c); S["cnt"].append(n)
            R, lo, hi = standardise(collapse(cnt[0, u]), collapse(pop[0, u]))
            tc, tp = cnt[0, u].sum(), pop[0, u].sum()
            S["aaP"].append(round(R, 1)); S["loP"].append(round(lo, 1)); S["hiP"].append(round(hi, 1))
            S["crP"].append(round(tc / tp * 1e5, 1)); S["cntP"].append(int(tc))
            S["ageCnt"].append([int(x) for x in collapse(cnt[0, u])])
        for y in range(1, 15):
            R, lo, hi = standardise(collapse(cnt[y].sum(0)), collapse(pop[y].sum(0)))
            S["nycAA"].append(round(R, 1)); S["nycLo"].append(round(lo, 1)); S["nycHi"].append(round(hi, 1))
            S["nycCr"].append(round(cnt[y].sum() / pop[y].sum() * 1e5, 1))
        S["nycAAP"] = round(standardise(collapse(cnt[0].sum(0)), collapse(pop[0].sum(0)))[0], 1)
        S["nycCrP"] = round(cnt[0].sum() / pop[0].sum() * 1e5, 1)
        out["sites"].append(S)
    return out, rows
