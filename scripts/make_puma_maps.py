#!/usr/bin/env python3
"""
NYC lung & bronchus cancer incidence maps by PUMA, 2010-2023
=============================================================
Makes one choropleth per year (same colour scale in every year, so years are comparable), a
14-year-average map, and a one-page overview sheet.

Inputs
  --xlsx  NYC_Lung_and_Bronchus_Cancer_Integrated_Table_2010_2023.xlsx   (sheets: Crude Incidence Rates, Case Counts)
  --repo  a clone of https://github.com/nycehs/NYC_geography              (PUMA_or_Subborough.geo.json, CD.geo.json)
  --out   output folder

Needs only numpy, pandas, openpyxl and matplotlib (puma_geo.py must sit next to this file).

Note on the join: the workbook's "PUMA Code" (0-54) is a running index, not the Census PUMA code.  Each of the
55 neighbourhood names is matched to a boundary polygon through PUMA_OF below.  Two Manhattan units in the
workbook are split differently from the 2010 PUMA file (Chelsea & Hell's Kitchen = Community District 4 only;
Midtown, East Midtown & Flatiron = CD5 + CD6), so PUMA 3807 is cut along the CD4|CD5 line.
"""
import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import PathPatch, Rectangle

from puma_geo import (_key, bbox, inside, interior_point, load_features, orient, path_of, polys_from_geometry,
                      ring_area, shared_chain, split_ring, area_km2)

YEARS = list(range(2010, 2024))

# ------------------------------------------------------------------ look (see dataviz skill: one-hue sequential)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
COAST = "#b9b7ae"
BREAKS = [30, 40, 50, 60, 75, 90]                                       # 7 classes, fixed for every year
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # blue 100,200,...,700
CLASS_LABELS = ["< 30", "30–40", "40–50", "50–60", "60–75", "75–90", "90 +"]

# ------------------------------------------------------------------ crosswalk: workbook "PUMA Code" -> 2010 PUMA code
PUMA_OF = {0: 3809, 2: 3806, 3: 3805, 4: 3802, 5: 3803, 6: 3804, 7: 3801, 8: 3810,
           10: 3708, 11: 3707, 12: 3706, 13: 3701, 14: 3709, 15: 3703, 16: 3704, 17: 3702, 18: 3710, 19: 3705,
           20: 4001, 21: 4004, 22: 4003, 23: 4002, 24: 4008, 25: 4005, 26: 4012, 27: 4006, 28: 4011, 29: 4013,
           30: 4017, 31: 4014, 32: 4018, 33: 4015, 34: 4016, 35: 4007, 36: 4010, 37: 4009,
           38: 4101, 39: 4109, 40: 4102, 41: 4107, 42: 4110, 43: 4108, 44: 4103, 45: 4106, 46: 4111, 47: 4113,
           48: 4104, 49: 4112, 50: 4105, 51: 4114, 52: 3903, 53: 3902, 54: 3901}
SPECIAL = {1: "CD4 only (western part of 2010 PUMA 3807)",
           9: "CD5 + CD6 (eastern part of 2010 PUMA 3807 + all of 2010 PUMA 3808)"}

BOROUGH_LABELS = [("MANHATTAN", -6.0, 12.5, "right"), ("BRONX", 20.5, 21.0, "left"), ("QUEENS", 19.5, -3.5, "center"),
                  ("BROOKLYN", 1.0, -17.0, "center"), ("STATEN ISLAND", -14.5, -2.5, "center")]


# =============================================================================================== data
def load_table(xlsx):
    def read(sheet):
        raw = pd.read_excel(xlsx, sheet_name=sheet, header=None)
        hdr = [int(h) if isinstance(h, (int, float, np.integer, np.floating)) and not pd.isna(h) else h
               for h in raw.iloc[3]]
        df = raw.iloc[4:].copy()
        df.columns = hdr
        return df

    r, c = read("Crude Incidence Rates"), read("Case Counts")
    nyc = r[r["PUMA Neighborhood Name"] == "NYC Average / Total"].iloc[0]
    r, c = r[r["PUMA Code"].notna()].copy(), c[c["PUMA Code"].notna()].copy()
    for df in (r, c):
        df["PUMA Code"] = df["PUMA Code"].astype(int)
    r, c = r.set_index("PUMA Code").sort_index(), c.set_index("PUMA Code").sort_index()
    assert len(r) == 55 and (r.index == c.index).all()
    assert (r["PUMA Neighborhood Name"] == c["PUMA Neighborhood Name"]).all()
    for y in YEARS:
        r[y], c[y] = r[y].astype(float), c[y].astype(float)
    r["Overall Rate (14-Yr Avg)"] = r["Overall Rate (14-Yr Avg)"].astype(float)
    assert not r[YEARS].isna().any().any()
    return r, c, nyc


def build_units(repo):
    """unit id (workbook code) -> list of parts; part = polys (MultiPolygon); polygon = [rings]."""
    pf = load_features(os.path.join(repo, "PUMA_or_Subborough.geo.json"))
    cf = load_features(os.path.join(repo, "CD.geo.json"))
    puma = {f["properties"]["PUMA"]: polys_from_geometry(f["geometry"]) for f in pf}
    names = {f["properties"]["PUMA"]: f["properties"]["GEONAME"] for f in pf}
    cd = {f["properties"]["GEOCODE"]: polys_from_geometry(f["geometry"]) for f in cf
          if f["properties"]["GEOCODE"] in (104, 105)}
    assert len(puma) == 55
    units = {code: [puma[pc]] for code, pc in PUMA_OF.items()}
    # Midtown: cut PUMA 3807 along the CD4|CD5 boundary
    assert len(puma[3807]) == 1
    chain = shared_chain(cd[104][0][0], cd[105][0][0])
    a_ring, c_ring, dist = split_ring(puma[3807][0][0], chain, interior_point(cd[104]))
    assert max(dist) < 0.05, dist
    a_polys, c_polys = orient([[a_ring]]), orient([[c_ring]])
    assert abs(area_km2(a_polys) + area_km2(c_polys) - area_km2(puma[3807])) < 1e-6
    units[1] = [a_polys]
    units[9] = [c_polys, puma[3808]]
    assert sorted(units) == list(range(55))
    return units, puma, names


def classify_edges(units, delta=0.02):
    """Split all polygon edges into: boundaries between two different units / coast line (one side is water)."""
    edges = {}
    for uid, parts in units.items():
        for polys in parts:
            for poly in polys:
                for r in poly:
                    for k in range(len(r) - 1):
                        a, b = r[k], r[k + 1]
                        ka, kb = _key(a), _key(b)
                        edges.setdefault((ka, kb) if ka < kb else (kb, ka), (a, b))
    segs = np.array(list(edges.values()))
    d = segs[:, 1] - segs[:, 0]
    ln = np.hypot(d[:, 0], d[:, 1])
    keep = ln > 1e-9
    segs, d, ln = segs[keep], d[keep], ln[keep]
    mid = segs.mean(1)
    nrm = np.column_stack([-d[:, 1], d[:, 0]]) / ln[:, None]
    side = []
    for sgn in (1, -1):
        pts = mid + sgn * delta * nrm
        ids = np.full(len(pts), -1)
        for uid, parts in units.items():
            hit = np.zeros(len(pts), bool)
            for polys in parts:
                hit |= inside(polys, pts)
            ids[hit & (ids < 0)] = uid
        side.append(ids)
    s1, s2 = side
    internal = (s1 >= 0) & (s2 >= 0) & (s1 != s2)
    coast = (s1 >= 0) ^ (s2 >= 0)
    return segs[internal], segs[coast]


def klass(v):
    return int(np.digitize(v, BREAKS))


# =============================================================================================== drawing
def draw_map(ax, paths, color_of, seg_in, seg_coast, xlim, ylim, lw_line=0.8, lw_coast=0.4):
    for uid, plist in paths.items():
        for p in plist:
            ax.add_patch(PathPatch(p, fc=color_of[uid], ec=color_of[uid], lw=0.4, joinstyle="round", zorder=1))
    ax.add_collection(LineCollection(seg_in, colors=SURFACE, linewidths=lw_line, capstyle="round", zorder=2))
    ax.add_collection(LineCollection(seg_coast, colors=COAST, linewidths=lw_coast, capstyle="round", zorder=3))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()


def borough_labels(ax, fs):
    for txt, x, y, ha in BOROUGH_LABELS:
        t = ax.text(x, y, txt, ha=ha, va="center", fontsize=fs, color=INK2, fontweight="medium", zorder=6)
        t.set_path_effects([pe.withStroke(linewidth=2.6, foreground=SURFACE)])


def legend_vertical(ax, x0, y_top, fs):
    ax.text(x0, y_top, "Cases per 100,000\nresidents per year", ha="left", va="top", fontsize=fs, color=INK2,
            linespacing=1.3, zorder=5)
    y = y_top - 4.6
    sw_w, sw_h, gap = 2.7, 1.9, 0.45
    for k in range(len(RAMP) - 1, -1, -1):
        ax.add_patch(Rectangle((x0, y - sw_h), sw_w, sw_h, fc=RAMP[k], ec="none", zorder=5))
        ax.text(x0 + sw_w + 0.8, y - sw_h / 2, CLASS_LABELS[k], ha="left", va="center", fontsize=fs, color=INK2, zorder=5)
        y -= sw_h + gap


def stats_block(year_stats):
    hi, lo = year_stats["hi"], year_stats["lo"]
    parts = ["NYC average  %.1f" % year_stats["avg"] if year_stats.get("avg") is not None else None,
             "{:,} new cases".format(year_stats["cases"]),
             "",
             textwrap.fill("Highest:  %s  %.1f" % (hi[0], hi[1]), 34, subsequent_indent="    "),
             textwrap.fill("Lowest:   %s  %.1f" % (lo[0], lo[1]), 34, subsequent_indent="    ")]
    return "\n".join(p for p in parts if p is not None)


def render_single(path, year_label, rates_by_uid, stats, paths, seg_in, seg_coast, xlim, ylim, dpi,
                  title, subtitle, footnote):
    color_of = {u: RAMP[klass(v)] for u, v in rates_by_uid.items()}
    fig = plt.figure(figsize=(8.0, 9.0), dpi=dpi, facecolor=SURFACE)
    fig.text(0.05, 0.965, title, fontsize=19, fontweight="semibold", color=INK, va="top", ha="left")
    fig.text(0.05, 0.925, subtitle, fontsize=10.5, color=INK2, va="top", ha="left")
    ax = fig.add_axes([0.01, 0.075, 0.98, 0.835])
    draw_map(ax, paths, color_of, seg_in, seg_coast, xlim, ylim, lw_line=0.9)
    borough_labels(ax, 8)
    legend_vertical(ax, xlim[0] + 1.0, ylim[1] - 1.0, 9.5)
    ax.text(xlim[1] - 0.8, ylim[0] + 0.6, stats_block(stats), ha="right", va="bottom", fontsize=9, color=INK2,
            linespacing=1.45, zorder=5)
    fig.text(0.05, 0.032, textwrap.fill(footnote, 148), fontsize=6.8, color=MUTED, va="center", ha="left", linespacing=1.4)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def render_overview(path, panels, paths, seg_in, seg_coast, xlim, ylim, dpi, title, subtitle, footnote):
    ncol, nrow = 5, 3
    cw = 3.25
    ch = cw * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    top_h, bot_h = 1.05, 1.45
    W, H = ncol * cw + 0.3, nrow * ch + top_h + bot_h
    fig = plt.figure(figsize=(W, H), dpi=dpi, facecolor=SURFACE)
    fig.text(0.15 / W, 1 - 0.28 / H, title, fontsize=21, fontweight="semibold", color=INK, va="top", ha="left")
    fig.text(0.15 / W, 1 - 0.68 / H, subtitle, fontsize=11, color=INK2, va="top", ha="left")
    for i, (label, sub, rates_by_uid) in enumerate(panels):
        r, c = divmod(i, ncol)
        ax = fig.add_axes([(0.15 + c * cw) / W, (bot_h + (nrow - 1 - r) * ch) / H, cw / W, ch / H])
        color_of = {u: RAMP[klass(v)] for u, v in rates_by_uid.items()}
        draw_map(ax, paths, color_of, seg_in, seg_coast, xlim, ylim, lw_line=0.5, lw_coast=0.3)
        ax.text(xlim[0] + 0.8, ylim[1] - 0.8, label, ha="left", va="top", fontsize=16, fontweight="semibold",
                color=INK, zorder=5)
        ax.text(xlim[0] + 0.8, ylim[1] - 5.4, sub, ha="left", va="top", fontsize=8.5, color=INK2, zorder=5,
                linespacing=1.35)
    # shared horizontal legend + footnote (explicit heights in inches from the bottom edge)
    x0 = 0.15 + 0.02
    y_title, y_sw, y_lab, y_note = 1.18, 0.78, 0.62, 0.24
    fig.text(x0 / W, y_title / H, "Cases per 100,000 residents per year  (same scale in every panel)",
             fontsize=10, color=INK2, va="center", ha="left")
    sw_w, sw_h, gap = 0.95, 0.2, 0.05
    for k, (col, lab) in enumerate(zip(RAMP, CLASS_LABELS)):
        x = x0 + k * (sw_w + gap)
        fig.add_artist(Rectangle((x / W, y_sw / H), sw_w / W, sw_h / H, fc=col, ec="none", transform=fig.transFigure))
        fig.text((x + sw_w / 2) / W, y_lab / H, lab, fontsize=9, color=INK2, va="center", ha="center")
    fig.text(x0 / W, y_note / H, textwrap.fill(footnote, 240), fontsize=7.2, color=MUTED, va="center", ha="left",
             linespacing=1.4)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--only", help="comma-separated list of years/labels to render (testing)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "axes.unicode_minus": False})

    rates, cases, nyc = load_table(args.xlsx)
    units, puma, boundary_names = build_units(args.repo)
    allb = bbox([p for parts in units.values() for part in parts for p in part])
    pad = 1.3
    xlim, ylim = (allb[0] - pad, allb[2] + pad), (allb[1] - pad, allb[3] + pad)
    paths = {u: [path_of(p) for p in parts] for u, parts in units.items()}
    seg_in, seg_coast = classify_edges(units)
    print("extent km:", np.round(xlim, 1), np.round(ylim, 1), "| internal edges:", len(seg_in), "| coast edges:", len(seg_coast))

    name = rates["PUMA Neighborhood Name"]
    footnote = ("Data: NYC Spatial-Temporal Cancer Dataset (crude incidence rates, lung and bronchus). Boundaries: 2010 PUMAs from the "
                "nycehs/NYC_geography repository on GitHub (shapes mostly originally from NYC Dept. of City Planning); the two Midtown "
                "units are re-cut at the Community District 4|5 line to match the table. "
                "Crude (not age-adjusted) rates: older neighborhoods run higher; one-year rates in small PUMAs are noisy.")

    only = set(args.only.split(",")) if args.only else None
    for y in YEARS:
        if only and str(y) not in only:
            continue
        v = rates[y]
        stats = dict(hi=(name[v.idxmax()], v.max()), lo=(name[v.idxmin()], v.min()), avg=float(nyc[y]), cases=int(cases[y].sum()))
        render_single(os.path.join(args.out, "lung_cancer_incidence_%d.png" % y), str(y), v.to_dict(), stats, paths, seg_in,
                      seg_coast, xlim, ylim, args.dpi, "Lung and bronchus cancer incidence, %d" % y,
                      "Crude rate per 100,000 residents, by Public Use Microdata Area (55 PUMAs, New York City)", footnote)
        print("wrote", y)

    avg = rates["Overall Rate (14-Yr Avg)"]
    tot = int(cases[YEARS].values.sum())
    stats_avg = dict(hi=(name[avg.idxmax()], avg.max()), lo=(name[avg.idxmin()], avg.min()), avg=None, cases=tot)
    if not only or "avg" in only:
        render_single(os.path.join(args.out, "lung_cancer_incidence_2010-2023_average.png"), "2010–2023", avg.to_dict(),
                      stats_avg, paths, seg_in, seg_coast, xlim, ylim, args.dpi,
                      "Lung and bronchus cancer incidence, 2010–2023 average",
                      "14-year pooled crude rate per 100,000 residents, by PUMA (total new cases shown below)", footnote)
        print("wrote average")

    if not only or "overview" in only:
        panels = []
        for y in YEARS:
            panels.append((str(y), "NYC avg %.1f\n{:,} cases".format(int(cases[y].sum())) % float(nyc[y]), rates[y].to_dict()))
        panels.append(("2010–2023", "14-year average\n{:,} cases".format(tot), avg.to_dict()))
        render_overview(os.path.join(args.out, "lung_cancer_incidence_2010-2023_overview.png"), panels, paths, seg_in,
                        seg_coast, xlim, ylim, 200,
                        "Lung and bronchus cancer incidence across New York City, 2010–2023",
                        "Crude annual rate per 100,000 residents in each of the 55 Public Use Microdata Areas; "
                        "darker = higher.  One color scale for all panels, so years can be compared directly.",
                        footnote)
        print("wrote overview")

    # crosswalk for the record
    rows = []
    for code in range(55):
        if code in PUMA_OF:
            pc = PUMA_OF[code]
            rows.append(dict(workbook_code=code, workbook_name=name[code], puma_2010=pc, geoid_2010="36%05d" % pc,
                             boundary_file_name=boundary_names[pc], note=""))
        else:
            rows.append(dict(workbook_code=code, workbook_name=name[code], puma_2010="", geoid_2010="",
                             boundary_file_name="(re-cut from 3807/3808)", note=SPECIAL[code]))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "puma_crosswalk.csv"), index=False)

    # plausibility: implied population density per unit (2010)
    pop = cases[2010] / rates[2010] * 1e5
    area = {u: sum(area_km2(p) for p in parts) for u, parts in units.items()}
    dens = pd.Series({u: pop[u] / area[u] for u in units})
    print("people per km2 (implied 2010 pop / polygon area), lowest/highest:")
    print(pd.concat([name, dens.rename("per_km2").round(0)], axis=1).sort_values("per_km2").iloc[[0, 1, 2, 3, -4, -3, -2, -1]].to_string())


if __name__ == "__main__":
    main()
