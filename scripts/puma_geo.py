"""
Geometry helpers for the NYC PUMA lung-cancer maps (numpy + matplotlib only; no GIS libraries needed).

Polygon convention used everywhere below:
    ring     = (N, 2) float array of projected km coordinates, closed (first == last)
    polygon  = [outer_ring, hole_ring, ...]
    polys    = [polygon, polygon, ...]         (a MultiPolygon)
"""
import json
import numpy as np
from matplotlib.path import Path

# Local equirectangular projection centred on NYC (distortion is negligible at city scale)
LON0, LAT0 = -74.0, 40.70
KX = 111.320 * np.cos(np.radians(LAT0))   # km per degree of longitude
KY = 110.574                               # km per degree of latitude


def project(coords):
    a = np.asarray(coords, float)
    return np.column_stack([(a[:, 0] - LON0) * KX, (a[:, 1] - LAT0) * KY])


def ring_area(r):
    """Signed shoelace area of a closed ring (positive = counter-clockwise)."""
    x, y = r[:, 0], r[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def orient(polys):
    """Outer rings counter-clockwise, holes clockwise (needed for correct hole rendering)."""
    out = []
    for poly in polys:
        new = []
        for k, r in enumerate(poly):
            a = ring_area(r)
            if (a > 0) != (k == 0):
                r = r[::-1].copy()
            new.append(r)
        out.append(new)
    return out


def polys_from_geometry(geom):
    if geom["type"] == "Polygon":
        raw = [geom["coordinates"]]
    elif geom["type"] == "MultiPolygon":
        raw = geom["coordinates"]
    else:
        raise ValueError(geom["type"])
    return orient([[project(r) for r in poly] for poly in raw])


def area_km2(polys):
    tot = 0.0
    for poly in polys:
        tot += abs(ring_area(poly[0])) - sum(abs(ring_area(h)) for h in poly[1:])
    return tot


def path_of(polys):
    """One compound matplotlib Path for a MultiPolygon."""
    verts, codes = [], []
    for poly in polys:
        for r in poly:
            verts.append(r)
            codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 2) + [Path.CLOSEPOLY]
    return Path(np.vstack(verts), codes)


def inside(polys, pts):
    """Even-odd point-in-polygon test for an (M, 2) array of points."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    cnt = np.zeros(len(pts), int)
    for poly in polys:
        for r in poly:
            cnt += Path(r).contains_points(pts)
    return (cnt % 2) == 1


def bbox(polys):
    allp = np.vstack([r for poly in polys for r in poly])
    return allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()


def interior_point(polys, step=0.05):
    """A point that is certainly inside `polys` (inside-sample closest to the sample mean)."""
    x0, y0, x1, y1 = bbox(polys)
    gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    pts = pts[inside(polys, pts)]
    m = pts.mean(0)
    return pts[np.argmin(((pts - m) ** 2).sum(1))]


def load_features(path):
    return json.load(open(path))["features"]


# ----------------------------------------------------------------------------------------------
# Splitting one PUMA polygon along a boundary shared by two community districts
# ----------------------------------------------------------------------------------------------
def _key(v, scale=1e6):
    return (int(round(v[0] * scale)), int(round(v[1] * scale)))


def shared_chain(ring_a, ring_b):
    """Longest run of consecutive vertices of ring_a that are also vertices of ring_b (exact match)."""
    a = ring_a[:-1]
    keys_b = {_key(v) for v in ring_b[:-1]}
    flag = np.array([_key(v) in keys_b for v in a])
    if flag.all() or not flag.any():
        return a if flag.all() else a[:0]
    f = int(np.argmin(flag))                       # rotate so index 0 is a non-shared vertex
    a, flag = np.roll(a, -f, axis=0), np.roll(flag, -f)
    best, i, n = (0, 0), 0, len(a)
    while i < n:
        if flag[i]:
            j = i
            while j < n and flag[j]:
                j += 1
            if j - i > best[1]:
                best = (i, j - i)
            i = j
        else:
            i += 1
    s, ln = best
    return a[s:s + ln]


def _nearest_on_ring(pts, q):
    """Nearest point on the closed polyline through `pts` (open list) to q -> (dist, seg, t, point)."""
    p0, p1 = pts, np.roll(pts, -1, axis=0)
    d = p1 - p0
    l2 = (d ** 2).sum(1)
    l2[l2 == 0] = 1e-18
    t = np.clip(((q - p0) * d).sum(1) / l2, 0, 1)
    proj = p0 + t[:, None] * d
    dist = np.hypot(*(proj - q).T)
    i = int(np.argmin(dist))
    return float(dist[i]), i, float(t[i]), proj[i]


def split_ring(ring, chain, test_pt):
    """
    Cut a polygon ring in two with the polyline `chain` (its two end points lie on / very near the ring).
    Returns (ring_containing_test_pt, other_ring, snap_distances_km).
    """
    pts = ring[:-1].copy()
    snapped, dists = [], []
    for q in (chain[0], chain[-1]):
        dist, i, t, p = _nearest_on_ring(pts, q)
        dists.append(dist)
        if t < 1e-9:
            snapped.append(pts[i].copy())
        elif t > 1 - 1e-9:
            snapped.append(pts[(i + 1) % len(pts)].copy())
        else:
            pts = np.insert(pts, i + 1, p, axis=0)
            snapped.append(p.copy())

    def index_of(q):
        return int(np.argmin(np.hypot(*(pts - q).T)))

    i1, i2 = index_of(snapped[0]), index_of(snapped[1])
    n = len(pts)

    def arc(a, b):
        return pts[a:b + 1] if a <= b else np.vstack([pts[a:], pts[:b + 1]])

    mid = chain[1:-1]
    x = np.vstack([arc(i1, i2), mid[::-1]])
    y = np.vstack([arc(i2, i1), mid])
    x, y = np.vstack([x, x[:1]]), np.vstack([y, y[:1]])
    if Path(x).contains_point(test_pt):
        return x, y, dists
    return y, x, dists
