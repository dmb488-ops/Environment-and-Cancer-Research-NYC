"""Small statistics toolkit written with numpy / scipy only.

statsmodels is not required: the routines below are short, and `test_stats_lib.py` checks them against
scikit-learn and scipy.  Everything here is deterministic.
"""
import numpy as np
from scipy import stats
from scipy.cluster import hierarchy


# ----------------------------------------------------------------------------- basics
def zscore(a, ddof=0):
    a = np.asarray(a, float)
    return (a - np.nanmean(a, axis=0)) / np.nanstd(a, axis=0, ddof=ddof)


def rank_avg(a):
    return stats.rankdata(a, method="average")


def spearman(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def boot_spearman_ci(x, y, B=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for Spearman rho (resampling neighborhoods)."""
    rng = np.random.default_rng(seed)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    out = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        out[b] = stats.spearmanr(x[i], y[i])[0]
    return float(np.nanquantile(out, alpha / 2)), float(np.nanquantile(out, 1 - alpha / 2))


# ----------------------------------------------------------------------------- linear regression
def ols(X, y, w=None, cluster=None, cov="HC3", const=True, extra_k=0):
    """(Weighted) least squares with robust standard errors.

    cov: 'HC3' (default, small samples), 'HC1', 'classic'; ignored when `cluster` is given, in which case the
    Liang-Zeger cluster-robust estimator with the usual G/(G-1) * (n-1)/(n-k) correction is used and
    p-values come from a t distribution with G-1 degrees of freedom.
    extra_k: parameters that were absorbed beforehand (fixed effects) and count in the degrees-of-freedom correction.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    n = len(y)
    Xc = np.column_stack([np.ones(n), X]) if const else X
    k = Xc.shape[1]
    w = np.ones(n) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    Xw, yw = Xc * sw[:, None], y * sw
    bread = np.linalg.inv(Xw.T @ Xw)
    beta = bread @ (Xw.T @ yw)
    e = y - Xc @ beta
    s = (w * e)[:, None] * Xc
    kk = k + extra_k
    if cluster is not None:
        g = np.unique(cluster, return_inverse=True)[1]
        G = int(g.max()) + 1
        sg = np.zeros((G, k))
        np.add.at(sg, g, s)
        V = bread @ (sg.T @ sg) @ bread * (G / (G - 1)) * ((n - 1) / (n - kk))
        df = G - 1
    else:
        df = n - kk
        if cov == "classic":
            V = (np.sum(w * e ** 2) / df) * bread
        else:
            if cov == "HC3":
                h = w * np.einsum("ij,jk,ik->i", Xc, bread, Xc)
                ss = s / (1 - h)[:, None]
            elif cov == "HC1":
                ss = s * np.sqrt(n / (n - kk))
            else:
                raise ValueError(cov)
            V = bread @ (ss.T @ ss) @ bread
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df)
    tq = stats.t.ppf(0.975, df)
    ybar = np.sum(w * y) / np.sum(w)
    sst = np.sum(w * (y - ybar) ** 2) if const else np.sum(w * y ** 2)
    sse = np.sum(w * e ** 2)
    r2 = 1 - sse / sst
    adj = 1 - (1 - r2) * (n - (1 if const else 0)) / (n - kk)
    return dict(beta=beta, se=se, t=t, p=p, lo=beta - tq * se, hi=beta + tq * se, r2=r2, adj_r2=adj, n=n, k=k, df=df,
                resid=e, fitted=Xc @ beta, const=const)


def _wmean_by(v, g, w):
    """Weighted group means broadcast back to rows (v: (n,) or (n,p))."""
    v = np.asarray(v, float)
    G = int(g.max()) + 1
    num = np.zeros((G,) + v.shape[1:])
    den = np.zeros(G)
    ww = w if v.ndim == 1 else w[:, None]
    np.add.at(num, g, v * ww)
    np.add.at(den, g, w)
    m = num / (den if v.ndim == 1 else den[:, None])
    return m[g]


def demean_fe(M, ids, w=None, tol=1e-12, maxit=500):
    """Remove one or two sets of fixed effects by alternating (weighted) projections.
    M: (n,) or (n,p); ids: list of integer group arrays."""
    M = np.asarray(M, float).copy()
    n = M.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, float)
    gs = [np.unique(i, return_inverse=True)[1] for i in ids]
    for _ in range(maxit):
        before = M.copy()
        for g in gs:
            M = M - _wmean_by(M, g, w)
        if np.max(np.abs(M - before)) < tol:
            break
    return M


def fe_ols(X, y, pid, tid=None, w=None, cluster=None):
    """Fixed-effects ("within") regression. pid = neighborhood id; tid = year id (two-way FE when given).
    Standard errors clustered by `cluster` (default: pid). Year effects count towards the df correction."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    ids = [pid] if tid is None else [pid, tid]
    Xd = demean_fe(X, ids, w)
    yd = demean_fe(y, ids, w)
    extra = 0 if tid is None else len(np.unique(tid)) - 1
    return ols(Xd, yd, w=w, cluster=pid if cluster is None else cluster, const=False, extra_k=extra)


# ----------------------------------------------------------------------------- Poisson regression
def poisson(X, y, offset, cluster=None, const=True, tol=1e-10, maxit=100):
    """Poisson regression (log link) by IRLS with cluster-robust standard errors.
    Rates are modelled as counts with offset = log(person-years)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    n = len(y)
    Xc = np.column_stack([np.ones(n), X]) if const else X
    k = Xc.shape[1]
    beta = np.zeros(k)
    if const:
        beta[0] = np.log(y.sum() / np.exp(offset).sum())
    dev_old = np.inf
    for it in range(maxit):
        eta = Xc @ beta + offset
        mu = np.exp(np.clip(eta, -30, 30))
        z = eta - offset + (y - mu) / mu
        A = Xc.T @ (Xc * mu[:, None])
        beta_new = np.linalg.solve(A, Xc.T @ (mu * z))
        eta2 = Xc @ beta_new + offset
        mu2 = np.exp(np.clip(eta2, -30, 30))
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2 * np.sum(np.where(y > 0, y * np.log(y / mu2), 0.0) - (y - mu2))
        beta = beta_new
        if abs(dev_old - dev) < tol * (abs(dev) + 0.1):
            break
        dev_old = dev
    eta = Xc @ beta + offset
    mu = np.exp(np.clip(eta, -30, 30))
    bread = np.linalg.inv(Xc.T @ (Xc * mu[:, None]))
    s = (y - mu)[:, None] * Xc
    if cluster is None:
        V = bread @ (s.T @ s) @ bread * (n / (n - k))
        df = n - k
    else:
        g = np.unique(cluster, return_inverse=True)[1]
        G = int(g.max()) + 1
        sg = np.zeros((G, k))
        np.add.at(sg, g, s)
        V = bread @ (sg.T @ sg) @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
        df = G - 1
    se = np.sqrt(np.diag(V))
    zt = beta / se
    p = 2 * stats.t.sf(np.abs(zt), df)
    tq = stats.t.ppf(0.975, df)
    pearson = np.sum((y - mu) ** 2 / mu)
    return dict(beta=beta, se=se, z=zt, p=p, lo=beta - tq * se, hi=beta + tq * se, irr=np.exp(beta),
                irr_lo=np.exp(beta - tq * se), irr_hi=np.exp(beta + tq * se), deviance=float(dev), n=n, k=k,
                dispersion=float(pearson / (n - k)), mu=mu, const=const)


# ----------------------------------------------------------------------------- collinearity
def corr_matrix(X, method="spearman"):
    X = np.asarray(X, float)
    if method == "spearman":
        X = np.apply_along_axis(rank_avg, 0, X)
    return np.corrcoef(X, rowvar=False)


def vif_from_corr(C):
    """VIF_j = 1 / (1 - R2_j) = j-th diagonal element of the inverse correlation matrix."""
    return np.diag(np.linalg.inv(C))


def condition_number(C):
    ev = np.linalg.eigvalsh(C)
    return float(np.sqrt(ev.max() / ev.min()))


def cluster_order(C, thresh=0.3, method="average"):
    """Hierarchical clustering of variables on distance 1-|rho|. Returns leaf order and flat cluster labels
    (variables with |rho| >= 1-thresh chain together)."""
    D = 1 - np.abs(C)
    np.fill_diagonal(D, 0)
    iu = np.triu_indices_from(D, 1)
    Z = hierarchy.linkage(D[iu], method=method)
    order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, D[iu]))
    labels = hierarchy.fcluster(Z, t=thresh, criterion="distance")
    return order.tolist(), labels.tolist(), Z


def vif_prune_path(C, names, protect=(), cutoff=5.0):
    """Statistical (theory-blind) pruning: repeatedly drop the highest-VIF variable until all VIF < cutoff.
    Variables in `protect` are never dropped (a variable with the next highest VIF goes instead)."""
    keep = list(range(len(names)))
    path = []
    while True:
        sub = C[np.ix_(keep, keep)]
        v = vif_from_corr(sub)
        if v.max() < cutoff or len(keep) <= 2:
            break
        cand = [(v[i], keep[i]) for i in range(len(keep)) if names[keep[i]] not in protect]
        worst = max(cand)
        path.append((names[worst[1]], float(worst[0])))
        keep.remove(worst[1])
    sub = C[np.ix_(keep, keep)]
    return [names[i] for i in keep], path, dict(zip([names[i] for i in keep], vif_from_corr(sub).tolist()))
