"""Cross-validation, permutation importance and model factories (scikit-learn)."""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------- model factories
def make_rf(seed=0, leaf=3, n=300, mf=1 / 3):
    return lambda: RandomForestRegressor(n_estimators=n, max_features=mf, min_samples_leaf=leaf, random_state=seed, n_jobs=2)


def make_gbm(seed=0, leaf=3):
    return lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.7,
                                             min_samples_leaf=leaf, random_state=seed)


def make_enet(seed=0):
    return lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.3, 0.5, 0.7, 0.9, 1.0], alphas=50, cv=5,
                                                                random_state=seed, max_iter=20000))


def make_ols():
    return lambda: make_pipeline(StandardScaler(), LinearRegression())


def make_ridge():
    return lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 40)))


# ----------------------------------------------------------------------------- fold generators
def repeated_group_folds(groups, k=5, repeats=6, seed=0):
    """Random K-fold on whole groups (a neighborhood is never split between train and test), repeated."""
    groups = np.asarray(groups)
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    out = []
    for r in range(repeats):
        for ch in np.array_split(rng.permutation(ug), k):
            te = np.isin(groups, ch)
            out.append((r, np.where(~te)[0], np.where(te)[0]))
    return out


def borough_folds(boroughs, repeats=1):
    """Leave one borough out: predict a whole borough from the other four (a spatial extrapolation test)."""
    b = np.asarray(boroughs)
    return [(r, np.where(b != x)[0], np.where(b == x)[0]) for r in range(repeats) for x in np.unique(b)]


# ----------------------------------------------------------------------------- cross-validation
def cv_run(make, X, y, folds, n_perm=0, seed=0, group_sets=None):
    """Out-of-fold R2 / RMSE and (optionally) permutation importance on the held-out data.

    importance_j = increase in held-out MSE when feature j is shuffled, divided by Var(y): "R2 points lost".
    group_sets: {name: [column indices]} shuffled together (same row permutation) to handle correlated features.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    vy = np.var(y)
    reps = sorted({f[0] for f in folds})
    oof = {r: np.full(n, np.nan) for r in reps}
    imp, gimp = [], []
    for r, tr, te in folds:
        m = make()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        oof[r][te] = pred
        if n_perm:
            base = np.mean((y[te] - pred) ** 2)
            row = np.zeros(p)
            for j in range(p):
                d = 0.0
                for _ in range(n_perm):
                    Xp = X[te].copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    d += np.mean((y[te] - m.predict(Xp)) ** 2) - base
                row[j] = d / n_perm / vy
            imp.append(row)
            if group_sets:
                grow = []
                for name, cols in group_sets.items():
                    d = 0.0
                    for _ in range(n_perm):
                        Xp = X[te].copy()
                        Xp[:, cols] = Xp[:, cols][rng.permutation(len(te))]
                        d += np.mean((y[te] - m.predict(Xp)) ** 2) - base
                    grow.append(d / n_perm / vy)
                gimp.append(grow)
    r2 = [1 - np.nansum((y - oof[r]) ** 2) / np.sum((y - y.mean()) ** 2) for r in reps]
    rm = [np.sqrt(np.nanmean((y - oof[r]) ** 2)) for r in reps]
    res = dict(r2=float(np.mean(r2)), r2_sd=float(np.std(r2)), rmse=float(np.mean(rm)), n_folds=len(folds))
    if n_perm:
        res["imp"] = np.array(imp)
        if group_sets:
            res["gimp"] = np.array(gimp)
            res["group_names"] = list(group_sets)
    return res


def stability_selection(X, y, groups, alpha, l1, B=300, frac=0.5, seed=0):
    """Meinshausen-Buhlmann stability selection: how often does each feature keep a non-zero elastic-net coefficient
    when half of the neighborhoods are drawn at random?  X must be standardized."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    sel = np.zeros(X.shape[1])
    for _ in range(B):
        pick = rng.choice(ug, int(len(ug) * frac), replace=False)
        idx = np.isin(groups, pick)
        m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000).fit(X[idx], y[idx])
        sel += m.coef_ != 0
    return sel / B


def pdp_curves(model, X, names, resolution=15):
    """Partial dependence of each feature (original units) with a direction summary."""
    out = {}
    from scipy.stats import spearmanr
    for j, nm in enumerate(names):
        r = partial_dependence(model, X, [j], grid_resolution=resolution, kind="average")
        g, a = np.asarray(r["grid_values"][0]), np.asarray(r["average"][0])
        rho = spearmanr(g, a)[0] if np.ptp(a) > 0 else 0.0
        out[nm] = dict(x=g.tolist(), y=a.tolist(), range=float(np.ptp(a)), rho=float(rho))
    return out
