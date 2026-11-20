"""Checks stats_lib.py against scikit-learn / scipy and against identities that must hold.
Run:  python3 test_stats_lib.py"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, PoissonRegressor

import stats_lib as sl

rng = np.random.default_rng(1)

def ok(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (" " + extra if extra else ""))
    assert cond, name


# ---- OLS vs sklearn / scipy ------------------------------------------------------------------
n, p = 120, 4
X = rng.normal(size=(n, p)); X[:, 1] += 0.5 * X[:, 0]
y = 1.0 + X @ np.array([0.8, -0.4, 0.0, 0.3]) + rng.normal(size=n) * (1 + 0.5 * np.abs(X[:, 0]))
w = rng.uniform(0.5, 2.0, n)
f = sl.ols(X, y, cov="classic")
sk = LinearRegression().fit(X, y)
ok("ols coefficients = sklearn", np.allclose(f["beta"][1:], sk.coef_) and np.isclose(f["beta"][0], sk.intercept_))
fw = sl.ols(X, y, w=w, cov="classic")
skw = LinearRegression().fit(X, y, sample_weight=w)
ok("weighted ols = sklearn sample_weight", np.allclose(fw["beta"][1:], skw.coef_) and np.isclose(fw["beta"][0], skw.intercept_))
lr = stats.linregress(X[:, 0], y)
f1 = sl.ols(X[:, 0], y, cov="classic")
ok("classic SE and p = scipy.linregress", np.isclose(f1["se"][1], lr.stderr) and np.isclose(f1["p"][1], lr.pvalue))
ok("R2 = sklearn", np.isclose(f["r2"], sk.score(X, y)))

# ---- robust SEs ------------------------------------------------------------------------------
h1 = sl.ols(X, y, cov="HC1")
cl = sl.ols(X, y, cluster=np.arange(n))
ok("cluster with one obs per cluster = HC1", np.allclose(h1["se"], cl["se"]))
h3 = sl.ols(X, y, cov="HC3")
jk = np.array([sl.ols(np.delete(X, i, 0), np.delete(y, i), cov="classic")["beta"] for i in range(n)])
se_jk = np.sqrt((n - 1) / n * ((jk - jk.mean(0)) ** 2).sum(0))
ok("HC3 SE close to the jackknife SE", np.all(np.abs(h3["se"] / se_jk - 1) < 0.05), "max rel diff %.3f" % np.max(np.abs(h3["se"] / se_jk - 1)))

# ---- fixed effects = dummy variables ---------------------------------------------------------
N, T = 25, 8
pid = np.repeat(np.arange(N), T); tid = np.tile(np.arange(T), N)
Xp = rng.normal(size=(N * T, 3)) + 0.3 * pid[:, None] / N
a = rng.normal(size=N)[pid]; b = rng.normal(size=T)[tid]
yp = a + b + Xp @ np.array([0.5, 0.0, -0.7]) + rng.normal(size=N * T)
keep = rng.uniform(size=N * T) > 0.15                      # unbalanced panel
pid, tid, Xp, yp = pid[keep], tid[keep], Xp[keep], yp[keep]
wp = rng.uniform(0.5, 3, len(yp))
fe = sl.fe_ols(Xp, yp, pid, tid, w=wp)
D = np.column_stack([(pid == i).astype(float) for i in range(1, N)] + [(tid == j).astype(float) for j in range(1, T)])
dm = sl.ols(np.column_stack([Xp, D]), yp, w=wp, cov="classic")
ok("two-way FE (weighted, unbalanced) = dummy-variable regression", np.allclose(fe["beta"], dm["beta"][1:4], atol=1e-8))
fe1 = sl.fe_ols(Xp, yp, pid)
D1 = np.column_stack([(pid == i).astype(float) for i in range(1, N)])
dm1 = sl.ols(np.column_stack([Xp, D1]), yp, cov="classic")
ok("one-way FE = dummy-variable regression", np.allclose(fe1["beta"], dm1["beta"][1:4], atol=1e-8))

# ---- Poisson ----------------------------------------------------------------------------------
m = 400
Xq = rng.normal(size=(m, 3)); expo = rng.uniform(1e3, 5e4, m)
yq = rng.poisson(expo * np.exp(-6 + Xq @ np.array([0.3, -0.2, 0.0])))
pf = sl.poisson(Xq, yq, np.log(expo))
skp = PoissonRegressor(alpha=0, solver="newton-cholesky", tol=1e-12, max_iter=500).fit(Xq, yq / expo, sample_weight=expo)
ok("poisson coefficients = sklearn PoissonRegressor", np.allclose(pf["beta"][1:], skp.coef_, atol=1e-6) and np.isclose(pf["beta"][0], skp.intercept_, atol=1e-6))
Xc = np.column_stack([np.ones(m), Xq])
ok("poisson score equations are zero", np.allclose(Xc.T @ (yq - pf["mu"]), 0, atol=1e-5))
ok("poisson recovers simulated effects", np.all(np.abs(pf["beta"][1:] - np.array([0.3, -0.2, 0.0])) < 3 * pf["se"][1:]))

# ---- collinearity ----------------------------------------------------------------------------
Z = rng.normal(size=(200, 4)); Z[:, 3] = Z[:, 0] + 0.3 * rng.normal(size=200)
C = np.corrcoef(Z, rowvar=False)
v = sl.vif_from_corr(C)
r2 = [sl.ols(np.delete(Z, j, 1), Z[:, j])["r2"] for j in range(4)]
ok("VIF = 1/(1-R2)", np.allclose(v, 1 / (1 - np.array(r2))))
print("all tests passed")
