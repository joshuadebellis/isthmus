"""
ph_vs_hopkins.py - can persistent homology beat Hopkins-on-UMAP (Sainburg, Thielk & Gentner 2020,
PLoS Comp Biol) at telling discrete vocal repertoires from graded ones, on repertoires whose
answer is known?

Each synthetic repertoire: 600 whistle-like calls rendered as 32x32 spectrograms (their format).

Methods, run on the same calls:
  1. Hopkins on UMAP  - Sainburg's recipe: default 2D UMAP; reference points uniform over the
                        convex hull; distances to the power d; LOWER = more clusterable.
                        It gives a number, not a p-value.
  2. PH test (ours)   - spectrograms -> PCA-10. kNN density (log-density units via estimated
                        intrinsic dimension). H0 persistence of density peaks on the kNN graph.
                        Statistic = prominence of the 2nd-highest peak.
                        Graded null = Silverman critical bandwidth: blur the data just enough that
                        one density peak holds >= 98% of points, resample from that (variance-
                        preserving smoothed bootstrap), recompute the statistic.
                        p = (1 + #null >= observed) / (1 + n_null).
                        Number of types = 1 + #observed peaks above the null's 95th percentile.
  3. Silverman test   - same null; p = share of null samples that are still multimodal at h_crit.

PREDICTIONS (written before running):
  PH:        rejects (p < 0.05) each discrete repertoire in >= 4 of 5 replicates and each graded
             repertoire in <= 1 of 5. Estimated types ~8 for discrete_tight.
  Hopkins:   at least one graded repertoire scores <= the worst discrete_loose replicate
             (misranks graded vs discrete).
  Silverman: right direction, fewer discrete rejections than PH (known to be conservative).

Install:  uv pip install umap-learn scikit-learn scipy tqdm pandas
Run:      python ph_vs_hopkins.py --reps 2 --n_null 40     (smoke test, a few minutes)
          python ph_vs_hopkins.py                          (full: 5 reps x 5 repertoires)
"""
import argparse
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

N = 600
GRID = 32
K_TYPES = 8
K_NN = 15
UNIMODAL_SHARE = 0.98       # "one peak" = one mean-shift mode attracts >= 98% of the points
HOPKINS_REPEATS = 20
HOPKINS_FRAC = 0.1
LO = np.array([0.15, 0.15, -0.25, 0.6])   # start freq, end freq, bow, duration
HI = np.array([0.85, 0.85, 0.25, 1.0])
SPAN = HI - LO


# ------------------------------------------------------------------ synthetic whistles
def render(theta, rng):
    """Whistle contour -> 32x32 spectrogram (ridge along contour, duration-rescaled,
    zero-padded on the right, background noise)."""
    a, e, c, dur = theta
    ncol = int(np.clip(round(dur * GRID), 4, GRID))
    s = np.linspace(0.0, 1.0, ncol)
    f = np.clip(a + (e - a) * s + 4.0 * c * s * (1.0 - s), 0.03, 0.97)
    rows = np.arange(GRID)[:, None]
    spec = np.zeros((GRID, GRID))
    spec[:, :ncol] = np.exp(-0.5 * ((rows - f[None, :] * (GRID - 1)) / 0.8) ** 2)
    spec += np.abs(rng.normal(0.0, 0.05, spec.shape))
    return (spec / spec.max()).ravel()


def spread_prototypes(k, seed=123):
    r = np.random.default_rng(seed)
    cand = LO + SPAN * r.random((4000, 4))
    z = (cand - LO) / SPAN
    chosen = [0]
    dmin = np.linalg.norm(z - z[0], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(dmin))
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(z - z[j], axis=1))
    return cand[chosen]


PROTOS = spread_prototypes(K_TYPES)


def gen_discrete(rng, sd):
    lab = rng.integers(0, K_TYPES, N)
    return np.clip(PROTOS[lab] + sd * SPAN * rng.normal(size=(N, 4)), LO, HI)


def gen_graded_1d(rng):
    u = rng.random(N)
    th = np.column_stack([0.2 + 0.6 * u, 0.8 - 0.6 * u, 0.2 * np.sin(2 * np.pi * u), 0.65 + 0.3 * u])
    return np.clip(th + 0.01 * SPAN * rng.normal(size=(N, 4)), LO, HI)


def gen_graded_uniform(rng):
    return LO + SPAN * rng.random((N, 4))


def gen_graded_gauss(rng):
    return np.clip((LO + HI) / 2 + 0.15 * SPAN * rng.normal(size=(N, 4)), LO, HI)


REPERTOIRES = {
    "discrete_tight": lambda r: gen_discrete(r, 0.02),
    "discrete_loose": lambda r: gen_discrete(r, 0.08),
    "graded_1d": gen_graded_1d,
    "graded_uniform": gen_graded_uniform,
    "graded_gauss": gen_graded_gauss,
}
DISCRETE = ["discrete_tight", "discrete_loose"]
GRADED = ["graded_1d", "graded_uniform", "graded_gauss"]


# ------------------------------------------------------------------ Sainburg-style Hopkins
def hopkins(X, rng, frac=HOPKINS_FRAC, repeats=HOPKINS_REPEATS):
    """LOWER = more clusterable. Reference points uniform over the convex hull of X."""
    n, d = X.shape
    m = max(10, int(frac * n))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    hull = Delaunay(X) if d <= 3 else None
    lo, hi = X.min(axis=0), X.max(axis=0)
    vals = []
    for _ in range(repeats):
        Y = np.empty((0, d))
        while len(Y) < m:
            cand = lo + (hi - lo) * rng.random((4 * m, d))
            if hull is not None:
                cand = cand[hull.find_simplex(cand) >= 0]
            Y = np.vstack([Y, cand[: m - len(Y)]])
        u = nn.kneighbors(Y, n_neighbors=1)[0][:, 0]
        idx = rng.choice(n, m, replace=False)
        w = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
        vals.append(float((w ** d).sum() / ((u ** d).sum() + (w ** d).sum())))
    return float(np.mean(vals))


# ------------------------------------------------------------------ PH statistic
def prominences(Z, k=K_NN):
    """H0 persistence of kNN density peaks on the symmetric kNN graph.
    Density in log units: -d_hat * log(distance to k-th neighbour), d_hat = Levina-Bickel
    intrinsic dimension (MacKay-Ghahramani averaging), so thin and thick clouds are on the same scale.
    Points enter in decreasing density; when two peaks meet, the lower one dies with
    prominence = its peak - current level. Returns (finite prominences largest first, d_hat)."""
    n = len(Z)
    dist, ind = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    T = dist[:, 1:] + 1e-12
    d_hat = 1.0 / np.mean(np.log(T[:, -1:] / T[:, :-1]).sum(axis=1) / (k - 1))
    dens = -d_hat * np.log(T[:, -1])
    nbrs = [set(row[1:].tolist()) for row in ind]
    for i in range(n):
        for j in ind[i, 1:]:
            nbrs[j].add(i)
    parent = np.arange(n)
    peak = {}
    processed = np.zeros(n, bool)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    prom = []
    for i in np.argsort(-dens):
        roots = {find(j) for j in nbrs[i] if processed[j]}
        processed[i] = True
        if not roots:
            peak[i] = dens[i]
            continue
        roots = sorted(roots, key=lambda r: -peak[r])
        top = roots[0]
        parent[i] = top
        for r in roots[1:]:
            prom.append(peak[r] - dens[i])
            parent[r] = top
    final_roots = {find(i) for i in range(n)}
    if len(final_roots) > 1:                       # graph pieces that never met die at the floor
        best = max(final_roots, key=lambda r: peak[r])
        for r in final_roots:
            if r != best:
                prom.append(peak[r] - dens.min())
    return np.sort(np.array(prom))[::-1], float(d_hat)


def top_prominence(p):
    return float(p[0]) if len(p) else 0.0


# ------------------------------------------------------------------ Silverman machinery
def is_multimodal(Z, h, max_iter=300, tol=1e-4):
    """Gaussian-kernel mean shift from every point at bandwidth h. Unimodal iff one mode
    attracts >= UNIMODAL_SHARE of the points (tiny outlier modes are tolerated)."""
    Y = Z.copy()
    zsq = (Z ** 2).sum(axis=1)
    for _ in range(max_iter):
        d2 = np.maximum((Y ** 2).sum(axis=1)[:, None] + zsq[None, :] - 2.0 * Y @ Z.T, 0.0)
        W = np.exp(-(d2 - d2.min(axis=1, keepdims=True)) / (2.0 * h * h))
        Yn = (W @ Z) / W.sum(axis=1, keepdims=True)
        shift = np.abs(Yn - Y).max()
        Y = Yn
        if shift < tol * h:
            break
    centers, counts = [], []
    for y in Y:
        if centers:
            dc = np.linalg.norm(np.asarray(centers) - y, axis=1)
            j = int(np.argmin(dc))
            if dc[j] < 0.3 * h:
                counts[j] += 1
                continue
        centers.append(y)
        counts.append(1)
    return max(counts) < UNIMODAL_SHARE * len(Z)


def critical_bandwidth(Z, steps=14):
    """Smallest h (bisection on a log scale) at which the KDE is unimodal."""
    lo, hi = 0.02, 2.0
    while is_multimodal(Z, hi):
        hi *= 2.0
    for _ in range(steps):
        mid = math.sqrt(lo * hi)
        if is_multimodal(Z, mid):
            lo = mid
        else:
            hi = mid
    return hi


def smoothed_bootstrap(Z, h, rng):
    """Silverman/Efron variance-preserving smoothed bootstrap from the KDE at bandwidth h."""
    n, d = Z.shape
    mu, var = Z.mean(axis=0), Z.var(axis=0)
    Y = Z[rng.integers(0, n, n)] + h * rng.normal(size=(n, d))
    return mu + (Y - mu) / np.sqrt(1.0 + h * h / var)


def ph_and_silverman(Z, rng, n_null):
    obs, d_hat = prominences(Z)
    s_obs = top_prominence(obs)
    h = critical_bandwidth(Z)
    s_null, multi_null = [], []
    for _ in tqdm(range(n_null), desc="    null samples", leave=False):
        Zs = smoothed_bootstrap(Z, h, rng)
        s_null.append(top_prominence(prominences(Zs)[0]))
        multi_null.append(is_multimodal(Zs, h))
    s_null = np.array(s_null)
    thr = float(np.quantile(s_null, 0.95))
    return dict(
        p_ph=float((1 + (s_null >= s_obs).sum()) / (1 + n_null)),
        n_types_ph=1 + int((obs > thr).sum()),
        p_silverman=float(np.mean(multi_null)),
        s_obs=s_obs, s_null95=thr, h_crit=h, d_hat=d_hat,
    )


# ------------------------------------------------------------------ run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n_null", type=int, default=100)
    ap.add_argument("--out", default="ph_vs_hopkins_results.csv")
    a = ap.parse_args()
    t0 = time.time()
    names = list(REPERTOIRES)
    jobs = [(name, rep) for name in names for rep in range(a.reps)]
    records = []
    for name, rep in tqdm(jobs, desc="repertoire x replicate"):
        rng = np.random.default_rng(1000 * names.index(name) + rep)
        X = np.stack([render(th, rng) for th in REPERTOIRES[name](rng)])
        Z = PCA(n_components=10, random_state=0).fit_transform(X)
        Z = Z / np.sqrt((Z ** 2).mean())
        emb = umap.UMAP(random_state=rep).fit_transform(X)
        rec = dict(repertoire=name, rep=rep, hopkins_umap=hopkins(emb, np.random.default_rng(7)))
        rec.update(ph_and_silverman(Z, rng, a.n_null))
        records.append(rec)
        tqdm.write(f"{name:15s} rep {rep}: p_PH={rec['p_ph']:.3f} types={rec['n_types_ph']} "
                   f"p_Silv={rec['p_silverman']:.3f} Hopkins={rec['hopkins_umap']:.3f}")
    df = pd.DataFrame(records)
    df.to_csv(a.out, index=False)

    print("\nPH / Silverman: 'rejects' = p < 0.05 = evidence of discrete types.  "
          "Hopkins: LOWER = more clusterable (~0.5 random).")
    print(f"{'repertoire':15s} {'PH rej':>7s} {'PH p med':>9s} {'types med':>9s} "
          f"{'Silv rej':>8s} {'Silv p med':>10s} {'Hopkins mean':>12s} {'Hopkins min-max':>16s} "
          f"{'h_crit':>7s} {'d_hat':>6s}")
    for name in names:
        g = df[df.repertoire == name]
        print(f"{name:15s} {int((g.p_ph < 0.05).sum()):>3d}/{len(g):<3d} {g.p_ph.median():9.3f} "
              f"{g.n_types_ph.median():9.1f} {int((g.p_silverman < 0.05).sum()):>4d}/{len(g):<3d} "
              f"{g.p_silverman.median():10.3f} {g.hopkins_umap.mean():12.3f} "
              f"{g.hopkins_umap.min():7.3f}-{g.hopkins_umap.max():<8.3f} "
              f"{g.h_crit.median():7.3f} {g.d_hat.median():6.2f}")

    need_hit = math.ceil(0.8 * a.reps)
    max_false = max(1, math.floor(0.2 * a.reps))
    ph_ok = all((df[df.repertoire == n].p_ph < 0.05).sum() >= need_hit for n in DISCRETE) and \
        all((df[df.repertoire == n].p_ph < 0.05).sum() <= max_false for n in GRADED)
    silv_ok = all((df[df.repertoire == n].p_silverman < 0.05).sum() >= need_hit for n in DISCRETE) and \
        all((df[df.repertoire == n].p_silverman < 0.05).sum() <= max_false for n in GRADED)
    worst_loose = df[df.repertoire == "discrete_loose"].hopkins_umap.max()
    best_graded = df[df.repertoire.isin(GRADED)].hopkins_umap.min()
    print(f"\nPH test passes pre-registered rule (discrete >= {need_hit}/{a.reps}, graded <= {max_false}/{a.reps}): {ph_ok}")
    print(f"Silverman passes the same rule: {silv_ok}")
    print(f"Hopkins: lowest graded = {best_graded:.3f} vs worst discrete_loose = {worst_loose:.3f} -> "
          + ("MISRANKS graded vs discrete" if best_graded <= worst_loose else "ranks correctly here"))
    print(f"wrote {a.out}   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()