"""
gap_vs_hopkins.py - calibrated gap-based H0 persistence vs Hopkins-on-UMAP
on synthetic repertoires with known answers.

Gap statistic (from gap_ph.py): MST merges between groups of >= MIN_GROUP calls,
each edge divided by the median distance to the K-th neighbour of the calls involved.

Null: the statistic is scale-free, so chance gaps depend mainly on intrinsic dimension and n.
Simulate smooth gap-free clouds (uniform cube and Gaussian) of the same n in d = 1..5
dimensions, record the largest relative gap of each. For a repertoire with estimated
dimension d_hat, compare with the nulls at floor(d_hat) and ceil(d_hat), both shapes,
and keep the most lenient (largest threshold, largest p) -> conservative.
  p        = (1 + #null max gaps >= observed max gap) / (1 + n_null)
  n_types  = 1 + #observed gaps > null 95th percentile of the max gap
  n_types_1d = same against the 1D null only (worst case, extra conservative)

PREDICTIONS (written before running):
  gap-PH: rejects (p < 0.05) each discrete repertoire in >= 4/5 replicates and each graded
          repertoire in <= 1/5; median n_types ~8 for discrete_tight AND discrete_loose,
          1 for every graded repertoire.
  Hopkins-on-UMAP: lowest graded score <= worst discrete_loose score (misranks).

Needs gap_ph.py, diag_render.py, ph_vs_hopkins.py in the same folder.
Run:  python gap_vs_hopkins.py        (about 5 minutes)
"""
import math
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from diag_render import REPS, render_smooth
from gap_ph import K, relative_gaps
from ph_vs_hopkins import N, hopkins

N_REPS = 5
N_NULL = 200
DIMS = [1, 2, 3, 4, 5]
SHAPES = ["uniform", "gauss"]
DISCRETE = ["discrete_tight", "discrete_loose"]
GRADED = ["graded_1d", "graded_1d_fixeddur", "graded_uniform", "graded_gauss"]


def lb_dim(Z, k=K):
    """Levina-Bickel intrinsic dimension (MacKay-Ghahramani averaging) from k nearest neighbours."""
    T = np.sort(squareform(pdist(Z)), axis=1)[:, 1:k + 1] + 1e-12
    return float(1.0 / np.mean(np.log(T[:, -1:] / T[:, :-1]).sum(axis=1) / (k - 1)))


def max_gap(Z):
    g = relative_gaps(Z)
    return float(g[0]) if len(g) else 0.0


def build_null(n, rng):
    table = {}
    jobs = [(s, d) for d in DIMS for s in SHAPES]
    for shape, d in tqdm(jobs, desc="null clouds (shape x dim)"):
        vals = []
        for _ in range(N_NULL):
            Z = rng.random((n, d)) if shape == "uniform" else rng.normal(size=(n, d))
            vals.append(max_gap(Z))
        table[(shape, d)] = np.array(vals)
    return table


def test_against(gaps, table, dims):
    obs = float(gaps[0]) if len(gaps) else 0.0
    thr = max(np.quantile(table[(s, d)], 0.95) for s in SHAPES for d in dims)
    p = max((1 + (table[(s, d)] >= obs).sum()) / (1 + N_NULL) for s in SHAPES for d in dims)
    return float(p), 1 + int((gaps > thr).sum()), float(thr)


def main():
    t0 = time.time()
    table = build_null(N, np.random.default_rng(2026))
    print("\nnull: largest relative gap in smooth gap-free clouds (n = %d)" % N)
    print(f"{'dim':>3s} {'shape':>8s} {'median':>7s} {'95th':>6s} {'99th':>6s}")
    for d in DIMS:
        for s in SHAPES:
            v = table[(s, d)]
            print(f"{d:3d} {s:>8s} {np.median(v):7.2f} {np.quantile(v, 0.95):6.2f} {np.quantile(v, 0.99):6.2f}")

    records = []
    jobs = [(name, rep) for name in REPS for rep in range(N_REPS)]
    for name, rep in tqdm(jobs, desc="repertoire x replicate"):
        i = list(REPS).index(name)
        rng = np.random.default_rng(900 + 10 * i + rep)
        X = np.stack([render_smooth(th, rng) for th in REPS[name](rng)])
        Z = PCA(n_components=10, random_state=0).fit_transform(X)
        Z = Z / np.sqrt((Z ** 2).mean())
        gaps = relative_gaps(Z)
        d_hat = lb_dim(Z)
        dims = sorted({min(max(math.floor(d_hat), 1), DIMS[-1]), min(max(math.ceil(d_hat), 1), DIMS[-1])})
        p, n_types, thr = test_against(gaps, table, dims)
        _, n_types_1d, _ = test_against(gaps, table, [1])
        emb = umap.UMAP(random_state=rep).fit_transform(X)
        h = hopkins(emb, np.random.default_rng(7))
        records.append(dict(repertoire=name, rep=rep, d_hat=d_hat, max_gap=float(gaps[0]), thr=thr,
                            p_gap=p, n_types=n_types, n_types_1d=n_types_1d, hopkins_umap=h))
        tqdm.write(f"{name:20s} rep {rep}: d_hat={d_hat:.2f} max_gap={gaps[0]:.2f} thr={thr:.2f} "
                   f"p={p:.3f} types={n_types} (1D-null {n_types_1d})  Hopkins={h:.3f}")
    df = pd.DataFrame(records)
    df.to_csv("gap_vs_hopkins_results.csv", index=False)

    print("\ngap-PH: 'rej' = p < 0.05 (evidence of discrete types).  Hopkins: LOWER = more clusterable (~0.5 random)")
    print(f"{'repertoire':20s} {'gap rej':>7s} {'p med':>6s} {'types med':>9s} {'types 1D med':>12s} "
          f"{'Hopkins mean':>12s} {'Hopkins min-max':>16s} {'d_hat':>6s}")
    for name in REPS:
        g = df[df.repertoire == name]
        print(f"{name:20s} {int((g.p_gap < 0.05).sum()):>3d}/{len(g):<3d} {g.p_gap.median():6.3f} "
              f"{g.n_types.median():9.1f} {g.n_types_1d.median():12.1f} {g.hopkins_umap.mean():12.3f} "
              f"{g.hopkins_umap.min():7.3f}-{g.hopkins_umap.max():<8.3f} {g.d_hat.median():6.2f}")

    need_hit, max_false = math.ceil(0.8 * N_REPS), max(1, math.floor(0.2 * N_REPS))
    ok = all((df[df.repertoire == n].p_gap < 0.05).sum() >= need_hit for n in DISCRETE) and \
        all((df[df.repertoire == n].p_gap < 0.05).sum() <= max_false for n in GRADED)
    worst_loose = df[df.repertoire == "discrete_loose"].hopkins_umap.max()
    best_graded = df[df.repertoire.isin(GRADED)].hopkins_umap.min()
    print(f"\ngap-PH passes pre-registered rule (discrete >= {need_hit}/{N_REPS}, graded <= {max_false}/{N_REPS}): {ok}")
    print(f"Hopkins: lowest graded = {best_graded:.3f} vs worst discrete_loose = {worst_loose:.3f} -> "
          + ("MISRANKS graded vs discrete" if best_graded <= worst_loose else "ranks correctly here"))
    print(f"wrote gap_vs_hopkins_results.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()