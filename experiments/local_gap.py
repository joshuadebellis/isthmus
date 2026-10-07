"""
local_gap.py - gap-based H0 persistence with LOCAL normalization.

Each MST merge between groups of >= MIN_GROUP calls:
    local gap = edge length / median K-NN distance of the points in the K-neighbourhoods
                of the edge's two endpoints
so a gap is judged against spacing where it occurs, not against the dense core.
(The previous version divided by the median over both whole groups: sparse tails looked gappy.)

PREDICTIONS (written before running):
  1. Shape invariance: for each dimension 1..5, the null 95th percentile of the max local gap
     agrees across uniform / gauss / exponential within 20% (max/min <= 1.20), including 1D.
  2. Head-to-head (6 synthetic repertoires x 5 replicates, threshold = most lenient shape at
     floor/ceil(d_hat)): discrete rejected in >= 4/5, each graded in <= 1/5; median types 8
     for discrete_tight and discrete_loose.

Needs gap_ph.py, gap_vs_hopkins.py, diag_render.py, ph_vs_hopkins.py in the same folder.
Run:  python local_gap.py      (about 2-4 minutes)
"""
import math
import time

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from diag_render import REPS, render_smooth
from gap_ph import K, MIN_GROUP
from gap_vs_hopkins import lb_dim

N = 600
N_NULL = 200
N_REPS = 5
DIMS = [1, 2, 3, 4, 5]
SHAPES = ["uniform", "gauss", "exponential"]
DISCRETE = ["discrete_tight", "discrete_loose"]
GRADED = ["graded_1d", "graded_1d_fixeddur", "graded_uniform", "graded_gauss"]


def local_gaps(Z):
    """Locally normalized MST merge gaps between groups of >= MIN_GROUP points, largest first."""
    n = len(Z)
    D = squareform(pdist(Z)) + 1e-9
    np.fill_diagonal(D, 0.0)
    nbr = np.argsort(D, axis=1)[:, :K + 1]            # each point and its K nearest neighbours
    rk = D[np.arange(n), nbr[:, K]]                   # distance to K-th neighbour
    T = minimum_spanning_tree(D).tocoo()
    parent = np.arange(n)
    size = np.ones(n, int)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gaps = []
    for e in np.argsort(T.data):
        i, j, w = T.row[e], T.col[e], T.data[e]
        a, b = find(i), find(j)
        if a == b:
            continue
        if size[a] >= MIN_GROUP and size[b] >= MIN_GROUP:
            local = np.union1d(nbr[i], nbr[j])
            gaps.append(w / np.median(rk[local]))
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    return np.sort(np.array(gaps))[::-1]


def null_cloud(shape, d, rng):
    if shape == "uniform":
        return rng.random((N, d))
    if shape == "gauss":
        return rng.normal(size=(N, d))
    return rng.exponential(size=(N, d))


def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)
    table = {}
    for shape, d in tqdm([(s, d) for d in DIMS for s in SHAPES], desc="null clouds (shape x dim)"):
        vals = []
        for _ in range(N_NULL):
            g = local_gaps(null_cloud(shape, d, rng))
            vals.append(float(g[0]) if len(g) else 0.0)
        table[(shape, d)] = np.array(vals)

    print(f"\nnull: largest LOCAL gap in smooth gap-free clouds (n = {N}); 95th percentile by shape")
    print(f"{'dim':>3s} " + " ".join(f"{s:>12s}" for s in SHAPES) + f" {'max/min':>8s}")
    invariant = True
    for d in DIMS:
        q = [np.quantile(table[(s, d)], 0.95) for s in SHAPES]
        ratio = max(q) / min(q)
        invariant &= ratio <= 1.20
        print(f"{d:3d} " + " ".join(f"{v:12.2f}" for v in q) + f" {ratio:8.2f}")
    print(f"shape invariance (max/min <= 1.20 at every dim): {invariant}")

    records = []
    jobs = [(name, rep) for name in REPS for rep in range(N_REPS)]
    for name, rep in tqdm(jobs, desc="repertoire x replicate"):
        i = list(REPS).index(name)
        r = np.random.default_rng(900 + 10 * i + rep)
        X = np.stack([render_smooth(th, r) for th in REPS[name](r)])
        Z = PCA(n_components=10, random_state=0).fit_transform(X)
        Z = Z / np.sqrt((Z ** 2).mean())
        gaps = local_gaps(Z)
        d_hat = lb_dim(Z)
        dims = sorted({min(max(math.floor(d_hat), 1), DIMS[-1]), min(max(math.ceil(d_hat), 1), DIMS[-1])})
        obs = float(gaps[0]) if len(gaps) else 0.0
        thr = max(np.quantile(table[(s, d)], 0.95) for s in SHAPES for d in dims)
        p = max((1 + (table[(s, d)] >= obs).sum()) / (1 + N_NULL) for s in SHAPES for d in dims)
        n_types = 1 + int((gaps > thr).sum())
        records.append(dict(repertoire=name, rep=rep, d_hat=d_hat, max_gap=obs, thr=thr, p=p, n_types=n_types))
        tqdm.write(f"{name:20s} rep {rep}: d_hat={d_hat:.2f} top gaps={np.round(gaps[:9], 2)} "
                   f"thr={thr:.2f} p={p:.3f} types={n_types}")
    df = pd.DataFrame(records)
    df.to_csv("local_gap_results.csv", index=False)

    print(f"\n{'repertoire':20s} {'rej':>5s} {'p med':>6s} {'types med':>9s} {'max gap med':>11s} {'thr med':>8s} {'d_hat':>6s}")
    for name in REPS:
        g = df[df.repertoire == name]
        print(f"{name:20s} {int((g.p < 0.05).sum()):>2d}/{len(g):<2d} {g.p.median():6.3f} {g.n_types.median():9.1f} "
              f"{g.max_gap.median():11.2f} {g.thr.median():8.2f} {g.d_hat.median():6.2f}")
    need_hit, max_false = math.ceil(0.8 * N_REPS), max(1, math.floor(0.2 * N_REPS))
    ok = all((df[df.repertoire == n].p < 0.05).sum() >= need_hit for n in DISCRETE) and \
        all((df[df.repertoire == n].p < 0.05).sum() <= max_false for n in GRADED)
    print(f"\nlocal gap-PH passes pre-registered rule (discrete >= {need_hit}/{N_REPS}, graded <= {max_false}/{N_REPS}): {ok}")
    print(f"wrote local_gap_results.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()