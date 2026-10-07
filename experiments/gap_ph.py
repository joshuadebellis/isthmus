"""
gap_ph.py - H0 persistence as GAPS, not density bumps.

Minimum spanning tree of the calls (PCA-10 of 32x32 spectrograms) = single-linkage H0
persistence. Each MST edge that merges two groups of >= MIN_GROUP calls gets
    relative gap = edge length / median local spacing of the calls in both groups,
local spacing = distance to the K-th nearest neighbour.
Gradual bunching along a continuum barely changes this; empty space between types makes it big.

PREDICTIONS (written before running, threshold fixed in advance at GAP_THR = 2.0):
  discrete_tight, discrete_loose: 7 gaps > 2  -> 8 groups, every replicate.
  graded_1d, graded_1d_fixeddur:  no gap > 2  -> 1 group.
  graded_uniform, graded_gauss:   no gap > 2  -> 1 group.

Needs diag_render.py and ph_vs_hopkins.py in the same folder.
Run:  python gap_ph.py      (seconds)
"""
import time

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from diag_render import REPS, render_smooth

K = 15            # local spacing = distance to 15th nearest neighbour
MIN_GROUP = 10    # only merges between groups of >= 10 calls count
GAP_THR = 2.0     # pre-registered: a "real" gap is > 2x the local spacing
N_REPS = 3


def relative_gaps(Z):
    """Relative gaps of all MST merges between groups of >= MIN_GROUP points, largest first."""
    n = len(Z)
    D = squareform(pdist(Z)) + 1e-9          # tiny offset: zero distances would vanish from the sparse MST
    np.fill_diagonal(D, 0.0)
    rk = np.sort(D, axis=1)[:, K]            # column 0 is the point itself
    T = minimum_spanning_tree(D).tocoo()
    order = np.argsort(T.data)
    parent = np.arange(n)
    members = {i: [i] for i in range(n)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gaps = []
    for e in order:
        w, a, b = T.data[e], find(T.row[e]), find(T.col[e])
        if a == b:
            continue
        if len(members[a]) >= MIN_GROUP and len(members[b]) >= MIN_GROUP:
            gaps.append(w / np.median(rk[members[a] + members[b]]))
        if len(members[a]) < len(members[b]):
            a, b = b, a
        parent[b] = a
        members[a].extend(members.pop(b))
    return np.sort(np.array(gaps))[::-1]


def main():
    t0 = time.time()
    print(f"relative gap = MST merge length / median distance to {K}th neighbour "
          f"(groups >= {MIN_GROUP} calls); groups = 1 + #gaps > {GAP_THR}")
    print(f"{'repertoire':20s} {'rep':>3s} {'groups':>6s}   top relative gaps")
    for i, (name, gen) in enumerate(tqdm(list(REPS.items()), desc="repertoires")):
        for rep in range(N_REPS):
            rng = np.random.default_rng(500 + 10 * i + rep)
            X = np.stack([render_smooth(th, rng) for th in gen(rng)])
            Z = PCA(n_components=10, random_state=0).fit_transform(X)
            Z = Z / np.sqrt((Z ** 2).mean())
            g = relative_gaps(Z)
            groups = 1 + int((g > GAP_THR).sum())
            tqdm.write(f"{name:20s} {rep:3d} {groups:6d}   {np.round(g[:10], 2)}")
    print("\ntruth: discrete sets have 8 types (expect 7 big gaps); graded sets have none (expect 1 group)")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()