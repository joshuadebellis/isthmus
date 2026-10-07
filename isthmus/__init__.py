r"""gaptest - a calibrated topological test for discrete types in a set of vocalizations.

Idea: discrete call types are separated by empty space; a continuum is not, even if it is lumpy.
H0 persistence (minimum spanning tree) of the calls' feature vectors gives every merge between
groups of >= min_group calls a LOCAL gap = edge length / median distance-to-k-th-neighbour around
the edge. The largest local gap is compared with gaps in smooth, gap-free clouds (uniform, Gaussian,
exponential) of the same size and intrinsic dimension; the most lenient null is used.

Usage:
    from gaptest import gap_test
    res = gap_test(Z)          # Z: (n_calls, n_features), e.g. PCA-10 of 32x32 spectrograms
    res.p, res.is_discrete, res.n_types, res.groups

Run this file directly for a self-test on three known cases.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

K = 15
MIN_GROUP = 10
SHAPES = ("uniform", "gauss", "exponential")
_NULL_CACHE = {}


def intrinsic_dim(Z, k=K):
    """Levina-Bickel maximum-likelihood intrinsic dimension (MacKay-Ghahramani averaging)."""
    T = np.sort(squareform(pdist(Z)), axis=1)[:, 1:k + 1] + 1e-12
    return float(1.0 / np.mean(np.log(T[:, -1:] / T[:, :-1]).sum(axis=1) / (k - 1)))


def gap_tree(Z, k=K, min_group=MIN_GROUP):
    """MST of Z and the local gap of every merge between groups of >= min_group points.
    Returns (gaps, edge indices into T.data, T)."""
    n = len(Z)
    D = squareform(pdist(Z)) + 1e-9
    np.fill_diagonal(D, 0.0)
    nbr = np.argpartition(D, k, axis=1)[:, :k + 1]
    rk = D[np.arange(n)[:, None], nbr].max(axis=1)
    T = minimum_spanning_tree(D).tocoo()
    parent = np.arange(n)
    size = np.ones(n, int)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    gaps, edges = [], []
    for e in np.argsort(T.data):
        i, j, w = T.row[e], T.col[e], T.data[e]
        a, b = find(i), find(j)
        if a == b:
            continue
        if size[a] >= min_group and size[b] >= min_group:
            gaps.append(w / np.median(rk[np.union1d(nbr[i], nbr[j])]))
            edges.append(e)
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    return np.array(gaps), np.array(edges, int), T


def null_max_gaps(n, d, shape, n_null=200, seed=2026, k=K, min_group=MIN_GROUP):
    """Largest local gap in n_null smooth gap-free clouds of n points in d dimensions (cached)."""
    key = (n, d, shape, n_null, k, min_group)
    if key not in _NULL_CACHE:
        rng = np.random.default_rng(seed + 1000 * d + SHAPES.index(shape))
        vals = []
        for _ in tqdm(range(n_null), desc=f"null n={n} d={d} {shape}", leave=False):
            if shape == "uniform":
                Z = rng.random((n, d))
            elif shape == "gauss":
                Z = rng.normal(size=(n, d))
            else:
                Z = rng.exponential(size=(n, d))
            g = gap_tree(Z, k, min_group)[0]
            vals.append(float(g.max()) if len(g) else 0.0)
        _NULL_CACHE[key] = np.array(vals)
    return _NULL_CACHE[key]


def groups_from_cut(T, gaps, edges, thr, n, min_group=MIN_GROUP):
    keep = np.ones(len(T.data), bool)
    keep[edges[gaps > thr]] = False
    G = coo_matrix((np.ones(keep.sum()), (T.row[keep], T.col[keep])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    sizes = np.bincount(lab)
    lab = lab.copy()
    lab[sizes[lab] < min_group] = -1
    return lab


@dataclass
class GapTestResult:
    p: float
    alpha: float
    n_types: int
    groups: np.ndarray
    gaps: np.ndarray
    threshold: float
    d_hat: float
    null_dims: list = field(default_factory=list)

    @property
    def is_discrete(self):
        return self.p < self.alpha

    def __repr__(self):
        verdict = "TYPES" if self.is_discrete else "continuum"
        return (f"GapTestResult({verdict}: p={self.p:.3f}, n_types={self.n_types}, max_gap="
                f"{(self.gaps.max() if len(self.gaps) else 0):.2f}, threshold={self.threshold:.2f}, "
                f"d_hat={self.d_hat:.2f}, null_dims={self.null_dims})")


def gap_test(Z, n_null=200, alpha=0.05, k=K, min_group=MIN_GROUP, max_dim=10):
    """Calibrated test for discrete types. Z: (n_calls, n_features)."""
    Z = np.asarray(Z, float)
    n = len(Z)
    gaps, edges, T = gap_tree(Z, k, min_group)
    d_hat = intrinsic_dim(Z, k)
    dims = sorted({min(max(math.floor(d_hat), 1), max_dim), min(max(math.ceil(d_hat), 1), max_dim)})
    nulls = [null_max_gaps(n, d, s, n_null, k=k, min_group=min_group) for d in dims for s in SHAPES]
    obs = float(gaps.max()) if len(gaps) else 0.0
    thr = max(float(np.quantile(v, 1 - alpha)) for v in nulls)
    p = max(float((1 + (v >= obs).sum()) / (1 + len(v))) for v in nulls)
    n_types = 1 + int((gaps > thr).sum())
    groups = groups_from_cut(T, gaps, edges, thr, n, min_group)
    return GapTestResult(p=p, alpha=alpha, n_types=n_types, groups=groups, gaps=gaps,
                         threshold=thr, d_hat=d_hat, null_dims=dims)


def _self_test():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=10.0, size=(8, 5))
    clusters = np.vstack([c + rng.normal(size=(75, 5)) for c in centers])
    uniform = rng.random((600, 3))
    t = rng.random(600) * 6.0
    curve = np.column_stack([np.cos(t), np.sin(t), 0.3 * t] + [0.02 * rng.normal(size=600) for _ in range(7)])
    cases = [("8 separated clusters", clusters, "TYPES, 8"),
             ("uniform cloud (3-D)", uniform, "continuum"),
             ("thin curved filament (10-D)", curve, "continuum")]
    for name, Z, expected in cases:
        res = gap_test(Z)
        print(f"{name:30s} expected {expected:10s} -> {res}")


if __name__ == "__main__":
    _self_test()