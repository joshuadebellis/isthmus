"""
whistle_gap.py - calibrated gap-based H0 persistence vs Hopkins-on-UMAP on real dolphin whistles
(OpenWhistle contours, cached in openwhistle_contours.pkl).

Sets:  signature whistles (SW, positive control: one stereotyped type per dolphin)
       non-signature whistles (NSW, open question: discrete types or a continuum?)
Each traced contour -> 32x32 contour image (smooth duration) -> PCA-10.
Local gap = MST merge length / median K-NN distance around the edge's endpoints (groups >= 10).
Null: smooth gap-free clouds (uniform, gauss, exponential) at the set's n and floor/ceil(d_hat);
most lenient shape/dim is used. p = share of null max gaps >= observed max gap.
Groups = MST with the significant gaps cut; compared with labels by adjusted Rand index (ARI).

PREDICTIONS (written before running):
  SW:   p < 0.05; type count within +-2 of the number of SW categories; ARI(groups, labels) > 0.5.
  NSW:  no prediction.
  Hopkins-on-UMAP: near 0 for both sets (does not distinguish them).

Run:  python whistle_gap.py                     (roughly 8-12 minutes)
      python whistle_gap.py --n_null 50         (quicker, coarser p-values)
"""
import argparse
import math
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from gap_vs_hopkins import lb_dim
from openwhistle_data import clean
from ph_vs_hopkins import hopkins

GRID = 32
K = 15
MIN_GROUP = 10
MAX_N = 1500
SHAPES = ["uniform", "gauss", "exponential"]


# ------------------------------------------------------------------ gap statistic
def gap_tree(Z):
    """MST of Z plus the local gap of every merge between groups of >= MIN_GROUP points.
    Returns (gaps, edge indices into T.data, T)."""
    n = len(Z)
    D = squareform(pdist(Z)) + 1e-9
    np.fill_diagonal(D, 0.0)
    nbr = np.argpartition(D, K, axis=1)[:, :K + 1]             # self + K nearest (unordered)
    rk = D[np.arange(n)[:, None], nbr].max(axis=1)             # distance to K-th neighbour
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
        if size[a] >= MIN_GROUP and size[b] >= MIN_GROUP:
            gaps.append(w / np.median(rk[np.union1d(nbr[i], nbr[j])]))
            edges.append(e)
        if size[a] < size[b]:
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    return np.array(gaps), np.array(edges, int), T


def max_gap(Z):
    g = gap_tree(Z)[0]
    return float(g.max()) if len(g) else 0.0


def null_table(n, dims, rng, n_null):
    table = {}
    for shape, d in tqdm([(s, d) for d in dims for s in SHAPES], desc=f"  null clouds n={n}", leave=False):
        vals = []
        for _ in range(n_null):
            if shape == "uniform":
                Z = rng.random((n, d))
            elif shape == "gauss":
                Z = rng.normal(size=(n, d))
            else:
                Z = rng.exponential(size=(n, d))
            vals.append(max_gap(Z))
        table[(shape, d)] = np.array(vals)
    return table


def groups_from_cut(T, gaps, edges, thr, n):
    keep = np.ones(len(T.data), bool)
    keep[edges[gaps > thr]] = False
    G = coo_matrix((np.ones(keep.sum()), (T.row[keep], T.col[keep])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    sizes = np.bincount(lab)
    lab = lab.copy()
    lab[sizes[lab] < MIN_GROUP] = -1
    return lab


# ------------------------------------------------------------------ whistle images
def render_contour(t, lf, dur99, lo, hi):
    """Traced contour -> 32x32 image: time stretched by duration (fractional last column),
    frequency (log2 Hz) mapped onto rows over the global band [lo, hi], Gaussian ridge."""
    dur = float(t[-1])
    length = max(GRID * min(dur / dur99, 1.0), 2.0)
    j = np.arange(GRID)
    cover = np.clip(length - j, 0.0, 1.0)
    s = np.clip((j + 0.5) / length, 0.0, 1.0)
    f = np.interp(s * dur, t, lf)
    r = np.clip((f - lo) / (hi - lo), 0.0, 1.0) * (GRID - 1)
    rows = np.arange(GRID)[:, None]
    img = cover[None, :] * np.exp(-0.5 * ((rows - r[None, :]) / 0.8) ** 2)
    return (img / img.max()).ravel()


# ------------------------------------------------------------------ one set
def analyse(name, sub, dur99, lo, hi, rng, n_null, ax_umap, ax_bar):
    labels = sub.main_category.to_numpy().astype(str)
    X = np.stack([render_contour(t, lf, dur99, lo, hi) for t, lf in zip(sub.t, sub.lf)])
    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    Z = Z / np.sqrt((Z ** 2).mean())
    n = len(Z)
    gaps, edges, T = gap_tree(Z)
    d_hat = lb_dim(Z)
    dims = sorted({min(max(math.floor(d_hat), 1), 10), min(max(math.ceil(d_hat), 1), 10)})
    table = null_table(n, dims, rng, n_null)
    obs = float(gaps.max()) if len(gaps) else 0.0
    thr = max(np.quantile(table[(s, d)], 0.95) for s in SHAPES for d in dims)
    p = max((1 + (table[(s, d)] >= obs).sum()) / (1 + n_null) for s in SHAPES for d in dims)
    n_types = 1 + int((gaps > thr).sum())
    lab = groups_from_cut(T, gaps, edges, thr, n)
    ok = lab >= 0
    ari = adjusted_rand_score(labels[ok], lab[ok]) if ok.sum() > 1 and len(np.unique(lab[ok])) > 1 else float("nan")
    emb = umap.UMAP(random_state=0).fit_transform(X)
    hop = hopkins(emb, np.random.default_rng(7))

    cats = sorted(np.unique(labels))
    for c in cats:
        m = labels == c
        ax_umap.scatter(emb[m, 0], emb[m, 1], s=3, label=f"{c} ({m.sum()})")
    ax_umap.legend(fontsize=6, markerscale=3, loc="best")
    ax_umap.set_title(f"{name}: UMAP coloured by label   Hopkins = {hop:.3f}", fontsize=10)
    top = np.sort(gaps)[::-1][:30]
    ax_bar.bar(np.arange(len(top)), top, color=["crimson" if g > thr else "grey" for g in top])
    ax_bar.axhline(thr, color="k", ls="--", lw=1, label=f"null 95th pct = {thr:.2f}")
    ax_bar.set_title(f"{name}: largest local gaps   p = {p:.3f}   types = {n_types}   ARI = {ari:.2f}", fontsize=10)
    ax_bar.set_xlabel("rank")
    ax_bar.set_ylabel("gap / local spacing")
    ax_bar.legend(fontsize=8)

    print(f"\n=== {name}: n = {n}, categories = {len(cats)}, d_hat = {d_hat:.2f} (null dims {dims})")
    print(f"top local gaps: {np.round(np.sort(gaps)[::-1][:15], 2)}")
    print(f"null 95th pct = {thr:.2f}   p = {p:.3f}   types = {n_types}   ARI(groups, labels) = {ari:.3f}   Hopkins(UMAP) = {hop:.3f}")
    print("groups (rows, -1 = too small) x labels (columns):")
    print(pd.crosstab(pd.Series(lab, name="group"), pd.Series(labels, name="label")).to_string())
    return dict(set=name, n=n, n_categories=len(cats), d_hat=d_hat, max_gap=obs, thr=thr, p=p,
                n_types=n_types, ari=ari, hopkins_umap=hop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="openwhistle_contours.pkl")
    ap.add_argument("--n_null", type=int, default=200)
    ap.add_argument("--min_span", type=float, default=0.5)
    a = ap.parse_args()
    t0 = time.time()
    raw = pd.read_pickle(a.cache)
    df = clean(raw, max_rate=25.0, max_gap=0.12, conf0=0.08, jump_cost=1.5)
    keep = df.span_frac.isna() | (df.span_frac >= a.min_span)
    df = df[keep].reset_index(drop=True)
    print(f"traced whistles kept (span >= {a.min_span} of annotation): {len(df)}")
    print(df.groupby(["signature", "main_category"]).size().to_string())

    all_lf = np.concatenate(df.lf.to_list())
    lo, hi = np.quantile(all_lf, [0.01, 0.99])
    dur99 = float(np.quantile([t[-1] for t in df.t], 0.99))
    print(f"frequency band log2 Hz [{lo:.2f}, {hi:.2f}] = [{2 ** lo:.0f}, {2 ** hi:.0f}] Hz; duration scale {dur99:.2f} s")

    rng = np.random.default_rng(0)
    sets = []
    for name, mask in [("signature (SW)", df.signature), ("non-signature (NSW)", ~df.signature)]:
        sub = df[mask]
        if len(sub) > MAX_N:
            sub = sub.iloc[rng.choice(len(sub), MAX_N, replace=False)]
        sets.append((name, sub.reset_index(drop=True)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    results = []
    for row, (name, sub) in enumerate(tqdm(sets, desc="whistle sets")):
        results.append(analyse(name, sub, dur99, lo, hi, rng, a.n_null, axes[row, 0], axes[row, 1]))
    fig.tight_layout()
    fig.savefig("whistle_gap.png", dpi=110)
    pd.DataFrame(results).to_csv("whistle_gap_results.csv", index=False)

    print("\nsummary")
    print(pd.DataFrame(results).round(3).to_string(index=False))
    sw = results[0]
    print(f"\nSW positive control: p < 0.05 -> {sw['p'] < 0.05};  "
          f"|types - categories| <= 2 -> {abs(sw['n_types'] - sw['n_categories']) <= 2};  ARI > 0.5 -> {sw['ari'] > 0.5}")
    print(f"wrote whistle_gap.png, whistle_gap_results.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()