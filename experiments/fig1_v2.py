r"""fig1_v2.py - Fig 1 with island-aware outlines in the gap-test panel.
Writes figures/fig1_hook_v2.png (new name, so it cannot be confused with the old figure).

Run:  python fig1_v2.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, HDBSCAN
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from gaptest import gap_test
from make_figures import BLUE, INK, INK2, MUTED, ORANGE, OUT, finch_features, mouse_features, pca10


def outline(ax, pts, color):
    try:
        h = ConvexHull(pts)
    except Exception:
        return False
    loop = np.append(h.vertices, h.vertices[0])
    ax.plot(pts[loop, 0], pts[loop, 1], color=color, lw=1.6)
    return True


def outline_groups(ax, emb, labels, color, split_islands, eps):
    groups_drawn, report = 0, []
    for g in sorted(set(labels) - {-1}):
        pts_all = emb[labels == g]
        if len(pts_all) < 10:
            continue
        pieces = DBSCAN(eps=eps, min_samples=3).fit_predict(pts_all) if split_islands else np.zeros(len(pts_all), int)
        n_islands = 0
        for piece in sorted(set(pieces) - {-1}):
            pts = pts_all[pieces == piece]
            if len(pts) >= 5 and outline(ax, pts, color):
                n_islands += 1
        if n_islands:
            groups_drawn += 1
            report.append(n_islands)
    return groups_drawn, report


def main():
    OUT.mkdir(exist_ok=True)
    rows = [("Bengalese finch (Bird7)", finch_features("Bird7")),
            ("Mouse (BM007, clean calls)", mouse_features("BM007"))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for r, (name, X) in enumerate(tqdm(rows, desc="animals")):
        Z = pca10(X)
        emb = umap.UMAP(random_state=0).fit_transform(X)
        eps = 0.03 * float(np.ptp(emb, axis=0).max())
        hd = HDBSCAN(min_cluster_size=max(10, int(0.01 * len(X)))).fit_predict(emb)
        res = gap_test(Z)
        for c in (0, 1):
            axes[r, c].scatter(emb[:, 0], emb[:, 1], s=4, color=MUTED, linewidths=0)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
        k_hd, _ = outline_groups(axes[r, 0], emb, hd, ORANGE, split_islands=False, eps=eps)
        axes[r, 0].set_title(f"{name}\nHDBSCAN on UMAP: {k_hd} 'call types'", loc="left")
        if res.is_discrete:
            k_gap, islands = outline_groups(axes[r, 1], emb, res.groups, BLUE, split_islands=True, eps=eps)
            axes[r, 1].set_title(f"Gap test: TYPES (p = {res.p:.3f})\n{k_gap} types", loc="left")
            print(f"{name}: gap-test groups drawn={k_gap}; islands per group={islands}")
        else:
            axes[r, 1].set_title(f"Gap test: continuum (p = {res.p:.2f})\nno types", loc="left")
            print(f"{name}: gap test says continuum (p={res.p:.3f})")
        top = np.sort(res.gaps)[::-1][:30]
        ax = axes[r, 2]
        ax.bar(np.arange(len(top)), top, width=0.8, color=[BLUE if g > res.threshold else MUTED for g in top])
        ax.axhline(res.threshold, color=INK2, ls="--", lw=1)
        ax.text(len(top) - 0.5, res.threshold * 1.08, "chance threshold (95%)", ha="right", va="bottom", color=INK2, fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(0.3, 12)
        ax.set_xlabel("merge rank (largest first)")
        ax.set_ylabel("gap / local spacing")
        ax.set_title("Largest gaps between groups of calls", loc="left")
    fig.suptitle("Standard clustering finds call types in both animals. The gap test finds them only where empty space separates them.",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "fig1_hook_v2.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()