r"""make_figures.py - the three figures for the README / post.

Fig 1  figures/fig1_hook.png         finch (Bird7) vs mouse (BM007, clean calls): HDBSCAN-on-UMAP
                                      'types' vs gap-test groups on the same UMAP, plus gap barcodes.
Fig 2  figures/fig2_calibration.png  null calibration and synthetic results.
Fig 3  figures/fig3_finch_vs_mouse.png  largest gap / threshold per animal; types reported per animal.

Needs: gaptest.py, finch_gap.py, mice_gap.py and the CSVs from earlier runs; mice_events.csv;
the Koumura folder (finch_gap.DEFAULT_ROOT).
Run:  python make_figures.py
"""
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from finch_gap import DEFAULT_ROOT, MAX_N, load_annotations, syllable_image
from gaptest import SHAPES, gap_test, null_max_gaps
from mice_gap import CLEAN_DB, call_image

OUT = Path("figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b9b8b2", "#e6e5e0"
GRADED = ["graded_1d", "graded_1d_fixeddur", "graded_uniform", "graded_gauss"]
DISCRETE = ["discrete_tight", "discrete_loose"]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
})


def pca10(X):
    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    return Z / np.sqrt((Z ** 2).mean())


def finch_features(bird="Bird7"):
    bird_dir = Path(DEFAULT_ROOT) / bird
    syl, _ = load_annotations(bird_dir)
    rng = np.random.default_rng(0)
    if len(syl) > MAX_N:
        syl = [syl[i] for i in rng.choice(len(syl), MAX_N, replace=False)]
    audio, segs = {}, []
    for wav, start, length, _ in syl:
        if wav not in audio:
            fs, x = wavfile.read(bird_dir / "Wave" / wav)
            audio[wav] = (fs, x if x.ndim == 1 else x[:, 0])
        fs, x = audio[wav]
        segs.append(x[start:start + length])
    dur_max = float(np.quantile([n for _, _, n, _ in syl], 0.99)) / fs
    return np.stack([syllable_image(s, fs, dur_max) for s in tqdm(segs, desc=f"{bird} spectrograms", leave=False)])


def mouse_features(mouse="BM007"):
    ev = pd.read_csv("mice_events.csv")
    ev = ev[(ev.mouse == mouse) & (ev.tonality_db >= CLEAN_DB)].reset_index(drop=True)
    if len(ev) > MAX_N:
        ev = ev.iloc[np.random.default_rng(0).choice(len(ev), MAX_N, replace=False)].reset_index(drop=True)
    dur_max = float(np.quantile(ev.dur_ms, 0.99))
    audio, imgs = {}, []
    for r in tqdm(list(ev.itertuples()), desc=f"{mouse} spectrograms", leave=False):
        if r.file not in audio:
            fs, x = wavfile.read(r.file)
            audio[r.file] = (fs, (x if x.ndim == 1 else x[:, 0]).astype(float))
        fs, x = audio[r.file]
        imgs.append(call_image(x, fs, r.onset_s, r.offset_s, dur_max))
    return np.stack(imgs)


def draw_hulls(ax, emb, labels, color, split_islands=False, eps=0.6):
    """Outline each group. With split_islands, a group that UMAP spread over separate islands
    gets one outline per island instead of one outline spanning the empty space between them.
    Returns the number of groups drawn."""
    n = 0
    for g in sorted(set(labels) - {-1}):
        pts_all = emb[labels == g]
        if len(pts_all) < 10:
            continue
        if split_islands:
            pieces = DBSCAN(eps=eps, min_samples=3).fit_predict(pts_all)
        else:
            pieces = np.zeros(len(pts_all), int)
        drew = False
        for piece in sorted(set(pieces) - {-1}):
            pts = pts_all[pieces == piece]
            if len(pts) < 5:
                continue
            try:
                h = ConvexHull(pts)
            except Exception:
                continue
            loop = np.append(h.vertices, h.vertices[0])
            ax.plot(pts[loop, 0], pts[loop, 1], color=color, lw=1.6)
            drew = True
        n += int(drew)
    return n


def fig1():
    rows = [("Bengalese finch (Bird7)", finch_features("Bird7")),
            ("Mouse (BM007, clean calls)", mouse_features("BM007"))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for r, (name, X) in enumerate(tqdm(rows, desc="fig1 animals")):
        Z = pca10(X)
        emb = umap.UMAP(random_state=0).fit_transform(X)
        hd = HDBSCAN(min_cluster_size=max(10, int(0.01 * len(X)))).fit_predict(emb)
        res = gap_test(Z)
        for c in (0, 1):
            axes[r, c].scatter(emb[:, 0], emb[:, 1], s=4, color=MUTED, linewidths=0)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].grid(False)
        k_hd = draw_hulls(axes[r, 0], emb, hd, ORANGE)
        axes[r, 0].set_title(f"{name}\nHDBSCAN on UMAP: {k_hd} 'call types'", loc="left")
        if res.is_discrete:
            k_gap = draw_hulls(axes[r, 1], emb, res.groups, BLUE)
            axes[r, 1].set_title(f"Gap test: TYPES (p = {res.p:.3f})\n{k_gap} types", loc="left")
        else:
            axes[r, 1].set_title(f"Gap test: continuum (p = {res.p:.2f})\nno types", loc="left")
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
        tqdm.write(f"{name}: HDBSCAN-UMAP hulls={k_hd}  {res}")
    fig.suptitle("Standard clustering finds call types in both animals. The gap test finds them only where empty space separates them.",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig1_hook.png", dpi=200)


def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    for shape, col in zip(SHAPES, [BLUE, ORANGE, AQUA]):
        q = [np.quantile(null_max_gaps(600, d, shape), 0.95) for d in tqdm(range(1, 6), desc=f"null {shape}", leave=False)]
        ax.plot(range(1, 6), q, color=col, lw=2, marker="o", ms=6, label=shape)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("intrinsic dimension of the data")
    ax.set_ylabel("95th percentile of largest gap")
    ax.set_title("(a) How big a gap appears by chance\n(the test uses the highest curve: conservative)", loc="left")
    ax.legend(frameon=False, title="smooth, gap-free cloud", loc="lower right")

    ax = axes[1]
    syn = pd.read_csv("local_gap_results.csv")
    order = DISCRETE + GRADED
    for i, rep in enumerate(order):
        g = syn[syn.repertoire == rep]
        x = i + np.linspace(-0.15, 0.15, len(g))
        ax.scatter(x, g.max_gap, s=36, color=BLUE if rep in DISCRETE else ORANGE, zorder=3,
                   edgecolors="white", linewidths=1)
        ax.hlines(g.thr.median(), i - 0.3, i + 0.3, color=INK2, lw=1.2, ls="--")
    graded = syn[syn.repertoire.isin(GRADED)]
    n_false = int((graded.p < 0.05).sum())
    ax.text(3.5, 4.5, f"continua: {n_false} false alarm in {len(graded)} runs ({n_false / len(graded):.0%})",
            ha="center", va="top", color=INK2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
    ax.set_ylabel("largest gap / local spacing")
    ax.set_title("(b) Simulated repertoires: types (blue) vs\ncontinua (orange); dashes = threshold", loc="left")

    ax = axes[2]
    cc = pd.read_csv("cluster_compare_synthetic.csv")
    cc = cc[cc.repertoire.isin(GRADED)]
    methods = ["gap-PH", "HDBSCAN-PCA", "GMM-BIC", "HDBSCAN-UMAP"]
    med = [float(cc[cc.method == m].k.median()) for m in methods]
    lo = [float(cc[cc.method == m].k.min()) for m in methods]
    hi = [float(cc[cc.method == m].k.max()) for m in methods]
    ax.bar(range(4), med, width=0.6, color=BLUE)
    ax.errorbar(range(4), med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor=INK2, lw=1, capsize=3)
    for i, v in enumerate(med):
        ax.text(i, hi[i] + 0.6, f"{v:g}", ha="center", color=INK, fontsize=10)
    ax.axhline(1, color=INK2, ls="--", lw=1)
    ax.text(0.36, 1.6, "truth: 1", ha="left", color=INK2, fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["gap test", "HDBSCAN\nPCA", "GMM\nBIC", "HDBSCAN\nUMAP"])
    ax.set_ylabel("'call types' reported (median, range)")
    ax.set_title(f"(c) Types reported on continua with no types\n({len(cc) // 4} simulated runs)", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_calibration.png", dpi=200)


def fig3():
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter
    fr = pd.read_csv("finch_gap_results.csv")
    mr = pd.read_csv("mice_gap_results.csv")
    cf = pd.read_csv("cluster_compare_finch.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    f_ratio = fr.max_gap / fr.thr
    m_ratio = mr.max_gap / mr.thr
    ax.scatter(np.zeros(len(f_ratio)) + np.linspace(-0.12, 0.12, len(f_ratio)), f_ratio, s=40, color=BLUE,
               edgecolors="white", linewidths=1, zorder=3, label="Bengalese finches (11 birds)")
    ax.scatter(np.ones(len(m_ratio)) + np.linspace(-0.12, 0.12, len(m_ratio)), m_ratio, s=40, color=ORANGE,
               edgecolors="white", linewidths=1, zorder=3, label="mice (4 animals x 2 runs)")
    ax.axhline(1, color=INK2, ls="--", lw=1)
    ax.text(1.45, 1.04, "chance threshold", ha="right", va="bottom", color=INK2, fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3, 4, 6, 8]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.6, 9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["finches", "mice"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("largest gap / chance threshold")
    ax.set_title("(a) Finch syllables are separated by empty space;\nmouse calls are not", loc="left")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    finch_lab = cf[cf.method == "gap-PH"].n_labels.to_numpy()
    finch_gap = cf[cf.method == "gap-PH"].k.to_numpy()
    finch_hd = cf[cf.method == "HDBSCAN-UMAP"].k.to_numpy()
    mouse_gap = mr.gap_types.to_numpy()
    mouse_hd = mr.hdbscan_umap.to_numpy()
    series = [("hand labels", MUTED, [finch_lab, None]), ("gap test", BLUE, [finch_gap, mouse_gap]),
              ("HDBSCAN on UMAP", ORANGE, [finch_hd, mouse_hd])]
    for j, (name, col, vals) in enumerate(series):
        for i, v in enumerate(vals):
            if v is None:
                continue
            x = i + (j - 1) * 0.22 + np.linspace(-0.05, 0.05, len(v))
            ax.scatter(x, v, s=34, color=col, edgecolors="white", linewidths=1, zorder=3, label=name if i == 0 else None)
    ax.text(1 - 0.22, 1.5, "no hand labels\nfor mice", ha="center", va="bottom", color=INK2, fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["finches", "mice"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 27)
    ax.set_ylabel("call types reported per animal")
    ax.set_title("(b) How many call types?", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig3_finch_vs_mouse.png", dpi=200)


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    for f in tqdm([fig1, fig2, fig3], desc="figures"):
        f()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.png'))}   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()