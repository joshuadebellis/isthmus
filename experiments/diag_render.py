"""
diag_render.py - did whole-column duration rounding in ph_vs_hopkins.render() build fake
discreteness into the "graded" repertoires?

Needs ph_vs_hopkins.py in the same folder (imports its generators, hopkins, prominences).

PREDICTIONS (written before running):
  1. Old render: graded_1d in PCA/UMAP splits into ~10 strands/clumps that line up with
     the integer column count. Fixed render: one continuous curve.
  2. Raw PH peak count (prominence > 1.0) for graded_1d drops substantially with the fix.
  3. Hopkins-on-UMAP for fixed graded_1d and graded_1d_fixeddur stays near 0
     (a clean continuum scored as clustered) - a genuine Hopkins failure, not a render artifact.

Run:  python diag_render.py      (about 2-3 minutes, writes diag_graded1d.png)
"""
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from ph_vs_hopkins import (GRID, HI, LO, N, SPAN, gen_discrete, gen_graded_1d,
                           gen_graded_gauss, gen_graded_uniform, hopkins, prominences, render)

STRONG = 1.0   # prominence threshold (log-density units) for counting a "strong" peak, fixed in advance


def render_smooth(theta, rng):
    """Same whistle, but duration is continuous: the last occupied column is partially
    filled (coverage = fractional part), so a tiny duration change gives a tiny image change."""
    a, e, c, dur = theta
    length = dur * GRID
    j = np.arange(GRID)
    cover = np.clip(length - j, 0.0, 1.0)
    s = np.clip((j + 0.5) / length, 0.0, 1.0)
    f = np.clip(a + (e - a) * s + 4.0 * c * s * (1.0 - s), 0.03, 0.97)
    rows = np.arange(GRID)[:, None]
    spec = cover[None, :] * np.exp(-0.5 * ((rows - f[None, :] * (GRID - 1)) / 0.8) ** 2)
    spec = spec + np.abs(rng.normal(0.0, 0.05, spec.shape))
    return (spec / spec.max()).ravel()


def gen_graded_1d_fixeddur(rng):
    th = gen_graded_1d(rng)
    th[:, 3] = 0.8
    return th


REPS = {
    "discrete_tight": lambda r: gen_discrete(r, 0.02),
    "discrete_loose": lambda r: gen_discrete(r, 0.08),
    "graded_1d": gen_graded_1d,
    "graded_1d_fixeddur": gen_graded_1d_fixeddur,
    "graded_uniform": gen_graded_uniform,
    "graded_gauss": gen_graded_gauss,
}


def analyse(X):
    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    Z = Z / np.sqrt((Z ** 2).mean())
    emb = umap.UMAP(random_state=0).fit_transform(X)
    prom, d_hat = prominences(Z)
    return dict(Z=Z, emb=emb, hop=hopkins(emb, np.random.default_rng(7)),
                n_strong=1 + int((prom > STRONG).sum()), top=prom[:6], d_hat=d_hat)


def main():
    t0 = time.time()

    # ---- part 1: graded_1d, old vs fixed render, same calls
    theta = gen_graded_1d(np.random.default_rng(42))
    u = np.clip((theta[:, 0] - 0.2) / 0.6, 0, 1)
    ncol = np.clip(np.round(theta[:, 3] * GRID), 4, GRID).astype(int)
    print(f"graded_1d: distinct whole-column durations under the old render = {len(np.unique(ncol))}")
    res = {}
    for label, fn in tqdm([("old (whole columns)", render), ("fixed (smooth duration)", render_smooth)],
                          desc="graded_1d old vs fixed"):
        rng = np.random.default_rng(7)
        X = np.stack([fn(th, rng) for th in theta])
        res[label] = analyse(X)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, (label, r) in enumerate(res.items()):
        axes[row, 0].scatter(r["Z"][:, 0], r["Z"][:, 1], c=u, s=4, cmap="viridis")
        axes[row, 0].set_title(f"{label}\nPCA, colour = position along continuum", fontsize=10)
        axes[row, 1].scatter(r["emb"][:, 0], r["emb"][:, 1], c=u, s=4, cmap="viridis")
        axes[row, 1].set_title(f"UMAP, colour = position   Hopkins={r['hop']:.3f}", fontsize=10)
        axes[row, 2].scatter(r["emb"][:, 0], r["emb"][:, 1], c=ncol, s=4, cmap="tab10")
        axes[row, 2].set_title("UMAP, colour = whole-column duration", fontsize=10)
    fig.tight_layout()
    fig.savefig("diag_graded1d.png", dpi=110)

    print(f"\n{'graded_1d render':26s} {'Hopkins(UMAP)':>13s} {'PH strong peaks':>15s} {'d_hat':>6s}   top prominences")
    for label, r in res.items():
        print(f"{label:26s} {r['hop']:13.3f} {r['n_strong']:15d} {r['d_hat']:6.2f}   {np.round(r['top'], 2)}")

    # ---- part 2: every repertoire with the fixed render
    print(f"\nFIXED render, all repertoires (Hopkins: lower = more clusterable, ~0.5 random)")
    print(f"{'repertoire':20s} {'Hopkins(UMAP)':>13s} {'PH strong peaks':>15s} {'d_hat':>6s}   top prominences")
    for i, (name, gen) in enumerate(tqdm(list(REPS.items()), desc="fixed render")):
        rng = np.random.default_rng(500 + i)
        X = np.stack([render_smooth(th, rng) for th in gen(rng)])
        r = analyse(X)
        tqdm.write(f"{name:20s} {r['hop']:13.3f} {r['n_strong']:15d} {r['d_hat']:6.2f}   {np.round(r['top'], 2)}")
    print(f"\ntruth: discrete sets have 8 types; graded sets have none (1 peak expected)")
    print(f"wrote diag_graded1d.png   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()