"""
cluster_compare.py - is gap-PH a better way to CLUSTER vocalizations, or only a better test?

Methods (same calls, same 10-D PCA features unless stated):
  gap-PH        cut significant local gaps (our test's groups)
  HDBSCAN-UMAP  HDBSCAN on 2-D UMAP (Sainburg et al. 2020's clustering approach)
  HDBSCAN-PCA   HDBSCAN on PCA-10
  GMM-BIC       Gaussian mixture, number of components chosen by BIC
HDBSCAN min_cluster_size = max(10, 1% of n), fixed in advance.

Part A - synthetic (render_smooth, 600 calls, 3 replicates): clusters reported on continua
         (truth 1) and ARI on the discrete sets (truth 8).
Part B - Bengalese finches (11 birds, 1500 syllables each): ARI / AMI vs hand labels,
         |clusters - labels|, share of points left unassigned.

PREDICTIONS (written before running):
  A: gap-PH reports 1 cluster in >= 10 of 12 graded runs; HDBSCAN-UMAP reports > 1 in >= 8 of 12.
  B: gap-PH median ARI within 0.05 of HDBSCAN-UMAP (comparable, not better), and >= GMM-BIC.

Run:  python cluster_compare.py      (about 15 minutes)
"""
import argparse
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from diag_render import REPS, render_smooth
from finch_gap import DEFAULT_ROOT, syllable_image
from finch_split_spectra import bird_groups
from gap_vs_hopkins import lb_dim
from ph_vs_hopkins import K_TYPES, LO, N, PROTOS, SPAN, HI
from whistle_gap import MIN_GROUP, SHAPES, gap_tree, groups_from_cut, null_table

METHODS = ["gap-PH", "HDBSCAN-UMAP", "HDBSCAN-PCA", "GMM-BIC"]
GRADED = ["graded_1d", "graded_1d_fixeddur", "graded_uniform", "graded_gauss"]
N_REPS = 3
N_NULL = 200


def pca10(X):
    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    return Z / np.sqrt((Z ** 2).mean())


def other_methods(X, Z, kmax):
    mcs = max(MIN_GROUP, int(0.01 * len(X)))
    emb = umap.UMAP(random_state=0).fit_transform(X)
    out = {"HDBSCAN-UMAP": HDBSCAN(min_cluster_size=mcs).fit_predict(emb),
           "HDBSCAN-PCA": HDBSCAN(min_cluster_size=mcs).fit_predict(Z)}
    best_bic, best_pred = np.inf, None
    for k in range(1, kmax + 1):
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit(Z)
        b = g.bic(Z)
        if b < best_bic:
            best_bic, best_pred = b, g.predict(Z)
    out["GMM-BIC"] = best_pred
    return out


def n_clusters(pred):
    return len(set(np.unique(pred)) - {-1})


def score(truth, pred):
    return dict(ARI=adjusted_rand_score(truth, pred), AMI=adjusted_mutual_info_score(truth, pred),
                k=n_clusters(pred), unassigned=float(np.mean(pred == -1)))


# ------------------------------------------------------------------ part A: synthetic
def gen_with_labels(name, rng):
    if name in ("discrete_tight", "discrete_loose"):
        sd = 0.02 if name == "discrete_tight" else 0.08
        lab = rng.integers(0, K_TYPES, N)
        theta = np.clip(PROTOS[lab] + sd * SPAN * rng.normal(size=(N, 4)), LO, HI)
        return theta, lab
    return REPS[name](rng), np.zeros(N, int)


def part_a():
    rng_null = np.random.default_rng(2026)
    null_cache = {}
    rows = []
    jobs = [(name, rep) for name in REPS for rep in range(N_REPS)]
    for name, rep in tqdm(jobs, desc="A: synthetic"):
        rng = np.random.default_rng(700 + 10 * list(REPS).index(name) + rep)
        theta, lab = gen_with_labels(name, rng)
        X = np.stack([render_smooth(th, rng) for th in theta])
        Z = pca10(X)
        gaps, edges, T = gap_tree(Z)
        d_hat = lb_dim(Z)
        dims = tuple(sorted({min(max(math.floor(d_hat), 1), 10), min(max(math.ceil(d_hat), 1), 10)}))
        if dims not in null_cache:
            null_cache[dims] = null_table(N, list(dims), rng_null, N_NULL)
        table = null_cache[dims]
        thr = max(np.quantile(table[(s, d)], 0.95) for s in SHAPES for d in dims)
        preds = {"gap-PH": groups_from_cut(T, gaps, edges, thr, N)}
        preds.update(other_methods(X, Z, kmax=15))
        for m in METHODS:
            r = dict(repertoire=name, rep=rep, method=m, k=n_clusters(preds[m]),
                     unassigned=float(np.mean(preds[m] == -1)))
            if name.startswith("discrete"):
                r["ARI"] = adjusted_rand_score(lab, preds[m])
            rows.append(r)
        tqdm.write(f"{name:20s} rep {rep}: clusters " + "  ".join(f"{m}={n_clusters(preds[m])}" for m in METHODS))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ part B: finches
def part_b(root, results_csv):
    thr = pd.read_csv(results_csv).set_index("bird").thr.to_dict()
    birds = sorted([d for d in Path(root).iterdir() if d.is_dir() and d.name in thr],
                   key=lambda d: int("".join(ch for ch in d.name if ch.isdigit()) or 0))
    rng = np.random.default_rng(1)
    rows = []
    for bird in tqdm(birds, desc="B: finches"):
        df, audio = bird_groups(bird, float(thr[bird.name]), rng)
        fs = audio[df.wav.iloc[0]][0]
        dur_max = float(np.quantile(df.length, 0.99)) / fs
        X = np.stack([syllable_image(audio[w][1][s:s + n], audio[w][0], dur_max)
                      for w, s, n in zip(df.wav, df.start, df.length)])
        Z = pca10(X)
        preds = {"gap-PH": df.group.to_numpy()}
        preds.update(other_methods(X, Z, kmax=25))
        truth = df.label.to_numpy()
        n_labels = int((pd.Series(truth).value_counts() >= MIN_GROUP).sum())
        line = []
        for m in METHODS:
            s = score(truth, preds[m])
            rows.append(dict(bird=bird.name, method=m, n_labels=n_labels, **s))
            line.append(f"{m} ARI={s['ARI']:.2f} k={s['k']}")
        tqdm.write(f"{bird.name} (labels {n_labels}): " + "  ".join(line))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--results", default="finch_gap_results.csv")
    a = ap.parse_args()
    t0 = time.time()

    A = part_a()
    A.to_csv("cluster_compare_synthetic.csv", index=False)
    print("\nA. clusters reported (median over replicates); truth: discrete = 8, graded = 1")
    print(A.pivot_table(index="repertoire", columns="method", values="k", aggfunc="median")[METHODS].to_string())
    print("\nA. ARI on discrete sets (median)")
    print(A[A.repertoire.str.startswith("discrete")].pivot_table(index="repertoire", columns="method", values="ARI",
                                                                 aggfunc="median")[METHODS].round(3).to_string())
    g = A[A.repertoire.isin(GRADED)]
    one = g[g.method == "gap-PH"].k.le(1).sum()
    fake = g[g.method == "HDBSCAN-UMAP"].k.gt(1).sum()
    print(f"\nprediction A: gap-PH reports 1 cluster in {one}/{len(g) // len(METHODS)} graded runs (pred >= 10); "
          f"HDBSCAN-UMAP reports > 1 in {fake}/{len(g) // len(METHODS)} (pred >= 8)")

    B = part_b(a.root, a.results)
    B.to_csv("cluster_compare_finch.csv", index=False)
    B["k_err"] = (B.k - B.n_labels).abs()
    print("\nB. finches: per-bird ARI vs hand labels")
    print(B.pivot_table(index="bird", columns="method", values="ARI")[METHODS].round(3).to_string())
    summ = B.groupby("method").agg(ARI_median=("ARI", "median"), AMI_median=("AMI", "median"),
                                   k_error_median=("k_err", "median"), unassigned_mean=("unassigned", "mean"))
    print("\nB. summary")
    print(summ.loc[METHODS].round(3).to_string())
    gp, hu, gm = (summ.loc[m, "ARI_median"] for m in ["gap-PH", "HDBSCAN-UMAP", "GMM-BIC"])
    print(f"\nprediction B: |gap-PH - HDBSCAN-UMAP| median ARI = {abs(gp - hu):.3f} (pred <= 0.05): {abs(gp - hu) <= 0.05};  "
          f"gap-PH >= GMM-BIC: {gp >= gm}")
    print(f"wrote cluster_compare_synthetic.csv, cluster_compare_finch.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()