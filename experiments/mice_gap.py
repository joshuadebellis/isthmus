r"""mice_gap.py - the topological gate on mouse USVs (Goffinet et al. 2021 recordings).

Per mouse (calls from mice_events.csv, capped at 1500): 32x32 spectrograms (30-110 kHz, log-spaced
bins, 40 dB range, time stretched by log duration) -> PCA-10 -> gap-PH with a null matched to n and
d_hat -> gate. Same calls: HDBSCAN-UMAP, HDBSCAN-PCA, GMM-BIC cluster counts and Hopkins on UMAP.
Two runs: all detections, and clean calls only (median tonality >= 20 dB).

PREDICTIONS (written before running):
  gate says 'continuum' (p >= 0.05) for each mouse in both runs;
  HDBSCAN-UMAP reports > 1 cluster on the same calls;
  Hopkins on UMAP near 0 ('clustered').

Run:  python mice_gap.py
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
from scipy.io import wavfile
from scipy.signal import spectrogram
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from finch_gap import null_for
from gap_vs_hopkins import lb_dim
from ph_vs_hopkins import hopkins
from whistle_gap import MIN_GROUP, SHAPES, gap_tree, groups_from_cut

GRID = 32
FMIN, FMAX = 30_000.0, 110_000.0
DR = 4.0
NPERSEG, NOVERLAP = 512, 448
MAX_N = 1500
CLEAN_DB = 20.0
N_NULL = 200


def call_image(x, fs, on, off, dur_max_ms):
    seg = x[int(on * fs):int(off * fs)]
    if len(seg) < NPERSEG:
        seg = np.pad(seg, (0, NPERSEG - len(seg)))
    f, _, P = spectrogram(seg, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, mode="psd")
    band = (f >= FMIN) & (f <= FMAX)
    S = np.log10(P[band] + 1e-20)
    tf = np.geomspace(FMIN, FMAX, GRID)
    S = np.stack([np.interp(tf, f[band], S[:, k]) for k in range(S.shape[1])], axis=1)
    S = np.clip((S - (S.max() - DR)) / DR, 0.0, 1.0)
    dur_ms = (off - on) * 1000.0
    w = max(GRID * np.log1p(dur_ms) / np.log1p(dur_max_ms), 2.0)
    j = np.arange(GRID)
    cover = np.clip(w - j, 0.0, 1.0)
    xs = np.clip((j + 0.5) / w, 0.0, 1.0) * (S.shape[1] - 1)
    img = np.stack([np.interp(xs, np.arange(S.shape[1]), S[i]) for i in range(GRID)]) * cover[None, :]
    return img.ravel()


def analyse(name, ev, rng, ax_u, ax_b):
    if len(ev) > MAX_N:
        ev = ev.iloc[rng.choice(len(ev), MAX_N, replace=False)].reset_index(drop=True)
    audio = {}
    dur_max = float(np.quantile(ev.dur_ms, 0.99))
    imgs = []
    for r in ev.itertuples():
        if r.file not in audio:
            fs, x = wavfile.read(r.file)
            audio[r.file] = (fs, (x if x.ndim == 1 else x[:, 0]).astype(float))
        fs, x = audio[r.file]
        imgs.append(call_image(x, fs, r.onset_s, r.offset_s, dur_max))
    X = np.stack(imgs)
    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    Z = Z / np.sqrt((Z ** 2).mean())
    n = len(Z)
    gaps, edges, T = gap_tree(Z)
    d_hat = lb_dim(Z)
    dims = sorted({min(max(math.floor(d_hat), 1), 10), min(max(math.ceil(d_hat), 1), 10)})
    table = null_for(n, dims, rng, N_NULL)
    obs = float(gaps.max()) if len(gaps) else 0.0
    thr = max(np.quantile(table[(s, d)], 0.95) for s in SHAPES for d in dims)
    p = max((1 + (table[(s, d)] >= obs).sum()) / (1 + N_NULL) for s in SHAPES for d in dims)
    n_types = 1 + int((gaps > thr).sum())
    grp = groups_from_cut(T, gaps, edges, thr, n)
    emb = umap.UMAP(random_state=0).fit_transform(X)
    hop = hopkins(emb, np.random.default_rng(7))
    mcs = max(MIN_GROUP, int(0.01 * n))
    k_hu = len(set(HDBSCAN(min_cluster_size=mcs).fit_predict(emb)) - {-1})
    k_hp = len(set(HDBSCAN(min_cluster_size=mcs).fit_predict(Z)) - {-1})
    bics = [GaussianMixture(k, covariance_type="full", random_state=0).fit(Z).bic(Z) for k in range(1, 16)]
    k_gmm = int(np.argmin(bics)) + 1
    gate = "TYPES" if p < 0.05 else "continuum"

    ax_u.scatter(emb[:, 0], emb[:, 1], c=grp, s=3, cmap="tab10")
    ax_u.set_title(f"{name}: UMAP coloured by gap-PH group\nHopkins={hop:.3f}  HDBSCAN-UMAP clusters={k_hu}", fontsize=9)
    top = np.sort(gaps)[::-1][:30]
    ax_b.bar(np.arange(len(top)), top, color=["crimson" if g > thr else "grey" for g in top])
    ax_b.axhline(thr, color="k", ls="--", lw=1)
    ax_b.set_title(f"{name}: largest local gaps  p={p:.3f}  gate={gate}  types={n_types}", fontsize=9)

    tqdm.write(f"\n=== {name}: n={n} d_hat={d_hat:.2f} dims {dims}")
    tqdm.write(f"top gaps {np.round(np.sort(gaps)[::-1][:10], 2)}  thr={thr:.2f}  p={p:.3f}  gate={gate}  types={n_types}")
    tqdm.write(f"HDBSCAN-UMAP={k_hu}  HDBSCAN-PCA={k_hp}  GMM-BIC={k_gmm}  Hopkins={hop:.3f}")
    if n_types > 1:
        ev = ev.assign(group=grp)
        tqdm.write(ev.groupby("group").agg(n=("dur_ms", "size"), dur_ms=("dur_ms", "median"),
                                           peak_khz=("peak_khz", "median"), tonality_db=("tonality_db", "median"))
                   .round(1).to_string())
    return dict(run=name, n=n, d_hat=d_hat, max_gap=obs, thr=thr, p=p, gate=gate, gap_types=n_types,
                hdbscan_umap=k_hu, hdbscan_pca=k_hp, gmm_bic=k_gmm, hopkins_umap=hop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="mice_events.csv")
    t0 = time.time()
    ev = pd.read_csv(ap.parse_args().events)
    runs = []
    for mouse, g in ev.groupby("mouse"):
        runs.append((f"{mouse} all", g.reset_index(drop=True)))
        runs.append((f"{mouse} clean", g[g.tonality_db >= CLEAN_DB].reset_index(drop=True)))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(runs), 2, figsize=(14, 4.5 * len(runs)), squeeze=False)
    res = [analyse(name, e, rng, axes[i, 0], axes[i, 1]) for i, (name, e) in enumerate(tqdm(runs, desc="runs"))]
    fig.tight_layout()
    fig.savefig("mice_gap.png", dpi=100)
    df = pd.DataFrame(res)
    df.to_csv("mice_gap_results.csv", index=False)
    print("\nsummary")
    print(df.round(3).to_string(index=False))
    print(f"\nprediction - gate says continuum in every run: {(df.gate == 'continuum').all()}")
    print(f"prediction - HDBSCAN-UMAP reports > 1 cluster in every run: {(df.hdbscan_umap > 1).all()}")
    print(f"wrote mice_gap.png, mice_gap_results.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()