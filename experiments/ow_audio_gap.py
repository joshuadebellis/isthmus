r"""ow_audio_gap.py - the topological gate on dolphin whistle AUDIO (OpenWhistle, cached by ow_audio_fetch.py).

Each clip is one whistle. 32x32 spectrograms (3-21 kHz log-spaced, 40 dB range, time stretched by
log duration) -> PCA-10 -> gap-PH with null matched to n and d_hat. Same calls: HDBSCAN-UMAP,
HDBSCAN-PCA, GMM-BIC, Hopkins on UMAP. Each method's clusters are compared with the labels (ARI);
gap-PH groups are also compared with recording session (confound check).

PREDICTIONS (written before running):
  signature whistles (positive control): gate p < 0.05; |types - signature categories| <= 2;
    ARI(gap-PH groups, dolphin) > 0.5.  Expectation: partial pass (individuals overlap acoustically).
  non-signature whistles: no prediction.

Run:  python ow_audio_gap.py
"""
import argparse
import io
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
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from finch_gap import null_for
from gap_vs_hopkins import lb_dim
from openwhistle_data import is_signature
from ph_vs_hopkins import hopkins
from whistle_gap import MIN_GROUP, SHAPES, gap_tree, groups_from_cut

GRID = 32
FMIN, FMAX = 3000.0, 21000.0
DR = 4.0
NPERSEG, NOVERLAP = 1024, 896
MAX_N = 1500
N_NULL = 200


def spec_image(b, dur_max_ms):
    fs, x = wavfile.read(io.BytesIO(b))
    x = (x if x.ndim == 1 else x[:, 0]).astype(float)
    if len(x) < NPERSEG:
        x = np.pad(x, (0, NPERSEG - len(x)))
    f, _, P = spectrogram(x, fs=fs, nperseg=NPERSEG, noverlap=NOVERLAP, mode="psd")
    band = (f >= FMIN) & (f <= FMAX)
    S = np.log10(P[band] + 1e-20)
    tf = np.geomspace(FMIN, FMAX, GRID)
    S = np.stack([np.interp(tf, f[band], S[:, k]) for k in range(S.shape[1])], axis=1)
    S = np.clip((S - (S.max() - DR)) / DR, 0.0, 1.0)
    dur_ms = 1000.0 * len(x) / fs
    w = max(GRID * np.log1p(dur_ms) / np.log1p(dur_max_ms), 2.0)
    j = np.arange(GRID)
    cover = np.clip(w - j, 0.0, 1.0)
    xs = np.clip((j + 0.5) / w, 0.0, 1.0) * (S.shape[1] - 1)
    img = np.stack([np.interp(xs, np.arange(S.shape[1]), S[i]) for i in range(GRID)]) * cover[None, :]
    return img.ravel()


def analyse(name, sub, rng, ax_u, ax_b):
    if len(sub) > MAX_N:
        sub = sub.iloc[rng.choice(len(sub), MAX_N, replace=False)].reset_index(drop=True)
    dur_max = 1000.0 * float(np.quantile(sub.clip_s, 0.99))
    X = np.stack([spec_image(b, dur_max) for b in tqdm(sub.wav, desc=f"  spectrograms {name}", leave=False)])
    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    Z = Z / np.sqrt((Z ** 2).mean())
    n = len(Z)
    labels = sub.main_category.to_numpy()
    sessions = sub["name"].astype(str).to_numpy() if "name" in sub else np.zeros(n, str)
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
    preds = {"gap-PH": grp,
             "HDBSCAN-UMAP": HDBSCAN(min_cluster_size=mcs).fit_predict(emb),
             "HDBSCAN-PCA": HDBSCAN(min_cluster_size=mcs).fit_predict(Z)}
    bics = [(GaussianMixture(k, covariance_type="full", random_state=0).fit(Z), k) for k in range(1, 16)]
    best = min(bics, key=lambda gk: gk[0].bic(Z))[0]
    preds["GMM-BIC"] = best.predict(Z)
    gate = "TYPES" if p < 0.05 else "continuum"
    n_cats = int((pd.Series(labels).value_counts() >= MIN_GROUP).sum())

    for lab in sorted(np.unique(labels)):
        m = labels == lab
        ax_u.scatter(emb[m, 0], emb[m, 1], s=3, label=f"{lab} ({m.sum()})")
    ax_u.legend(fontsize=6, markerscale=3)
    ax_u.set_title(f"{name}: UMAP coloured by label  Hopkins={hop:.3f}", fontsize=9)
    top = np.sort(gaps)[::-1][:30]
    ax_b.bar(np.arange(len(top)), top, color=["crimson" if g > thr else "grey" for g in top])
    ax_b.axhline(thr, color="k", ls="--", lw=1)
    ax_b.set_title(f"{name}: local gaps  p={p:.3f}  gate={gate}  types={n_types}", fontsize=9)

    tqdm.write(f"\n=== {name}: n={n}, label categories={n_cats}, d_hat={d_hat:.2f} dims {dims}")
    tqdm.write(f"top gaps {np.round(np.sort(gaps)[::-1][:12], 2)}  thr={thr:.2f}  p={p:.3f}  gate={gate}  types={n_types}")
    tqdm.write(f"Hopkins(UMAP)={hop:.3f}")
    row = dict(set=name, n=n, categories=n_cats, d_hat=d_hat, max_gap=obs, thr=thr, p=p, gate=gate, gap_types=n_types,
               hopkins_umap=hop, ari_gap_vs_session=adjusted_rand_score(sessions, grp))
    for m, pr in preds.items():
        k = len(set(np.unique(pr)) - {-1})
        a = adjusted_rand_score(labels, pr)
        row[f"k_{m}"] = k
        row[f"ari_{m}"] = a
        tqdm.write(f"  {m:13s} clusters={k:3d}  ARI vs labels={a:.3f}")
    tqdm.write(f"  gap-PH groups vs recording session: ARI={row['ari_gap_vs_session']:.3f}")
    tqdm.write(pd.crosstab(pd.Series(grp, name="gap group"), pd.Series(labels, name="label")).to_string())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="openwhistle_audio.pkl")
    t0 = time.time()
    df = pd.read_pickle(ap.parse_args().cache)
    df["signature"] = df.main_category.map(is_signature)
    print(df.groupby(["signature", "main_category"]).size().to_string())
    sets = [("signature (SW)", df[df.signature].reset_index(drop=True)),
            ("non-signature (NSW)", df[~df.signature & df.main_category.str.upper().str.contains("NSW")].reset_index(drop=True))]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(sets), 2, figsize=(15, 5.5 * len(sets)), squeeze=False)
    res = [analyse(nm, s, rng, axes[i, 0], axes[i, 1]) for i, (nm, s) in enumerate(tqdm(sets, desc="sets"))]
    fig.tight_layout()
    fig.savefig("ow_audio_gap.png", dpi=100)
    out = pd.DataFrame(res)
    out.to_csv("ow_audio_gap_results.csv", index=False)
    print("\nsummary")
    print(out.round(3).T.to_string())
    sw = res[0]
    print(f"\nSW positive control: p < 0.05 -> {sw['p'] < 0.05};  |types - categories| <= 2 -> "
          f"{abs(sw['gap_types'] - sw['categories']) <= 2};  ARI(gap-PH, dolphin) > 0.5 -> {sw['ari_gap-PH'] > 0.5}")
    print(f"wrote ow_audio_gap.png, ow_audio_gap_results.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()