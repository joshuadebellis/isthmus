"""
finch_gap.py - real-audio positive control: Bengalese finch syllables (Koumura & Okanoya),
the same data Sainburg et al. 2020 used as their clean example.

Per bird: annotated syllables -> Sainburg-style 32x32 spectrograms (500-8000 Hz on log-spaced
bins, 40 dB range, time stretched by log duration, zero-padded) -> PCA-10 ->
calibrated local-gap H0 persistence (as in whistle_gap.py) + Hopkins on UMAP.

Annotation format: Sequence(WaveFileName, Position, Length, Note*), Note(Position, Length, Label);
Note positions are relative to their Sequence, so absolute start = Sequence.Position + Note.Position.

PREDICTIONS (written before running):
  gap-PH finds types (p < 0.05) in >= 80% of birds;
  median ARI(groups, hand labels) > 0.5;
  median |types - number of labels with >= 10 syllables in the sample| <= 2.

Run:  python finch_gap.py --birds Bird0 Bird1        (quick check, a few minutes)
      python finch_gap.py                            (all birds, roughly 20-40 minutes)
"""
import argparse
import math
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
import umap  # noqa: E402

from gap_vs_hopkins import lb_dim
from ph_vs_hopkins import hopkins
from whistle_gap import MIN_GROUP, SHAPES, gap_tree, groups_from_cut, max_gap

DEFAULT_ROOT = r"C:\Users\joshu\PycharmProjects\whaletalk\data\koumura"
GRID = 32
FMIN, FMAX = 500.0, 8000.0
DR = 4.0            # dynamic range in log10 power units = 40 dB
MAX_N = 1500


# ------------------------------------------------------------------ data
def load_annotations(bird_dir):
    root = ET.parse(bird_dir / "Annotation.xml").getroot()
    syl, bad = [], 0
    for seq in root.iter("Sequence"):
        wav = seq.findtext("WaveFileName")
        s_pos, s_len = int(seq.findtext("Position")), int(seq.findtext("Length"))
        for note in seq.iter("Note"):
            p, n = int(note.findtext("Position")), int(note.findtext("Length"))
            bad += (p + n > s_len)
            syl.append((wav, s_pos + p, n, note.findtext("Label")))
    return syl, bad


def syllable_image(seg, fs, dur_max):
    seg = seg.astype(float)
    if len(seg) < 256:
        seg = np.pad(seg, (0, 256 - len(seg)))
    f, _, P = spectrogram(seg, fs=fs, nperseg=256, noverlap=224, mode="psd")
    band = (f >= FMIN) & (f <= FMAX)
    S = np.log10(P[band] + 1e-12)
    fb = f[band]
    target_f = np.geomspace(FMIN, FMAX, GRID)
    S = np.stack([np.interp(target_f, fb, S[:, k]) for k in range(S.shape[1])], axis=1)
    S = np.clip((S - (S.max() - DR)) / DR, 0.0, 1.0)
    dur_ms = 1000.0 * len(seg) / fs
    w = max(GRID * np.log1p(dur_ms) / np.log1p(1000.0 * dur_max), 2.0)
    j = np.arange(GRID)
    cover = np.clip(w - j, 0.0, 1.0)
    x = np.clip((j + 0.5) / w, 0.0, 1.0) * (S.shape[1] - 1)
    frames = np.arange(S.shape[1])
    img = np.stack([np.interp(x, frames, S[i]) for i in range(GRID)]) * cover[None, :]
    return img.ravel()


# ------------------------------------------------------------------ null with caching
NULL_CACHE = {}


def null_for(n, dims, rng, n_null):
    for d in dims:
        for shape in SHAPES:
            key = (n, shape, d)
            if key in NULL_CACHE:
                continue
            vals = []
            for _ in tqdm(range(n_null), desc=f"  null n={n} {shape} d={d}", leave=False):
                if shape == "uniform":
                    Z = rng.random((n, d))
                elif shape == "gauss":
                    Z = rng.normal(size=(n, d))
                else:
                    Z = rng.exponential(size=(n, d))
                vals.append(max_gap(Z))
            NULL_CACHE[key] = np.array(vals)
    return {(s, d): NULL_CACHE[(n, s, d)] for s in SHAPES for d in dims}


# ------------------------------------------------------------------ one bird
def analyse_bird(bird_dir, rng, n_null, ax):
    syl, bad = load_annotations(bird_dir)
    if len(syl) > MAX_N:
        syl = [syl[i] for i in rng.choice(len(syl), MAX_N, replace=False)]
    wav_cache = {}
    imgs, labels = [], []
    durs = [n for _, _, n, _ in syl]
    fs_ref = None
    for wav, start, length, lab in syl:
        if wav not in wav_cache:
            fs, x = wavfile.read(bird_dir / "Wave" / wav)
            wav_cache[wav] = (fs, x if x.ndim == 1 else x[:, 0])
        fs, x = wav_cache[wav]
        fs_ref = fs
        imgs.append((x[start:start + length], lab))
    dur_max = float(np.quantile(durs, 0.99)) / fs_ref
    X = np.stack([syllable_image(seg, fs_ref, dur_max) for seg, _ in imgs])
    labels = np.array([lab for _, lab in imgs])

    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    Z = Z / np.sqrt((Z ** 2).mean())
    n = len(Z)
    gaps, edges, T = gap_tree(Z)
    d_hat = lb_dim(Z)
    dims = sorted({min(max(math.floor(d_hat), 1), 10), min(max(math.ceil(d_hat), 1), 10)})
    table = null_for(n, dims, rng, n_null)
    obs = float(gaps.max()) if len(gaps) else 0.0
    thr = max(np.quantile(table[(s, d)], 0.95) for s in SHAPES for d in dims)
    p = max((1 + (table[(s, d)] >= obs).sum()) / (1 + n_null) for s in SHAPES for d in dims)
    n_types = 1 + int((gaps > thr).sum())
    grp = groups_from_cut(T, gaps, edges, thr, n)
    ok = grp >= 0
    ari = adjusted_rand_score(labels[ok], grp[ok]) if len(np.unique(grp[ok])) > 1 else 0.0
    counts = pd.Series(labels).value_counts()
    n_labels = int((counts >= MIN_GROUP).sum())
    emb = umap.UMAP(random_state=0).fit_transform(X)
    hop = hopkins(emb, np.random.default_rng(7))

    for lab in sorted(np.unique(labels)):
        m = labels == lab
        ax.scatter(emb[m, 0], emb[m, 1], s=2, label=lab)
    ax.set_title(f"{bird_dir.name}: p={p:.3f} types={n_types} labels={n_labels}\nARI={ari:.2f} Hopkins={hop:.3f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

    tqdm.write(f"\n=== {bird_dir.name}: n={n} (annotation notes past sequence end: {bad}), labels with >= {MIN_GROUP}: {n_labels}, "
               f"d_hat={d_hat:.2f} dims {dims}")
    tqdm.write(f"top gaps {np.round(np.sort(gaps)[::-1][:12], 2)}  thr={thr:.2f}  p={p:.3f}  types={n_types}  "
               f"ARI={ari:.3f}  Hopkins={hop:.3f}")
    tqdm.write(pd.crosstab(pd.Series(grp, name="group"), pd.Series(labels, name="label")).to_string())
    return dict(bird=bird_dir.name, n=n, n_labels=n_labels, d_hat=d_hat, max_gap=obs, thr=thr, p=p,
                n_types=n_types, ari=ari, hopkins_umap=hop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--birds", nargs="*", default=None)
    ap.add_argument("--n_null", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    root = Path(a.root)
    birds = sorted([d for d in root.iterdir() if d.is_dir() and (d / "Annotation.xml").exists()],
                   key=lambda d: int("".join(ch for ch in d.name if ch.isdigit()) or 0))
    if a.birds:
        birds = [d for d in birds if d.name in a.birds]
    print(f"birds: {[d.name for d in birds]}")
    rng = np.random.default_rng(0)
    cols = 4
    rows = int(np.ceil(len(birds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    results = []
    for ax, bird in zip(axes.flat, tqdm(birds, desc="birds")):
        results.append(analyse_bird(bird, rng, a.n_null, ax))
    for ax in list(axes.flat)[len(birds):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("finch_umap.png", dpi=110)
    df = pd.DataFrame(results)
    df.to_csv("finch_gap_results.csv", index=False)
    print("\nsummary")
    print(df.round(3).to_string(index=False))
    hit = (df.p < 0.05).mean()
    print(f"\nfinds types (p<0.05) in {hit:.0%} of birds  (pred >= 80%): {hit >= 0.8}")
    print(f"median ARI = {df.ari.median():.3f}  (pred > 0.5): {df.ari.median() > 0.5}")
    print(f"median |types - labels| = {(df.n_types - df.n_labels).abs().median():.1f}  (pred <= 2): "
          f"{(df.n_types - df.n_labels).abs().median() <= 2}")
    print(f"wrote finch_umap.png, finch_gap_results.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()