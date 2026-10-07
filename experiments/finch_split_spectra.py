"""
finch_split_spectra.py - what do gap-PH's split syllables actually look like?

Recomputes the same groups as finch_context.py (same rng, same birds, same thresholds) and,
for every split label, plots the mean onset-aligned spectrogram of each group in REAL time
with 20 ms of context before onset and after offset (dashed lines = annotated onset / median
offset), plus the difference between the two largest groups.

Prints per split:
  edge ratio = mean |difference| in the first and last 10 ms of the syllable
               / mean |difference| in the interior. >> 1: difference sits at the boundaries
               (segmentation suspect). <= ~1: difference runs through the syllable (real variant).
  median duration of each group, and each group's most common previous syllable / run position.

Run:  python finch_split_spectra.py      (under a minute; writes split_spectra.png)
"""
import argparse
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

from finch_context import PURITY, load_with_context
from finch_gap import DEFAULT_ROOT, DR, FMAX, FMIN, MAX_N, syllable_image
from whistle_gap import MIN_GROUP, gap_tree, groups_from_cut

PAD_MS = 20.0
EDGE_MS = 10.0


def bird_groups(bird_dir, thr, rng):
    """Same computation as finch_context.analyse_bird, returning the table with groups and the audio."""
    df = load_with_context(bird_dir)
    if len(df) > MAX_N:
        df = df.iloc[rng.choice(len(df), MAX_N, replace=False)].reset_index(drop=True)
    audio = {}
    segs = []
    for wav, start, length in zip(df.wav, df.start, df.length):
        if wav not in audio:
            fs, x = wavfile.read(bird_dir / "Wave" / wav)
            audio[wav] = (fs, (x if x.ndim == 1 else x[:, 0]).astype(float))
        fs, x = audio[wav]
        segs.append(x[start:start + length])
    dur_max = float(np.quantile(df.length, 0.99)) / fs
    X = np.stack([syllable_image(s, fs, dur_max) for s in segs])
    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    Z = Z / np.sqrt((Z ** 2).mean())
    gaps, edges, T = gap_tree(Z)
    df["group"] = groups_from_cut(T, gaps, edges, thr, len(Z))
    return df, audio


def aligned_spec(x, fs, start, length, grid_ms):
    """Log spectrogram of [onset - PAD, offset + PAD], resampled onto grid_ms (ms from onset), NaN outside."""
    pad = int(PAD_MS * fs / 1000)
    a = max(start - pad, 0)
    seg = x[a:start + length + pad]
    if len(seg) < 256:
        seg = np.pad(seg, (0, 256 - len(seg)))
    f, t, P = spectrogram(seg, fs=fs, nperseg=256, noverlap=224, mode="psd")
    band = (f >= FMIN) & (f <= FMAX)
    S = np.log10(P[band] + 1e-12)
    S = np.clip((S - (S.max() - DR)) / DR, 0.0, 1.0)
    tm = t * 1000.0 - (start - a) * 1000.0 / fs
    out = np.full((S.shape[0], len(grid_ms)), np.nan)
    inside = (grid_ms >= tm[0]) & (grid_ms <= tm[-1])
    for i in range(S.shape[0]):
        out[i, inside] = np.interp(grid_ms[inside], tm, S[i])
    return f[band], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--results", default="finch_gap_results.csv")
    a = ap.parse_args()
    t0 = time.time()
    thr = pd.read_csv(a.results).set_index("bird").thr.to_dict()
    root = Path(a.root)
    birds = sorted([d for d in root.iterdir() if d.is_dir() and d.name in thr],
                   key=lambda d: int("".join(ch for ch in d.name if ch.isdigit()) or 0))
    rng = np.random.default_rng(1)

    panels = []
    for bird in tqdm(birds, desc="birds"):
        df, audio = bird_groups(bird, float(thr[bird.name]), rng)
        ct = pd.crosstab(df.group, df.label)
        ct = ct[ct.index >= 0]
        for lab in ct.columns:
            pure = [g for g in ct.index if ct.loc[g, lab] >= MIN_GROUP and ct.loc[g, lab] / ct.loc[g].sum() >= PURITY]
            if len(pure) < 2:
                continue
            sub = df[(df.label == lab) & df.group.isin(pure)]
            fs = audio[sub.wav.iloc[0]][0]
            max_ms = 1000.0 * sub.length.max() / fs
            grid = np.arange(-PAD_MS, max_ms + PAD_MS, 1.0)
            pure = sorted(pure, key=lambda g: -(sub.group == g).sum())
            means, info = {}, {}
            for g in pure:
                rows = sub[sub.group == g]
                specs = []
                for wav, start, length in zip(rows.wav, rows.start, rows.length):
                    fb, s = aligned_spec(audio[wav][1], fs, start, length, grid)
                    specs.append(s)
                means[g] = np.nanmean(np.stack(specs), axis=0)
                prev_top = rows.prev.value_counts()
                run_top = rows.run_pos.value_counts()
                info[g] = dict(n=len(rows), dur=float(np.median(rows.length)) * 1000.0 / fs,
                               prev=f"{prev_top.index[0]} ({prev_top.iloc[0] / len(rows):.0%})",
                               run=f"{run_top.index[0]} ({run_top.iloc[0] / len(rows):.0%})")
            gA, gB = pure[0], pure[1]
            L = min(info[gA]["dur"], info[gB]["dur"])
            d = np.nanmean(np.abs(means[gA] - means[gB]), axis=0)
            edge_m = ((grid >= 0) & (grid < EDGE_MS)) | ((grid > L - EDGE_MS) & (grid <= L))
            int_m = (grid >= EDGE_MS) & (grid <= L - EDGE_MS)
            ratio = float(np.nanmean(d[edge_m]) / np.nanmean(d[int_m])) if int_m.any() else float("nan")
            panels.append(dict(bird=bird.name, label=lab, groups=pure, means=means, info=info, grid=grid,
                               fb=fb, ratio=ratio, gA=gA, gB=gB))
            tqdm.write(f"\n{bird.name} label {lab}: edge ratio = {ratio:.2f}")
            for g in pure:
                i = info[g]
                tqdm.write(f"  group {g}: n={i['n']}  median duration {i['dur']:.0f} ms  "
                           f"prev = {i['prev']}  run position = {i['run']}")

    if not panels:
        print("no split labels found")
        return
    ncol = 1 + max(len(p["groups"]) for p in panels)
    fig, axes = plt.subplots(len(panels), ncol, figsize=(3.6 * ncol, 2.6 * len(panels)), squeeze=False)
    for r, p in enumerate(panels):
        ext = [p["grid"][0], p["grid"][-1], p["fb"][0] / 1000, p["fb"][-1] / 1000]
        for c, g in enumerate(p["groups"]):
            ax = axes[r, c]
            ax.imshow(p["means"][g], origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=1)
            i = p["info"][g]
            ax.axvline(0, color="w", ls="--", lw=0.8)
            ax.axvline(i["dur"], color="w", ls="--", lw=0.8)
            ax.set_title(f"{p['bird']} lab {p['label']} grp {g} n={i['n']}\nprev {i['prev']}, {i['run']}", fontsize=7)
            ax.tick_params(labelsize=6)
        ax = axes[r, len(p["groups"])]
        diff = p["means"][p["gA"]] - p["means"][p["gB"]]
        ax.imshow(diff, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.set_title(f"grp {p['gA']} - grp {p['gB']}   edge ratio {p['ratio']:.2f}", fontsize=7)
        ax.tick_params(labelsize=6)
        for c in range(len(p["groups"]) + 1, ncol):
            axes[r, c].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("ms from annotated onset", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("kHz", fontsize=7)
    fig.tight_layout()
    fig.savefig("split_spectra.png", dpi=110)
    print(f"\nwrote split_spectra.png   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()