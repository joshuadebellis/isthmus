r"""mice_detect.py - detect mouse USVs in Goffinet et al. 'truncated_audio' chunks.

Rules (fixed in advance):
  band 30-110 kHz; frame is tonal if its peak is >= TONAL_DB above the band median;
  tonal frames < MERGE_MS apart merge; events of MIN_MS-MAX_MS kept;
  events whose peak frequency jumps a median > MAX_JUMP_KHZ per frame are dropped (noise).
Writes mice_events.csv and mice_detect_check.png (24 random detections) for an eyeball check.

Run:  python mice_detect.py --root data\goffinet
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram
from tqdm.auto import tqdm

FMIN, FMAX = 30_000.0, 110_000.0
NPERSEG, HOP = 512, 128
TONAL_DB = 15.0
MERGE_MS = 10.0
MIN_MS, MAX_MS = 5.0, 300.0
MAX_JUMP_KHZ = 6.0


def detect(x, fs):
    f, t, S = spectrogram(x, fs=fs, nperseg=NPERSEG, noverlap=NPERSEG - HOP, mode="psd")
    band = (f >= FMIN) & (f <= FMAX)
    fb = f[band]
    D = 10 * np.log10(S[band] + 1e-20)
    tonal = D.max(axis=0) - np.median(D, axis=0)
    peak_f = fb[D.argmax(axis=0)]
    active = np.flatnonzero(tonal >= TONAL_DB)
    if len(active) == 0:
        return []
    dt = t[1] - t[0]
    gap = max(1, int(MERGE_MS / 1000 / dt))
    runs, s = [], active[0]
    for a, b in zip(active[:-1], active[1:]):
        if b - a > gap:
            runs.append((s, a))
            s = b
    runs.append((s, active[-1]))
    events = []
    for s, e in runs:
        dur_ms = (e - s + 1) * dt * 1000
        if dur_ms < MIN_MS or dur_ms > MAX_MS:
            continue
        fr = np.arange(s, e + 1)
        fr = fr[tonal[fr] >= TONAL_DB]
        jumps = np.abs(np.diff(peak_f[fr])) / 1000
        if len(jumps) == 0 or np.median(jumps) > MAX_JUMP_KHZ:
            continue
        on = max(t[s] - NPERSEG / 2 / fs, 0.0)
        off = min(t[e] + NPERSEG / 2 / fs, len(x) / fs)
        events.append(dict(onset_s=on, offset_s=off, dur_ms=(off - on) * 1000,
                           peak_khz=float(np.median(peak_f[fr])) / 1000, tonality_db=float(np.median(tonal[fr]))))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"data\goffinet")
    root = Path(ap.parse_args().root)
    rows = []
    wavs = sorted(root.rglob("truncated_audio/*.wav"))
    print(f"{len(wavs)} wav files under {root.resolve()}")
    for p in tqdm(wavs, desc="detecting"):
        fs, x = wavfile.read(p)
        x = (x if x.ndim == 1 else x[:, 0]).astype(float)
        for ev in detect(x, fs):
            rows.append(dict(mouse=p.parent.parent.name, file=str(p), fs=fs, **ev))
    df = pd.DataFrame(rows)
    df.to_csv("mice_events.csv", index=False)
    print("\ncalls per mouse:")
    print(df.groupby("mouse").size().to_string())
    print("\nduration ms percentiles 5/25/50/75/95:", np.round(np.percentile(df.dur_ms, [5, 25, 50, 75, 95]), 1))
    print("peak kHz percentiles 5/25/50/75/95:", np.round(np.percentile(df.peak_khz, [5, 25, 50, 75, 95]), 1))

    pick = df.sample(n=min(24, len(df)), random_state=0)
    fig, axes = plt.subplots(4, 6, figsize=(18, 10))
    for ax, (_, r) in zip(axes.flat, pick.iterrows()):
        fs, x = wavfile.read(r.file)
        x = (x if x.ndim == 1 else x[:, 0]).astype(float)
        pad = int(0.01 * fs)
        a, b = max(int(r.onset_s * fs) - pad, 0), min(int(r.offset_s * fs) + pad, len(x))
        f, t, S = spectrogram(x[a:b], fs=fs, nperseg=NPERSEG, noverlap=NPERSEG - HOP, mode="psd")
        m = (f >= 20_000) & (f <= 120_000)
        ax.imshow(10 * np.log10(S[m] + 1e-20), origin="lower", aspect="auto", cmap="magma",
                  extent=[0, (b - a) / fs * 1000, 20, 120])
        ax.set_title(f"{r.mouse} {r.dur_ms:.0f} ms {r.peak_khz:.0f} kHz", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig("mice_detect_check.png", dpi=100)
    print(f"\nwrote mice_events.csv ({len(df)} calls) and mice_detect_check.png")


if __name__ == "__main__":
    main()