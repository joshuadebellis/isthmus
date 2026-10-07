r"""inspect_mice.py - file types, sample rates, durations and energy by band for a mouse folder.
Run:  python inspect_mice.py --root data\goffinet\BM010
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch
from tqdm.auto import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"data\goffinet\BM010")
    root = Path(ap.parse_args().root)
    print("looking in:", root.resolve())
    files = [p for p in root.rglob("*") if p.is_file()]
    print("file types:", dict(Counter(p.suffix.lower() for p in files)))
    wavs = sorted(p for p in files if p.suffix.lower() == ".wav")
    rates, durs, psd, freqs = Counter(), [], None, None
    for p in tqdm(wavs, desc="reading wavs"):
        fs, x = wavfile.read(p)
        x = x if x.ndim == 1 else x[:, 0]
        rates[fs] += 1
        durs.append(len(x) / fs)
        f, P = welch(x.astype(float), fs=fs, nperseg=1024)
        psd, freqs = (P, f) if psd is None else ((psd + P) if len(P) == len(psd) else psd, freqs)
    durs = np.array(durs)
    print(f"wav files: {len(wavs)}  dtype: {x.dtype}  sample rates: {dict(rates)}")
    print(f"durations (s): min {durs.min():.2f}  median {np.median(durs):.2f}  max {durs.max():.2f}  total {durs.sum() / 60:.1f} min")
    db = 10 * np.log10(psd + 1e-20)
    for lo in range(0, int(freqs[-1] // 1000), 10):
        m = (freqs >= lo * 1000) & (freqs < (lo + 10) * 1000)
        print(f"  {lo:3d}-{lo + 10:3d} kHz  {db[m].mean() - db.max():7.1f} dB")


if __name__ == "__main__":
    main()