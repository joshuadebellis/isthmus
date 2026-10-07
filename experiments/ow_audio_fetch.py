r"""ow_audio_fetch.py - cache OpenWhistle whistle audio (WAV bytes) and labels; skips the image columns.
Prints category counts, sample rates, and clip length vs annotated duration (is each clip just the whistle?).

Run:  python ow_audio_fetch.py
"""
import io

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.io import wavfile
from tqdm.auto import tqdm

from openwhistle_data import _class_names, _parquet_files, _split_of

COLS = ["audio", "main_category", "name", "onset", "offset", "duration", "recording_duration", "snr_db", "whistle_type"]
OUT = "openwhistle_audio.pkl"


def main():
    fs, files = _parquet_files()
    frames, names = [], {}
    for f in tqdm(files, desc="parquet files"):
        pf = pq.ParquetFile(fs.open(f))
        names.update(_class_names(pf))
        cols = [c for c in COLS if c in pf.schema_arrow.names]
        t = pf.read(columns=cols).to_pandas()
        t["split"] = _split_of(f)
        t["wav"] = [a.get("bytes") if isinstance(a, dict) else None for a in t.pop("audio")]
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    if np.issubdtype(df.main_category.dtype, np.integer) and "main_category" in names:
        df["main_category"] = df.main_category.map(dict(enumerate(names["main_category"])))
    df["main_category"] = df.main_category.astype(str)
    df = df[df.wav.notna()].reset_index(drop=True)
    sr, clip = [], []
    for b in tqdm(df.wav, desc="reading wav headers"):
        r, x = wavfile.read(io.BytesIO(b))
        sr.append(r)
        clip.append(len(x) / r)
    df["sr"] = sr
    df["clip_s"] = clip
    df.to_pickle(OUT)

    print(f"\ncached {len(df)} clips -> {OUT}  ({df.wav.map(len).sum() / 1e6:.0f} MB of audio)")
    print("\nclips per category:")
    print(df.main_category.value_counts().to_string())
    print("\nsample rates:", df.sr.value_counts().to_dict())
    q = [5, 25, 50, 75, 95]
    print("clip length s   5/25/50/75/95:", np.round(np.percentile(df.clip_s, q), 3))
    if "duration" in df:
        d = pd.to_numeric(df.duration, errors="coerce")
        print("annotated dur s 5/25/50/75/95:", np.round(np.nanpercentile(d, q), 3))
        print("clip / duration 5/25/50/75/95:", np.round(np.nanpercentile(df.clip_s / d, q), 3))
    if "onset" in df:
        print("onset  5/50/95:", np.round(np.nanpercentile(pd.to_numeric(df.onset, errors="coerce"), [5, 50, 95]), 3))
        print("offset 5/50/95:", np.round(np.nanpercentile(pd.to_numeric(df.offset, errors="coerce"), [5, 50, 95]), 3))


if __name__ == "__main__":
    main()