"""
openwhistle_data.py  —  load OpenWhistle 'Whistle-Classification' contours.

Source: Hugging Face dataset  dolphinteam/Whistle-Classification
        (OpenWhistle, Mustun et al., arXiv 2609.34839; paper states CC-BY 4.0 — cite it).

Only the light columns are read (f0 contour, labels, session, SNR); the audio and
960x960 spectrogram images are skipped, so the transfer stays small.

  python openwhistle_data.py --inspect          # print files, schema, class names, sample rows
  python openwhistle_data.py --fetch            # build the local cache (openwhistle_contours.pkl)

Everything downstream reads the cache: a pandas pickle with one row per whistle:
  split, t (np.array, s), f (np.array, Hz), conf (np.array or None), f0_ok,
  main_category (str), whistle_type (int), whistle_name, session, snr_db, duration
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

REPO = "dolphinteam/Whistle-Classification"
CACHE = "openwhistle_contours.pkl"
WANT = ["f0_time", "f0_hz", "f0_conf", "f0_ok", "f0_bad_reason", "main_category", "class label",
        "whistle_type", "whistle_name", "name", "snr_db", "duration", "onset", "offset"]

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ------------------------------------------------------------------ HF access
def _parquet_files():
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{REPO}/**/*.parquet"))
    return fs, files


def _split_of(path):
    m = re.search(r"(train|validation|valid|val|test)", os.path.basename(path)) or \
        re.search(r"/(train|validation|valid|val|test)/", path)
    s = m.group(1) if m else "unknown"
    return {"valid": "validation", "val": "validation"}.get(s, s)


def _class_names(pf):
    """ClassLabel names stored by HF in the parquet 'huggingface' metadata."""
    meta = pf.schema_arrow.metadata or {}
    raw = meta.get(b"huggingface")
    out = {}
    if raw:
        feats = json.loads(raw).get("info", {}).get("features", {})
        for k, v in feats.items():
            if isinstance(v, dict) and v.get("_type") == "ClassLabel":
                out[k] = v.get("names")
    return out


def inspect():
    import pyarrow.parquet as pq
    fs, files = _parquet_files()
    print(f"{len(files)} parquet files")
    for f in files:
        print("  ", f, f"[{_split_of(f)}]")
    pf = pq.ParquetFile(fs.open(files[0]))
    print("\nschema:")
    for fld in pf.schema_arrow:
        print(f"  {fld.name:22s} {fld.type}")
    print("\nClassLabel names:", json.dumps(_class_names(pf), indent=1))
    light = [c for c in WANT if c in pf.schema_arrow.names and not c.startswith("f0_")]
    print("\nfirst rows (light columns):")
    print(pf.read_row_group(0, columns=light).to_pandas().head(10).to_string())
    t = pf.read_row_group(0, columns=[c for c in ("f0_time", "f0_hz", "f0_conf") if c in pf.schema_arrow.names]).to_pandas().head(2)
    for c in t.columns:
        v = np.asarray(t[c].iloc[0])
        print(f"\n{c}: len={len(v)} first={v[:5]} dtype={v.dtype}")


def fetch(cache=CACHE):
    import pyarrow.parquet as pq
    try:
        fs, files = _parquet_files()
        opener = fs.open
    except Exception as e:  # fall back to downloading the parquet files
        print(f"[fetch] column-wise remote read unavailable ({e}); downloading parquet files")
        from huggingface_hub import snapshot_download
        root = snapshot_download(REPO, repo_type="dataset", allow_patterns=["*.parquet", "**/*.parquet"])
        files = sorted(os.path.join(dp, f) for dp, _, fn in os.walk(root) for f in fn if f.endswith(".parquet"))
        opener = open
    frames, names = [], {}
    for f in tqdm(files, desc="parquet files"):
        pf = pq.ParquetFile(opener(f, "rb"))
        names.update(_class_names(pf))
        cols = [c for c in WANT if c in pf.schema_arrow.names]
        df = pf.read(columns=cols).to_pandas()
        df["split"] = _split_of(f)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    cat_col = "main_category" if "main_category" in raw else "class label"
    cat = raw[cat_col]
    if np.issubdtype(cat.dtype, np.integer) and cat_col in names:
        cat = cat.map(dict(enumerate(names[cat_col])))
    out = pd.DataFrame({
        "split": raw["split"],
        "t": [np.asarray(x, float) for x in raw["f0_time"]],
        "f": [np.asarray(x, float) for x in raw["f0_hz"]],
        "conf": [np.asarray(x, float) for x in raw["f0_conf"]] if "f0_conf" in raw else None,
        # missing quality flag -> treated as BAD (conservative); counts are reported by the run
        "f0_ok": raw.get("f0_ok", pd.Series(True, index=raw.index)).fillna(False).astype(bool),
        "main_category": cat.astype(str),
        "whistle_type": raw.get("whistle_type", pd.Series(-1, index=raw.index)),
        "whistle_name": raw.get("whistle_name", pd.Series("", index=raw.index)).astype(str),
        "session": raw.get("name", pd.Series("unknown", index=raw.index)).astype(str),
        "snr_db": raw.get("snr_db", pd.Series(np.nan, index=raw.index)),
        "duration": raw.get("duration", pd.Series(np.nan, index=raw.index)),
    })
    out.to_pickle(cache)
    print(f"[fetch] cached {len(out)} whistles -> {cache}")
    print(out.groupby(["split"]).size().to_string())
    print(out.main_category.value_counts().to_string())
    return out


# ------------------------------------------------------------------ cleaning
def is_signature(cat: str) -> bool:
    """'4SW_Luna' -> signature; anything containing 'NSW' -> non-signature."""
    c = cat.upper()
    return ("SW" in c) and ("NSW" not in c)


def _despike(lf, win=5, thr=0.25, passes=2):
    """Keep-mask removing points > thr octaves from the running median of their neighbours.
    Reflect padding, so a spike at either END is judged against its neighbours, not itself."""
    keep = np.ones(len(lf), bool)
    for _ in range(passes):
        idx = np.where(keep)[0]
        if len(idx) < win:
            break
        x = lf[idx]
        h = win // 2
        xp = np.pad(x, h, mode="reflect")
        med = np.median(np.lib.stride_tricks.sliding_window_view(xp, win), axis=1)
        bad = np.abs(x - med) > thr
        if not bad.any():
            break
        keep[idx[bad]] = False
    return keep


def trace_track(t, lf, w, max_rate=25.0, tol=0.04, max_gap=0.04, jump_cost=2.0, rate_dt_cap=0.015):
    """Best smooth path through a pitch track (dynamic programming).
    Consecutive chosen points must satisfy  0 < dt <= max_gap  and
    |d log2 f| <= tol + max_rate * dt   (max_rate in octaves/s; 25 oct/s ~ 0.125 oct per 5 ms).
    The rate allowance stops growing after rate_dt_cap seconds, so skipping points across a
    dropout cannot buy a large frequency leap. Path score = sum of point weights minus
    jump_cost * |d log2 f| per link (smooth whistles pay ~nothing; leaps and jittery noise
    bands pay a lot). Returns indices of the best-scoring path.
    Scattered junk cannot form a long smooth chain; a faint but continuous whistle can;
    spikes break continuity and are skipped; dropouts up to max_gap are bridged."""
    n = len(t)
    best, prev = w.astype(float).copy(), np.full(n, -1)
    j0 = 0
    for i in range(n):
        while t[i] - t[j0] > max_gap:
            j0 += 1
        if j0 < i:
            js = np.arange(j0, i)
            dt = t[i] - t[js]
            jump = np.abs(lf[i] - lf[js])
            ok = (dt > 0) & (jump <= tol + max_rate * np.minimum(dt, rate_dt_cap))
            if ok.any():
                cand = np.where(ok, best[js] - jump_cost * jump, -np.inf)
                k = int(np.argmax(cand))
                if cand[k] + w[i] > best[i]:
                    best[i], prev[i] = cand[k] + w[i], js[k]
    i = int(np.argmax(best))
    path = []
    while i >= 0:
        path.append(i)
        i = prev[i]
    return np.array(path[::-1])


def clean_track(t, f, conf=None, min_conf=None, max_rate=25.0, tol=0.04, max_gap=0.04,
                conf0=0.08, jump_cost=2.0, min_points=5, **_ignored):
    """Clean one pitch track. Returns (t from 0, log2 f, info) or (None, None, info).
      1. finite, positive frequencies (optional absolute confidence floor min_conf)
      2. trace the best smooth path, scoring each point (confidence - conf0): points below
         conf0 (junk) cost score, so they enter the path only to bridge a real dropout.
         Continuity separates whistle from junk; confidence decides between chains."""
    t, f = np.asarray(t, float), np.asarray(f, float)
    n0 = len(t)
    c = np.asarray(conf, float) if (conf is not None and np.ndim(conf) and len(conf) == len(f)) else np.ones_like(f)
    m = np.isfinite(t) & np.isfinite(f) & (f > 0) & np.isfinite(c)
    if min_conf is not None:
        m &= c >= min_conf
    t, f, c = t[m], f[m], c[m]
    info = {"n_raw": n0, "valid": len(t)}
    if len(t) < min_points:
        return None, None, info
    o = np.argsort(t, kind="stable")
    t, lf, c = t[o], np.log2(f[o]), c[o]
    idx = trace_track(t, lf, c - conf0, max_rate=max_rate, tol=tol, max_gap=max_gap, jump_cost=jump_cost)
    t, lf, c = t[idx], lf[idx], c[idx]
    k = _despike(lf)                                     # final pass: isolated spikes the tracer bridged to
    t, lf, c = t[k], lf[k], c[k]
    info.update(final=len(t), t0=float(t[0]), span=float(t[-1] - t[0]), mean_conf=float(c.mean()))
    if len(t) < min_points or t[-1] - t[0] <= 0:
        return None, None, info
    return t - t[0], lf, info


def clean(df, min_points=5, min_conf=None, require_ok=True, max_rate=25.0, max_gap=0.04, conf0=0.08, jump_cost=2.0, **_ignored):
    """Drop bad tracks and trace the rest; adds t (s, from 0), lf (log2 Hz), kept_point_frac,
    span_frac (traced span / annotated duration) and signature."""
    keep, T, LF, kept_frac, span_frac = [], [], [], [], []
    for i, r in df.iterrows():
        if require_ok and not r.f0_ok:
            continue
        conf = r.conf if ("conf" in df.columns and r.conf is not None) else None
        t, lf, info = clean_track(r.t, r.f, conf, min_conf=min_conf, max_rate=max_rate,
                                  max_gap=max_gap, conf0=conf0, jump_cost=jump_cost, min_points=min_points)
        if t is None:
            continue
        keep.append(i)
        T.append(t)
        LF.append(lf)
        kept_frac.append(info["final"] / max(info["n_raw"], 1))
        dur = r.duration if ("duration" in df.columns and np.isfinite(r.duration) and r.duration > 0) else np.nan
        span_frac.append(info["span"] / dur if np.isfinite(dur) else np.nan)
    out = df.loc[keep].reset_index(drop=True).copy()
    out["t"] = T
    out["lf"] = LF
    out["kept_point_frac"] = kept_frac
    out["span_frac"] = span_frac
    out["signature"] = out.main_category.map(is_signature)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--cache", default=CACHE)
    a = ap.parse_args()
    if a.inspect:
        inspect()
    if a.fetch:
        fetch(a.cache)