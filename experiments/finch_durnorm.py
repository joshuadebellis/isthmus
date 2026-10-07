"""
finch_durnorm.py - do gap-PH's label splits survive when duration is removed?

Same syllable sample and original groups as finch_context.py / finch_split_spectra.py.
New representation: every syllable stretched to the full 32-column width (no duration
information), same frequency axis and dynamic range. gap-PH rerun with each bird's threshold
from finch_gap_results.csv (valid while d_hat stays near the original 4.5-5.7; printed).

A split SURVIVES if, under the duration-normalized grouping, the label still splits into
>= 2 pure groups (>= 80% that label, >= MIN_GROUP of its syllables) AND the split agrees with
the original split on the same syllables (ARI >= 0.5).

PREDICTIONS (written before running):
  survive: Bird4 lab 0, Bird6 lab 0, Bird8 lab 0, Bird9 lab 1   (spectral 'colour' variants)
           Bird5 lab 2, Bird2 lab 1                             (possibly different syllables)
  vanish:  Bird0 lab 0, Bird0 lab 6, Bird5 lab 1, Bird9 lab 0   (length / tempo variants)

Run:  python finch_durnorm.py      (under a minute)
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

from finch_context import PURITY
from finch_gap import DEFAULT_ROOT, DR, FMAX, FMIN, GRID
from finch_split_spectra import bird_groups
from gap_vs_hopkins import lb_dim
from whistle_gap import MIN_GROUP, gap_tree, groups_from_cut

PREDICTED_SURVIVE = {("Bird4", "0"), ("Bird6", "0"), ("Bird8", "0"), ("Bird9", "1"), ("Bird5", "2"), ("Bird2", "1")}
PREDICTED_VANISH = {("Bird0", "0"), ("Bird0", "6"), ("Bird5", "1"), ("Bird9", "0")}


def syllable_image_fullwidth(seg, fs):
    """Like finch_gap.syllable_image, but every syllable fills all 32 columns (duration removed)."""
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
    x = (np.arange(GRID) + 0.5) / GRID * (S.shape[1] - 1)
    frames = np.arange(S.shape[1])
    return np.stack([np.interp(x, frames, S[i]) for i in range(GRID)]).ravel()


def pure_groups(df, col, lab):
    ct = pd.crosstab(df[col], df.label)
    ct = ct[ct.index >= 0]
    if lab not in ct.columns:
        return []
    return [g for g in ct.index if ct.loc[g, lab] >= MIN_GROUP and ct.loc[g, lab] / ct.loc[g].sum() >= PURITY]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--results", default="finch_gap_results.csv")
    a = ap.parse_args()
    t0 = time.time()
    res0 = pd.read_csv(a.results).set_index("bird")
    root = Path(a.root)
    birds = sorted([d for d in root.iterdir() if d.is_dir() and d.name in res0.index],
                   key=lambda d: int("".join(ch for ch in d.name if ch.isdigit()) or 0))
    rng = np.random.default_rng(1)
    rows = []
    for bird in tqdm(birds, desc="birds"):
        thr = float(res0.loc[bird.name, "thr"])
        df, audio = bird_groups(bird, thr, rng)
        X = np.stack([syllable_image_fullwidth(audio[w][1][s:s + n], audio[w][0])
                      for w, s, n in zip(df.wav, df.start, df.length)])
        Z = PCA(n_components=10, random_state=0).fit_transform(X)
        Z = Z / np.sqrt((Z ** 2).mean())
        gaps, edges, T = gap_tree(Z)
        df["group_dn"] = groups_from_cut(T, gaps, edges, thr, len(Z))
        d_hat = lb_dim(Z)
        ok = df.group_dn >= 0
        ari_labels = adjusted_rand_score(df.label[ok], df.group_dn[ok])
        n_types = 1 + int((gaps > thr).sum())
        tqdm.write(f"\n{bird.name}: duration-normalized d_hat={d_hat:.2f} (original {res0.loc[bird.name, 'd_hat']:.2f})  "
                   f"types={n_types} (original {int(res0.loc[bird.name, 'n_types'])})  ARI vs labels={ari_labels:.3f} "
                   f"(original {res0.loc[bird.name, 'ari']:.3f})")
        for lab in sorted(df.label.unique()):
            orig = pure_groups(df, "group", lab)
            if len(orig) < 2:
                continue
            sub = df[(df.label == lab) & df.group.isin(orig)]
            new = pure_groups(df, "group_dn", lab)
            in_new = sub.group_dn.isin(new)
            agree = adjusted_rand_score(sub.group[in_new], sub.group_dn[in_new]) if in_new.sum() > 1 else 0.0
            survives = len(new) >= 2 and agree >= 0.5
            key = (bird.name, lab)
            pred = "survive" if key in PREDICTED_SURVIVE else ("vanish" if key in PREDICTED_VANISH else "none")
            rows.append(dict(bird=bird.name, label=lab, n=len(sub), orig_groups=len(orig), new_pure_groups=len(new),
                             agreement_ARI=round(float(agree), 3), survives=survives, predicted=pred,
                             prediction_correct=(pred == "survive") == survives if pred != "none" else None))
            tqdm.write(f"  label {lab}: original split into {len(orig)} groups; duration-normalized: {len(new)} pure groups, "
                       f"agreement ARI={agree:.3f} -> {'SURVIVES' if survives else 'vanishes'} (predicted {pred})")
            tqdm.write("    " + pd.crosstab(sub.group.rename("original"), sub.group_dn.rename("dur-normalized"))
                       .to_string().replace("\n", "\n    "))
    out = pd.DataFrame(rows)
    out.to_csv("finch_durnorm_results.csv", index=False)
    print("\nsummary")
    print(out.to_string(index=False))
    scored = out[out.predicted != "none"]
    print(f"\npredictions correct: {int(scored.prediction_correct.sum())} of {len(scored)}")
    print(f"wrote finch_durnorm_results.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()