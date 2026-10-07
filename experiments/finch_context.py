"""
finch_context.py - are gap-PH's splits of single hand labels real song structure?

For each bird: rerun gap-PH on a fresh 1500-syllable sample, cutting at the bird's threshold
from finch_gap_results.csv. A label is SPLIT if >= 2 groups each hold >= MIN_GROUP of its
syllables and are >= 80% that label. For each split, test group vs context (chi-square,
Cramer's V, Bonferroni over all tests run):
  run_pos   1st / 2nd / 3rd / 4th+ syllable in a run of the same label     (pre-registered)
  prev      previous syllable label, or START                              (pre-registered)
  pos_bin   early / middle / late third of the bout                        (pre-registered)
  recording which quarter of the bird's wav files (control: drift / recording conditions)

PREDICTIONS (written before running):
  label-0 splits: run_pos or pos_bin predicts group with Bonferroni p < 0.05 and V > 0.3.
  other splits: no prediction.

Run:  python finch_context.py      (a few minutes; needs finch_gap_results.csv)
"""
import argparse
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

from finch_gap import DEFAULT_ROOT, MAX_N, syllable_image
from whistle_gap import MIN_GROUP, gap_tree, groups_from_cut

PURITY = 0.8
VARS = ["run_pos", "prev", "pos_bin", "recording"]


def load_with_context(bird_dir):
    root = ET.parse(bird_dir / "Annotation.xml").getroot()
    rows = []
    for seq in root.iter("Sequence"):
        wav = seq.findtext("WaveFileName")
        s_pos = int(seq.findtext("Position"))
        notes = [(int(n.findtext("Position")), int(n.findtext("Length")), n.findtext("Label")) for n in seq.iter("Note")]
        labels = [lab for _, _, lab in notes]
        m = len(notes)
        for k, (p, length, lab) in enumerate(notes):
            run = 0
            while k - run - 1 >= 0 and labels[k - run - 1] == lab:
                run += 1
            rel = k / (m - 1) if m > 1 else 0.0
            rows.append(dict(wav=wav, start=s_pos + p, length=length, label=lab,
                             prev=labels[k - 1] if k > 0 else "START",
                             run_pos=["1st", "2nd", "3rd", "4th+"][min(run, 3)],
                             pos_bin=["early", "middle", "late"][min(int(rel * 3), 2)],
                             wav_num=int("".join(ch for ch in Path(wav).stem if ch.isdigit()) or 0)))
    df = pd.DataFrame(rows)
    df["recording"] = pd.qcut(df.wav_num.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    return df


def cramers_v(chi2, n, shape):
    k = min(shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else float("nan")


def analyse_bird(bird_dir, thr, rng):
    df = load_with_context(bird_dir)
    if len(df) > MAX_N:
        df = df.iloc[rng.choice(len(df), MAX_N, replace=False)].reset_index(drop=True)
    wav_cache = {}
    segs = []
    for wav, start, length in zip(df.wav, df.start, df.length):
        if wav not in wav_cache:
            fs, x = wavfile.read(bird_dir / "Wave" / wav)
            wav_cache[wav] = (fs, x if x.ndim == 1 else x[:, 0])
        fs, x = wav_cache[wav]
        segs.append(x[start:start + length])
    dur_max = float(np.quantile(df.length, 0.99)) / fs
    X = np.stack([syllable_image(s, fs, dur_max) for s in segs])
    Z = PCA(n_components=10, random_state=0).fit_transform(X)
    Z = Z / np.sqrt((Z ** 2).mean())
    gaps, edges, T = gap_tree(Z)
    df["group"] = groups_from_cut(T, gaps, edges, thr, len(Z))
    ct = pd.crosstab(df.group, df.label)
    ct = ct[ct.index >= 0]
    tests = []
    for lab in ct.columns:
        pure = [g for g in ct.index if ct.loc[g, lab] >= MIN_GROUP and ct.loc[g, lab] / ct.loc[g].sum() >= PURITY]
        if len(pure) < 2:
            continue
        sub = df[(df.label == lab) & (df.group.isin(pure))]
        tqdm.write(f"\n--- {bird_dir.name} label {lab}: split into groups "
                   + ", ".join(f"{g} (n={int((sub.group == g).sum())})" for g in pure))
        for var in VARS:
            tab = pd.crosstab(sub.group, sub[var])
            tab = tab.loc[:, tab.sum(axis=0) >= 5]
            if tab.shape[1] < 2:
                tqdm.write(f"  {var}: only one usable category, skipped")
                continue
            chi2, p, dof, _ = chi2_contingency(tab)
            v = cramers_v(chi2, int(tab.values.sum()), tab.shape)
            tests.append(dict(bird=bird_dir.name, label=lab, groups=str(pure), var=var, n=int(tab.values.sum()),
                              chi2=float(chi2), dof=int(dof), p=float(p), V=v))
            tqdm.write(f"  {var}: chi2={chi2:.1f} dof={dof} p={p:.2e} V={v:.2f}")
            if var != "prev" or tab.shape[1] <= 8:
                tqdm.write("    " + tab.to_string().replace("\n", "\n    "))
    return tests


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
    tests = []
    for bird in tqdm(birds, desc="birds"):
        tests += analyse_bird(bird, float(thr[bird.name]), rng)
    res = pd.DataFrame(tests)
    if res.empty:
        print("no split labels found")
        return
    m = len(res)
    res["p_bonf"] = np.minimum(res.p * m, 1.0)
    res.to_csv("finch_context_results.csv", index=False)
    print(f"\n{m} tests in total (Bonferroni factor {m})")
    print(res[["bird", "label", "var", "n", "V", "p", "p_bonf"]].round(4).to_string(index=False))
    print("\nstrongest context per split (Bonferroni p < 0.05):")
    sig = res[res.p_bonf < 0.05].sort_values("V", ascending=False).groupby(["bird", "label"]).head(1)
    print(sig[["bird", "label", "var", "V", "p_bonf"]].round(4).to_string(index=False) if len(sig) else "  none")
    lab0 = res[res.label == "0"]
    ok = lab0[(lab0["var"].isin(["run_pos", "pos_bin"])) & (lab0.p_bonf < 0.05) & (lab0.V > 0.3)]
    print(f"\nprediction (label-0 splits explained by run_pos or pos_bin, V > 0.3): "
          f"{ok[['bird']].drop_duplicates().shape[0]} of {lab0[['bird']].drop_duplicates().shape[0]} label-0 splits")
    print(f"wrote finch_context_results.csv   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()