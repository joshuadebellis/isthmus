r"""ow_identity_check.py - do the gate's 32x32 dolphin spectrograms carry label information?

--set sw  : signature whistles, label = which dolphin (as before)
--set nsw : non-signature whistles, label = NSW category
Same images as ow_audio_gap.py. Logistic regression and 5-NN, cross-validated grouped by recording
session (honest) and plain stratified 5-fold (upper bound). Balanced accuracy; chance = 1/#labels.
Labels with < MIN_PER_LABEL clips are dropped. Balanced accuracy by SNR tertile is also reported.

DECISION RULES (written before running), session-grouped balanced accuracy on pixels:
  sw : <= 0.25 little identity; >= 0.50 'distinguishable but not discrete'; else weak.
  nsw: <= 0.45 little category information; >= 0.65 'distinguishable but not discrete'; else weak.

Run:  python ow_identity_check.py --set sw
      python ow_identity_check.py --set nsw
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

from openwhistle_data import is_signature
from ow_audio_gap import spec_image

MIN_PER_LABEL = 20
RULES = {"sw": (0.25, 0.50), "nsw": (0.45, 0.65)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="openwhistle_audio.pkl")
    ap.add_argument("--set", choices=["sw", "nsw"], default="sw")
    a = ap.parse_args()
    t0 = time.time()
    df = pd.read_pickle(a.cache)
    sig = df.main_category.map(is_signature)
    if a.set == "sw":
        sub = df[sig]
    else:
        sub = df[~sig & df.main_category.str.upper().str.contains("NSW")]
    counts = sub.main_category.value_counts()
    dropped = counts[counts < MIN_PER_LABEL]
    sub = sub[sub.main_category.isin(counts[counts >= MIN_PER_LABEL].index)].reset_index(drop=True)
    y = sub.main_category.to_numpy()
    groups = sub["name"].astype(str).to_numpy()
    k = len(np.unique(y))
    print(f"set={a.set}  clips: {len(sub)}  labels: {k}  sessions: {len(np.unique(groups))}  chance = {1 / k:.3f}")
    print(sub.main_category.value_counts().to_string())
    if len(dropped):
        print("dropped (fewer than", MIN_PER_LABEL, "clips):", dropped.to_dict())
    if k < 2:
        print("fewer than 2 labels left - nothing to classify")
        return

    dur_max = 1000.0 * float(np.quantile(sub.clip_s, 0.99))
    X = np.stack([spec_image(b, dur_max) for b in tqdm(sub.wav, desc="spectrograms")])
    Z = PCA(n_components=10, random_state=0).fit_transform(X)

    try:
        cv_group = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, y, groups))
        group_note = "grouped by session"
    except ValueError as e:
        cv_group = None
        group_note = f"session-grouped CV not possible ({e})"
    cv_plain = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X, y))

    models = {
        "logreg pixels": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)), X),
        "logreg PCA-10": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")), Z),
        "5-NN PCA-10": (make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)), Z),
    }
    rows, grouped_pred = [], None
    for name, (model, feats) in tqdm(models.items(), desc="classifiers"):
        for cv_name, cv in [("session-grouped", cv_group), ("plain 5-fold", cv_plain)]:
            if cv is None:
                continue
            pred = cross_val_predict(model, feats, y, cv=cv)
            rows.append(dict(model=name, cv=cv_name, balanced_acc=round(balanced_accuracy_score(y, pred), 3)))
            if name == "logreg pixels" and cv_name == "session-grouped":
                grouped_pred = pred
    res = pd.DataFrame(rows)
    print(f"\n{group_note}")
    print(res.pivot(index="model", columns="cv", values="balanced_acc").to_string())

    if grouped_pred is not None and "snr_db" in sub:
        snr = pd.to_numeric(sub.snr_db, errors="coerce")
        tert = pd.qcut(snr, 3, labels=["low SNR", "mid SNR", "high SNR"])
        print("\nlogreg pixels, session-grouped, by SNR tertile:")
        for t in ["low SNR", "mid SNR", "high SNR"]:
            m = (tert == t).to_numpy()
            print(f"  {t:9s} n={m.sum():5d}  SNR {snr[m].min():.1f}-{snr[m].max():.1f} dB  "
                  f"balanced acc={balanced_accuracy_score(y[m], grouped_pred[m]):.3f}")

    lo, hi = RULES[a.set]
    key = res[(res.model == "logreg pixels") & (res.cv == ("session-grouped" if cv_group is not None else "plain 5-fold"))]
    ba = float(key.balanced_acc.iloc[0])
    verdict = ("little label information -> gate result uninformative" if ba <= lo else
               "labels visible but no gaps -> 'distinguishable but not discrete'" if ba >= hi else "weak label signal")
    print(f"\ndecision ({key.cv.iloc[0]} logreg on pixels = {ba:.3f}; rule {lo}/{hi}): {verdict}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()