"""Train the screening model and write model/diabetes_rf.joblib + model/metadata.json.

    python app/train.py                      # from the repo root
    python app/train.py --data path/to.csv

The same steps as Lab 1–2, as a script that anyone can re-run and get the same model:
clean (0 → NaN) → stratified 80/20 split → imputer + random forest (hyperparameters from GridSearchCV in Lab 2)
→ threshold from out-of-fold probabilities on train (recall ≥ 0.80) → one evaluation on test → save.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline

from model_utils import FEATURES, META_PATH, MODEL_PATH, ZERO_AS_MISSING

ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
TARGET_RECALL = 0.80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "labs" / "data" / "diabetes.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)
    X, y = df[FEATURES], df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, max_features="sqrt",
                               random_state=RANDOM_STATE, n_jobs=-1),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    threshold = float(thr[rec[:-1] >= TARGET_RECALL].max())

    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= threshold).astype(int)
    metrics = {"roc_auc": roc_auc_score(y_test, p), "recall": recall_score(y_test, pred),
               "precision": precision_score(y_test, pred)}

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    meta = {
        "model": "random_forest", "features": FEATURES, "threshold": round(threshold, 4),
        "target_recall": TARGET_RECALL, "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "n_train": int(len(X_train)), "n_test": int(len(X_test)), "trained_on": date.today().isoformat(),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"saved {MODEL_PATH.relative_to(ROOT)} ({MODEL_PATH.stat().st_size / 1e6:.2f} MB)")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
