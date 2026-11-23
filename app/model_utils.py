"""Load the trained pipeline and turn one person's inputs into a risk, a decision and a SHAP explanation.

Shared by app.py (the web UI) and tests/ — keep all model logic here, not in the UI.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "diabetes_rf.joblib"
META_PATH = MODEL_DIR / "metadata.json"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"]
# 0 in these columns means "not measured" in the training data → the pipeline's imputer fills them
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@lru_cache(maxsize=1)
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def to_frame(person: dict) -> pd.DataFrame:
    """One row in training column order; None / 0 (for ZERO_AS_MISSING) become NaN, as in training."""
    missing = set(FEATURES) - set(person)
    if missing:
        raise ValueError(f"missing inputs: {sorted(missing)}")
    row = {f: (np.nan if person[f] is None else float(person[f])) for f in FEATURES}
    for f in ZERO_AS_MISSING:
        if row[f] == 0:
            row[f] = np.nan
    return pd.DataFrame([row], columns=FEATURES)


def predict(person: dict) -> dict:
    """Return probability of diabetes, the screening decision at the saved threshold, and the threshold."""
    model, meta = load_model(), load_metadata()
    p = float(model.predict_proba(to_frame(person))[0, 1])
    t = float(meta["threshold"])
    return {"probability": p, "threshold": t, "refer": p >= t}


def explain(person: dict):
    """SHAP explanation (class 1 = diabetes) of one person, ready for shap.plots.waterfall."""
    model = load_model()
    X = to_frame(person)
    X_imp = pd.DataFrame(model[:-1].transform(X), columns=FEATURES)
    explanation = _explainer()(X_imp)
    return explanation[0, :, 1]


@lru_cache(maxsize=1)
def _explainer():
    import shap
    return shap.TreeExplainer(load_model()[-1])
