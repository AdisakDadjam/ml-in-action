"""Run from the repo root:  python -m pytest app/tests -q"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model_utils import FEATURES, explain, load_metadata, predict, to_frame  # noqa: E402

HIGH = {"Pregnancies": 7, "Glucose": 194, "BloodPressure": 68, "SkinThickness": 28, "Insulin": None,
        "BMI": 35.9, "DiabetesPedigreeFunction": 0.745, "Age": 41}
LOW = {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29, "Insulin": None,
       "BMI": 22.0, "DiabetesPedigreeFunction": 0.2, "Age": 23}


def test_metadata_matches_features():
    meta = load_metadata()
    assert meta["features"] == FEATURES
    assert 0 < meta["threshold"] < 1


def test_sklearn_version_matches_training():
    import sklearn
    assert sklearn.__version__ == load_metadata()["scikit_learn"], "retrain or pin scikit-learn in app/requirements.txt"


def test_zero_and_none_become_missing():
    row = to_frame({**LOW, "Insulin": 0, "SkinThickness": None})
    assert math.isnan(row.loc[0, "Insulin"]) and math.isnan(row.loc[0, "SkinThickness"])
    assert row.loc[0, "Pregnancies"] == 1


def test_missing_key_raises():
    with pytest.raises(ValueError):
        predict({k: v for k, v in LOW.items() if k != "Glucose"})


def test_high_vs_low_risk():
    hi, lo = predict(HIGH), predict(LOW)
    assert hi["refer"] and not lo["refer"]
    assert hi["probability"] > lo["probability"]


def test_shap_adds_up_to_prediction():
    e = explain(HIGH)
    assert np.isclose(e.base_values + e.values.sum(), predict(HIGH)["probability"])
