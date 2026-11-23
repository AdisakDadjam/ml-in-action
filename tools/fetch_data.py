"""Download the two course datasets and write them with column names to labs/data/ (run once by the instructor).

    python tools/fetch_data.py

- diabetes.csv  Pima Indians Diabetes (Smith et al., 1988; originally NIDDK) — 768 rows, target `Outcome`
- heart.csv     UCI Heart Disease, Cleveland subset (Janosi et al., 1989, CC BY 4.0) — 303 rows, target `target`
"""
from pathlib import Path
import io
import urllib.request

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "labs" / "data"
PIMA = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
CLEVELAND = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age", "Outcome"]
    pima = pd.read_csv(io.StringIO(get(PIMA)), header=None, names=cols)
    assert pima.shape == (768, 9), pima.shape
    pima.to_csv(OUT / "diabetes.csv", index=False)

    hcols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
             "slope", "ca", "thal", "num"]
    heart = pd.read_csv(io.StringIO(get(CLEVELAND)), header=None, names=hcols, na_values="?")
    heart["target"] = (heart.pop("num") > 0).astype(int)      # 0 = no disease, 1–4 = disease
    for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        heart[c] = heart[c].astype("Int64")
    assert heart.shape == (303, 14), heart.shape
    heart.to_csv(OUT / "heart.csv", index=False)
    print("wrote", OUT / "diabetes.csv", pima.shape, "and", OUT / "heart.csv", heart.shape)


if __name__ == "__main__":
    main()
