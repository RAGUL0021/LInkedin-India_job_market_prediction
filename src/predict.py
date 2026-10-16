"""Predict applicants for new jobs.  Usage:

    python src/predict.py sample/sample_input.csv              # -> sample/sample_input_predictions.csv
    python src/predict.py my_jobs.csv -o results.csv

Input CSV: either the friendly template columns (see sample/sample_input.csv)
or the original raw scrape columns (job, location, no_of_employ, ...).
Missing columns are fine - the model handles blanks.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import build_features, standardize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "best_model.joblib")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found - run `python src/train.py` first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def level_of(n):
    return "Low" if n < 15 else ("Medium" if n < 70 else "High")


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    b = get_bundle()
    feats = build_features(df)
    log_pred = b["model"].predict(feats)
    lo_q, hi_q = b["interval"]
    pred = np.clip(np.expm1(log_pred), 1, 200)
    low = np.clip(np.expm1(log_pred + lo_q), 1, 200)
    high = np.clip(np.expm1(log_pred + hi_q), 1, 200)
    out = df.copy()
    out["predicted_applicants"] = np.round(pred).astype(int)
    out["range_low_80pct"] = np.round(low).astype(int)
    out["range_high_80pct"] = np.round(high).astype(int)
    out["demand_level"] = [level_of(p) for p in pred]
    out["role_family_detected"] = feats["role_family"].values
    out["city_detected"] = feats["city"].values
    return out


def main():
    ap = argparse.ArgumentParser(description="Batch-predict job applicants from a CSV")
    ap.add_argument("input_csv")
    ap.add_argument("-o", "--output", help="output CSV path")
    a = ap.parse_args()
    df = pd.read_csv(a.input_csv)
    res = predict_df(df)
    out = a.output or os.path.splitext(a.input_csv)[0] + "_predictions.csv"
    res.to_csv(out, index=False)
    print(f"Saved {len(res)} predictions -> {out}")
    print(res[[c for c in ["job_title", "job", "predicted_applicants", "range_low_80pct",
                           "range_high_80pct", "demand_level"] if c in res.columns]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
