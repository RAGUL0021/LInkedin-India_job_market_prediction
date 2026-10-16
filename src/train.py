"""Train + compare several ML models, keep the best, and export dashboard data.

Run:  python src/train.py
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import CAT_COLS, NUM_COLS, TEXT_COL, load_training_data  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "job_market.csv")
MODELS = os.path.join(ROOT, "models")
SEED = 42


def make_preprocessor(dense=False):
    cat = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=8,
        sparse_output=not dense
    )

    num = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler())
    ])

    txt = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=150
    )

    return ColumnTransformer(
        [
            ("cat", cat, CAT_COLS),
            ("num", num, NUM_COLS),
            ("txt", txt, TEXT_COL)
        ],
        sparse_threshold=0.0 if dense else 0.3
    )


def candidates():
    return {
        "Ridge Regression": Ridge(alpha=5.0),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=SEED
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.8,
            random_state=SEED
        ),
    }





def demand_level(n):
    """Business-friendly label for predicted applicants."""
    return np.where(n < 15, "Low", np.where(n < 70, "Medium", "High"))


def metrics(y_true_log, y_pred_log):
    yt, yp = np.expm1(y_true_log), np.clip(np.expm1(y_pred_log), 0, 200)
    return {"r2_log": float(r2_score(y_true_log, y_pred_log)),
            "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
            "mae_applicants": float(mean_absolute_error(yt, yp)),
            "rmse_applicants": float(np.sqrt(mean_squared_error(yt, yp))),
            "r2_applicants": float(r2_score(yt, yp)),
            "level_accuracy": float((demand_level(yt) == demand_level(yp)).mean())}


def main():
    t0 = time.time()
    X, y, clean = load_training_data(DATA)
    ylog = np.log1p(y)
    print(f"Training rows: {len(X)}  | applicants mean={y.mean():.1f} median={y.median():.0f}")

    Xtr, Xte, ytr, yte = train_test_split(X, ylog, test_size=0.2, random_state=SEED)
    kf = KFold(5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, est in candidates().items():
        pipe = Pipeline([
    ("prep", make_preprocessor(dense=False)),
    ("model", est)
])
        
        cv_pred = cross_val_predict(pipe, Xtr, ytr, cv=kf)
        cv_m = metrics(ytr.values, cv_pred)
        pipe.fit(Xtr, ytr)
        te_m = metrics(yte.values, pipe.predict(Xte))
        results[name] = {"cv": cv_m, "test": te_m}
        fitted[name] = (pipe, cv_pred)
        print(f"{name:22s} CV R2(log)={cv_m['r2_log']:.3f}  test R2(log)={te_m['r2_log']:.3f}  "
              f"test MAE={te_m['mae_applicants']:.1f} applicants  level acc={te_m['level_accuracy']:.2f}")

    best = max(results, key=lambda k: results[k]["cv"]["r2_log"])
    print(f"\nBEST MODEL (by 5-fold CV R2): {best}")

    # Refit the winner on ALL data for production use; keep honest hold-out metrics from above.
    best_est = candidates()[best]
    final = Pipeline([
    ("prep", make_preprocessor(dense=False)),
    ("model", best_est)
])
    final.fit(X, ylog)

    # Prediction interval from out-of-fold residuals (80% band, in log space)
    resid = ytr.values - fitted[best][1]
    lo_q, hi_q = np.quantile(resid, [0.10, 0.90])

    # Permutation importance on hold-out (grouped by raw input feature)
    pipe_best = fitted[best][0]
    pi = permutation_importance(pipe_best, Xte, yte, n_repeats=8, random_state=SEED, scoring="r2")
    imp = sorted(zip(X.columns, pi.importances_mean), key=lambda t: -t[1])
    imp = [{"feature": k, "importance": float(max(v, 0))} for k, v in imp]

    # Actual vs predicted (hold-out)
    te_pred = np.clip(np.expm1(pipe_best.predict(Xte)), 0, 200)
    scatter = [{"actual": float(a), "pred": float(p)} for a, p in zip(np.expm1(yte.values)[:400], te_pred[:400])]

    # Categories the UI should offer
    options = {c: [str(v) for v in X[c].value_counts().index[:40]] for c in
               ["role_family", "city", "work_type", "employment_type", "seniority", "company_size", "industry"]}

    joblib.dump({"model": final, "best_name": best, "interval": [float(lo_q), float(hi_q)],
                 "feature_cols": list(X.columns)}, os.path.join(MODELS, "best_model.joblib"))

    report = {"best_model": best, "n_train_rows": int(len(X)), "results": results,
              "importance": imp[:12], "scatter": scatter, "options": options,
              "interval_log": [float(lo_q), float(hi_q)]}
    with open(os.path.join(MODELS, "model_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    export_dashboard_data(clean)
    print(f"Done in {time.time() - t0:.0f}s -> models/best_model.joblib, models/model_report.json, models/dashboard_data.json")


def export_dashboard_data(clean):
    """Pre-aggregate the EDA numbers used by the dashboard charts."""
    raw = pd.read_csv(DATA)
    from preprocess import build_features
    F = build_features(raw)
    F["applicants"] = pd.to_numeric(raw["no_of_application"].astype(str).str.replace(",", ""), errors="coerce")

    def top(series, n=10):
        vc = series.value_counts().head(n)
        return {"labels": [str(i) for i in vc.index], "values": [int(v) for v in vc.values]}

    def avg_by(col, n=10, minc=15):
        g = F.dropna(subset=["applicants"]).groupby(col)["applicants"].agg(["mean", "count"])
        g = g[g["count"] >= minc].sort_values("mean", ascending=False).head(n)
        return {"labels": [str(i) for i in g.index], "values": [round(float(v), 1) for v in g["mean"]],
                "counts": [int(c) for c in g["count"]]}

    ok = F.dropna(subset=["applicants", "posted_hours"]).copy()
    ok["bucket"] = pd.cut(ok["posted_hours"], [-1, 3, 6, 12, 24, 48, 96, 1000],
                          labels=["<3h", "3-6h", "6-12h", "12-24h", "1-2d", "2-4d", "4d+"])
    tb = ok.groupby("bucket", observed=True)["applicants"].mean()

    cities = F[~F.city.isin(["India (unspecified)", "Unknown"])]
    data = {
        "kpi": {"total_jobs": int(len(raw)), "with_applicant_count": int(F["applicants"].notna().sum()),
                "avg_applicants": round(float(F["applicants"].mean()), 1),
                "median_applicants": float(F["applicants"].median()),
                "pct_capped_200": round(float((F["applicants"] >= 200).mean() / F["applicants"].notna().mean() * 100), 1),
                "remote_share": round(float((F.work_type == "Remote").mean() * 100), 1),
                "companies": int(F.company.nunique())},
        "top_roles": top(F.role_family, 12),
        "top_cities": top(cities.city, 10),
        "work_type": top(F.work_type[F.work_type != "Unknown"], 4),
        "seniority": top(F.seniority, 6),
        "top_companies": top(F.company[F.company != "Unknown"], 10),
        "top_industries": top(F.industry[F.industry != "Unknown"], 8),
        "avg_by_role": avg_by("role_family", 12, 20),
        "avg_by_work": avg_by("work_type", 4, 20),
        "avg_by_city": avg_by("city", 10, 40),
        "avg_by_seniority": avg_by("seniority", 6, 10),
        "avg_by_age": {"labels": [str(i) for i in tb.index], "values": [round(float(v), 1) for v in tb.values]},
    }
    with open(os.path.join(MODELS, "dashboard_data.json"), "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
