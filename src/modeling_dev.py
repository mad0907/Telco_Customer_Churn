"""
Development/verification script for feature engineering + modeling.
Not a deliverable. Run with: source venv/bin/activate && python src/modeling_dev.py
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from pipeline_dev import clean, load_raw

warnings.filterwarnings("ignore")

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive num_add_on_services, avg_monthly_spend, and tenure_bucket from
    the cleaned frame. See notebook Section 6 for the justification behind
    each derived feature.
    """
    out = df.copy()

    add_on_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    # These columns encode "No internet service" as a third category.
    # For a service count, that's equivalent to "No" -- collapse it so the
    # derived count reflects actual add-ons subscribed, not an artifact of
    # the internet-service branch in the source system.
    out["num_add_on_services"] = (
        out[add_on_cols].apply(lambda c: (c == "Yes").astype(int)).sum(axis=1)
    )

    # avg monthly spend implied by total billing vs. observed tenure;
    # guards divide-by-zero for tenure==0 by falling back to MonthlyCharges
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = out["TotalCharges"] / out["tenure"].replace(0, np.nan)
    out["avg_monthly_spend"] = implied.fillna(out["MonthlyCharges"])

    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    )

    return out


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    """
    Build a ColumnTransformer: one-hot encode categoricals, and either
    StandardScaler or passthrough numeric columns depending on
    scale_numeric (True for Logistic Regression, False for tree ensembles).
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transform = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transform, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical_cols),
        ]
    )


def main():
    """Train and evaluate all three models once, printing headline metrics for each."""
    raw = load_raw(DATA_PATH)
    clean_df = clean(raw)
    feat_df = engineer_features(clean_df)

    y = feat_df["Churn"]
    X = feat_df.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print("train shape:", X_train.shape, "test shape:", X_test.shape)
    print("train churn rate:", y_train.mean(), "test churn rate:", y_test.mean())

    results = {}

    # baseline: logistic regression, scaled numerics, class_weight balanced
    lr_pre = build_preprocessor(X_train, scale_numeric=True)
    lr_pipe = Pipeline([
        ("pre", lr_pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    lr_pipe.fit(X_train, y_train)
    proba = lr_pipe.predict_proba(X_test)[:, 1]
    pred = lr_pipe.predict(X_test)
    results["logreg"] = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }

    # bagging: random forest, no scaling needed
    rf_pre = build_preprocessor(X_train, scale_numeric=False)
    rf_pipe = Pipeline([
        ("pre", rf_pre),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    rf_param_dist = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [4, 6, 8, 12, None],
        "clf__min_samples_leaf": [1, 2, 4, 8],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=10, scoring="roc_auc", cv=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    proba = rf_best.predict_proba(X_test)[:, 1]
    pred = rf_best.predict(X_test)
    results["random_forest"] = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "best_params": rf_search.best_params_,
    }

    # boosting: xgboost, no scaling needed
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pre = build_preprocessor(X_train, scale_numeric=False)
    xgb_pipe = Pipeline([
        ("pre", xgb_pre),
        ("clf", XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE,
            scale_pos_weight=scale_pos_weight, n_jobs=-1,
        )),
    ])
    xgb_param_dist = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipe, xgb_param_dist, n_iter=10, scoring="roc_auc", cv=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    proba = xgb_best.predict_proba(X_test)[:, 1]
    pred = xgb_best.predict(X_test)
    results["xgboost"] = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "best_params": xgb_search.best_params_,
    }

    for name, metrics in results.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
