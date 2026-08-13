"""
Standalone feature engineering module, used by the Streamlit prototype
(app.py) at inference time.

This intentionally mirrors the clean() / engineer_features() logic defined
inline in notebooks/telco_churn_analysis.ipynb rather than importing from
the notebook. Notebooks are kept self-contained deliverables here so they
can be read top to bottom without external file dependencies; this module
exists so a serving surface (app.py, and by extension deployment/score.py
in a real deployment) has a single importable source for the same
transformations, rather than re-deriving them independently.

This duplication between notebook and module is exactly the train/serve
skew risk the architecture document's Azure ML Managed Feature Store
recommendation is meant to eliminate in a real deployment -- here, at
take-home-exercise scale, keeping the two in careful sync by hand is an
accepted, explicitly noted trade-off.
"""
import numpy as np
import pandas as pd

ADD_ON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning rules used in notebook Section 4:
    - TotalCharges: blank strings (all tenure == 0) become 0.0, a
      structural zero rather than a missing value.
    - SeniorCitizen: normalized to Yes/No for encoding consistency.
    - customerID: dropped if present (identifier, not a feature).
    - Churn: mapped to 0/1 if present.
    """
    out = df.copy()

    if out["TotalCharges"].dtype == object:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    zero_tenure_mask = out["tenure"] == 0
    out.loc[zero_tenure_mask & out["TotalCharges"].isnull(), "TotalCharges"] = 0.0

    if out["SeniorCitizen"].dtype != object:
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})

    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    if "Churn" in out.columns and out["Churn"].dtype == object:
        out["Churn"] = out["Churn"].map({"Yes": 1, "No": 0})

    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same derived features used in notebook Section 6:
    num_add_on_services, avg_monthly_spend, tenure_bucket.
    """
    out = df.copy()

    out["num_add_on_services"] = (
        out[ADD_ON_COLUMNS].apply(lambda c: (c == "Yes").astype(int)).sum(axis=1)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        implied_spend = out["TotalCharges"] / out["tenure"].replace(0, np.nan)
    out["avg_monthly_spend"] = implied_spend.fillna(out["MonthlyCharges"])

    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    )

    return out


def prepare_for_scoring(raw_record: dict) -> pd.DataFrame:
    """
    Full pipeline from a raw customer record (as collected by app.py's
    form) to a single-row DataFrame ready for model.predict_proba().
    Raises ValueError on a missing required field, rather than letting a
    KeyError surface from deep inside pandas.
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in raw_record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    df = pd.DataFrame([raw_record])
    df = clean(df)
    df = engineer_features(df)
    return df
