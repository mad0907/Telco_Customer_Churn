"""
Telco Churn Intelligence -- interactive prototype.

Loads the XGBoost pipeline trained and persisted in
notebooks/telco_churn_analysis.ipynb (Section 10) and lets a user enter a
customer profile to see: churn probability, risk tier, the specific SHAP
drivers behind that score, and a recommended retention action.

This is a decision-support prototype, not the production serving surface --
the production equivalent of this logic is deployment/score.py, deployed
behind an Azure ML managed online endpoint per the architecture document.
Run with: streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.churn_features import prepare_for_scoring

MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODELS_DIR / "churn_xgboost_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

st.set_page_config(page_title="Telco Churn Intelligence", layout="wide")


@st.cache_resource
def load_model_and_metadata():
    """Load the persisted pipeline and its metadata once per session (cached)."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found in models/. Run "
            "notebooks/telco_churn_analysis.ipynb end to end first -- "
            "Section 10 persists the model this app depends on."
        )
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_resource
def build_explainer(_model):
    """Build (and cache) a SHAP TreeExplainer for the pipeline's classifier step."""
    return shap.TreeExplainer(_model.named_steps["clf"])


def assign_risk_tier(probability: float, metadata: dict) -> str:
    """Map a churn probability to a Low/Medium/High/Critical tier using the model's saved bins."""
    bins = metadata["risk_tier_bins"]
    labels = metadata["risk_tier_labels"]
    for i in range(len(bins) - 1):
        if bins[i] < probability <= bins[i + 1]:
            return labels[i]
    return labels[0]


def recommend_action(tier: str, top_drivers: list) -> str:
    """Turn a risk tier and its top SHAP drivers into a specific, actionable recommendation."""
    driver_names = {name for name, _ in top_drivers}
    if tier in ("Critical", "High"):
        if any(n.startswith("Contract") or n.startswith("PaymentMethod") for n in driver_names):
            return (
                "Prioritize for retention outreach. Contract type or payment "
                "method is among the top drivers for this customer -- lead "
                "with a contract-term or payment-method offer rather than a "
                "generic discount."
            )
        return "Prioritize for retention outreach with a tailored offer."
    if tier == "Medium":
        return "Include in a lower-cost, broader retention campaign rather than individual outreach."
    return "No action needed based on churn risk."


EXAMPLE_PROFILES = {
    "High-risk: new fiber customer, month-to-month": dict(
        gender="Female", SeniorCitizen=0, Partner="No", Dependents="No",
        tenure=2, PhoneService="Yes", MultipleLines="No",
        InternetService="Fiber optic", OnlineSecurity="No", OnlineBackup="No",
        DeviceProtection="No", TechSupport="No", StreamingTV="No",
        StreamingMovies="No", Contract="Month-to-month", PaperlessBilling="Yes",
        PaymentMethod="Electronic check", MonthlyCharges=85.0, TotalCharges=170.0,
    ),
    "Low-risk: long-tenure, two-year contract": dict(
        gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes",
        tenure=60, PhoneService="Yes", MultipleLines="Yes",
        InternetService="DSL", OnlineSecurity="Yes", OnlineBackup="Yes",
        DeviceProtection="Yes", TechSupport="Yes", StreamingTV="No",
        StreamingMovies="No", Contract="Two year", PaperlessBilling="No",
        PaymentMethod="Bank transfer (automatic)", MonthlyCharges=55.0, TotalCharges=3300.0,
    ),
}

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}


st.title("Telco Churn Intelligence")
st.caption(
    "Prototype decision-support tool -- enter a customer profile to see "
    "predicted churn risk, the specific drivers behind that score, and a "
    "recommended action. Backed by the XGBoost model from "
    "notebooks/telco_churn_analysis.ipynb."
)

try:
    model, metadata = load_model_and_metadata()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

example_choice = st.selectbox(
    "Load an example customer (optional)",
    ["-- manual entry --"] + list(EXAMPLE_PROFILES.keys()),
)
defaults = EXAMPLE_PROFILES.get(example_choice, {})

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account")
        gender = st.selectbox("Gender", CATEGORICAL_OPTIONS["gender"],
                               index=CATEGORICAL_OPTIONS["gender"].index(defaults.get("gender", "Female")))
        senior = st.checkbox("Senior citizen", value=bool(defaults.get("SeniorCitizen", 0)))
        partner = st.selectbox("Has partner", CATEGORICAL_OPTIONS["Partner"],
                                index=CATEGORICAL_OPTIONS["Partner"].index(defaults.get("Partner", "No")))
        dependents = st.selectbox("Has dependents", CATEGORICAL_OPTIONS["Dependents"],
                                   index=CATEGORICAL_OPTIONS["Dependents"].index(defaults.get("Dependents", "No")))
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100,
                                  value=int(defaults.get("tenure", 12)))
        contract = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"],
                                 index=CATEGORICAL_OPTIONS["Contract"].index(defaults.get("Contract", "Month-to-month")))

    with col2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", CATEGORICAL_OPTIONS["PhoneService"],
                              index=CATEGORICAL_OPTIONS["PhoneService"].index(defaults.get("PhoneService", "Yes")))
        multiple_lines = st.selectbox("Multiple lines", CATEGORICAL_OPTIONS["MultipleLines"],
                                       index=CATEGORICAL_OPTIONS["MultipleLines"].index(defaults.get("MultipleLines", "No")))
        internet = st.selectbox("Internet service", CATEGORICAL_OPTIONS["InternetService"],
                                 index=CATEGORICAL_OPTIONS["InternetService"].index(defaults.get("InternetService", "DSL")))
        online_security = st.selectbox("Online security", CATEGORICAL_OPTIONS["OnlineSecurity"],
                                        index=CATEGORICAL_OPTIONS["OnlineSecurity"].index(defaults.get("OnlineSecurity", "No")))
        online_backup = st.selectbox("Online backup", CATEGORICAL_OPTIONS["OnlineBackup"],
                                      index=CATEGORICAL_OPTIONS["OnlineBackup"].index(defaults.get("OnlineBackup", "No")))
        device_protection = st.selectbox("Device protection", CATEGORICAL_OPTIONS["DeviceProtection"],
                                          index=CATEGORICAL_OPTIONS["DeviceProtection"].index(defaults.get("DeviceProtection", "No")))

    with col3:
        st.subheader("Billing")
        tech_support = st.selectbox("Tech support", CATEGORICAL_OPTIONS["TechSupport"],
                                     index=CATEGORICAL_OPTIONS["TechSupport"].index(defaults.get("TechSupport", "No")))
        streaming_tv = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"],
                                     index=CATEGORICAL_OPTIONS["StreamingTV"].index(defaults.get("StreamingTV", "No")))
        streaming_movies = st.selectbox("Streaming movies", CATEGORICAL_OPTIONS["StreamingMovies"],
                                         index=CATEGORICAL_OPTIONS["StreamingMovies"].index(defaults.get("StreamingMovies", "No")))
        paperless = st.selectbox("Paperless billing", CATEGORICAL_OPTIONS["PaperlessBilling"],
                                  index=CATEGORICAL_OPTIONS["PaperlessBilling"].index(defaults.get("PaperlessBilling", "Yes")))
        payment_method = st.selectbox("Payment method", CATEGORICAL_OPTIONS["PaymentMethod"],
                                       index=CATEGORICAL_OPTIONS["PaymentMethod"].index(defaults.get("PaymentMethod", "Electronic check")))
        monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0,
                                           value=float(defaults.get("MonthlyCharges", 70.0)))
        total_charges = st.number_input("Total charges ($)", min_value=0.0, max_value=10000.0,
                                         value=float(defaults.get("TotalCharges", 840.0)))

    submitted = st.form_submit_button("Score this customer")

if submitted:
    raw_record = dict(
        gender=gender, SeniorCitizen=1 if senior else 0, Partner=partner,
        Dependents=dependents, tenure=tenure, PhoneService=phone,
        MultipleLines=multiple_lines, InternetService=internet,
        OnlineSecurity=online_security, OnlineBackup=online_backup,
        DeviceProtection=device_protection, TechSupport=tech_support,
        StreamingTV=streaming_tv, StreamingMovies=streaming_movies,
        Contract=contract, PaperlessBilling=paperless,
        PaymentMethod=payment_method, MonthlyCharges=monthly_charges,
        TotalCharges=total_charges,
    )

    try:
        scoring_df = prepare_for_scoring(raw_record)
        scoring_df = scoring_df[metadata["feature_columns"]]
        probability = float(model.predict_proba(scoring_df)[0, 1])
    except Exception as exc:
        st.error(f"Scoring failed: {exc}")
        st.stop()

    tier = assign_risk_tier(probability, metadata)

    try:
        explainer = build_explainer(model)
        transformed = model.named_steps["pre"].transform(scoring_df)
        feature_names = model.named_steps["pre"].get_feature_names_out()
        shap_values = explainer.shap_values(transformed)[0]
        top_idx = np.argsort(-np.abs(shap_values))[:5]
        top_drivers = [
            (feature_names[i].split("__", 1)[-1], float(shap_values[i]))
            for i in top_idx
        ]
    except Exception as exc:
        st.warning(f"Could not compute driver explanation: {exc}")
        top_drivers = []

    st.divider()
    result_col, driver_col = st.columns([1, 1.4])

    with result_col:
        st.metric("Churn probability", f"{probability:.1%}")
        tier_color = {"Critical": "red", "High": "orange", "Medium": "blue", "Low": "green"}
        st.markdown(f"**Risk tier:** :{tier_color.get(tier, 'grey')}[{tier}]")
        st.write("**Recommended action**")
        st.info(recommend_action(tier, top_drivers))

    with driver_col:
        st.write("**Top drivers for this prediction**")
        if top_drivers:
            driver_df = pd.DataFrame(top_drivers, columns=["feature", "shap_contribution"])
            driver_df["direction"] = driver_df["shap_contribution"].apply(
                lambda v: "increases risk" if v > 0 else "decreases risk"
            )
            st.dataframe(driver_df, use_container_width=True, hide_index=True)
        else:
            st.write("Not available for this prediction.")

    with st.expander("Model metadata"):
        st.json(metadata)
