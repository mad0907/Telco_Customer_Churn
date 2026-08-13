"""
Development/verification script for the churn pipeline.
Not a deliverable -- used to validate logic before it is transcribed into
the notebook. Run with: source venv/bin/activate && python src/pipeline_dev.py
"""
import sys
import numpy as np
import pandas as pd

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV, raising a clear error if it is missing."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Dataset not found at {path}") from exc
    return df


def audit(df: pd.DataFrame) -> None:
    """Print shape, dtypes, null counts, blank-TotalCharges rows, and the churn base rate."""
    print("shape:", df.shape)
    print("\ndtypes:\n", df.dtypes)
    print("\nnull counts:\n", df.isnull().sum()[df.isnull().sum() > 0])
    # TotalCharges is read as object because of blank strings
    blanks = df[df["TotalCharges"].str.strip() == ""]
    print("\nblank TotalCharges rows:", len(blanks))
    print(blanks[["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])
    print("\nduplicate customerID count:", df["customerID"].duplicated().sum())
    print("\nChurn distribution:\n", df["Churn"].value_counts(normalize=True))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce TotalCharges to numeric, fill the structural-zero rows
    (tenure == 0), normalize SeniorCitizen, drop customerID, and map
    Churn to 0/1. See notebook Section 4 for the full rationale.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    # All rows with a null TotalCharges have tenure == 0 (new customers,
    # no billing cycle completed yet). This is not missing-at-random noise,
    # it is a structural zero, so we set it to 0.0 rather than imputing
    # with median/mean, and rather than dropping the rows (would discard
    # otherwise-complete records and bias the tenure=0 cohort out of EDA).
    zero_tenure_mask = out["tenure"] == 0
    assert out.loc[out["TotalCharges"].isnull(), "tenure"].eq(0).all(), (
        "Found a null TotalCharges row with nonzero tenure -- "
        "structural-zero assumption does not hold, revisit imputation."
    )
    out.loc[zero_tenure_mask, "TotalCharges"] = 0.0
    out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})
    out = out.drop(columns=["customerID"])
    out["Churn"] = out["Churn"].map({"Yes": 1, "No": 0})
    return out


def main():
    """Run the audit -> clean -> outlier-scan sequence and print results."""
    df = load_raw(DATA_PATH)
    audit(df)
    clean_df = clean(df)
    print("\nclean dtypes:\n", clean_df.dtypes)
    print("\nclean nulls:", clean_df.isnull().sum().sum())
    print("\nMonthlyCharges describe:\n", clean_df["MonthlyCharges"].describe())
    print("\ntenure describe:\n", clean_df["tenure"].describe())

    # outlier scan via IQR on the two continuous fields
    for col in ["MonthlyCharges", "TotalCharges", "tenure"]:
        q1, q3 = clean_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((clean_df[col] < lo) | (clean_df[col] > hi)).sum()
        print(f"{col}: IQR bounds [{lo:.2f}, {hi:.2f}], outliers={n_out}")


if __name__ == "__main__":
    sys.exit(main())
