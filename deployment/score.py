"""
Entry script for the Azure ML managed online deployment.

Responsibilities beyond calling model.predict():
  1. Validate the incoming payload against the expected schema before it
     reaches the model (a guardrail, not just error handling -- a malformed
     or out-of-range request should never reach the model silently).
  2. Score the request and attach a confidence-based routing flag so the
     application layer knows when a prediction needs human review rather
     than automated action.
  3. Log every request/response pair as a signed, immutable prediction
     event before the response is returned to the caller (see
     governance_logging.md for the full design this implements).
  4. Fail closed: if governance logging fails, the request fails rather
     than silently serving an unlogged prediction.
"""
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone

import mlflow
import pandas as pd

logger = logging.getLogger("telco_churn_scoring")

EXPECTED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Below this confidence, the prediction is routed for human review instead
# of an automated retention action -- this is the guardrail referenced in
# the governance section, not just a UI hint.
LOW_CONFIDENCE_MARGIN = 0.10

MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")


class SchemaValidationError(ValueError):
    pass


def init():
    """
    Azure ML managed online endpoint entry point, called once per worker
    at container startup. Loads the model into the module-global `model`
    so run() can reuse it across requests without reloading.
    """
    global model
    model_path = os.environ["AZUREML_MODEL_DIR"]
    try:
        model = mlflow.pyfunc.load_model(model_path)
    except Exception as exc:
        logger.exception("Model failed to load at startup")
        raise RuntimeError("Scoring service could not initialize its model") from exc


def _validate_payload(record: dict) -> None:
    """Raise SchemaValidationError if a required field is missing or out of range."""
    missing = [c for c in EXPECTED_COLUMNS if c not in record]
    if missing:
        raise SchemaValidationError(f"Missing required fields: {missing}")

    if not isinstance(record["tenure"], (int, float)) or record["tenure"] < 0:
        raise SchemaValidationError("tenure must be a non-negative number")

    if not isinstance(record["MonthlyCharges"], (int, float)) or record["MonthlyCharges"] < 0:
        raise SchemaValidationError("MonthlyCharges must be a non-negative number")


def _hash_input(record: dict) -> str:
    """SHA-256 hash of the canonicalized input record, used as the ledger's input reference."""
    canonical = json.dumps(record, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _log_prediction_event(input_hash: str, prediction: int, probability: float) -> None:
    """
    Writes a signed, immutable audit record before the caller receives a
    response. This is a fail-closed dependency: if this raises, run()
    re-raises and the request fails, rather than serving a prediction with
    no audit trail. See governance_logging.md for the full Confidential
    Ledger + Key Vault design.
    """
    event = {
        "input_hash": input_hash,
        "model_name": "telco-churn-xgboost",
        "model_version": MODEL_VERSION,
        "prediction": prediction,
        "probability": round(probability, 6),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        # Illustrative call -- the production implementation signs `event`
        # with the Key Vault key referenced by AZURE_KEY_VAULT_URI and
        # appends it to the Confidential Ledger referenced by
        # CONFIDENTIAL_LEDGER_ENDPOINT. Both are environment-scoped per
        # deployment (see managed_online_deployment.yml).
        from governance_client import append_to_ledger  # local module, not shown here
        append_to_ledger(event)
    except Exception as exc:
        logger.exception("Immutable audit logging failed for prediction event")
        raise RuntimeError(
            "Prediction could not be recorded to the audit trail; refusing "
            "to return an unlogged prediction"
        ) from exc


def run(raw_data: str):
    """
    Azure ML managed online endpoint entry point, called once per scoring
    request. raw_data is a JSON string of the form {"data": [record, ...]}.
    Validates, scores, and audit-logs each record; returns a per-record
    result list (each item either a prediction or an {"error": ...} entry).
    """
    start = time.time()
    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        return {"error": f"Malformed JSON payload: {exc}"}, 400

    records = payload.get("data")
    if not isinstance(records, list) or not records:
        return {"error": "Payload must contain a non-empty 'data' list"}, 400

    results = []
    for record in records:
        try:
            _validate_payload(record)
        except SchemaValidationError as exc:
            results.append({"error": str(exc)})
            continue

        input_hash = _hash_input(record)
        df = pd.DataFrame([record])[EXPECTED_COLUMNS]

        try:
            probability = float(model.predict_proba(df)[0, 1])
        except Exception as exc:
            logger.exception("Model inference failed")
            results.append({"error": "Inference failed", "input_hash": input_hash})
            continue

        # Cost-minimizing threshold from Section 9.3, not the default 0.5
        threshold = 0.30
        prediction = int(probability >= threshold)
        needs_review = abs(probability - threshold) < LOW_CONFIDENCE_MARGIN

        _log_prediction_event(input_hash, prediction, probability)

        results.append({
            "input_hash": input_hash,
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "needs_human_review": needs_review,
            "model_version": MODEL_VERSION,
        })

    logger.info("Scored %d records in %.1f ms", len(records), (time.time() - start) * 1000)
    return {"results": results}
