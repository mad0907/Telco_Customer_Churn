# Immutable Prediction Logging - Design Notes

Referenced from `score.py`. This is the design that `append_to_ledger()`
implements; it is documented separately from the code so the reasoning
survives independent of any one implementation.

## Why a ledger instead of a normal application log

A normal application log (Blob Storage, a database table, Application
Insights) can be edited or deleted by anyone with write access to it,
including automated processes. For a model that can influence what
retention offer a customer receives, that is a governance gap: if a
prediction is later disputed - by a customer, a regulator, or an internal
audit - there needs to be a record that provably was not altered after
the fact.

Azure Confidential Ledger is used for this because it is a managed service
built specifically for this property: entries are cryptographically hashed
and chained, and the service runs inside hardware-backed confidential
compute, so tampering is detectable rather than merely discouraged by
access controls. Azure Key Vault holds the signing key used to sign each
prediction event before it is written, so the signature itself cannot be
forged by anything without Key Vault access, which is a separate
permission boundary from the scoring service's own identity.

## What gets logged

Every prediction event contains:

- `input_hash` - a SHA-256 hash of the input record, not the raw record
  itself. This lets an auditor confirm a specific input produced a specific
  output without the ledger becoming a second copy of customer PII that
  now also needs its own data-retention and access-control policy.
- `model_name` and `model_version` - which exact model version produced
  the prediction, tying every prediction back to a specific entry in the
  model registry and, from there, back to the training run that produced
  it.
- `prediction` and `probability` - the actual output.
- `timestamp_utc`.

The raw input record itself is written to the separate, non-immutable
scored-data store referenced in `monitoring_job.yml` (`telco_predictions`),
which is where the input actually needed for drift monitoring lives, under
normal data-retention policy. The ledger is the tamper-evident record that
a prediction happened and what it was, not the system of record for
customer data.

## Fail-closed, not fail-open

`score.py` raises if `append_to_ledger()` fails, which means a prediction
that cannot be logged is not returned to the caller. This is a deliberate
choice: the alternative (log best-effort, return the prediction regardless)
would mean the audit trail has silent gaps precisely when the logging
infrastructure is unhealthy - which is exactly when an incident is most
likely to also be happening upstream. A brief scoring outage during a
logging incident is judged to be a better failure mode than an unlogged
automated decision.

## What this does not cover

This design covers integrity and non-repudiation of the prediction record
itself. It does not, on its own, cover:

- **Bias and fairness testing** - handled separately as a pre-deployment
  gate (see the AI Governance section of the main write-up), not by the
  logging layer.
- **Explainability at serving time** - `score.py` does not compute or
  return SHAP values per request, since that would add meaningful latency
  to every scoring call. The recommended pattern is to compute and cache
  SHAP explanations in the batch monitoring job, and expose an on-demand
  explanation endpoint (separate from the main scoring path) that a
  retention agent can query for a specific customer when the automated
  decision is questioned.
- **Retention offer accountability** - the ledger records what the model
  predicted, not what the downstream retention workflow did with that
  prediction. Logging the actual retention action taken is the
  responsibility of the application/UI layer in the architecture diagram,
  and should follow the same immutable-logging pattern if it is to be
  auditable end to end.
