"""
Builds the architecture diagrams used in the write-up document, as rendered
PNGs (not Mermaid) so they display reliably in Word/PDF without needing a
Mermaid-aware renderer.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

plt.rcParams["font.family"] = "DejaVu Sans"

BOX_FACE = "#F2F4F7"
BOX_EDGE = "#344054"
ACCENT_FACE = "#E6EEF9"
ACCENT_EDGE = "#1D4ED8"
GOV_FACE = "#FDF2E9"
GOV_EDGE = "#B54708"
TEXT_COLOR = "#101828"


def box(ax, x, y, w, h, text, face=BOX_FACE, edge=BOX_EDGE, fontsize=9.5, weight="normal"):
    """Draw a rounded rectangle with centered label text at the given axes coordinates."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=TEXT_COLOR, weight=weight, wrap=True)
    return patch


def arrow(ax, start, end, color="#475467", style="-|>", connectionstyle="arc3,rad=0.0", lw=1.4):
    a = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        color=color, linewidth=lw, connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


def label(ax, x, y, text, fontsize=8, color="#475467", style="italic", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color, style=style)


def new_canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


# ===========================================================================
# Diagram 1: End-to-end system architecture
# ===========================================================================
fig, ax = new_canvas(13, 8.8)

ax.text(0.3, 8.4, "System Architecture — Churn Scoring as a Deployed Service",
        fontsize=13, weight="bold", color=TEXT_COLOR)

# Row 1: sources
box(ax, 0.3, 7.0, 2.3, 0.9, "CRM / Billing\nSystem")
box(ax, 2.9, 7.0, 2.3, 0.9, "Support Ticket\nSystem")
box(ax, 5.5, 7.0, 2.3, 0.9, "Usage / Event\nTelemetry")

arrow(ax, (1.45, 7.0), (2.9, 6.2), connectionstyle="arc3,rad=-0.15")
arrow(ax, (4.05, 7.0), (4.05, 6.2))
arrow(ax, (6.65, 7.0), (5.2, 6.2), connectionstyle="arc3,rad=0.15")

# Row 2: ingestion / feature store
box(ax, 2.4, 5.3, 3.3, 0.9, "Azure Data Factory\n(ingestion + validation)", face=ACCENT_FACE, edge=ACCENT_EDGE)
arrow(ax, (4.05, 5.3), (4.05, 4.5))
box(ax, 2.4, 3.6, 3.3, 0.9, "Azure ML Managed\nFeature Store", face=ACCENT_FACE, edge=ACCENT_EDGE)

# Training pipeline (right side, connected to feature store)
arrow(ax, (5.7, 4.05), (7.3, 4.05))
box(ax, 7.3, 3.6, 3.0, 0.9, "Azure ML Training\nPipeline (scheduled +\ndrift-triggered)", face=ACCENT_FACE, edge=ACCENT_EDGE)
arrow(ax, (8.8, 4.5), (8.8, 5.3))
box(ax, 7.3, 5.3, 3.0, 0.9, "Model Registry\n(versioned, champion/\nchallenger evaluation)", face=ACCENT_FACE, edge=ACCENT_EDGE)

# Row 3: serving path
arrow(ax, (4.05, 3.6), (4.05, 2.8))
box(ax, 2.4, 1.9, 3.3, 0.9, "Managed Online Endpoint\n(scoring API)", face=ACCENT_FACE, edge=ACCENT_EDGE)
arrow(ax, (8.8, 5.3), (5.7, 2.35), connectionstyle="arc3,rad=0.25")
label(ax, 7.2, 4.15, "promote on pass", fontsize=7.5)

arrow(ax, (5.7, 2.35), (7.3, 2.35))
box(ax, 7.3, 1.9, 3.0, 0.9, "Application / UI Layer\n(retention workflow)", face=BOX_FACE, edge=BOX_EDGE)

# Governance / logging, spans bottom
box(ax, 0.3, 0.5, 10.0, 0.9,
    "Governance layer: request/response logged to Azure Confidential Ledger (immutable) "
    "+ Key Vault-signed | Azure Monitor / App Insights",
    face=GOV_FACE, edge=GOV_EDGE, fontsize=8.5)
arrow(ax, (4.05, 1.9), (2.5, 1.4), connectionstyle="arc3,rad=-0.2", color=GOV_EDGE)
arrow(ax, (8.8, 1.9), (7.5, 1.4), connectionstyle="arc3,rad=0.2", color=GOV_EDGE)

# monitoring feedback loop
arrow(ax, (10.3, 2.35), (11.6, 2.35), color="#B54708")
box(ax, 11.6, 1.9, 1.1, 0.9, "Monitoring\n& Drift\nDetection", face=GOV_FACE, edge=GOV_EDGE, fontsize=8)
arrow(ax, (12.15, 2.8), (12.15, 4.05), connectionstyle="arc3,rad=0.0", color=GOV_EDGE)
arrow(ax, (12.15, 4.05), (10.3, 4.05), color=GOV_EDGE)
label(ax, 12.15, 3.3, "retrain\ntrigger", fontsize=7, color=GOV_EDGE, ha="left")

plt.tight_layout()
plt.savefig("../assets/architecture_diagram.png", dpi=220, bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# Diagram 2: MLOps CI/CD pipeline
# ===========================================================================
fig, ax = new_canvas(13, 6.5)
ax.text(0.3, 6.05, "MLOps Pipeline — Retraining and Promotion", fontsize=13, weight="bold", color=TEXT_COLOR)

stages = [
    ("Git repo\n(code + pipeline\ndefinitions)", 0.3),
    ("CI: lint, unit\ntest, data\nvalidation", 2.5),
    ("Azure ML\npipeline: train\n+ evaluate", 4.7),
    ("Champion/\nchallenger\ncompare vs. prod", 6.9),
    ("Register model\n(if it wins)", 9.1),
]
for text, x in stages:
    box(ax, x, 4.0, 1.9, 1.1, text, fontsize=8.5)

for i in range(len(stages) - 1):
    x0 = stages[i][1] + 1.9
    x1 = stages[i + 1][1]
    arrow(ax, (x0, 4.55), (x1, 4.55))

# gate branch
arrow(ax, (10.0, 4.0), (10.0, 3.2), connectionstyle="arc3,rad=0")
box(ax, 8.6, 2.3, 2.8, 0.9, "Manual approval gate\n(release manager sign-off)", face=GOV_FACE, edge=GOV_EDGE, fontsize=8.5)
arrow(ax, (10.0, 2.3), (10.0, 1.5))
box(ax, 8.6, 0.6, 2.8, 0.9, "CD: deploy to Managed\nOnline Endpoint (canary)", face=ACCENT_FACE, edge=ACCENT_EDGE, fontsize=8.5)

arrow(ax, (8.6, 1.05), (6.4, 1.05))
box(ax, 3.8, 0.6, 2.6, 0.9, "Full rollout after\ncanary healthy", face=BOX_FACE, edge=BOX_EDGE, fontsize=8.5)

arrow(ax, (3.8, 1.05), (1.6, 1.05))
box(ax, 0.3, 0.6, 1.3, 0.9, "Champion\nreplaced", face=BOX_FACE, edge=BOX_EDGE, fontsize=8)

label(ax, 1.25, 3.8, "if challenger loses: no promotion, prior champion stays live",
      fontsize=8, color="#B54708", ha="left")

plt.tight_layout()
plt.savefig("../assets/mlops_pipeline_diagram.png", dpi=220, bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# Diagram 3: Immutable prediction logging / governance blueprint
# ===========================================================================
fig, ax = new_canvas(12, 7.8)
ax.text(0.3, 7.35, "Immutable Prediction Logging & Governance Blueprint", fontsize=13, weight="bold", color=TEXT_COLOR)

box(ax, 0.3, 5.9, 2.6, 0.9, "Inference request\n(customer features)")
arrow(ax, (2.9, 6.35), (4.0, 6.35))
box(ax, 4.0, 5.9, 2.6, 0.9, "Managed Online\nEndpoint (model)", face=ACCENT_FACE, edge=ACCENT_EDGE)
arrow(ax, (6.6, 6.35), (7.7, 6.35))
box(ax, 7.7, 5.9, 3.5, 0.9, "Prediction event object:\ninput hash, model version,\noutput, confidence, timestamp")

arrow(ax, (9.45, 5.9), (9.45, 5.0))
box(ax, 7.9, 4.1, 3.1, 0.9, "Azure Key Vault\n(sign / encrypt event)", face=GOV_FACE, edge=GOV_EDGE)

arrow(ax, (9.45, 4.1), (9.45, 3.2))
box(ax, 7.5, 2.3, 3.9, 0.9,
    "Azure Confidential Ledger\n(hash-chained, tamper-evident\nimmutable append-only log)",
    face=GOV_FACE, edge=GOV_EDGE)

arrow(ax, (7.5, 2.75), (5.6, 2.75))
box(ax, 2.9, 2.3, 2.7, 0.9, "Immutable Blob Storage\n(WORM policy, bulk archive)", face=GOV_FACE, edge=GOV_EDGE, fontsize=8.5)

arrow(ax, (9.45, 2.3), (9.45, 1.5))
box(ax, 7.4, 0.5, 4.0, 1.0,
    "Audit / compliance query interface\n(model risk review, regulator or\ninternal-audit access)")

label(ax, 0.3, 5.35, "Every prediction that could influence a retention offer is logged "
      "before the response is returned to the caller -- not batched or sampled --\n"
      "so the audit trail cannot have gaps for the predictions that matter most.",
      fontsize=8.5, ha="left")

plt.tight_layout()
plt.savefig("../assets/governance_logging_diagram.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print("Diagrams written to ../assets/")
