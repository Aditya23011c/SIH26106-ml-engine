"""
main.py - FastAPI microservice exposing the fraud-detection model,
plus the blockchain evidentiary-integrity endpoints (/anchor, /verify)
and the MongoDB-native analytics endpoint (/analytics/trends).

Detection: tries to load the fine-tuned DistilBERT V4 model first - from a
local folder if present (dev machine), otherwise downloads it straight
from HuggingFace Hub (deployment, e.g. Render, where the model folder
isn't in the repo). If DistilBERT can't be loaded at all, falls back to
the TF-IDF + Gradient Boosting baseline automatically.

Confidence calibration: if a calibration.json (temperature scaling factor)
exists next to the local model folder, DistilBERT's raw logits are divided
by that temperature before softmax. This does NOT change which label wins
(argmax is unaffected) - it only makes the confidence number more honest,
so genuinely uncertain predictions can correctly fall into the
"suspicious" band instead of being confidently wrong. See
models/calibrate_confidence.py for how this value was produced.

Per-class suspicious thresholds: phishing and bec are held to different
confidence thresholds before being downgraded to "suspicious" (see
SUSPICIOUS_THRESHOLDS below). This exists because BEC's own real-world
confidence tends to run lower than phishing's even when correct, so a
single shared threshold was causing real BEC cases to be under-flagged as
"suspicious" too often. Revisit these numbers as more real-world labeled
data is collected - they are a starting point, not a final answer.

Indicator-gate override [NEW]: if the model predicts phishing/bec but NONE
of the hand-coded red-flag indicators fired at all (no urgency phrasing,
no impersonation phrasing, no BEC phrasing, no links, no attachment
mention, no excessive punctuation, no generic greeting), that prediction
is downgraded to "suspicious" regardless of confidence. This exists
because content-only classifiers were observed to fire phishing/bec at
95%+ confidence on completely benign finance/order/security-alert emails
purely from topic-vocabulary overlap with the training data - a case with
literally zero corroborating red flags is treated as insufficiently
supported to auto-flag outright. Trade-off: a genuinely well-written
attack with no obvious tells will also get downgraded to "suspicious"
instead of a hard flag - documented as an intentional precision-over-recall
choice for this layer, with the backend's forensics correlation rule
(DKIM/DMARC/risk_band) as the actual root-cause fix.

Output contract:
{ classification, confidence_score, matched_indicators, processed_at }

Blockchain: /anchor and /verify wrap anchor_case.py / verify_case.py so
the Node.js backend can call this over HTTP instead of importing Python
directly (matches the shared microservice architecture - all handoffs
travel as HTTPS JSON).

Analytics: /analytics/trends runs aggregation pipelines directly against
the same MongoDB `cases` collection the platform writes to - no separate
sync step, no separate warehouse.

Run from project root: uvicorn main:app --reload --port 8001
"""
import json
import re
import joblib
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from blockchain.anchor_case import anchor_case
from blockchain.verify_case import verify_case
from analytics.trend_queries import (
    phishing_bec_volume_by_country_weekly,
    fastest_growing_campaigns,
    high_risk_infrastructure_breakdown,
)

BASELINE_PATH = Path("models/baseline_model.joblib")

# V4 fine-tuned DistilBERT model
DISTILBERT_PATH = Path("models/distilbert-phishing-final-v4")

# HuggingFace fallback - used when the local V4 folder isn't present
# (e.g. on a deployment host where it's excluded from git via .gitignore
# to stay under GitHub's 100MB file limit).
HF_MODEL_REPO = "adityaprakashgupta-07/sih26106-distilbert-phishing"

app = FastAPI(title="SIH26106 Email Detection Engine")

LABELS = ["legitimate", "phishing", "bec"]

# Per-class confidence threshold below which a phishing/bec prediction is
# downgraded to "suspicious" instead. Kept separate (not one shared value)
# because BEC's genuine-positive confidence tends to run lower than
# phishing's - a single 0.7 cutoff was pushing real BEC cases into
# "suspicious" too often. Tune these as more real-world labeled examples
# come in; there's nothing special about these exact numbers yet.
SUSPICIOUS_THRESHOLDS = {
    "phishing": 0.7,
    "bec": 0.5,
}

URGENCY_PHRASES = [
    "urgent", "immediately", "act now", "verify your account", "suspended",
    "act within", "final notice", "action required", "expire", "limited time",
]
IMPERSONATION_PHRASES = [
    "ceo", "cfo", "president", "on behalf of", "confidential request",
    "wire transfer", "keep this between us", "don't discuss", "urgent request from",
]
BEC_PHRASES = [
    "invoice attached", "updated bank details", "payment details have changed",
    "remit payment", "new vendor account", "beneficiary account", "swift code",
    "please process payment", "outstanding invoice",
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+\.\S+", " EMAILTOKEN ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def extract_indicators(text: str) -> dict:
    text_lower = text.lower()
    return {
        "urgency_score": sum(p in text_lower for p in URGENCY_PHRASES),
        "impersonation_score": sum(p in text_lower for p in IMPERSONATION_PHRASES),
        "bec_score": sum(p in text_lower for p in BEC_PHRASES),
        "url_count": len(re.findall(r"http\S+", text)),
        "has_attachment_mention": bool(re.search(r"attach(ed|ment)", text_lower)),
        "excessive_punctuation": bool(re.search(r"[!?]{2,}", text)),
        "generic_greeting": bool(
            re.search(r"^(dear (customer|user|sir/madam)|to whom it may concern)", text_lower)
        ),
    }


def has_any_indicator(indicators: dict) -> bool:
    """
    True if at least one hand-coded red-flag indicator fired. Used by the
    indicator-gate override below: a phishing/bec call with zero
    corroborating indicators is treated as unsupported by anything other
    than the model's own learned vocabulary association.
    """
    return (
        indicators["urgency_score"] > 0
        or indicators["impersonation_score"] > 0
        or indicators["bec_score"] > 0
        or indicators["url_count"] > 0
        or indicators["has_attachment_mention"]
        or indicators["excessive_punctuation"]
        or indicators["generic_greeting"]
    )


# ---------- Model loading: try DistilBERT first (local, then HF Hub), fall back to baseline ----------

MODEL_TYPE = None
distilbert_model = None
distilbert_tokenizer = None
baseline_bundle = None

# Temperature-scaling factor for calibrating DistilBERT confidence scores.
# 1.0 = no change. Only loaded when the local V4 model folder is used
# directly (the calibration was fit specifically against that model's
# logits) - if the HF Hub fallback path is used instead, defaults to 1.0
# since we can't guarantee the calibration still applies.
CALIBRATION_TEMPERATURE = 1.0

try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    if DISTILBERT_PATH.exists():
        model_source = str(DISTILBERT_PATH)
        print(f"Loading DistilBERT V4 from local folder: {model_source}")

        calibration_path = DISTILBERT_PATH / "calibration.json"
        if calibration_path.exists():
            with open(calibration_path) as f:
                CALIBRATION_TEMPERATURE = json.load(f)["temperature"]
            print(f"Loaded calibration temperature: {CALIBRATION_TEMPERATURE:.3f}")
        else:
            print("No calibration.json found next to the model - confidence scores will be uncalibrated (temperature=1.0).")
    else:
        model_source = HF_MODEL_REPO
        print(f"Local DistilBERT V4 folder not found. Downloading from HuggingFace Hub: {HF_MODEL_REPO}")
        print("Note: calibration is not applied on this path (temperature=1.0) - it was only fit against the local V4 weights.")

    distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(model_source)
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_source)
    distilbert_model.eval()
    MODEL_TYPE = "distilbert"
    print("Loaded DistilBERT model successfully.")
except Exception as e:
    print(f"DistilBERT load failed or unavailable ({e}). Falling back to baseline model.")
    baseline_bundle = joblib.load(BASELINE_PATH)
    MODEL_TYPE = "baseline"
    print("Loaded baseline (TF-IDF + Gradient Boosting) model successfully.")


def predict_distilbert(text: str):
    import torch
    inputs = distilbert_tokenizer(
        text, return_tensors="pt", truncation=True, padding="max_length", max_length=256
    )
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        # Apply temperature scaling before softmax. When CALIBRATION_TEMPERATURE
        # is 1.0 (no calibration file, or HF Hub fallback in use), this is a
        # no-op division and behaves exactly as before.
        scaled_logits = outputs.logits / CALIBRATION_TEMPERATURE
        probs = torch.softmax(scaled_logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs))
    confidence = float(probs[pred_idx])
    label = distilbert_model.config.id2label[pred_idx]
    return label, confidence


def predict_baseline(text: str):
    vectorizer = baseline_bundle["vectorizer"]
    model = baseline_bundle["model"]
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    pred_idx = proba.argmax()
    confidence = float(proba[pred_idx])
    label = model.classes_[pred_idx]
    return label, confidence


# ---------- Request / response models ----------

class EmailInput(BaseModel):
    subject: str
    body: str
    sender: Optional[str] = None
    metadata: Optional[dict] = None


class ClassificationOutput(BaseModel):
    classification: str
    confidence_score: float
    matched_indicators: dict
    processed_at: str


class CaseAnchorRequest(BaseModel):
    case_id: str
    case_document: dict


class CaseVerifyRequest(BaseModel):
    case_id: str
    case_document: dict


# ---------- Routes ----------

@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_type": MODEL_TYPE,
        "labels": LABELS,
        "team_name": "Brute Force",
        "project_name": "Mail Rakshak",
        "problem_statement_id": "SIH26106",
    }


@app.post("/classify", response_model=ClassificationOutput)
def classify_email(email: EmailInput):
    raw_text = f"{email.subject} {email.body}"
    text = clean_text(raw_text)

    if MODEL_TYPE == "distilbert":
        label, confidence = predict_distilbert(text)
    else:
        label, confidence = predict_baseline(text)

    indicators = extract_indicators(raw_text)

    # confidence-band override: low-confidence phishing/bec becomes "suspicious".
    # Uses a per-class threshold (SUSPICIOUS_THRESHOLDS) rather than one shared
    # value - see the module docstring for why.
    if label in SUSPICIOUS_THRESHOLDS and confidence < SUSPICIOUS_THRESHOLDS[label]:
        label = "suspicious"

    # indicator-gate override [NEW]: even a HIGH-confidence phishing/bec call
    # gets downgraded to "suspicious" if none of the hand-coded red-flag
    # indicators fired at all. See module docstring for the rationale and
    # the precision/recall trade-off this accepts.
    if label in ("phishing", "bec") and not has_any_indicator(indicators):
        label = "suspicious"

    return ClassificationOutput(
        classification=label,
        confidence_score=round(confidence * 100, 1),
        matched_indicators=indicators,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/anchor")
def anchor_case_endpoint(request: CaseAnchorRequest):
    try:
        result = anchor_case(request.case_id, request.case_document)
        return result
    except ValueError as e:
        # case already anchored on-chain
        raise HTTPException(status_code=409, detail=str(e))
    except ConnectionError as e:
        # RPC endpoint unreachable
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        # transaction reverted on-chain
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.post("/verify")
def verify_case_endpoint(request: CaseVerifyRequest):
    try:
        result = verify_case(request.case_id, request.case_document)
        return result
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.get("/analytics/trends")
def get_analytics_trends(
    months_back: int = 6,
    risk_threshold: float = 0.7,
):
    """
    Returns a combined analytics snapshot for the jury demo dashboard:
    weekly phishing/BEC volume by country, fastest-growing campaigns,
    and a high-risk infrastructure-type breakdown. Reads straight from
    the MongoDB `cases` collection - no separate warehouse or sync step.
    """
    try:
        return {
            "volume_by_country_weekly": phishing_bec_volume_by_country_weekly(
                months_back=months_back
            ),
            "fastest_growing_campaigns": fastest_growing_campaigns(),
            "high_risk_infrastructure_breakdown": high_risk_infrastructure_breakdown(
                risk_threshold=risk_threshold
            ),
        }
    except RuntimeError as e:
        # e.g. MONGO_URI missing from .env
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics query failed: {e}")