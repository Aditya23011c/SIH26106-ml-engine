"""
main.py - FastAPI microservice exposing the fraud-detection model,
plus the blockchain evidentiary-integrity endpoints (/anchor, /verify)
and the MongoDB-native analytics endpoint (/analytics/trends).

Detection: tries to load the fine-tuned DistilBERT model first - from a
local folder if present (dev machine), otherwise downloads it straight
from HuggingFace Hub (deployment, e.g. Render, where the model folder
isn't in the repo). If DistilBERT can't be loaded at all, falls back to
the TF-IDF + Gradient Boosting baseline automatically. Output contract
stays identical:
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
DISTILBERT_PATH = Path("models/distilbert-phishing-final")

# HuggingFace Hub repo the fine-tuned DistilBERT model was uploaded to.
# Used as a fallback when the local model folder isn't present -
# e.g. on Render, where the model folder is excluded from the git repo
# via .gitignore to keep the repo small and avoid GitHub's 100MB file limit.
HF_MODEL_REPO = "adityaprakashgupta-07/sih26106-distilbert-phishing"

app = FastAPI(title="SIH26106 Email Detection Engine")

LABELS = ["legitimate", "phishing", "bec"]
SUSPICIOUS_THRESHOLD = 0.7

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


# ---------- Model loading: try DistilBERT first (local, then HF Hub), fall back to baseline ----------

MODEL_TYPE = None
distilbert_model = None
distilbert_tokenizer = None
baseline_bundle = None

try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

    if DISTILBERT_PATH.exists():
        model_source = str(DISTILBERT_PATH)
        print(f"Loading DistilBERT from local folder: {model_source}")
    else:
        model_source = HF_MODEL_REPO
        print(f"Local DistilBERT folder not found. Downloading from HuggingFace Hub: {HF_MODEL_REPO}")

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
        probs = torch.softmax(outputs.logits, dim=1)[0]
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
    return {"status": "ok", "model_type": MODEL_TYPE, "labels": LABELS}


@app.post("/classify", response_model=ClassificationOutput)
def classify_email(email: EmailInput):
    raw_text = f"{email.subject} {email.body}"
    text = clean_text(raw_text)

    if MODEL_TYPE == "distilbert":
        label, confidence = predict_distilbert(text)
    else:
        label, confidence = predict_baseline(text)

    # confidence-band override: low-confidence phishing/bec becomes "suspicious"
    if label in ("phishing", "bec") and confidence < SUSPICIOUS_THRESHOLD:
        label = "suspicious"

    indicators = extract_indicators(raw_text)

    return ClassificationOutput(
        classification=label,
        confidence_score=round(confidence * 10, 1),
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