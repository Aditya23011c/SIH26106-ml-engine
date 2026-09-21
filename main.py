"""
main.py - FastAPI microservice exposing the fraud-detection model,
plus the blockchain evidentiary-integrity endpoints (/anchor, /verify)
and the MongoDB-native analytics endpoint (/analytics/trends).

Detection: tries to load the fine-tuned DistilBERT model first - from a
local folder if present (dev machine), otherwise downloads it straight
from HuggingFace Hub (deployment, e.g. Render, where the model folder
isn't in the repo). If DistilBERT can't be loaded at all, falls back
to the TF-IDF + Gradient Boosting baseline automatically.

Output:
{
    classification,
    confidence_score,
    model_probabilities,
    matched_indicators,
    processed_at
}

Blockchain: /anchor and /verify wrap anchor_case.py / verify_case.py.

Analytics: /analytics/trends runs aggregation pipelines directly
against the same MongoDB `cases` collection.
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


# ============================================================
# PATHS / MODEL CONFIGURATION
# ============================================================

BASELINE_PATH = Path("models/baseline_model.joblib")

# V4 fine-tuned DistilBERT model
DISTILBERT_PATH = Path("models/distilbert-phishing-final-v4")

# HuggingFace fallback
HF_MODEL_REPO = "adityaprakashgupta-07/sih26106-distilbert-phishing"


app = FastAPI(title="SIH26106 Email Detection Engine")


# ============================================================
# LABELS / THRESHOLDS
# ============================================================

LABELS = [
    "legitimate",
    "phishing",
    "bec",
]

SUSPICIOUS_THRESHOLD = 0.7


# ============================================================
# RULE-BASED INDICATORS
# ============================================================

URGENCY_PHRASES = [
    "urgent",
    "immediately",
    "act now",
    "verify your account",
    "suspended",
    "act within",
    "final notice",
    "action required",
    "expire",
    "limited time",
]

IMPERSONATION_PHRASES = [
    "ceo",
    "cfo",
    "president",
    "on behalf of",
    "confidential request",
    "wire transfer",
    "keep this between us",
    "don't discuss",
    "urgent request from",
]

BEC_PHRASES = [
    "invoice attached",
    "updated bank details",
    "payment details have changed",
    "remit payment",
    "new vendor account",
    "beneficiary account",
    "swift code",
    "please process payment",
    "outstanding invoice",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+\.\S+", " EMAILTOKEN ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


# ============================================================
# RULE-BASED INDICATOR EXTRACTION
# ============================================================

def extract_indicators(text: str) -> dict:
    text_lower = text.lower()

    return {
        "urgency_score": sum(
            p in text_lower for p in URGENCY_PHRASES
        ),

        "impersonation_score": sum(
            p in text_lower for p in IMPERSONATION_PHRASES
        ),

        "bec_score": sum(
            p in text_lower for p in BEC_PHRASES
        ),

        "url_count": len(
            re.findall(r"http\S+", text)
        ),

        "has_attachment_mention": bool(
            re.search(r"attach(ed|ment)", text_lower)
        ),

        "excessive_punctuation": bool(
            re.search(r"[!?]{2,}", text)
        ),

        "generic_greeting": bool(
            re.search(
                r"^(dear (customer|user|sir/madam)|to whom it may concern)",
                text_lower
            )
        ),
    }


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_TYPE = None

distilbert_model = None
distilbert_tokenizer = None

baseline_bundle = None


try:
    import torch

    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
    )

    # --------------------------------------------------------
    # Prefer local V4 model
    # --------------------------------------------------------

    if DISTILBERT_PATH.exists():

        model_source = str(DISTILBERT_PATH)

        print(
            f"Loading DistilBERT V4 from local folder: "
            f"{model_source}"
        )

    else:

        # ----------------------------------------------------
        # Fallback to HuggingFace
        # ----------------------------------------------------

        model_source = HF_MODEL_REPO

        print(
            "Local DistilBERT V4 folder not found. "
            f"Downloading from HuggingFace Hub: {HF_MODEL_REPO}"
        )

    # --------------------------------------------------------
    # Load tokenizer + model
    # --------------------------------------------------------

    distilbert_tokenizer = (
        DistilBertTokenizerFast.from_pretrained(
            model_source
        )
    )

    distilbert_model = (
        DistilBertForSequenceClassification.from_pretrained(
            model_source
        )
    )

    distilbert_model.eval()

    MODEL_TYPE = "distilbert"

    print("Loaded DistilBERT model successfully.")


except Exception as e:

    print(
        f"DistilBERT load failed or unavailable ({e}). "
        "Falling back to baseline model."
    )

    baseline_bundle = joblib.load(
        BASELINE_PATH
    )

    MODEL_TYPE = "baseline"

    print(
        "Loaded baseline "
        "(TF-IDF + Gradient Boosting) model successfully."
    )


# ============================================================
# DISTILBERT PREDICTION
# ============================================================

def predict_distilbert(text: str):

    import torch

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    inputs = distilbert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = distilbert_model(
            **inputs
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    pred_idx = int(
        torch.argmax(probs)
    )

    confidence = float(
        probs[pred_idx]
    )

    label = distilbert_model.config.id2label[
        pred_idx
    ]

    # --------------------------------------------------------
    # Individual class probabilities
    # --------------------------------------------------------

    probabilities = {
        "legitimate": float(probs[0]),
        "phishing": float(probs[1]),
        "bec": float(probs[2]),
    }

    return (
        label,
        confidence,
        probabilities,
    )


# ============================================================
# BASELINE PREDICTION
# ============================================================

def predict_baseline(text: str):

    vectorizer = baseline_bundle[
        "vectorizer"
    ]

    model = baseline_bundle[
        "model"
    ]

    X = vectorizer.transform(
        [text]
    )

    proba = model.predict_proba(
        X
    )[0]

    pred_idx = proba.argmax()

    confidence = float(
        proba[pred_idx]
    )

    label = model.classes_[
        pred_idx
    ]

    probabilities = {
        "legitimate": 0.0,
        "phishing": 0.0,
        "bec": 0.0,
    }

    # Fill probabilities according to
    # the baseline model's class ordering
    for class_name, probability in zip(
        model.classes_,
        proba
    ):
        if class_name in probabilities:
            probabilities[class_name] = float(
                probability
            )

    return (
        label,
        confidence,
        probabilities,
    )


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class EmailInput(BaseModel):

    subject: str
    body: str
    sender: Optional[str] = None
    metadata: Optional[dict] = None


class ClassificationOutput(BaseModel):

    classification: str

    confidence_score: float

    model_probabilities: dict

    matched_indicators: dict

    processed_at: str


class CaseAnchorRequest(BaseModel):

    case_id: str
    case_document: dict


class CaseVerifyRequest(BaseModel):

    case_id: str
    case_document: dict


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return RedirectResponse(
        url="/docs"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

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


# ============================================================
# EMAIL CLASSIFICATION
# ============================================================

@app.post(
    "/classify",
    response_model=ClassificationOutput
)
def classify_email(
    email: EmailInput
):

    # --------------------------------------------------------
    # Combine subject + body
    # --------------------------------------------------------

    raw_text = (
        f"{email.subject} {email.body}"
    )

    text = clean_text(
        raw_text
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if MODEL_TYPE == "distilbert":

        (
            label,
            confidence,
            probabilities,
        ) = predict_distilbert(
            text
        )

    else:

        (
            label,
            confidence,
            probabilities,
        ) = predict_baseline(
            text
        )

    # --------------------------------------------------------
    # Confidence-band override
    #
    # Low-confidence phishing/BEC becomes suspicious.
    # --------------------------------------------------------

    if (
        label in ("phishing", "bec")
        and confidence < SUSPICIOUS_THRESHOLD
    ):

        label = "suspicious"

    # --------------------------------------------------------
    # Rule-based indicators
    # --------------------------------------------------------

    indicators = extract_indicators(
        raw_text
    )

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return ClassificationOutput(

        classification=label,

        confidence_score=round(
            confidence * 100,
            1
        ),

        model_probabilities={
            key: round(
                value * 100,
                2
            )
            for key, value in probabilities.items()
        },

        matched_indicators=indicators,

        processed_at=datetime.now(
            timezone.utc
        ).isoformat(),
    )


# ============================================================
# BLOCKCHAIN - ANCHOR
# ============================================================

@app.post("/anchor")
def anchor_case_endpoint(
    request: CaseAnchorRequest
):

    try:

        result = anchor_case(
            request.case_id,
            request.case_document
        )

        return result

    except ValueError as e:

        # Case already anchored on-chain
        raise HTTPException(
            status_code=409,
            detail=str(e)
        )

    except ConnectionError as e:

        # RPC endpoint unreachable
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except RuntimeError as e:

        # Transaction reverted on-chain
        raise HTTPException(
            status_code=502,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {e}"
        )


# ============================================================
# BLOCKCHAIN - VERIFY
# ============================================================

@app.post("/verify")
def verify_case_endpoint(
    request: CaseVerifyRequest
):

    try:

        result = verify_case(
            request.case_id,
            request.case_document
        )

        return result

    except ConnectionError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {e}"
        )


# ============================================================
# ANALYTICS
# ============================================================

@app.get("/analytics/trends")
def get_analytics_trends(
    months_back: int = 6,
    risk_threshold: float = 0.7,
):

    """
    Returns a combined analytics snapshot for the jury demo
    dashboard:

    - Weekly phishing/BEC volume by country
    - Fastest-growing campaigns
    - High-risk infrastructure breakdown

    Reads directly from the MongoDB `cases` collection.
    """

    try:

        return {

            "volume_by_country_weekly":
                phishing_bec_volume_by_country_weekly(
                    months_back=months_back
                ),

            "fastest_growing_campaigns":
                fastest_growing_campaigns(),

            "high_risk_infrastructure_breakdown":
                high_risk_infrastructure_breakdown(
                    risk_threshold=risk_threshold
                ),
        }

    except RuntimeError as e:

        # e.g. MONGO_URI missing from .env
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Analytics query failed: {e}"
        )