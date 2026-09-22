"""
models/calibrate_confidence.py

Temperature scaling calibration for the fine-tuned DistilBERT model.

WHY THIS EXISTS:
The model is often overconfident (99.9%) on emails it actually gets wrong -
e.g. genuine security-alert/order-confirmation emails classified as
phishing/BEC with near-certainty. This does NOT retrain or change what the
model predicts (the label/argmax stays exactly the same) - it only rescales
the confidence numbers so they honestly reflect how sure the model really
is. After calibration, a case the model is genuinely unsure about will
show a lower confidence score, which lets the EXISTING "suspicious"
threshold logic in main.py catch it automatically instead of confidently
mislabeling it.

This is a standard technique called temperature scaling (Guo et al., 2017).
It fits a single scalar T on a held-out set by minimizing negative log
likelihood, then every future prediction does: softmax(logits / T) instead
of softmax(logits). T > 1 makes the model less confident (spreads out the
probability distribution); T = 1 changes nothing.

Run from project root (with the venv activated):
    python models/calibrate_confidence.py \
        --model_path models/distilbert-phishing-final-v4 \
        --holdout_csv calibration_holdout_set.csv

Requires: the holdout CSV must NOT have been used in training (that would
bias the calibration and defeat the purpose). See calibration_holdout_set.csv
for one already built from examples that were only ever used for manual
testing, never added to training_set.csv.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


def get_logits(model, tokenizer, texts, max_length=256, batch_size=16):
    all_logits = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", truncation=True,
                padding="max_length", max_length=max_length,
            )
            outputs = model(**inputs)
            all_logits.append(outputs.logits)
    return torch.cat(all_logits, dim=0)


def nll_for_temperature(T, logits, true_idx):
    scaled = logits / T
    log_probs = torch.log_softmax(scaled, dim=1)
    nll = -log_probs[torch.arange(len(true_idx)), true_idx].mean()
    return nll.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/distilbert-phishing-final-v4")
    parser.add_argument("--holdout_csv", type=str, default="calibration_holdout_set.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--label_col", type=str, default="label")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model folder not found: {model_path}")

    holdout_path = Path(args.holdout_csv)
    if not holdout_path.exists():
        raise SystemExit(f"Holdout CSV not found: {holdout_path}")

    print(f"Loading model from {model_path} ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
    label2id = model.config.label2id
    print(f"Model label2id: {label2id}")

    df = pd.read_csv(holdout_path)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(f"CSV must have columns '{args.text_col}' and '{args.label_col}'.")

    unknown_labels = set(df[args.label_col].unique()) - set(label2id.keys())
    if unknown_labels:
        raise SystemExit(f"Holdout CSV has labels not in the model's label set: {unknown_labels}")

    texts = df[args.text_col].tolist()
    true_idx = torch.tensor([label2id[l] for l in df[args.label_col]])

    print(f"Computing logits for {len(texts)} held-out examples...")
    logits = get_logits(model, tokenizer, texts)

    # --- Accuracy and confidence BEFORE calibration ---
    probs_before = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs_before, dim=1)
    acc = (preds == true_idx).float().mean().item()
    avg_conf_before = probs_before.max(dim=1).values.mean().item()
    print(f"\nHeld-out accuracy: {acc:.3f}")
    print(f"Average confidence BEFORE calibration: {avg_conf_before*100:.1f}%")

    # --- Fit temperature by minimizing NLL ---
    result = minimize_scalar(
        lambda T: nll_for_temperature(T, logits, true_idx),
        bounds=(0.5, 10.0), method="bounded",
    )
    T_optimal = result.x
    print(f"\nOptimal temperature: {T_optimal:.3f}")

    probs_after = torch.softmax(logits / T_optimal, dim=1)
    avg_conf_after = probs_after.max(dim=1).values.mean().item()
    print(f"Average confidence AFTER calibration: {avg_conf_after*100:.1f}%")
    print("(Accuracy/labels are unchanged by calibration - only the confidence numbers shift.)")

    # --- Show exactly what calibration did to the misclassified examples ---
    # This is the part that actually matters: does calibration push the
    # WRONG predictions below the 0.7 "suspicious" threshold, or does it
    # just lower confidence uniformly across the board (including cases
    # that were already correct)?
    id2label = model.config.id2label
    wrong_mask = (preds != true_idx)
    if wrong_mask.any():
        print(f"\n{wrong_mask.sum().item()} misclassified example(s) - before vs after calibration:")
        print("-" * 80)
        for i in torch.where(wrong_mask)[0].tolist():
            true_label = id2label[true_idx[i].item()]
            pred_label = id2label[preds[i].item()]
            conf_before = probs_before[i, preds[i]].item()
            conf_after = probs_after[i, preds[i]].item()
            below_threshold = " -> now below 0.7, becomes 'suspicious'" if conf_after < 0.7 else " -> STILL above 0.7, stays wrongly confident"
            print(f"  True: {true_label:12s} | Predicted: {pred_label:12s} | "
                  f"Before: {conf_before*100:5.1f}% | After: {conf_after*100:5.1f}%{below_threshold}")
            print(f"    Text: {texts[i][:100]}...")
        print("-" * 80)
    else:
        print("\nNo misclassified examples in this holdout set.")

    # --- Save calibration alongside the model ---
    calibration_path = model_path / "calibration.json"
    with open(calibration_path, "w") as f:
        json.dump({"temperature": T_optimal}, f, indent=2)
    print(f"\nSaved calibration temperature to {calibration_path}")
    print("Next step: update predict_distilbert() in main.py to load and apply this temperature.")


if __name__ == "__main__":
    main()