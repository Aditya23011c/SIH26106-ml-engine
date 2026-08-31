"""
train.py - trains TF-IDF + Gradient Boosting baseline classifier
on the 200k merged training_set.csv (legitimate + phishing + BEC),
with class balancing to handle any remaining imbalance.

Run from project root: python models/train.py
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/training_set.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

LABELS = ["legitimate", "phishing", "bec"]


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    print("Loaded dataset, shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.15,
        stratify=df["label"],
        random_state=42,
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        max_features=15000,   # increased from 8000 since we have far more data now
        ngram_range=(1, 2),
        min_df=3,             # slightly higher to filter noise from a much bigger corpus
        stop_words="english",
    )
    print("\nFitting TF-IDF vectorizer (this may take a few minutes on 170k+ rows)...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=1,   # prints progress since this dataset is much bigger now
    )

    print("\nTraining model... (this WILL take longer than before - 200k rows vs 18k. "
          "Expect 20-45+ minutes on CPU depending on your machine.)")
    clf.fit(X_train_vec, y_train, sample_weight=sample_weights)
    print("Training complete.")

    y_pred = clf.predict(X_test_vec)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, labels=LABELS, target_names=LABELS))

    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    print("CONFUSION MATRIX")
    print("(rows = actual, columns = predicted)")
    print(cm_df)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Email Fraud Detection Baseline (200k dataset)")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\nSaved confusion matrix image to {MODEL_DIR / 'confusion_matrix.png'}")

    joblib.dump(
        {"vectorizer": vectorizer, "model": clf, "labels": LABELS},
        MODEL_DIR / "baseline_model.joblib",
    )
    print(f"\nSaved model bundle to {MODEL_DIR / 'baseline_model.joblib'}")


if __name__ == "__main__":
    main()