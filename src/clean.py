"""
clean.py - loads raw phishing dataset + synthetic BEC data,
cleans text, merges everything, saves final processed CSV.
Run from project root: python src/clean.py
"""
import re
import os
import pandas as pd

RAW_PATH = "data/raw/Phishing_Email.csv"
BEC_PATH = "data/processed/bec_synthetic.csv"
OUT_PATH = "data/processed/labeled_emails.csv"


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+\.\S+", " EMAILTOKEN ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_phishing_legitimate():
    """Load and clean the original legitimate/phishing dataset."""
    df = pd.read_csv(RAW_PATH)
    print("Loaded raw CSV, shape:", df.shape)

    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c], errors="ignore")
    df = df.rename(columns={"Email Text": "text", "Email Type": "label_raw"})
    df = df.dropna(subset=["text"])

    label_map = {
        "Safe Email": "legitimate",
        "Phishing Email": "phishing",
    }
    df["label"] = df["label_raw"].map(label_map)
    df = df.dropna(subset=["label"])

    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0]

    return df[["text_clean", "label"]].rename(columns={"text_clean": "text"})


def load_bec():
    """Load and clean the synthetic BEC dataset (has separate subject/body)."""
    if not os.path.exists(BEC_PATH):
        print(f"WARNING: {BEC_PATH} not found, skipping BEC examples.")
        return pd.DataFrame(columns=["text", "label"])

    df = pd.read_csv(BEC_PATH)
    print("Loaded BEC CSV, shape:", df.shape)

    # combine subject + body into a single text field to match the main schema
    df["combined"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["text_clean"] = df["combined"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0]
    df["label"] = "bec"

    return df[["text_clean", "label"]].rename(columns={"text_clean": "text"})


def main():
    print("Working directory:", os.getcwd())
    print("Looking for raw file at:", os.path.abspath(RAW_PATH))
    print("Raw file exists?", os.path.exists(RAW_PATH))
    print("Looking for BEC file at:", os.path.abspath(BEC_PATH))
    print("BEC file exists?", os.path.exists(BEC_PATH))
    print("Processed folder exists?", os.path.exists("data/processed"))
    print()

    legit_phish = load_phishing_legitimate()
    bec = load_bec()

    # merge both sources
    out = pd.concat([legit_phish, bec], ignore_index=True)

    # drop exact duplicate texts (can happen across sources)
    before = len(out)
    out = out.drop_duplicates(subset=["text"])
    after = len(out)
    if before != after:
        print(f"Dropped {before - after} duplicate rows after merge.")

    # shuffle so classes aren't grouped in blocks
    out = out.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nFinal merged label distribution:")
    print(out["label"].value_counts())
    print(f"\nTotal rows: {len(out)}")

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved merged dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()