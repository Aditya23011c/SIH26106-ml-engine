"""
build_training_set.py - merges multiple raw phishing/legitimate CSVs
(different schemas) plus our synthetic BEC data into one unified
50k+ row training set.

Handles the common column-name variants seen across the Nazario /
SpamAssassin / Enron / CEAS_08 / Nigerian_Fraud / Ling corpora that
ship as separate CSVs in the Kaggle "Phishing Email Dataset" bundle.

Run from project root: python models/build_training_set.py
"""
import re
import glob
import pandas as pd
from pathlib import Path

RAW_KAGGLE_DIR = "data/raw_kaggle"
BEC_PATH = "data/bec_synthetic.csv"
OLD_LABELED_PATH = "data/training_set.csv"  # if it already exists from before, we still fold it in
OUT_PATH = "data/training_set.csv"

# Column name variants seen across these corpora - checked in order
TEXT_COL_CANDIDATES = [
    "text", "Email Text", "body", "Body", "text_combined", "message", "content",
]
SUBJECT_COL_CANDIDATES = ["subject", "Subject"]
LABEL_COL_CANDIDATES = [
    "label", "Label", "Email Type", "class", "Class", "type", "target",
]

# Label value normalization - maps whatever the raw dataset calls things
# into our three canonical labels. Anything not matched is dropped.
LABEL_VALUE_MAP = {
    # legitimate variants
    "safe email": "legitimate", "ham": "legitimate", "legitimate": "legitimate",
    "0": "legitimate", 0: "legitimate", "not spam": "legitimate", "normal": "legitimate",
    # phishing variants
    "phishing email": "phishing", "phishing": "phishing", "spam": "phishing",
    "1": "phishing", 1: "phishing", "fraud": "phishing", "nigerian fraud": "phishing",
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " URLTOKEN ", text)
    text = re.sub(r"\S+@\S+\.\S+", " EMAILTOKEN ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_one_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as e:
        print(f"  Skipping {path}: could not read ({e})")
        return pd.DataFrame(columns=["text", "label"])

    text_col = find_column(df, TEXT_COL_CANDIDATES)
    subject_col = find_column(df, SUBJECT_COL_CANDIDATES)
    label_col = find_column(df, LABEL_COL_CANDIDATES)

    if text_col is None or label_col is None:
        print(f"  Skipping {path}: could not find text/label columns "
              f"(found columns: {list(df.columns)})")
        return pd.DataFrame(columns=["text", "label"])

    out = pd.DataFrame()
    if subject_col:
        out["text"] = df[subject_col].fillna("").astype(str) + " " + df[text_col].fillna("").astype(str)
    else:
        out["text"] = df[text_col].fillna("").astype(str)

    # normalize label values (case-insensitive match against our map)
    def map_label(v):
        if pd.isna(v):
            return None
        key = str(v).strip().lower()
        return LABEL_VALUE_MAP.get(key, LABEL_VALUE_MAP.get(v, None))

    out["label"] = df[label_col].apply(map_label)
    out = out.dropna(subset=["label"])
    print(f"  Loaded {path}: {len(out)} usable rows -> {out['label'].value_counts().to_dict()}")
    return out


def load_bec():
    if not Path(BEC_PATH).exists():
        print(f"WARNING: {BEC_PATH} not found, skipping BEC examples.")
        return pd.DataFrame(columns=["text", "label"])
    df = pd.read_csv(BEC_PATH)
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    df["label"] = "bec"
    return df[["text", "label"]]


def main():
    all_dfs = []

    csv_files = glob.glob(f"{RAW_KAGGLE_DIR}/*.csv")
    if not csv_files:
        print(f"No CSVs found in {RAW_KAGGLE_DIR}/ — make sure you extracted the Kaggle download there.")
        return

    print(f"Found {len(csv_files)} raw CSVs to merge:\n")
    for path in csv_files:
        all_dfs.append(load_one_csv(path))

    bec_df = load_bec()
    print(f"\nLoaded BEC synthetic: {len(bec_df)} rows")
    all_dfs.append(bec_df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged["text"] = merged["text"].apply(clean_text)
    merged = merged[merged["text"].str.len() > 10]  # drop near-empty rows

    before = len(merged)
    merged = merged.drop_duplicates(subset=["text"])
    print(f"\nDropped {before - len(merged)} duplicates after merge.")

    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n" + "=" * 50)
    print("FINAL MERGED DATASET")
    print("=" * 50)
    print(merged["label"].value_counts())
    print(f"\nTotal rows: {len(merged)}")

    if len(merged) < 50000:
        print(f"\n⚠️  WARNING: total is {len(merged)}, below the 50k target. "
              f"Check that all Kaggle CSVs were extracted into {RAW_KAGGLE_DIR}/")

    merged.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()