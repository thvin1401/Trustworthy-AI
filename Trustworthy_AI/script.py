"""Fairness audit for the German Credit dataset.

This script loads the German Credit data, applies the saved XGBoost model,
then computes simple fairness indicators by gender.
"""

from pathlib import Path
import pickle
import zipfile

import pandas as pd
from fairlearn.metrics import demographic_parity_difference


# Keep all file references relative to this script so it works no matter
# where the command is launched from.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = BASE_DIR / "statlog+german+credit+data.zip"
DATA_PATH = DATA_DIR / "german.data"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
MODEL_PATH = BASE_DIR / "german_credit_model.pkl"

GERMAN_CREDIT_COLUMNS = [
    "Attribute1",
    "Attribute2",
    "Attribute3",
    "Attribute4",
    "Attribute5",
    "Attribute6",
    "Attribute7",
    "Attribute8",
    "Attribute9",
    "Attribute10",
    "Attribute11",
    "Attribute12",
    "Attribute13",
    "Attribute14",
    "Attribute15",
    "Attribute16",
    "Attribute17",
    "Attribute18",
    "Attribute19",
    "Attribute20",
    "target",
]

# Personal status and sex codes from the German Credit dataset description.
GENDER_MAPPING = {
    "A91": "Male",
    "A92": "Female",
    "A93": "Male",
    "A94": "Male",
    "A95": "Female",
}


def ensure_dataset_extracted() -> None:
    """Extract the dataset if the expected file does not exist yet."""
    if DATA_PATH.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)


def load_dataset() -> pd.DataFrame:
    """Load the German Credit dataset and normalize the target label."""
    ensure_dataset_extracted()

    df = pd.read_csv(DATA_PATH, sep=" ", header=None)
    df.columns = GERMAN_CREDIT_COLUMNS

    # Original labels: 1 = good credit, 2 = bad credit.
    # For binary classification we map them to 1 and 0.
    df["target"] = df["target"].map({1: 1, 2: 0})

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode features and align columns with the training schema."""
    df_encoded = pd.get_dummies(df.drop("target", axis=1))

    with open(FEATURE_COLUMNS_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    # The inference input must match the exact feature layout used at training time.
    for col in feature_cols:
        if col not in df_encoded:
            df_encoded[col] = 0

    return df_encoded[feature_cols]


def load_model():
    """Load the saved XGBoost model artifact."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def add_gender_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a clean gender column from Attribute9."""
    df = df.copy()
    df["gender"] = df["Attribute9"].map(GENDER_MAPPING)

    if df["gender"].isna().any():
        unknown_codes = sorted(df.loc[df["gender"].isna(), "Attribute9"].unique())
        raise ValueError(f"Unknown gender codes found in Attribute9: {unknown_codes}")

    return df


def main() -> None:
    # ================================
    # 1. LOAD DATASET
    # ================================
    df = load_dataset()

    # ================================
    # 2. ENCODE FEATURES
    # ================================
    X = build_feature_matrix(df)

    # ================================
    # 3. LOAD FEATURE COLUMNS
    # ================================
    # Feature-column alignment is handled inside build_feature_matrix().

    # ================================
    # 4. LOAD MODEL & PREDICT
    # ================================
    model = load_model()
    df["y_pred"] = model.predict(X)

    # ================================
    # 5. EXTRACT GENDER
    # ================================
    df = add_gender_column(df)

    # ================================
    # 6. CALCULATE APPROVAL RATE
    # ================================
    approval_rate = df.groupby("gender")["y_pred"].mean()
    rate_male = approval_rate["Male"]
    rate_female = approval_rate["Female"]

    print("=== APPROVAL RATE ===")
    print(f"Male   : {rate_male:.3f}")
    print(f"Female : {rate_female:.3f}")

    # ================================
    # 7. DEMOGRAPHIC PARITY
    # ================================
    # Demographic parity focuses on whether approval rates differ by group.
    dp = demographic_parity_difference(
        y_true=df["target"],
        y_pred=df["y_pred"],
        sensitive_features=df["gender"],
    )

    print("\n=== DEMOGRAPHIC PARITY ===")
    print(f"DP Difference: {dp:.3f}")

    # ================================
    # 8. DISPARATE IMPACT
    # ================================
    # Disparate impact is the ratio between the protected group's approval rate
    # and the reference group's approval rate.
    dir_ratio = rate_female / rate_male

    print("\n=== DISPARATE IMPACT ===")
    print(f"DIR: {dir_ratio:.3f}")

    # ================================
    # 9. CONCLUSION
    # ================================
    print("\n=== CONCLUSION ===")

    if abs(dp) > 0.1:
        print("-> Model shows BIAS based on Demographic Parity (DP > 0.1)")
    else:
        print("-> Model is relatively FAIR (DP ~= 0)")

    if dir_ratio < 0.8:
        print("-> Potential discrimination (DIR < 0.8 - 80% rule violated)")
    else:
        print("-> No strong discrimination signal based on DIR")


if __name__ == "__main__":
    main()
