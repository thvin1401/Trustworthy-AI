# ================================
# FAIRNESS AUDIT - GERMAN CREDIT
# ================================

import pandas as pd
import pickle
import zipfile
from fairlearn.metrics import demographic_parity_difference

# ================================
# 1. LOAD DATASET
# ================================

zip_path = "statlog+german+credit+data.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("data")

df = pd.read_csv("data/german.data", sep=" ", header=None)

# Column names
columns = [
    "Attribute1","Attribute2","Attribute3","Attribute4","Attribute5",
    "Attribute6","Attribute7","Attribute8","Attribute9","Attribute10",
    "Attribute11","Attribute12","Attribute13","Attribute14","Attribute15",
    "Attribute16","Attribute17","Attribute18","Attribute19","Attribute20",
    "target"
]

df.columns = columns

# Convert target: 1 = good (approve), 2 = bad (reject)
df["target"] = df["target"].map({1: 1, 2: 0})

# ================================
# 2. ENCODE FEATURES
# ================================

df_encoded = pd.get_dummies(df.drop("target", axis=1))

# ================================
# 3. LOAD FEATURE COLUMNS
# ================================

with open("feature_columns.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Add missing columns
for col in feature_cols:
    if col not in df_encoded:
        df_encoded[col] = 0

# Ensure correct order
X = df_encoded[feature_cols]

# ================================
# 4. LOAD MODEL & PREDICT
# ================================

with open("german_credit_model.pkl", "rb") as f:
    model = pickle.load(f)

df["y_pred"] = model.predict(X)

# ================================
# 5. EXTRACT GENDER
# ================================

def get_gender(val):
    return "Female" if val == "A92" else "Male"

df["gender"] = df["Attribute9"].apply(get_gender)

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

dp = demographic_parity_difference(
    y_true=df["target"],
    y_pred=df["y_pred"],
    sensitive_features=df["gender"]
)

print("\n=== DEMOGRAPHIC PARITY ===")
print(f"DP Difference: {dp:.3f}")

# ================================
# 8. DISPARATE IMPACT
# ================================

dir_ratio = rate_female / rate_male

print("\n=== DISPARATE IMPACT ===")
print(f"DIR: {dir_ratio:.3f}")

# ================================
# 9. CONCLUSION
# ================================

print("\n=== CONCLUSION ===")

if abs(dp) > 0.1:
    print("→ Model shows BIAS based on Demographic Parity (DP > 0.1)")
else:
    print("→ Model is relatively FAIR (DP ≈ 0)")

if dir_ratio < 0.8:
    print("→ Potential discrimination (DIR < 0.8 - 80% rule violated)")
else:
    print("→ No strong discrimination signal based on DIR")

# ================================
# END
# ================================