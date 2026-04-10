import os
import re
import pickle
import zipfile
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ================================
# 0. Global style
# ================================
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 240
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.grid"] = False

OUTPUT_DIR = "output_readable"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
BACKGROUND_SIZE = 200
TOP_N_GLOBAL = 15
TOP_N_LOCAL = 15
TOP_N_DEPENDENCE = 3

# ================================
# 1. Feature Mapping Logic
# ================================

COLUMN_MAPPING = {
    "Attribute1": "Checking_Status",
    "Attribute2": "Duration_Months",
    "Attribute3": "Credit_History",
    "Attribute4": "Purpose",
    "Attribute5": "Credit_Amount",
    "Attribute6": "Savings",
    "Attribute7": "Employment_Duration",
    "Attribute8": "Installment_Rate",
    "Attribute9": "Personal_Status_Sex",
    "Attribute10": "Guarantors",
    "Attribute11": "Residence_Duration",
    "Attribute12": "Property",
    "Attribute13": "Age",
    "Attribute14": "Other_Plans",
    "Attribute15": "Housing",
    "Attribute16": "Existing_Credits_Count",
    "Attribute17": "Job",
    "Attribute18": "Dependents",
    "Attribute19": "Telephone",
    "Attribute20": "Foreign_Worker",
}

VALUE_MAPPING = {
    "A11": "< 0 DM",
    "A12": "0-200 DM",
    "A13": "> 200 DM",
    "A14": "No account",
    "A30": "No credits",
    "A31": "All paid",
    "A32": "Paid till now",
    "A33": "Past delay",
    "A34": "Critical account",
    "A40": "New car",
    "A41": "Used car",
    "A42": "Furniture",
    "A43": "Radio/TV",
    "A44": "Appliances",
    "A45": "Repairs",
    "A46": "Education",
    "A49": "Business",
    "A61": "< 100 DM",
    "A62": "100-500 DM",
    "A63": "500-1000 DM",
    "A64": "> 1000 DM",
    "A65": "No savings",
    "A71": "Unemployed",
    "A72": "< 1yr",
    "A73": "1-4yrs",
    "A74": "4-7yrs",
    "A75": "> 7yrs",
    "A91": "Male:Divorced",
    "A92": "Female:Married/Div",
    "A93": "Male:Single",
    "A94": "Male:Married",
    "A95": "Female:Single",
    "A101": "None",
    "A102": "Co-applicant",
    "A103": "Guarantor",
    "A121": "Real estate",
    "A122": "Life insurance",
    "A123": "Car/Other",
    "A124": "No property",
    "A141": "Bank",
    "A142": "Stores",
    "A143": "None",
    "A151": "Rent",
    "A152": "Own",
    "A153": "Free",
    "A171": "Unskilled",
    "A172": "Unskilled Res",
    "A173": "Skilled",
    "A174": "Management",
    "A191": "None",
    "A192": "Yes",
    "A201": "Yes",
    "A202": "No",
}


def get_readable_name(col):
    read_col = col
    # Sort mapping keys by length descending to match Attribute10 before Attribute1
    sorted_attr_keys = sorted(COLUMN_MAPPING.keys(), key=len, reverse=True)

    # Replace base attribute part
    for attr in sorted_attr_keys:
        if col.startswith(attr):
            read_col = col.replace(attr, COLUMN_MAPPING[attr])
            break

    # Replace value part (e.g., _A14)
    # Handle cases like Attribute1_A14 or Attribute1A14 depending on how dummies were made
    for val, meaning in VALUE_MAPPING.items():
        if f"_{val}" in read_col:
            read_col = read_col.replace(f"_{val}", f"_{meaning}")
        elif read_col.endswith(val) and any(
            read_col.startswith(v) for v in COLUMN_MAPPING.values()
        ):
            # Fallback for some naming conventions
            read_col = read_col.replace(val, f"_{meaning}")

    return read_col


# ================================
# 2. Helper functions
# ================================
def clean_filename(name):
    name = str(name)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120]


def save_current_fig(path, width=12, height=7):
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()


def get_gender(attr9):
    return "Female" if attr9 in ("A92", "A95") else "Male"


def normalize_shap_output(explanation):
    values = explanation.values
    base_values = explanation.base_values

    if len(np.array(values).shape) == 3:
        values = values[:, :, 1]

        base_arr = np.array(base_values)
        if len(base_arr.shape) == 2:
            base_values = base_arr[:, 1]
        elif len(base_arr.shape) == 1 and len(base_arr) == 2:
            base_values = np.repeat(base_arr[1], values.shape[0])

        explanation = shap.Explanation(
            values=values,
            base_values=base_values,
            data=explanation.data,
            feature_names=explanation.feature_names,
        )

    return explanation


def select_representative_cases(df):
    approved_candidates = df[df["y_pred"] == 1]
    rejected_candidates = df[df["y_pred"] == 0]

    if len(approved_candidates) > 0:
        approved_idx = approved_candidates["y_proba"].idxmax()
    else:
        approved_idx = df["y_proba"].idxmax()

    if len(rejected_candidates) > 0:
        rejected_idx = rejected_candidates["y_proba"].idxmin()
    else:
        rejected_idx = df["y_proba"].idxmin()

    borderline_idx = (df["y_proba"] - 0.5).abs().idxmin()

    return {
        "approved": int(approved_idx),
        "rejected": int(rejected_idx),
        "borderline": int(borderline_idx),
    }


def create_local_contribution_table(X, shap_values, idx, case_name):
    local_df = pd.DataFrame(
        {
            "case": case_name,
            "feature": X.columns,
            "feature_value": X.iloc[idx].values,
            "shap_value": shap_values.values[idx],
            "abs_shap": np.abs(shap_values.values[idx]),
        }
    ).sort_values("abs_shap", ascending=False)

    local_df["direction"] = np.where(
        local_df["shap_value"] >= 0, "push_up", "push_down"
    )
    return local_df


def write_text_report(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def markdown_table_from_df(df, columns=None, max_rows=None):
    if columns is not None:
        df = df[columns].copy()
    if max_rows is not None:
        df = df.head(max_rows).copy()

    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = []
        for col in headers:
            val = row[col]
            if isinstance(val, (float, np.float64, np.float32)):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


# ================================
# 3. Load dataset
# ================================
zip_path = os.path.join(os.path.dirname(__file__), "statlog+german+credit+data.zip")

if not os.path.exists("data/german.data"):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data")

df = pd.read_csv("data/german.data", sep=" ", header=None)

original_cols = [
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
df.columns = original_cols

df["target"] = df["target"].map({1: 1, 2: 0})
df["gender"] = df["Attribute9"].apply(get_gender)

# ================================
# 4. Preprocess
# ================================
X_raw = df.drop(["target", "gender"], axis=1)
X_encoded = pd.get_dummies(X_raw)

with open(os.path.join(os.path.dirname(__file__), "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)

for col in feature_columns:
    if col not in X_encoded.columns:
        X_encoded[col] = 0

X = X_encoded[feature_columns].copy()
X = X.fillna(0).astype(np.float64)

# Apply Human-Readable mapping to X columns
readable_cols = [get_readable_name(c) for c in X.columns]
X.columns = readable_cols

print("X shape:", X.shape)
print("X dtype:", X.to_numpy().dtype)

# ================================
# 5. Load model
# ================================
with open(
    os.path.join(os.path.dirname(__file__), "german_credit_model.pkl"), "rb"
) as f:
    model = pickle.load(f)

# ================================
# 6. Predict
# ================================
y_pred = model.predict(
    X_encoded[feature_columns].fillna(0).astype(np.float64)
)  # Use original encoded names for model

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(
        X_encoded[feature_columns].fillna(0).astype(np.float64)
    )[:, 1]
else:
    y_proba = y_pred.astype(float)

df["y_pred"] = y_pred
df["y_proba"] = y_proba

# ================================
# 8. Build SHAP explainer
# ================================
# We use original X_encoded for the model but readable for the explanation display
X_for_model = X_encoded[feature_columns].fillna(0).astype(np.float64)

if len(X_for_model) > BACKGROUND_SIZE:
    background = shap.sample(X_for_model, BACKGROUND_SIZE, random_state=RANDOM_STATE)
else:
    background = X_for_model.copy()

try:
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional"
    )
    shap_values = explainer(X_for_model)
except Exception:
    explainer = shap.Explainer(model, background)
    shap_values = explainer(X_for_model)

shap_values = normalize_shap_output(shap_values)

# Inject readable names into SHAP explanation
shap_values.feature_names = readable_cols

# ================================
# 8. Global importance tables
# ================================
importance_df = pd.DataFrame(
    {
        "feature": X.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        "mean_shap": shap_values.values.mean(axis=0),
    }
)

# Proxy Analysis: Correlation with Gender
valid_cols = [c for c in X_encoded.columns if c in feature_columns]
correlations = (
    X_encoded[valid_cols].corrwith((df["gender"] == "Female").astype(int)).abs()
)
importance_df["gender_corr"] = correlations.reindex(feature_columns).fillna(0).values

# Now we can safely sort
importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
importance_df["rank"] = np.arange(1, len(importance_df) + 1)
importance_df = importance_df[
    ["rank", "feature", "mean_abs_shap", "mean_shap", "gender_corr"]
]

# ================================
# 9.5 Fairness Audit (from diagnose_bias.py)
# ================================
approval_data = df.groupby("gender")["y_pred"].mean().reset_index()
male_mask = approval_data["gender"] == "Male"
female_mask = approval_data["gender"] == "Female"

male_rate = float(approval_data.loc[male_mask, "y_pred"].values[0])
female_rate = float(approval_data.loc[female_mask, "y_pred"].values[0])
dir_val = female_rate / male_rate if male_rate > 0 else 0
# Calculate Bias Potential: Mean Abs SHAP * Gender Correlation
importance_df["bias_potential"] = (
    importance_df["mean_abs_shap"] * importance_df["gender_corr"]
)
top_bias_drivers = (
    importance_df.sort_values(by="bias_potential", ascending=False).head(5).copy()
)

importance_df.to_csv(f"{OUTPUT_DIR}/01_shap_importance.csv", index=False)

top10 = importance_df.head(10).copy()
top_features = list(importance_df["feature"].iloc[:TOP_N_DEPENDENCE].values)

print("\nTop 10 feature quan trọng nhất (Readable):")
print(top10.drop(columns=["bias_potential", "mean_shap"], errors="ignore"))

# ================================
# 10. Fairness and Global plots
# ================================
# Subplot: Approval Rate by Gender
sns.barplot(data=approval_data, x="gender", y="y_pred", palette="muted")
plt.title("Approval Rate by Gender")
plt.ylim(0, 1.1)
plt.ylabel("Approval Rate")
plt.tight_layout()
save_current_fig(f"{OUTPUT_DIR}/11_approval_rates.png", width=6, height=6)

# PLOT 12: Disparate Impact Ratio (DIR)
# Normalize relative to the majority group (Male)
dir_ratio_df = pd.DataFrame(
    {
        "Gender": ["Male (Reference)", "Female"],
        "DIR Ratio": [1.0, female_rate / male_rate if male_rate > 0 else 0],
    }
)
plt.figure(figsize=(10, 6))
ax = sns.barplot(x="Gender", y="DIR Ratio", data=dir_ratio_df, palette="viridis")
plt.title("Disparate Impact Ratio (DIR)", fontsize=16)
plt.ylim(0, 1.2)
plt.axhline(
    0.8, color="red", linestyle="--", alpha=0.5, label="80% Rule (Fairness Threshold)"
)
plt.axhline(1.0, color="gray", linestyle="-", alpha=0.3, label="Parity (1.0)")

# Add values on top of bars
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.4f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 10),
        textcoords="offset points",
    )

plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/12_disparate_impact.png")
plt.close()

# PLOT 13: Bias Potential (Top Proxy Drivers)
plt.figure(figsize=(12, 6))
sns.barplot(data=top_bias_drivers, y="feature", x="bias_potential", color="steelblue")
plt.title("Top Proxy Features (High Bias Potential)", fontsize=16)
plt.xlabel("Bias Potential (SHAP Importance x Gender Correlation)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/13_bias_potential.png")
plt.close()

# PLOT 14: Top 10 Features Correlated with Gender
plt.figure(figsize=(12, 7))
top_corr_features = importance_df.sort_values(by="gender_corr", ascending=False).head(10).copy()
plot_df = top_corr_features.melt(id_vars="feature", value_vars=["gender_corr", "mean_abs_shap"], 
                                 var_name="Metric", value_name="Value")

sns.barplot(data=plot_df, y="feature", x="Value", hue="Metric", palette="muted")
plt.title("Top 10 Features Correlated with Gender & Importance", fontsize=16)
plt.xlabel("Magnitude", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.xlim(0, max(1.1, plot_df["Value"].max() * 1.1))
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/14_gender_correlation.png")
plt.close()

shap.plots.bar(shap_values, max_display=TOP_N_GLOBAL, show=False)
plt.title("Global Feature Importance (SHAP)")
save_current_fig(f"{OUTPUT_DIR}/02_global_bar.png", width=12, height=10)

shap.plots.beeswarm(shap_values, max_display=TOP_N_GLOBAL, show=False)
plt.title("Global SHAP Beeswarm")
save_current_fig(f"{OUTPUT_DIR}/03_global_beeswarm.png", width=14, height=10)

# ================================
# 10. Decision plot
# ================================
try:
    sample_n = min(150, len(X))
    sample_idx = np.linspace(0, len(X) - 1, sample_n, dtype=int)
    base_value = np.array(shap_values.base_values).reshape(-1)[0]

    shap.decision_plot(
        base_value,
        shap_values.values[sample_idx],
        X.iloc[sample_idx],
        feature_order="hclust",
        show=False,
    )
    plt.title("Decision Plot (Sampled Cases)")
    save_current_fig(f"{OUTPUT_DIR}/04_decision_plot_sampled.png", width=14, height=10)
except Exception as e:
    print("Skip decision plot:", str(e))

# ================================
# 11. Representative local cases
# ================================
cases = select_representative_cases(df)

for case_name, idx in cases.items():
    try:
        shap.plots.waterfall(shap_values[idx], max_display=TOP_N_LOCAL, show=False)
        plt.title(
            f"Local Explanation - {case_name.capitalize()} Case (index={idx}, proba={df.loc[idx, 'y_proba']:.4f})"
        )
        save_current_fig(
            f"{OUTPUT_DIR}/05_waterfall_{case_name}.png", width=12, height=10
        )
    except Exception as e:
        print(f"Skip waterfall for {case_name}:", str(e))

    local_table = create_local_contribution_table(X, shap_values, idx, case_name)
    local_table.to_csv(
        f"{OUTPUT_DIR}/06_local_contributors_{case_name}.csv", index=False
    )

# ================================
# 12. Gender cohort comparison
# ================================
try:
    male_mask = df["gender"].values == "Male"
    female_mask = df["gender"].values == "Female"

    shap.plots.bar(
        {"Male": shap_values[male_mask], "Female": shap_values[female_mask]},
        max_display=12,
        show=False,
    )
    plt.title("SHAP Cohort Comparison by Gender")
    save_current_fig(f"{OUTPUT_DIR}/07_gender_cohort_bar.png", width=14, height=10)
except Exception as e:
    print("Skip gender cohort plot:", str(e))

# ================================
# 13. Dependence plots
# ================================
for i, feature_name in enumerate(top_features, start=1):
    try:
        # Use index if possible or find correct feature name in Explanation
        shap.plots.scatter(shap_values[:, feature_name], show=False)
        plt.title(f"Dependence Plot - {feature_name}")
        safe_name = clean_filename(feature_name)
        save_current_fig(
            f"{OUTPUT_DIR}/08_dependence_{i}_{safe_name}.png", width=11, height=7
        )
    except Exception as e:
        print(f"Skip dependence plot for {feature_name}: {e}")

# ================================
# 14. Summary & Report
# ================================
summary_stats = pd.DataFrame(
    {
        "metric": [
            "n_samples",
            "n_features",
            "approval_rate_pred",
            "male_approval_rate_pred",
            "female_approval_rate_pred",
            "approved_case_index",
            "rejected_case_index",
            "borderline_case_index",
        ],
        "value": [
            len(df),
            X.shape[1],
            float(df["y_pred"].mean()),
            float(df.loc[df["gender"] == "Male", "y_pred"].mean()),
            float(df.loc[df["gender"] == "Female", "y_pred"].mean()),
            cases["approved"],
            cases["rejected"],
            cases["borderline"],
        ],
    }
)
summary_stats.to_csv(f"{OUTPUT_DIR}/09_summary_stats.csv", index=False)

top_positive_df = (
    importance_df.sort_values("mean_shap", ascending=False).head(10).copy()
)
top_negative_df = importance_df.sort_values("mean_shap", ascending=True).head(10).copy()

lines = []
lines.append("# Detailed Model Explainability Report (Human Readable)")
lines.append("")
lines.append("## 1. Summary")
lines.append(f"- Approval rate: **{df['y_pred'].mean():.4f}**")
lines.append(
    f"- Male approval: **{df.loc[df['gender'] == 'Male', 'y_pred'].mean():.4f}**"
)
lines.append(
    f"- Female approval: **{df.loc[df['gender'] == 'Female', 'y_pred'].mean():.4f}**"
)
lines.append("")

lines.append("## 2. Global Importance")
lines.append(
    markdown_table_from_df(
        top10, columns=["rank", "feature", "mean_abs_shap", "mean_shap"]
    )
)
lines.append("")

lines.append("## 3. Top Positive Influencers")
lines.append(
    markdown_table_from_df(
        top_positive_df, columns=["rank", "feature", "mean_abs_shap", "mean_shap"]
    )
)
lines.append("")

lines.append("## 4. Top Negative Influencers")
lines.append(
    markdown_table_from_df(
        top_negative_df, columns=["rank", "feature", "mean_abs_shap", "mean_shap"]
    )
)
lines.append("")

lines.append("## 5. Local Representative Cases")

for case_name, idx in cases.items():
    local_csv = f"{OUTPUT_DIR}/06_local_contributors_{case_name}.csv"
    if os.path.exists(local_csv):
        local_df = pd.read_csv(local_csv).head(10)
        lines.append(f"### Case: {case_name.capitalize()} (Index {idx})")
        lines.append(f"- Probability: **{df.loc[idx, 'y_proba']:.4f}**")
        lines.append(f"- Gender: **{df.loc[idx, 'gender']}**")
        lines.append("")
        lines.append(
            markdown_table_from_df(
                local_df,
                columns=["feature", "feature_value", "shap_value", "direction"],
            )
        )
        lines.append("")

lines.append("")
lines.append("## 6. Fairness Audit Results")
lines.append(f"- **Disparate Impact Ratio (DIR)**: **{dir_val:.4f}**")
if dir_val < 0.8:
    lines.append(
        "  - ⚠️ **Warning**: DIR is below 0.8 (Four-fifths rule). The model shows significant bias."
    )
else:
    lines.append(
        "  - ✅ **Pass**: DIR is above 0.8. The model's bias is within common legal thresholds."
    )
lines.append(f"- **Approval Rate Gap**: **{abs(male_rate - female_rate) * 100:.2f}%**")
lines.append("")
lines.append("### Proxy Feature Analysis")
lines.append("Features with high correlation to Gender may act as proxies:")
lines.append("")
proxy_df = importance_df.sort_values(by="gender_corr", ascending=False).head(10)[
    ["rank", "feature", "gender_corr"]
]
lines.append(markdown_table_from_df(proxy_df))
lines.append("")
# Bias Source Analysis Logic
importance_df["bias_potential"] = (
    importance_df["mean_abs_shap"] * importance_df["gender_corr"]
)
top_bias_drivers = importance_df.sort_values(by="bias_potential", ascending=False).head(
    10
)

# Check if gender features are in top contributors
gender_related_features = [
    f for f in importance_df["feature"] if "Personal_Status_Sex" in f
]
gender_in_top_5 = any(
    f in top_bias_drivers["feature"].values for f in gender_related_features
)

lines.append("## 7. Nhận xét nguồn gốc Bias (Bias Source Analysis)")
lines.append("### 7.1. Model có bias không? (Is the model biased?)")
if dir_val < 0.8:
    lines.append(
        f"- **Trả lời**: **Có**. Chỉ số DIR (**{dir_val:.4f}**) thấp hơn ngưỡng 0.8, cho thấy sự chênh lệch đáng kể trong tỷ lệ phê duyệt giữa Nam và Nữ."
    )
else:
    lines.append(
        f"- **Trả lời**: **Không đáng kể**. Chỉ số DIR (**{dir_val:.4f}**) đạt trên ngưỡng 0.8. Tuy nhiên vẫn tồn tại khoảng cách phê duyệt {abs(male_rate - female_rate) * 100:.1f}%."
    )

lines.append("")
lines.append("### 7.2. Gender có ảnh hưởng mạnh không? (Is Gender highly influential?)")
gender_rank = (
    importance_df[importance_df["feature"].isin(gender_related_features)]["rank"].min()
    if gender_related_features
    else "N/A"
)
if isinstance(gender_rank, (int, float)) and gender_rank <= 15:
    lines.append(
        f"- **Trả lời**: **Có ảnh hưởng trực tiếp**. Biến Gender (Personal_Status_Sex) xuất hiện trong top {gender_rank} các yếu tố quan trọng nhất của model."
    )
else:
    lines.append(
        "- **Trả lời**: **Ảnh hưởng trực tiếp thấp**. Biến Gender không nằm trong nhóm yếu tố quyết định hàng đầu, bias có thể đến từ các biến gián tiếp."
    )

lines.append("")
lines.append("### 7.3. Bias đến từ đâu? (Where does bias come from?)")
lines.append(
    "Bias đến từ sự kết hợp giữa ảnh hưởng của biến Gender và các **Proxy Features** (biến đại diện)."
)
lines.append("")
lines.append("#### Top Proxy Features Drivers:")
lines.append(
    markdown_table_from_df(
        top_bias_drivers,
        columns=["rank", "feature", "mean_abs_shap", "gender_corr", "bias_potential"],
    )
)
lines.append("")
lines.append("![Bias Potential Visualization](13_bias_potential.png)")
lines.append("")
if gender_in_top_5:
    lines.append(
        "- **Phân tích**: Bias đến từ cả biến Gender trực tiếp và các yếu tố liên quan."
    )
else:
    lines.append(
        "- **Phân tích**: Mặc dù Gender không quá quan trọng, nhưng bias được truyền dẫn qua các 'Proxy' như Age hoặc Job (có correlation cao với giới tính)."
    )
lines.append("")

lines.append("## 8. Generated Files")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    lines.append(f"- {filename}")

write_text_report(f"{OUTPUT_DIR}/10_executive_summary.md", lines)

# Print final results at the Very End
print("\n--- Fairness Audit Summary ---")
print(f"Disparate Impact Ratio: {dir_val:.4f}")
print(f"Approval Rate Gap: {abs(male_rate - female_rate) * 100:.2f}%")

print("\n=== TOP 10 FEATURES CORRELATED WITH GENDER ===")
# Sorted by absolute correlation with gender
analysis_by_corr = importance_df.sort_values(by="gender_corr", ascending=False)
print(
    analysis_by_corr[["feature", "gender_corr", "mean_abs_shap"]]
    .head(10)
    .to_string(index=False)
)

print("\n--- Potential Proxy Features (Top 10 by Bias Potential) ---")
print(top_bias_drivers[["feature", "gender_corr", "bias_potential"]].to_string())

print("\n--- Proxy Analysis ---")
if gender_in_top_5:
    print("Bias is driven by both direct Gender features and proxies.")
else:
    print("Bias is primarily driven by indirect proxy features.")
