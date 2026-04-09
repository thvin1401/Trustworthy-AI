import os
import re
import pickle
import zipfile
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

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

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
BACKGROUND_SIZE = 200
TOP_N_GLOBAL = 15
TOP_N_LOCAL = 15
TOP_N_DEPENDENCE = 3


# ================================
# 1. Helper functions
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
    # dùng đúng mapping phổ biến cho German Credit
    return "Female" if attr9 in ("A92", "A95") else "Male"


def normalize_shap_output(explanation):
    """
    Chuẩn hóa output SHAP về dạng 2 chiều:
    (n_samples, n_features)
    """
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
            feature_names=explanation.feature_names
        )

    return explanation


def select_representative_cases(df):
    """
    Chọn 3 case:
    - approved mạnh nhất
    - rejected mạnh nhất
    - borderline gần 0.5 nhất
    """
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
    local_df = pd.DataFrame({
        "case": case_name,
        "feature": X.columns,
        "feature_value": X.iloc[idx].values,
        "shap_value": shap_values.values[idx],
        "abs_shap": np.abs(shap_values.values[idx])
    }).sort_values("abs_shap", ascending=False)

    local_df["direction"] = np.where(local_df["shap_value"] >= 0, "push_up", "push_down")
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
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)
# ================================
# 2. Load dataset
# ================================
zip_path = "statlog+german+credit+data.zip"

if not os.path.exists("data/german.data"):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data")

df = pd.read_csv("data/german.data", sep=" ", header=None)

columns = [
    "Attribute1", "Attribute2", "Attribute3", "Attribute4", "Attribute5",
    "Attribute6", "Attribute7", "Attribute8", "Attribute9", "Attribute10",
    "Attribute11", "Attribute12", "Attribute13", "Attribute14", "Attribute15",
    "Attribute16", "Attribute17", "Attribute18", "Attribute19", "Attribute20",
    "target"
]
df.columns = columns

# 1 = good, 2 = bad -> map về 1/0
df["target"] = df["target"].map({1: 1, 2: 0})
df["gender"] = df["Attribute9"].apply(get_gender)

# ================================
# 3. Preprocess giống pipeline cũ
# ================================
X_raw = df.drop(["target", "gender"], axis=1)
X_encoded = pd.get_dummies(X_raw)

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

for col in feature_columns:
    if col not in X_encoded.columns:
        X_encoded[col] = 0

X = X_encoded[feature_columns].copy()
X = X.fillna(0).astype(np.float64)

print("X shape:", X.shape)
print("X dtype:", X.to_numpy().dtype)

# ================================
# 4. Load model
# ================================
with open("german_credit_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================================
# 5. Predict
# ================================
y_pred = model.predict(X)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X)[:, 1]
else:
    y_proba = y_pred.astype(float)

df["y_pred"] = y_pred
df["y_proba"] = y_proba

# ================================
# 6. Build SHAP explainer
# ================================
if len(X) > BACKGROUND_SIZE:
    background = shap.sample(X, BACKGROUND_SIZE, random_state=RANDOM_STATE)
else:
    background = X.copy()

try:
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional"
    )
    shap_values = explainer(X)
except Exception:
    explainer = shap.Explainer(model, background)
    shap_values = explainer(X)

shap_values = normalize_shap_output(shap_values)

# ================================
# 7. Global importance tables
# ================================
importance_df = pd.DataFrame({
    "feature": X.columns,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    "mean_shap": shap_values.values.mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

importance_df["rank"] = np.arange(1, len(importance_df) + 1)
importance_df = importance_df[["rank", "feature", "mean_abs_shap", "mean_shap"]]
importance_df.to_csv(f"{OUTPUT_DIR}/01_shap_importance.csv", index=False)

top10 = importance_df.head(10)
top_features = importance_df["feature"].head(TOP_N_DEPENDENCE).tolist()

print("\nTop 10 feature quan trọng nhất:")
print(top10)

# ================================
# 8. Global plots
# ================================
shap.plots.bar(shap_values, max_display=TOP_N_GLOBAL, show=False)
plt.title("Global Feature Importance (SHAP)")
save_current_fig(f"{OUTPUT_DIR}/02_global_bar.png", width=12, height=8)

shap.plots.beeswarm(shap_values, max_display=TOP_N_GLOBAL, show=False)
plt.title("Global SHAP Beeswarm")
save_current_fig(f"{OUTPUT_DIR}/03_global_beeswarm.png", width=13, height=8)

# ================================
# 9. Decision plot (sampled)
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
        show=False
    )
    plt.title("Decision Plot (Sampled Cases)")
    save_current_fig(f"{OUTPUT_DIR}/04_decision_plot_sampled.png", width=14, height=8)
except Exception as e:
    print("Skip decision plot:", str(e))

# ================================
# 10. Representative local cases
# ================================
cases = select_representative_cases(df)

for case_name, idx in cases.items():
    try:
        shap.plots.waterfall(shap_values[idx], max_display=TOP_N_LOCAL, show=False)
        plt.title(
            f"Local Explanation - {case_name.capitalize()} Case (index={idx}, proba={df.loc[idx, 'y_proba']:.4f})"
        )
        save_current_fig(
            f"{OUTPUT_DIR}/05_waterfall_{case_name}.png",
            width=12,
            height=8
        )
    except Exception as e:
        print(f"Skip waterfall for {case_name}:", str(e))

    local_table = create_local_contribution_table(X, shap_values, idx, case_name)
    local_table.to_csv(f"{OUTPUT_DIR}/06_local_contributors_{case_name}.csv", index=False)

# ================================
# 11. Gender cohort comparison
# ================================
try:
    male_mask = df["gender"].values == "Male"
    female_mask = df["gender"].values == "Female"

    shap.plots.bar(
        {
            "Male": shap_values[male_mask],
            "Female": shap_values[female_mask]
        },
        max_display=12,
        show=False
    )
    plt.title("SHAP Cohort Comparison by Gender")
    save_current_fig(f"{OUTPUT_DIR}/07_gender_cohort_bar.png", width=13, height=8)
except Exception as e:
    print("Skip gender cohort plot:", str(e))

# ================================
# 12. Dependence plots for top features
# ================================
for i, feature_name in enumerate(top_features, start=1):
    try:
        shap.plots.scatter(shap_values[:, feature_name], show=False)
        plt.title(f"Dependence Plot - {feature_name}")
        safe_name = clean_filename(feature_name)
        save_current_fig(
            f"{OUTPUT_DIR}/08_dependence_{i}_{safe_name}.png",
            width=11,
            height=7
        )
    except Exception as e:
        print(f"Skip dependence plot for {feature_name}: {e}")

# ================================
# 13. Executive summary tables
# ================================
summary_stats = pd.DataFrame({
    "metric": [
        "n_samples",
        "n_features",
        "approval_rate_pred",
        "male_approval_rate_pred",
        "female_approval_rate_pred",
        "approved_case_index",
        "rejected_case_index",
        "borderline_case_index"
    ],
    "value": [
        len(df),
        X.shape[1],
        float(df["y_pred"].mean()),
        float(df.loc[df["gender"] == "Male", "y_pred"].mean()),
        float(df.loc[df["gender"] == "Female", "y_pred"].mean()),
        cases["approved"],
        cases["rejected"],
        cases["borderline"]
    ]
})
summary_stats.to_csv(f"{OUTPUT_DIR}/09_summary_stats.csv", index=False)

# ================================
# 14. Executive summary markdown
# ================================
# ================================
# 14. Detailed markdown report
# ================================
top_positive_df = importance_df.sort_values("mean_shap", ascending=False).head(10).copy()
top_negative_df = importance_df.sort_values("mean_shap", ascending=True).head(10).copy()

lines = []
lines.append("# Detailed Model Explainability Report")
lines.append("")
lines.append("## 1. Report Objective")
lines.append("")
lines.append(
    "Mục tiêu của báo cáo này là giải thích mô hình XGBoost đã được huấn luyện cho bài toán phê duyệt tín dụng "
    "bằng phương pháp SHAP (SHapley Additive exPlanations). "
    "Phần giải thích được thực hiện ở cả hai cấp độ:"
)
lines.append("")
lines.append("- **Global explainability**: mô hình nhìn chung đang dựa vào các đặc trưng nào để đưa ra dự đoán.")
lines.append("- **Local explainability**: với từng hồ sơ cụ thể, những đặc trưng nào đã kéo dự đoán lên hoặc xuống.")
lines.append("")
lines.append(
    "Báo cáo này cũng liên hệ kết quả explainability với fairness analysis ở Task 2, "
    "đặc biệt là theo nhóm giới tính."
)
lines.append("")

lines.append("## 2. Dataset, Preprocessing and Model Context")
lines.append("")
lines.append(f"- Số lượng mẫu dữ liệu dùng để giải thích: **{len(df)}**")
lines.append(f"- Số lượng đặc trưng sau khi preprocessing: **{X.shape[1]}**")
lines.append(
    "- Dữ liệu gốc được tiền xử lý lại theo đúng pipeline đã dùng ở phần trước, "
    "bao gồm one-hot encoding cho các biến phân loại và đồng bộ cột với `feature_columns.pkl`."
)
lines.append(
    "- Việc tái sử dụng đúng pipeline cũ là bắt buộc để đảm bảo SHAP đang giải thích đúng đầu vào mà model `.pkl` thực sự nhìn thấy."
)
lines.append(
    "- Mô hình được giải thích trong báo cáo này là mô hình XGBoost đã huấn luyện sẵn từ nhóm trước, "
    "được load trực tiếp từ file `german_credit_model.pkl`."
)
lines.append("")

lines.append("## 3. Prediction Summary")
lines.append("")
lines.append(f"- Predicted approval rate (toàn bộ dữ liệu): **{df['y_pred'].mean():.4f}**")
lines.append(f"- Male predicted approval rate: **{df.loc[df['gender'] == 'Male', 'y_pred'].mean():.4f}**")
lines.append(f"- Female predicted approval rate: **{df.loc[df['gender'] == 'Female', 'y_pred'].mean():.4f}**")
lines.append("")
lines.append(
    "Các tỷ lệ trên được tính từ dự đoán của mô hình hiện tại. "
    "Chúng không tự động chứng minh mô hình công bằng hay thiếu công bằng, "
    "nhưng là cơ sở để liên hệ với fairness analysis ở Task 2."
)
lines.append("")

lines.append("## 4. SHAP Methodology")
lines.append("")
lines.append(
    "SHAP được sử dụng để lượng hóa đóng góp của từng đặc trưng đối với đầu ra dự đoán của mô hình."
)
lines.append("")
lines.append("- **mean(|SHAP|)**: đo độ quan trọng trung bình của feature trên toàn bộ dataset.")
lines.append("- **mean(SHAP)**: cho biết xu hướng trung bình của feature là kéo dự đoán lên hay kéo xuống.")
lines.append("- **Waterfall plot**: giải thích chi tiết cho từng hồ sơ cụ thể.")
lines.append("- **Beeswarm plot**: thể hiện đồng thời độ quan trọng, hướng tác động và độ phân tán của từng feature.")
lines.append("- **Cohort plot theo giới tính**: so sánh sự khác biệt về ảnh hưởng của feature giữa các nhóm giới tính.")
lines.append("")

lines.append("## 5. Global Feature Importance")
lines.append("")
lines.append(
    "Bảng dưới đây liệt kê các đặc trưng quan trọng nhất của mô hình theo chỉ số **mean absolute SHAP value**."
)
lines.append("")
lines.append(markdown_table_from_df(
    top10,
    columns=["rank", "feature", "mean_abs_shap", "mean_shap"]
))
lines.append("")
lines.append(
    "Diễn giải:"
)
lines.append("")
lines.append("- `mean_abs_shap` càng lớn thì feature đó càng có ảnh hưởng mạnh đến dự đoán của mô hình.")
lines.append("- `mean_shap` dương cho thấy feature đó có xu hướng đẩy đầu ra về phía lớp positive (approval) nhiều hơn.")
lines.append("- `mean_shap` âm cho thấy feature đó có xu hướng kéo dự đoán về phía lớp negative (rejection) nhiều hơn.")
lines.append("")

lines.append("## 6. Features Pushing Prediction Up")
lines.append("")
lines.append(
    "Các feature dưới đây có xu hướng trung bình làm **tăng** đầu ra dự đoán (nghiêng về approval) mạnh nhất:"
)
lines.append("")
lines.append(markdown_table_from_df(
    top_positive_df,
    columns=["rank", "feature", "mean_abs_shap", "mean_shap"]
))
lines.append("")

lines.append("## 7. Features Pushing Prediction Down")
lines.append("")
lines.append(
    "Các feature dưới đây có xu hướng trung bình làm **giảm** đầu ra dự đoán (nghiêng về rejection) mạnh nhất:"
)
lines.append("")
lines.append(markdown_table_from_df(
    top_negative_df,
    columns=["rank", "feature", "mean_abs_shap", "mean_shap"]
))
lines.append("")

lines.append("## 8. Global Visual Explanations")
lines.append("")
lines.append("Các biểu đồ toàn cục đã được sinh ra:")
lines.append("")
lines.append("- `02_global_bar.png`: cho biết feature nào quan trọng nhất trên toàn bộ mô hình.")
lines.append("- `03_global_beeswarm.png`: cho biết mỗi feature tác động theo hướng nào và mức độ phân tán ra sao.")
lines.append("- `04_decision_plot_sampled.png`: cho biết các feature đóng góp thế nào trên nhiều mẫu cùng lúc.")
lines.append("")
lines.append(
    "Ba biểu đồ này kết hợp với nhau để trả lời câu hỏi: "
    "**mô hình đang quan tâm điều gì, theo hướng nào, và mức độ nhất quán của ảnh hưởng đó ra sao.**"
)
lines.append("")

lines.append("## 9. Representative Local Cases")
lines.append("")
lines.append(
    "Thay vì chọn ngẫu nhiên một dòng dữ liệu, báo cáo chọn ba trường hợp đại diện:"
)
lines.append("")
lines.append(f"- **Approved case**: index = **{cases['approved']}**, predicted probability = **{df.loc[cases['approved'], 'y_proba']:.4f}**")
lines.append(f"- **Rejected case**: index = **{cases['rejected']}**, predicted probability = **{df.loc[cases['rejected'], 'y_proba']:.4f}**")
lines.append(f"- **Borderline case**: index = **{cases['borderline']}**, predicted probability = **{df.loc[cases['borderline'], 'y_proba']:.4f}**")
lines.append("")
lines.append(
    "Cách chọn này giúp phần local explanation có giá trị hơn nhiều so với việc chọn ngẫu nhiên, "
    "vì nó đại diện cho ba tình huống rõ ràng: được duyệt mạnh, bị từ chối mạnh, và trường hợp sát ngưỡng quyết định."
)
lines.append("")

for case_name, idx in cases.items():
    local_csv = f"{OUTPUT_DIR}/06_local_contributors_{case_name}.csv"
    if os.path.exists(local_csv):
        local_df = pd.read_csv(local_csv)
        top_local = local_df.head(10).copy()

        push_up = top_local[top_local["direction"] == "push_up"].head(5).copy()
        push_down = top_local[top_local["direction"] == "push_down"].head(5).copy()

        lines.append(f"### 9.{list(cases.keys()).index(case_name)+1} Case: {case_name.capitalize()} (index={idx})")
        lines.append("")
        lines.append(f"- Predicted label: **{int(df.loc[idx, 'y_pred'])}**")
        lines.append(f"- Predicted probability: **{df.loc[idx, 'y_proba']:.4f}**")
        lines.append(f"- Gender: **{df.loc[idx, 'gender']}**")
        lines.append("")
        lines.append("Top contributors theo độ lớn SHAP:")
        lines.append("")
        lines.append(markdown_table_from_df(
            top_local,
            columns=["feature", "feature_value", "shap_value", "abs_shap", "direction"],
            max_rows=10
        ))
        lines.append("")

        if len(push_up) > 0:
            lines.append("Các yếu tố đẩy dự đoán **lên**:")
            lines.append("")
            lines.append(markdown_table_from_df(
                push_up,
                columns=["feature", "feature_value", "shap_value", "abs_shap"]
            ))
            lines.append("")

        if len(push_down) > 0:
            lines.append("Các yếu tố kéo dự đoán **xuống**:")
            lines.append("")
            lines.append(markdown_table_from_df(
                push_down,
                columns=["feature", "feature_value", "shap_value", "abs_shap"]
            ))
            lines.append("")

        lines.append(
            f"Biểu đồ tương ứng: `05_waterfall_{case_name}.png`"
        )
        lines.append("")

lines.append("## 10. Gender Cohort Analysis")
lines.append("")
lines.append(
    "Biểu đồ `07_gender_cohort_bar.png` dùng để so sánh mức độ ảnh hưởng của feature giữa hai nhóm giới tính."
)
lines.append("")
lines.append(
    "Ý nghĩa của phần này không phải để kết luận ngay rằng mô hình công bằng hay không công bằng, "
    "mà để trả lời câu hỏi sâu hơn: "
    "**nếu approval rate giữa các nhóm có khác nhau, thì các feature ảnh hưởng mạnh nhất ở mỗi nhóm có giống nhau không?**"
)
lines.append("")
lines.append(
    "Đây là cầu nối trực tiếp giữa Task 2 (fairness) và Task 3 (explainability). "
    "Nếu cohort plot cho thấy sự khác biệt đáng kể về feature importance giữa Male và Female, "
    "thì đó là tín hiệu cần đào sâu thêm trong fairness audit."
)
lines.append("")

lines.append("## 11. Dependence Analysis")
lines.append("")
lines.append(
    "Các dependence plot (`08_dependence_*.png`) được tạo cho các feature quan trọng nhất để quan sát sâu hơn mối quan hệ giữa giá trị feature và tác động SHAP."
)
lines.append("")
lines.append(
    "Phần này đặc biệt hữu ích khi muốn trả lời:"
)
lines.append("")
lines.append("- feature quan trọng có tác động tuyến tính hay phi tuyến?")
lines.append("- tác động đó ổn định hay thay đổi mạnh giữa các vùng giá trị?")
lines.append("- có xuất hiện interaction ngầm với các feature khác không?")
lines.append("")

lines.append("## 12. Main Findings")
lines.append("")
lines.append("Từ toàn bộ kết quả trên, có thể rút ra một số nhận định chính:")
lines.append("")
lines.append("1. Mô hình XGBoost không còn là hộp đen hoàn toàn, vì có thể xác định các feature ảnh hưởng mạnh nhất bằng SHAP.")
lines.append("2. Ảnh hưởng của feature có thể được phân tích ở cả hai cấp độ: toàn cục và từng hồ sơ cụ thể.")
lines.append("3. Các trường hợp local explanation cho thấy cùng một mô hình nhưng lý do dẫn đến approval và rejection có thể rất khác nhau giữa các cá nhân.")
lines.append("4. Cohort analysis theo giới tính tạo ra cầu nối thực tế giữa explainability và fairness, giúp phân tích model một cách có trách nhiệm hơn.")
lines.append("")

lines.append("## 13. Limitations")
lines.append("")
lines.append("- Các feature hiện đang hiển thị dưới dạng mã thuộc tính (`Attribute...`) hoặc cột one-hot, nên để diễn giải nghiệp vụ sâu hơn cần đối chiếu thêm data dictionary của German Credit.")
lines.append("- SHAP giải thích hành vi của mô hình hiện tại, không chứng minh rằng mô hình đúng về mặt nhân quả.")
lines.append("- Chênh lệch approval rate giữa các nhóm không tự động đồng nghĩa với bias; cần kết hợp với fairness metrics của Task 2.")
lines.append("- Local explanation chỉ phản ánh các case được chọn, không đại diện cho toàn bộ mọi trường hợp cá biệt.")
lines.append("")

lines.append("## 14. Conclusion")
lines.append("")
lines.append(
    "Phần explainability đã chứng minh rằng mô hình có thể được giải thích một cách hệ thống bằng SHAP, "
    "bao gồm cả global feature importance, direction of impact, local case-level reasoning, "
    "và cohort comparison theo giới tính. "
    "Điều này giúp tăng tính minh bạch của mô hình và hỗ trợ đánh giá Trustworthy AI một cách đầy đủ hơn."
)
lines.append("")

lines.append("## 15. Generated Artifacts")
lines.append("")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    lines.append(f"- {filename}")

write_text_report(f"{OUTPUT_DIR}/10_executive_summary.md", lines)

# ================================
# 15. Console summary
# ================================
print("\nĐã sinh file trong output/:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print("-", filename)