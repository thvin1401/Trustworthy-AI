# Đề xuất cấu trúc slide báo cáo

## Slide 1 — Trang bìa
- Tên đề tài: **Auditing Bias in Credit Scoring — Trustworthy AI**
- Dataset: UCI German Credit
- Hướng: Ethical AI / Interpretable AI
- Thành viên nhóm, GVHD, ngày báo cáo

## Slide 2 — Đặt vấn đề (Motivation)
- Mô hình tài chính dễ kế thừa **bias lịch sử** theo giới tính/khu vực
- Hậu quả: từ chối tín dụng bất công, rủi ro pháp lý (EEOC, GDPR)
- Câu hỏi nghiên cứu: *Mô hình credit scoring có đối xử công bằng giữa các nhóm giới tính không? Biến `Gender` có nằm trong top feature quan trọng?*

## Slide 3 — Mục tiêu & Phạm vi
- Train mô hình **XGBoost** trên German Credit
- Audit fairness bằng **Fairlearn**: Demographic Parity, Disparate Impact
- Giải thích mô hình bằng **SHAP**
- Phân tích **direct bias vs proxy bias**

## Slide 4 — Dataset: UCI German Credit
- 1000 mẫu, 20 thuộc tính (tài chính + nhân khẩu học)
- Target: Good (1) / Bad (0) credit
- Biến nhạy cảm: `Attribute9` (Personal_Status_Sex) → suy ra giới tính
- Đặc điểm: low-resource, tabular, train nhanh

## Slide 5 — Phương pháp / Pipeline
1. Load & giải nén dữ liệu
2. Đặt tên cột, map target nhị phân
3. One-hot encoding (`pd.get_dummies`)
4. Train **XGBClassifier**
5. Inference + tách nhóm nhạy cảm theo giới tính
6. Fairness audit + SHAP explainability

*(Sơ đồ pipeline minh hoạ)*

## Slide 6 — Kiến trúc & triển khai
- Cấu trúc repo (`script.py`, `*.pkl`, `data/`, `Result/`)
- Đóng gói **Docker / docker-compose** để tái lập thí nghiệm
- `requirements.txt`: xgboost, fairlearn, shap, pandas, scikit-learn

## Slide 7 — Kết quả Fairness Audit
- **Approval Rate Gap: 11.45%** (Nam > Nữ)
- **Disparate Impact Ratio: 0.8329** → vượt ngưỡng 0.8 (Four-fifths rule)
- Hình: `11_approval_rates.png`, `12_disparate_impact.png`
- Nhận định: hợp pháp nhưng vẫn lệch đáng kể về kỹ thuật

## Slide 8 — Giải thích mô hình bằng SHAP
- Top features: `Checking_Status`, `Duration_Months`, `Credit_Amount`
- `Personal_Status_Sex` nằm **ngoài Top 10** tầm quan trọng tổng thể
- Hình: `02_global_bar.png` (SHAP beeswarm/bar)

## Slide 9 — Nguồn gốc của Bias (Direct vs Proxy)
- **Direct bias:** `Personal_Status_Sex` SHAP tổng ~0.18
- **Proxy features:**
  - `Age` (corr ~0.16, rank 7)
  - `Employment Duration < 1 năm`
  - `Housing_Own`
- Hình: `13_bias_potential.png`, `14_gender_correlation.png`

## Slide 10 — Thảo luận
- Bias không biến mất khi loại bỏ biến giới tính → **proxy bias**
- Trade-off: fairness vs accuracy
- Giới hạn: dataset nhỏ (1000 mẫu), nhãn giới tính suy ra gián tiếp từ marital status

## Slide 11 — Trạng thái hoàn thành so với đề bài
- ✅ Train mô hình **XGBoost** trên German Credit
- ✅ Audit fairness bằng **Fairlearn** — Demographic Parity, Disparate Impact
- ✅ Phân tích **SHAP** (global bar, beeswarm, dependence, waterfall, gender cohort)
- ✅ Kết luận chính thức: `Gender` **không thuộc top-3** feature, nhưng xuất hiện qua proxy
- ✅ Pipeline tái lập bằng **Docker / docker-compose**
- ⚠️ Hạn chế phương pháp: **fairness audit chạy trên full dataset** (không có train/test split); model nạp từ pickle nên không tái lập được quá trình train (hyperparameter, seed)
- ⚠️ Hạn chế dữ liệu: dataset nhỏ (1000 mẫu), nhãn giới tính suy ra gián tiếp từ `Attribute9` (Personal_Status_Sex)

## Slide 12 — Hướng phát triển (ngoài phạm vi đề bài)
- Thêm training pipeline có cross-validation & lưu seed/config
- Bổ sung fairness metrics khác: Equal Opportunity, Equalized Odds
- Thử **bias mitigation** (Reweighing, `ExponentiatedGradient`) để so sánh DIR trước/sau
- Mở rộng sang nhóm nhạy cảm khác (tuổi, foreign worker)
- So sánh nhiều mô hình (LogReg, RandomForest, XGBoost)

## Slide 13 — Kết luận
- Mô hình XGBoost **đạt ngưỡng pháp lý** nhưng vẫn có chênh lệch giới tính 11.45%
- `Gender` **không nằm trong top-3** features, nhưng bias đến từ **proxy** (Age, Employment, Housing)
- Khuyến nghị: kết hợp fairness-aware training + giám sát liên tục

## Slide 14 — Demo & Q&A
- Demo chạy `script.py` qua Docker
- Hiển thị output trong `Trustworthy_AI/Result` & `output_readable/`

---
Gợi ý: nên chèn ảnh từ `Trustworthy_AI/output_readable/` (slide 7–9) và sơ đồ pipeline (slide 5) để trực quan.
