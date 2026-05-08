# Trustworthy AI: Bias Audit trong Credit Scoring

Project này là một repo Python dùng để kiểm tra độ công bằng của một mô hình phân loại tín dụng trên bộ dữ liệu German Credit.

Repo bám theo đề bài về `Auditing Bias in Credit Scoring`, thuộc hướng `Ethical AI / Interpretable AI`. Mục tiêu không chỉ là đo fairness bằng `Fairlearn`, mà còn phải giải thích mô hình bằng `SHAP` để xem biến liên quan đến giới tính có nằm trong nhóm feature quan trọng nhất hay không.

Hiện tại repo mới hoàn thành một phần:

- đã có dữ liệu và artifact model
- đã có script audit fairness cơ bản
- chưa có training pipeline đầy đủ trong repo
- chưa có phần giải thích bằng `SHAP`

## Mục tiêu project

- Dùng bộ dữ liệu `UCI German Credit Dataset`
- Huấn luyện một mô hình `XGBoost` cho bài toán credit scoring
- Audit fairness của mô hình bằng `Fairlearn`
- Tính `Demographic Parity` giữa các nhóm nhạy cảm, đặc biệt là giới tính
- Dùng `SHAP` để giải thích mô hình
- Kiểm tra xem feature liên quan đến `Gender` có nằm trong top-3 feature quan trọng hay không

## Yêu cầu đề bài

Đề bài gốc yêu cầu:

- **Problem**: mô hình tài chính có thể kế thừa bias lịch sử theo giới tính hoặc khu vực
- **Low-resource edge**: dùng dữ liệu tabular nên nhẹ, train nhanh
- **Methodology**:
  - train `XGBoost`
  - dùng `Fairlearn` để tính `Demographic Parity`
  - dùng `SHAP` để xem `Gender` có phải top-3 feature hay không
- **Dataset**: UCI German Credit Dataset

README này vì vậy sẽ mô tả rõ:

- phần nào của đề bài repo đã có
- phần nào còn thiếu
- phần nào sẽ được cập nhật sau khi training code được thêm vào

## Trạng thái hiện tại của repo

So với đề bài, repo hiện đang ở trạng thái:

- [x] Đã đưa bộ dữ liệu German Credit vào repo
- [x] Đã có model artifact `XGBoost` đã train sẵn
- [x] Đã có script audit fairness cơ bản
- [x] Đã tính approval rate theo giới tính
- [x] Đã tính `Demographic Parity`
- [x] Đã tính `Disparate Impact Ratio`
- [x] Đã có phân tích `SHAP` (global bar, beeswarm, dependence, waterfall, gender cohort)
- [x] Đã có kết luận chính thức: `Gender` (`Personal_Status_Sex`) **không nằm trong top-3 feature**; top-3 là `Checking_Status_No account`, `Duration_Months`, `Credit_Amount`
- [x] Đã sửa logic mapping giới tính từ `Attribute9` (A91/A93/A94 = Male; A92/A95 = Female)
- [x] Đã đóng gói Docker / docker-compose để tái lập thí nghiệm
- [ ] Chưa có training pipeline rõ ràng cho `XGBoost` (model dùng dạng pickle có sẵn)
- [ ] Chưa tách train/test — fairness audit hiện chạy trên full dataset

## Cấu trúc repo

```text
.
├── README.md
├── requirements.txt
├── dockerfile
├── docker-compose.yaml
└── Trustworthy_AI/
    ├── script.py
    ├── german_credit_model.pkl
    ├── feature_columns.pkl
    ├── statlog+german+credit+data.zip
    ├── Result
    └── data/
        ├── german.data
        ├── german.data-numeric
        ├── german.doc
        └── Index
```

Ý nghĩa các file chính:

- `Trustworthy_AI/script.py`: script chính để chạy fairness audit
- `Trustworthy_AI/german_credit_model.pkl`: model `XGBClassifier` đã train sẵn
- `Trustworthy_AI/feature_columns.pkl`: danh sách cột đầu vào mà model mong đợi sau bước one-hot encoding
- `Trustworthy_AI/statlog+german+credit+data.zip`: file zip của dataset gốc
- `Trustworthy_AI/data/german.doc`: mô tả ý nghĩa từng thuộc tính trong dataset
- `Trustworthy_AI/Result`: output mẫu đã được chạy sẵn

## Kiến trúc hiện tại

Kiến trúc của project:

1. Dataset được giải nén từ file zip.
2. File `german.data` được đọc vào `pandas.DataFrame`.
3. Các cột được đặt tên lại theo dạng `Attribute1 ... Attribute20`.
4. Biến mục tiêu `target` được map về nhị phân:
   - `1`: good credit
   - `0`: bad credit
5. Toàn bộ feature được one-hot encode bằng `pd.get_dummies`.
6. Script nạp `feature_columns.pkl` để ép dữ liệu suy luận khớp với schema lúc train.
7. Script nạp model pickle và chạy `predict`.
8. Từ `Attribute9`, script suy ra giới tính để chia nhóm nhạy cảm.
9. Script tính approval rate theo nhóm và in ra kết luận fairness.

Nói ngắn gọn: hiện tại đây mới là một pipeline suy luận + đánh giá fairness cơ bản. Để khớp hoàn toàn với đề bài, repo còn cần thêm training pipeline và phần giải thích mô hình bằng `SHAP`.

## Luồng chạy chi tiết

Khi chạy `script.py`, script thực hiện lần lượt:

1. Giải nén `statlog+german+credit+data.zip`
2. Đọc `data/german.data`
3. Gán tên cột
4. Chuyển `target` về nhị phân
5. One-hot encode dữ liệu
6. Bổ sung các cột còn thiếu theo `feature_columns.pkl`
7. Sắp xếp đúng thứ tự cột
8. Load `german_credit_model.pkl`
9. Predict nhãn `y_pred`
10. Tạo cột `gender`
11. Tính:
    - approval rate theo giới tính
    - demographic parity difference
    - disparate impact ratio
12. In kết luận cuối cùng

## Kết quả mẫu hiện có

File `Trustworthy_AI/Result` đang cho kết quả:

```text
=== APPROVAL RATE ===
Male   : 0.686
Female : 0.571

=== DEMOGRAPHIC PARITY ===
DP Difference: 0.115

=== DISPARATE IMPACT ===
DIR: 0.833
```

Cách hiểu nhanh:

- Nhóm nam được approve khoảng `68.6%`
- Nhóm nữ được approve khoảng `57.1%`
- Chênh lệch approval rate là khoảng `11.5 điểm %`
- Tỷ lệ `0.833` nghĩa là approval rate của nữ bằng khoảng `83.3%` của nam

Theo logic hiện tại của script:

- `|DP| > 0.1` thì xem là có dấu hiệu bias
- `DIR < 0.8` thì xem là có dấu hiệu phân biệt đối xử mạnh theo quy tắc 80%

Với output mẫu này:

- Script kết luận có bias theo `Demographic Parity`
- Nhưng chưa vi phạm mạnh theo ngưỡng `DIR < 0.8`

Lưu ý: đây mới chỉ là phần audit fairness cơ bản. Theo đúng đề bài, báo cáo cuối cùng còn cần thêm:

- phần huấn luyện mô hình `XGBoost`
- phần phân tích `SHAP`
- kết luận xem feature `Gender` có nằm trong top-3 hay không

## Cách chạy project

### Chạy local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd Trustworthy_AI
python script.py
```

Lưu ý: script đang dùng relative path, nên cần chạy từ bên trong thư mục `Trustworthy_AI`.

### Chạy bằng Docker

Build image:

```bash
docker build -t my-python-dev .
```

Mở container dev:

```bash
docker-compose run dev bash
```

Chạy script:

```bash
cd Trustworthy_AI
python script.py
```

## Dependency chính

- `pandas`: xử lý dữ liệu
- `scikit-learn`: tương thích pipeline/model serialization
- `xgboost`: nạp và dùng model `XGBClassifier`
- `fairlearn`: tính metric fairness

## Những lưu ý project đang làm

- Có sẵn dataset, model và feature schema nên có thể chạy audit ngay
- Có dùng `fairlearn` thay vì tự viết metric fairness
- Có thể dùng như demo ngắn cho môn học hoặc bài trình bày về AI fairness

## Các vấn đề quan trọng cần hiểu đúng

Đây là phần phản biện quan trọng. Kết quả hiện tại có giá trị minh họa, nhưng chưa đủ mạnh để xem là một fairness assessment hoàn chỉnh.

### 1. Mapping giới tính đang có lỗi logic

Trong dataset German Credit, `Attribute9` gồm:

- `A91`, `A93`, `A94`: nam
- `A92`, `A95`: nữ

Nhưng script hiện tại đang dùng:

```python
return "Female" if val == "A92" else "Male"
```

Điều đó có nghĩa là `A95` đang bị xếp nhầm sang nhóm nam. Đây là lỗi quan trọng vì nó làm sai sensitive group, kéo theo approval rate, DP và DIR đều có thể sai.

### 2. Audit chỉ dùng một góc nhìn fairness

Project hiện chỉ nhìn fairness qua:

- Demographic Parity
- Disparate Impact

Trong bối cảnh đề bài, `Demographic Parity` là metric trung tâm nên việc dùng nó là hợp lý. Tuy vậy, README vẫn cần nói rõ rằng:

- đây là metric được chọn theo yêu cầu đề bài
- nó không đại diện cho toàn bộ fairness
- kết luận cuối cùng cần diễn giải cẩn thận

Nhưng trong bài toán tín dụng, chỉ nhìn selection rate là chưa đủ. Nên xem thêm:

- Equal Opportunity
- Equalized Odds
- Group-wise confusion matrix
- False positive / false negative rate theo nhóm

Một mô hình có thể đạt demographic parity tương đối ổn nhưng vẫn bất công ở lỗi dự đoán giữa các nhóm.

### 3. Không có train/test split trong repo hiện tại

Repo chỉ chứa model đã train sẵn và chạy audit trực tiếp trên toàn bộ dataset đầu vào. Vì vậy:

- không biết model được train trên tập nào
- không biết audit đang chạy trên train set, test set hay full set
- không đánh giá được fairness generalization

Kết quả hiện tại nên được hiểu là audit trên dữ liệu đang có, không phải bằng chứng mạnh về fairness ngoài thực tế.

### 4. Không có pipeline huấn luyện để tái lập kết quả

Repo chưa có:

- code train model
- code chọn hyperparameter
- code lưu artifact
- seed/experiment config đầy đủ

Điều này khiến project khó tái lập và khó kiểm chứng nguồn gốc của model pickle.

Đây là khoảng trống quan trọng vì đề bài yêu cầu rõ việc `Train an XGBoost model`.

### 4b. Chưa có SHAP nên chưa khớp hoàn toàn với đề bài

Đề bài yêu cầu dùng `SHAP` để kiểm tra xem `Gender` có phải top-3 feature hay không. Repo hiện chưa có:

- dependency `shap`
- code tính SHAP values
- bảng xếp hạng feature importance theo SHAP
- kết luận chính thức về biến `Gender`

Nói cách khác, repo hiện mới đáp ứng phần fairness audit bằng `Fairlearn`, nhưng chưa đáp ứng trọn vẹn phần `Interpretable AI`.

### 5. Relative path đang bị cứng

Script chỉ chạy ổn nếu đang đứng trong thư mục `Trustworthy_AI`. Đây là điểm yếu về tính ổn định vì:

- chạy từ repo root sẽ lỗi path
- khó đóng gói thành module hay CI job

### 6. Phần kết luận đang hơi cứng ngưỡng

Hiện tại script dùng ngưỡng:

- `abs(dp) > 0.1`
- `dir_ratio < 0.8`

Đây là rule of thumb để minh họa, không phải chân lý tuyệt đối. Trong thực tế cần diễn giải theo:

- bối cảnh nghiệp vụ
- quy định pháp lý
- cost của false positive/false negative
- kích thước mẫu từng nhóm

## Kết luận học thuật

Nếu xem đây là một project học tập, repo này đủ tốt để minh họa:

- cách nạp model đã train
- cách chia sensitive group
- cách tính một vài fairness metric cơ bản

Nhưng nếu xem đây là một fairness audit nghiêm túc, thì repo hiện còn thiếu:

- sửa lỗi mapping giới tính
- bổ sung nhiều fairness metric hơn
- tách train/test rõ ràng
- thêm pipeline huấn luyện và tái lập thí nghiệm
- thêm phân tích SHAP theo đúng đề bài
- làm rõ ngữ cảnh nghiệp vụ và cách diễn giải kết quả

## Hướng cải thiện tiếp theo

Các bước nên làm tiếp:

1. Sửa logic xác định giới tính từ `Attribute9`
2. Refactor script để không phụ thuộc relative path
3. Thêm `train.py` hoặc notebook huấn luyện `XGBoost`
4. Thêm `SHAP` để kiểm tra feature importance và vị trí của `Gender`
5. Tính thêm confusion matrix và fairness metrics khác
6. Ghi kết quả ra file báo cáo thay vì chỉ `print`
7. Bổ sung test cơ bản cho data loading và fairness logic
