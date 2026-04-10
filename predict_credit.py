"""
Script dự đoán rủi ro tín dụng (Inference Script)
Sử dụng mô hình XGBoost đã huấn luyện.
"""

import pandas as pd
import joblib
import numpy as np

def predict_from_input(model_path="german_credit_model.pkl", features_path="feature_columns.pkl"):
    """
    Tải mô hình và thực hiện dự đoán mẫu.
    """
    try:
        # 1. Tải mô hình và danh sách cột
        model = joblib.load(model_path)
        feature_columns = joblib.load(features_path)
        print(f"--- Đã tải mô hình và {len(feature_columns)} features thành công ---")

        # 2. Tạo một mẫu dữ liệu giả định để test (Dữ liệu sau khi mã hóa)
        # Lưu ý: Thực tế bạn cần nhận dữ liệu thô, sau đó qua bước preprocess tương tự train
        # Ở đây mình tạo nhanh một DataFrame với đúng các cột đã lưu
        sample_data = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Giả định một vài giá trị cho mẫu test
        # Ví dụ: duration=24, amount=3000, installment=4...
        if 'Attribute2' in sample_data.columns: sample_data['Attribute2'] = 24    # Duration
        if 'Attribute5' in sample_data.columns: sample_data['Attribute5'] = 3000  # Amount
        if 'Attribute8' in sample_data.columns: sample_data['Attribute8'] = 4     # Installment rate
        
        # 3. Dự đoán
        prediction = model.predict(sample_data)[0]
        probability = model.predict_proba(sample_data)[0, 1]

        result_label = "Good Credit" if prediction == 1 else "Bad Credit"
        
        print("\n[KẾT QUẢ DỰ ĐOÁN MẪU]")
        print(f"  - Trạng thái: {result_label}")
        print(f"  - Xác suất (Good): {probability:.4f}")
        print("-" * 30)

    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file model. Vui lòng chạy 'german_credit_xgboost.py' để huấn luyện và lưu mô hình trước.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    predict_from_input()
