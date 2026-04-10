print("DEBUG: Script started")
"""
UCI German Credit Dataset (ID=144) - XGBoost Classifier (Refactored)
Nguồn: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
Dự đoán: Good Credit (1) / Bad Credit (0)
Output: Accuracy / F1 / ROC-AUC + Saved Model (.pkl)
"""

import numpy as np
import pandas as pd
import joblib
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

def load_data(dataset_id=144):
    """Tải dữ liệu từ UCI repository với fallback nếu server lỗi."""
    print(f"\n[1/5] Đang tải dữ liệu (dataset ID={dataset_id})...")
    
    local_filename = "german_credit.csv"
    
    # Ưu tiên 1: Thử file local
    if os.path.exists(local_filename):
        print(f"    [OK] Đang tải từ file local: {local_filename}")
        df = pd.read_csv(local_filename)
        return extract_from_df(df)

    # Ưu tiên 2: Thử tải từ UCI
    try:
        print(f"    [*] Đang thử kết nối UCI Repository (ID={dataset_id})...")
        german_credit = fetch_ucirepo(id=dataset_id)
        X = german_credit.data.features
        y = german_credit.data.targets.squeeze()
        print("    [OK] Tải dữ liệu thành công từ UCI.")
        return X, y
    except Exception as e:
        print(f"    [!] Lỗi khi kết nối UCI: {e}")

    # Ưu tiên 3: Thử tải từ mirror
    print("    [!] Đang thử tải từ mirror dự phòng (GitHub)...")
    mirror_url = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
    try:
        df = pd.read_csv(mirror_url)
        # Lưu lại để dùng cho lần sau
        df.to_csv(local_filename, index=False)
        print(f"    Tải từ mirror thành công và đã lưu vào {local_filename}.")
        return extract_from_df(df)
    except Exception as mirror_e:
        raise ConnectionError(f"Không thể tải dữ liệu từ cả UCI, file local và mirror: {mirror_e}")

def extract_from_df(df):
    """Hàm bổ trợ để tách features và target từ DataFrame."""
    if 'credit_risk' in df.columns:
        X = df.drop(columns=['credit_risk'])
        y = df['credit_risk']
    else:
        # Trường hợp không có tên cột 'credit_risk', lấy cột cuối cùng
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    return X, y

def preprocess_data(X_raw, y_raw):
    """Tiền xử lý: mã hóa nhãn và One-Hot Encoding cho features."""
    print("\n[2/5] Tiền xử lý dữ liệu...")
    
    # Nhãn gốc UCI: 1 = good, 2 = bad
    # Mirror Selva86: 1 = good, 0 = bad
    # Chúng ta muốn thống nhất về: 1 = good, 0 = bad
    y = y_raw.replace({2: 0}).astype(int)
    
    # One-Hot Encoding cho categorical, giữ nguyên numerical
    categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    
    # XGBoost không cho phép ký tự [, ] hoặc < trong tên feature
    # Các ký tự này thường xuất hiện khi pd.get_dummies lấy giá trị từ mirror
    X.columns = [
        col.replace("[", "_").replace("]", "_").replace("<", "_") 
        for col in X.columns
    ]
    
    print(f"    Phân phối nhãn: Good (1): {(y==1).sum()}, Bad (0): {(y==0).sum()}")
    print(f"    Tính năng sau mã hoá: {X.shape[1]} features")
    
    return X, y

def train_model(X_train, y_train, X_test, y_test):
    """Huấn luyện XGBoost với xử lý imbalance."""
    print("\n[3/5] Huấn luyện XGBoost Classifier...")
    
    # Tính class weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print("    Huấn luyện hoàn tất!")
    return model

def evaluate_model(model, X_test, y_test, X, y):
    """Đánh giá mô hình trên tập test và qua Cross-Validation."""
    print("\n[4/5] Đánh giá mô hình...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "=" * 50)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("=" * 50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Bad", "Good"]))
    
    # Cross-validation
    print("\nCross-Validation (5-fold):")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

def save_assets(model, feature_names, model_path="german_credit_model.pkl", features_path="feature_columns.pkl"):
    """Lưu mô hình và danh sách cột ra file pkl."""
    print(f"\n[5/5] Đang lưu assets...")
    joblib.dump(model, model_path)
    joblib.dump(feature_names, features_path)
    print(f"    Đã lưu mô hình: {model_path}")
    print(f"    Đã lưu danh sách cột: {features_path}")

def main():
    # 1. Load
    X_raw, y_raw = load_data()
    
    # 2. Preprocess
    X, y = preprocess_data(X_raw, y_raw)
    feature_names = X.columns.tolist()
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train
    model = train_model(X_train, y_train, X_test, y_test)
    
    # 5. Evaluate
    evaluate_model(model, X_test, y_test, X, y)
    
    # 6. Save
    save_assets(model, feature_names)
    
    print("\n[Done] Pipeline hoàn thành và mô hình đã được lưu!\n")

if __name__ == "__main__":
    main()
