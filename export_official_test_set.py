import pandas as pd
import numpy as np
import os
import glob
import logging
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cấu hình phải khớp chính xác với train_system.py
TEST_SIZE = 0.2
RANDOM_STATE = 42 # QUAN TRỌNG: Phải là 42 để khớp với lúc train!
SAMPLE_FRAC_CICIDS = 0.05 # Phải khớp với lúc train CICIDS

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'official_test_set')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ==========================================
# COMMON FUNCTION: LOAD DATA (Tái tạo từ train_system.py)
# ==========================================
def load_data(dataset_type, current_dir, sample_frac=0.05):
    """ Tải dữ liệu thô và tiền xử lý """
    if dataset_type == 'NSL-KDD':
        path = os.path.join(current_dir, 'data', 'NSL-KDD', 'NSL_Binary.csv')
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        df = df.drop_duplicates()
        df.drop(['num_outbound_cmds', 'is_host_login'], axis=1, inplace=True, errors='ignore')
        df['class'] = df['class'].apply(lambda x: 1 if x == 'anomaly' else 0)
        X = df.drop('class', axis=1)
        X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'])
        y = df['class']
        # Đảm bảo ép kiểu cho NSL-KDD để tránh lỗi
        for col in X.select_dtypes(include=['float64', 'int64']).columns:
            X[col] = pd.to_numeric(X[col], downcast='float') if 'float' in str(X[col].dtype) else pd.to_numeric(X[col], downcast='integer')
        return X, y
    
    elif dataset_type == 'CICIDS2017':
        folder_path = os.path.join(current_dir, 'data', 'CICIDS2017')
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not all_files: return None
        
        df_list = []
        for f in all_files:
            try:
                cols = pd.read_csv(f, nrows=1).columns
                dtypes = {col: 'float32' for col in cols if col.strip() != 'Label'}
                df = pd.read_csv(f, dtype=dtypes).sample(frac=sample_frac, random_state=42)
                df_list.append(df)
            except Exception as e:
                logging.warning(f"Error loading {os.path.basename(f)}: {e}")
                
        if not df_list: return None
        
        df = pd.concat(df_list, ignore_index=True)
        df = df.drop_duplicates()
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.dropna(inplace=True)

        if 'Label' not in df.columns: return None

        # Gán nhãn cho dễ đọc
        df['class'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # Xóa các cột không cần thiết để có X
        X = df.drop(['class', 'Label'], axis=1).select_dtypes(include=[np.number])
        y = df['class']

        # Đảm bảo ép kiểu cho CICIDS
        for col in X.select_dtypes(include=['float64', 'int64']).columns:
            X[col] = pd.to_numeric(X[col], downcast='float') if 'float' in str(X[col].dtype) else pd.to_numeric(X[col], downcast='integer')
        
        # Thêm lại cột Label gốc để tiện cho việc lưu file
        X['Label'] = df['Label']
        return X, y

# ==========================================
# HÀM CHÍNH: XUẤT FILE
# ==========================================
def export_test_set(dataset_type, sample_frac):
    logging.info(f"Xuất bộ dữ liệu Test chính thức cho: {dataset_type}...")
    
    # 1. Load Data
    data = load_data(dataset_type, current_dir, sample_frac)
    if data is None:
        logging.warning(f"Không thể tải dữ liệu gốc cho {dataset_type}. Bỏ qua.")
        return

    X_full, y_full = data
    
    # 2. Tái tạo chính xác bước split ban đầu
    # Lưu ý: Lần chạy này chỉ dùng để lấy X_test, y_test; X_train sẽ bị bỏ qua.
    X_train_dummy, X_test, y_train_dummy, y_test = train_test_split(
        X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_full
    )
    
    logging.info(f"Kích thước bộ Test chính thức: {X_test.shape[0]} dòng.")

    # 3. Chuẩn bị file CÓ Đáp án (Labeled)
    # NSL: Thêm cột 'class' vào X_test
    if dataset_type == 'NSL-KDD':
        df_labeled = X_test.copy()
        df_labeled['class'] = y_test.map({1: 'anomaly', 0: 'normal'})
    # CICIDS: Cột 'Label' đã có sẵn trong X_test (được thêm ở hàm load_data)
    elif dataset_type == 'CICIDS2017':
        df_labeled = X_test.copy()
        # CICIDS dùng cột Label gốc, không dùng cột 'class' 0/1 đã tạo ra
        
    labeled_path = os.path.join(output_dir, f'{dataset_type}_Official_Test_LABELED.csv')
    df_labeled.to_csv(labeled_path, index=False)
    logging.info(f"✅ Đã lưu file Labeled: {labeled_path}")

    # 4. Chuẩn bị file KHÔNG Đáp án (Unlabeled)
    # Loại bỏ cột nhãn khỏi file test
    if dataset_type == 'NSL-KDD':
        df_unlabeled = X_test.drop(columns=['protocol_type', 'service', 'flag'], errors='ignore')
    elif dataset_type == 'CICIDS2017':
        # Loại bỏ cột nhãn 'Label' (và 'class' nếu có)
        df_unlabeled = X_test.drop(columns=['Label'], errors='ignore')
        
    unlabeled_path = os.path.join(output_dir, f'{dataset_type}_Official_Test_UNLABELED.csv')
    df_unlabeled.to_csv(unlabeled_path, index=False)
    logging.info(f"✅ Đã lưu file Unlabeled: {unlabeled_path}")


if __name__ == "__main__":
    export_test_set('NSL-KDD', 0) # NSL không dùng sample_frac
    export_test_set('CICIDS2017', SAMPLE_FRAC_CICIDS)
    logging.info("Hoàn tất xuất bộ dữ liệu Test chính thức 100% độc lập!")