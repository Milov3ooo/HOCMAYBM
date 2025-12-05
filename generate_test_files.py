import pandas as pd
import os
import glob

# ==========================================
# CẤU HÌNH
# ==========================================
# Số lượng dòng muốn tạo cho mỗi file test
NUM_SAMPLES = 1000 
# Random seed khác với lúc train (42) để đảm bảo dữ liệu xáo trộn khác đi
RANDOM_SEED = 999    

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'test_data_generated')

# Tạo thư mục chứa file test nếu chưa có
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_files(dataset_type):
    print(f"\n--- Đang xử lý: {dataset_type} ---")
    df = None
    
    # 1. Đọc dữ liệu gốc
    try:
        if dataset_type == 'NSL-KDD':
            path = os.path.join(current_dir, 'data', 'NSL-KDD', 'NSL_Binary.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Xử lý sơ bộ cột nhãn NSL
                if 'class' in df.columns:
                    pass # Giữ nguyên để lưu file
            else:
                print(f"❌ Không tìm thấy file dữ liệu NSL-KDD tại: {path}")
                return

        elif dataset_type == 'CICIDS2017':
            path = os.path.join(current_dir, 'data', 'CICIDS2017')
            files = glob.glob(os.path.join(path, "*.csv"))
            if files:
                # Lấy file đầu tiên hoặc gộp (ở đây lấy file đầu tiên cho nhanh)
                print(f"Đọc dữ liệu từ: {os.path.basename(files[0])}")
                df = pd.read_csv(files[0])
                # Xử lý khoảng trắng trong tên cột CICIDS (quan trọng)
                df.columns = df.columns.str.strip()
            else:
                print(f"❌ Không tìm thấy file CSV nào trong thư mục: {path}")
                return

        # 2. Lấy mẫu ngẫu nhiên (Dữ liệu test)
        if df is not None and not df.empty:
            # Lấy tối đa NUM_SAMPLES dòng (nếu file gốc nhỏ hơn thì lấy hết)
            real_n = min(NUM_SAMPLES, len(df))
            df_sample = df.sample(n=real_n, random_state=RANDOM_SEED)
            
            # 3. Lưu file CÓ đáp án (Labeled)
            lbl_path = os.path.join(output_dir, f'{dataset_type}_Test_Labeled.csv')
            df_sample.to_csv(lbl_path, index=False)
            print(f"✅ Đã tạo file có đáp án: {lbl_path}")
            
            # 4. Lưu file KHÔNG có đáp án (Unlabeled)
            # Xác định cột Label để xóa
            label_col = 'class' if dataset_type == 'NSL-KDD' else 'Label'
            
            if label_col in df_sample.columns:
                df_unlabeled = df_sample.drop(columns=[label_col])
                unlbl_path = os.path.join(output_dir, f'{dataset_type}_Test_Unlabeled.csv')
                df_unlabeled.to_csv(unlbl_path, index=False)
                print(f"✅ Đã tạo file không đáp án: {unlbl_path}")
            else:
                print(f"⚠️ Cảnh báo: Không tìm thấy cột '{label_col}' để xóa.")
                
    except Exception as e:
        print(f"❌ Lỗi khi xử lý {dataset_type}: {e}")

if __name__ == "__main__":
    generate_files('NSL-KDD')
    generate_files('CICIDS2017')
    print(f"\n🎉 Hoàn tất! Bạn hãy vào thư mục '{output_dir}' để lấy file.")