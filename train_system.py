import pandas as pd
import numpy as np
import os
import joblib
import glob
import shutil
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# COMMON FUNCTION: LOAD DATA
# ==========================================
def load_data(dataset_type, current_dir, sample_frac=0.05):
    """
    Load and preprocess data for the specified dataset type.
    """
    logging.info(f"Loading data for: {dataset_type}...")

    if dataset_type == 'NSL-KDD':
        path = os.path.join(current_dir, 'data', 'NSL-KDD', 'NSL_Binary.csv')
        if not os.path.exists(path):
            logging.warning(f"Data not found for {dataset_type}. Skipping.")
            return None
        try:
            df = pd.read_csv(path)
            df = df.drop_duplicates()
            df.drop(['num_outbound_cmds', 'is_host_login'], axis=1, inplace=True, errors='ignore')
            df['class'] = df['class'].apply(lambda x: 1 if x == 'anomaly' else 0)
            X = df.drop('class', axis=1)
            X = pd.get_dummies(X, columns=['protocol_type', 'service', 'flag'])
            y = df['class']
            
            for col in X.select_dtypes(include=['float64', 'int64']).columns:
                X[col] = pd.to_numeric(X[col], downcast='float') if 'float' in str(X[col].dtype) else pd.to_numeric(X[col], downcast='integer')
            return X, y
        except Exception as e:
            logging.error(f"Error loading NSL-KDD: {e}")
            return None

    elif dataset_type == 'CICIDS2017':
        folder_path = os.path.join(current_dir, 'data', 'CICIDS2017')
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not all_files:
            logging.warning(f"No CSV files found for {dataset_type}. Skipping.")
            return None

        df_list = []
        for f in all_files:
            try:
                cols = pd.read_csv(f, nrows=1).columns
                dtypes = {col: 'float32' for col in cols if col.strip() != 'Label'}
                
                df = pd.read_csv(f, dtype=dtypes).sample(frac=sample_frac, random_state=42)
                logging.info(f"Loaded and sampled: {os.path.basename(f)}")
                df_list.append(df)
            except Exception as e:
                logging.warning(f"Error loading {os.path.basename(f)}: {e}")

        if not df_list:
            logging.warning(f"No valid data loaded for {dataset_type}.")
            return None

        df = pd.concat(df_list, ignore_index=True)
        df = df.drop_duplicates()
        
        df.columns = df.columns.str.strip()
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        df.dropna(inplace=True)

        if 'Label' not in df.columns:
            logging.warning("'Label' column not found in CICIDS2017 data after cleaning. Skipping.")
            return None

        df['class'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        df.drop('Label', axis=1, inplace=True)
        
        X = df.drop('class', axis=1).select_dtypes(include=[np.number])
        y = df['class']
        
        for col in X.select_dtypes(include=['float64', 'int64']).columns:
            X[col] = pd.to_numeric(X[col], downcast='float') if 'float' in str(X[col].dtype) else pd.to_numeric(X[col], downcast='integer')
        return X, y

# ==========================================
# FUNCTION: GENERATE & SAVE VISUALS
# ==========================================
def generate_and_save_visuals(save_path, model_performance, trained_models, X_test_pca, y_test):
    """
    Tạo và lưu các biểu đồ chứng minh hiệu suất và bảng so sánh CSV.
    """
    logging.info("Generating visual proofs and CSV report...")
    
    # 1. LƯU BẢNG SO SÁNH (CSV)
    df_perf = pd.DataFrame(model_performance).T.reset_index().rename(columns={'index': 'Model'})
    csv_path = os.path.join(save_path, 'model_comparison.csv')
    df_perf.to_csv(csv_path, index=False)
    logging.info(f"Saved comparison CSV: {csv_path}")
    
    # 2. BIỂU ĐỒ SO SÁNH (Bar Chart)
    df_long = df_perf.melt(id_vars='Model', var_name='Metric', value_name='Value')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_long, x='Model', y='Value', hue='Metric', palette='viridis')
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0.5, 1.05)
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    chart_path = os.path.join(save_path, '1_Model_Comparison.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    logging.info(f"Saved comparison chart: {chart_path}")

    # 3. CONFUSION MATRIX (CHO TẤT CẢ MÔ HÌNH)
    logging.info("Generating Confusion Matrices for all models...")
    
    for name, model in trained_models.items():
        try:
            y_pred = model.predict(X_test_pca)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
            plt.title(f'Confusion Matrix - {name}', fontsize=12, fontweight='bold')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            
            # Lưu file với tên riêng cho từng model, thay khoảng trắng bằng _
            safe_name = name.replace(" ", "_")
            cm_path = os.path.join(save_path, f'2_Confusion_Matrix_{safe_name}.png')
            plt.savefig(cm_path, dpi=300)
            plt.close()
            logging.info(f"Saved CM for {name}")
            
        except Exception as e:
            logging.warning(f"Could not generate CM for {name}: {e}")

# ==========================================
# COMMON FUNCTION: TRAIN & SAVE
# ==========================================
def train_and_save(dataset_type, sample_frac=0.05):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Load Data
    data = load_data(dataset_type, current_dir, sample_frac)
    if data is None:
        return

    X, y = data
    logging.info(f"Data for {dataset_type} ready: {X.shape}")

    # 2. Split & Preprocess
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Drop highly correlated
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_train.drop(to_drop, axis=1, inplace=True)
    X_test.drop(to_drop, axis=1, inplace=True)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # PCA
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    logging.info(f"PCA reduced dimensions to: {X_train_pca.shape[1]}")

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_pca, y_train)
    logging.info(f"After SMOTE: {X_train_res.shape}")

    # 3. Training Loop
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "k-NN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(criterion='gini', random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    }

    trained_models = {}
    model_performance = {}

    logging.info(f"Starting training for 5 models on {dataset_type}...")
    for name, model in models.items():
        logging.info(f"Training {name}...")
        try:
            # Lưu CV F1-score vào dictionary để lưu ra file
            cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5, scoring='f1')
            cv_mean = cv_scores.mean()
            logging.info(f"{name} CV F1-score: {cv_mean:.4f}")

            model.fit(X_train_res, y_train_res)
            y_pred = model.predict(X_test_pca)
            trained_models[name] = model
            model_performance[name] = {
                'CV F1-Score': cv_mean,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
                'Recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            }
        except Exception as e:
            logging.error(f"Error training {name}: {e}")

    # 4. Save Artifacts
    save_path = os.path.join(current_dir, 'model_storage', dataset_type)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # --- GỌI HÀM LƯU ẢNH & CSV ---
    generate_and_save_visuals(save_path, model_performance, trained_models, X_test_pca, y_test)
    # -----------------------------

    metadata = {
        'dataset_type': dataset_type,
        'final_columns': list(X_train.columns),
        'dropped_cols': to_drop,
        'pca_components': pca.n_components_
    }

    joblib.dump(trained_models, os.path.join(save_path, 'all_models.pkl'))
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(save_path, 'pca.pkl'))
    joblib.dump(metadata, os.path.join(save_path, 'metadata.pkl'))
    joblib.dump(model_performance, os.path.join(save_path, 'performance.pkl'))

    logging.info(f"Saved models, csv report and all visual proofs to: model_storage/{dataset_type}")

# ==========================================
# MAIN PROGRAM
# ==========================================
if __name__ == "__main__":
    train_and_save('NSL-KDD')
    train_and_save('CICIDS2017')
    logging.info("Completed all training processes!")  