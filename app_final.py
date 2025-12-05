import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# ==========================================
# 1. CONFIG & CSS
# ==========================================
st.set_page_config(page_title="NIDS Report", layout="wide")
pd.set_option("styler.render.max_elements", 5000000)
current_dir = os.path.dirname(os.path.abspath(__file__))

# COLOR SCHEME (Giữ nguyên theo yêu cầu của bạn)
RED_MAIN = '#FF0000'  # Red for key metrics values
BLACK_MAIN = '#000000' # Black for text
GRAY_SUB = '#808080'   # Gray for subtext

st.markdown(f"""
    <style>
    /* Font */
    h1, h2, h3 {{ font-family: 'Arial', sans-serif; color: {BLACK_MAIN}; }}
    
    /* Buttons */
    .stButton>button {{
        width: 100%; border-radius: 4px; height: 45px; 
        font-weight: bold; text-transform: uppercase;
        background-color: white; color: black; border: 1px solid black;
    }}
    .stButton>button:hover {{ background-color: #f0f0f0; color: black; }}
    
    /* Tabs */
    .stTabs [aria-selected="true"] {{ border-bottom: 4px solid {RED_MAIN}; color: {RED_MAIN}; font-weight: bold; }}
    
    /* Metrics */
    div[data-testid="stMetricLabel"] {{
        color: {GRAY_SUB}; font-size: 16px; font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
        color: {RED_MAIN}; font-size: 30px; font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD SYSTEM
# ==========================================
@st.cache_resource
def load_system(ds_name):
    path = os.path.join(current_dir, 'model_storage', ds_name)
    try:
        models = joblib.load(os.path.join(path, 'all_models.pkl'))
        scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        pca = joblib.load(os.path.join(path, 'pca.pkl'))
        metadata = joblib.load(os.path.join(path, 'metadata.pkl'))
        base_perf = joblib.load(os.path.join(path, 'performance.pkl'))
        return models, scaler, pca, metadata, base_perf
    except Exception as e:
        # st.error(f"Error loading system: {e}") # Có thể bỏ qua lỗi này khi chạy lần đầu
        return None, None, None, None, None

# SIDEBAR
st.sidebar.title("SETUP")
dataset = st.sidebar.selectbox("Dataset:", ("NSL-KDD", "CICIDS2017"))
st.sidebar.markdown("---")
mode = st.sidebar.radio("Function:", ["1. Audit", "2. Benchmark", "3. Dashboard"])

models, scaler, pca, metadata, base_perf = load_system(dataset)
if not models:
    st.error(f"Error: Không tìm thấy models cho {dataset}. Vui lòng chạy file `train_system.py` trước."); st.stop()

# PREPROCESS
def preprocess(df_in):
    df = df_in.copy()
    ignore = ['class', 'Label', ' Label', 'id', 'Destination Port', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Timestamp']
    df.drop([c for c in ignore if c in df.columns], axis=1, inplace=True, errors='ignore')
    
    if dataset == 'CICIDS2017':
        df.columns = df.columns.str.strip()
        df.replace([np.inf, -np.inf], np.nan, inplace=True); df.fillna(0, inplace=True)
        df = df.select_dtypes(include=[np.number])
    if dataset == 'NSL-KDD':
        df.drop(['num_outbound_cmds', 'is_host_login'], axis=1, inplace=True, errors='ignore')
        categorical_cols = [col for col in ['protocol_type', 'service', 'flag'] if col in df.columns]
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols)
        
    # Reindex to match training columns
    df = df.reindex(columns=metadata['final_columns'], fill_value=0)
    
    # Scale and apply PCA
    scaled = scaler.transform(df)
    transformed = pca.transform(scaled)
    return transformed

if 'pred_done' not in st.session_state: 
    st.session_state.update({'pred_done': False, 'y_pred': None, 'df_input': None})

# ==========================================
# 3. MAIN INTERFACE
# ==========================================
st.title(f"Hệ thống Phát hiện Xâm nhập (NIDS) - {dataset}")
st.markdown("---")

# --- MODE 1: AUDIT ---
if mode == "1. Audit":
    st.subheader("Kiểm thử Từng Mô hình")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.subheader("Tham số")
        model_name = st.selectbox("Mô hình:", list(models.keys()), index=3)
        st.markdown("---")
        f_in = st.file_uploader("1. Input Data (.csv)", type="csv")
    with c2:
        if f_in:
            df = pd.read_csv(f_in)
            st.write(f"Dữ liệu đầu vào: `{f_in.name}` ({len(df)} dòng)")
            if st.button("THỰC HIỆN PHÂN LOẠI"):
                try:
                    st.session_state.y_pred = models[model_name].predict(preprocess(df))
                    st.session_state.df_input = df
                    st.session_state.pred_done = True
                    st.success("Phân loại hoàn tất!")
                except Exception as e: st.error(f"Lỗi: {e}")

        if st.session_state.pred_done:
            y_p = st.session_state.y_pred
            st.subheader("Kết quả Phân loại")
            k1, k2 = st.columns(2)
            k1.metric("Bình thường (0)", np.sum(y_p == 0))
            k2.metric("Tấn công (1)", np.sum(y_p == 1))
            st.markdown("---")
            
            f_tr = st.file_uploader("2. Ground Truth (.csv) để so sánh", type="csv")
            if f_tr:
                df_tr = pd.read_csv(f_tr)
                if st.button("SO SÁNH & TÌM LỖI"):
                    lbl = next((c for c in ['class', 'Label', ' Label'] if c in df_tr.columns), None)
                    if lbl and len(df_tr) == len(st.session_state.df_input):
                        y_t = df_tr[lbl].apply(lambda x: 0 if str(x) in ['0', 'normal', 'BENIGN'] else 1).values
                        
                        # Tính CM để lấy 4 giá trị
                        cm_array = confusion_matrix(y_t, y_p)
                        TN, FP, FN, TP = cm_array.ravel()
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Accuracy", f"{accuracy_score(y_t, y_p):.2%}")
                        m2.metric("Precision", f"{precision_score(y_t, y_p, zero_division=0):.2%}")
                        m3.metric("Recall (Độ nhạy)", f"{recall_score(y_t, y_p, zero_division=0):.2%}")
                        m4.metric("F1-Score", f"{f1_score(y_t, y_p, zero_division=0):.2%}")
                        
                        st.markdown("#### Phân Tích Độ Sai Lệch (Confusion Matrix)")
                        c_cm_m1, c_cm_m2, c_cm_m3, c_cm_m4 = st.columns(4)
                        
                        c_cm_m1.metric("TN (Bình thường đúng)", f"{TN}", delta="Gói tin an toàn", delta_color="normal")
                        c_cm_m2.metric("TP (Tấn công đúng)", f"{TP}", delta="Phát hiện thành công", delta_color="inverse")
                        c_cm_m3.metric("FP (Báo động giả)", f"{FP}", delta="Lỗi cảnh báo sai", delta_color="inverse")
                        c_cm_m4.metric("FN (Bỏ sót tấn công)", f"{FN}", delta="Lỗi nguy hiểm", delta_color="inverse")
                        
                        err = np.where(y_t != y_p)[0]
                        if len(err) > 0:
                            st.error(f"Sai lệch: {len(err)} mẫu.")
                            
                            df_debug = st.session_state.df_input.copy()
                            df_debug['Actual'] = y_t; df_debug['Predicted'] = y_p
                            
                            ec1, ec2 = st.columns(2)
                            with ec1: 
                                missed = df_debug[(df_debug['Actual']==1) & (df_debug['Predicted']==0)]
                                st.write(f"**Bỏ sót tấn công (False Negatives): {len(missed)}**")
                                if not missed.empty: st.dataframe(missed.head(100))
                            with ec2:
                                false_alarm = df_debug[(df_debug['Actual']==0) & (df_debug['Predicted']==1)]
                                st.write(f"**Báo động giả (False Positives): {len(false_alarm)}**")
                                if not false_alarm.empty: st.dataframe(false_alarm.head(100))
                                
                            fig_cm = px.imshow(confusion_matrix(y_t, y_p), text_auto=True, aspect="equal", 
                                               color_continuous_scale='Greys', 
                                               x=['Normal', 'Anomaly'], y=['Normal', 'Anomaly'],
                                               labels=dict(x="Predicted", y="Actual"))
                            fig_cm.update_layout(title="Confusion Matrix", width=400, height=400)
                            fig_cm.update_coloraxes(showscale=False)
                            # FIX: Thêm key duy nhất
                            st.plotly_chart(fig_cm, key="audit_cm_chart")
                        else: 
                            st.success("Chính xác 100%.")
                    else: st.error("Lỗi file nhãn hoặc số dòng không khớp.")

# --- MODE 2: BENCHMARK ---
elif mode == "2. Benchmark":
    st.subheader("Đánh giá Toàn diện (Benchmark)")
    st.write("Chạy so sánh 5 mô hình trên tập dữ liệu mới.")
    
    c1, c2 = st.columns(2)
    with c1:
        f_features = st.file_uploader("Tải File Dữ liệu Test (KHÔNG CÓ NHÃN)", key='bench_x', type="csv")
    with c2:
        f_labels = st.file_uploader("Tải File Nhãn Gốc (GROUND TRUTH)", key='bench_y', type="csv")
    
    if f_features and f_labels:
        df_x = pd.read_csv(f_features)
        df_y = pd.read_csv(f_labels)
        
        lbl = next((c for c in ['class', 'Label', ' Label'] if c in df_y.columns), None)
        
        if lbl and len(df_x) == len(df_y):
            st.success(f"Hai file khớp nhau ({len(df_x)} dòng). Sẵn sàng Benchmark.")
            if st.button("BẮT ĐẦU ĐÁNH GIÁ TẤT CẢ MÔ HÌNH"):
                
                X = preprocess(df_x)
                y_t = df_y[lbl].apply(lambda x: 0 if str(x) in ['0', 'normal', 'BENIGN'] else 1).values
                
                res = {}
                prog = st.progress(0, text="Đang dự đoán và tính toán hiệu năng...")
                
                for i, (name, m) in enumerate(models.items()):
                    yp = m.predict(X)
                    cm_array = confusion_matrix(y_t, yp)
                    TN, FP, FN, TP = cm_array.ravel() 
                    
                    res[name] = {
                        'Accuracy': accuracy_score(y_t, yp), 'F1-Score': f1_score(y_t, yp, zero_division=0),
                        'Recall': recall_score(y_t, yp, zero_division=0), 'Precision': precision_score(y_t, yp, zero_division=0),
                        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
                        'CM': cm_array
                    }
                    prog.progress((i+1)/5, text=f"Đang đánh giá {name}...")
                
                prog.empty()
                st.markdown("---")
                st.subheader("2. Kết Quả Benchmark")
                
                # 1. BẢNG XẾP HẠNG
                st.markdown("### Bảng Xếp Hạng")
                df_res = pd.DataFrame(res).T.reset_index().rename(columns={'index':'Model'})
                
                df_display = df_res.drop(columns=['CM'], errors='ignore') 
                
                metric_cols_all = ['Accuracy', 'F1-Score', 'Recall', 'Precision', 'TN', 'FP', 'FN', 'TP']
                
                df_styled = df_display.style.highlight_max(axis=0, color='#FFC0CB', subset=metric_cols_all)
                # FIX WARNING: Dùng width thay vì use_container_width (nếu phiên bản cũ, giữ nguyên tham số cũ cũng được, ở đây tôi dùng use_container_width cho bản mới)
                st.dataframe(df_styled, use_container_width=True)
                
                # Biểu đồ cột so sánh
                df_chart = df_display.drop(columns=['TN', 'FP', 'FN', 'TP']).melt(id_vars='Model', value_vars=['Accuracy', 'F1-Score', 'Recall'])
                fig = px.bar(df_chart, x='Model', y='value', color='variable', barmode='group', 
                             text_auto='.2%', height=400, 
                             color_discrete_sequence=[RED_MAIN, BLACK_MAIN, GRAY_SUB], 
                             title="Comparison of Key Metrics")
                # FIX: Thêm key duy nhất
                st.plotly_chart(fig, key="bench_bar_chart")
                
                # 2. So sánh Train vs Test 
                st.markdown("### So sánh với lúc huấn luyện (Train vs Test)")
                df_train = pd.DataFrame(base_perf).T.reset_index().rename(columns={'index': 'Model'})
                df_train.columns = [c + ' (Train)' if c != 'Model' else c for c in df_train.columns]
                
                df_bench = df_display.drop(columns=['TN', 'FP', 'FN', 'TP'], errors='ignore').copy() 
                df_bench.columns = [c + ' (Benchmark)' if c != 'Model' else c for c in df_bench.columns]
                df_merged = pd.merge(df_train, df_bench, on='Model', how='inner')
                
                for m in ['Accuracy', 'F1-Score']:
                    df_merged[m + ' Delta'] = df_merged[m + ' (Benchmark)'] - df_merged[m + ' (Train)']
                
                def color_delta(val):
                    if isinstance(val, (int, float)):
                        return f'color: {RED_MAIN}' if val < -0.05 else f'color: {BLACK_MAIN}'
                    return 'color: black'
                
                try:
                    st.dataframe(df_merged.style.map(color_delta, subset=[c for c in df_merged.columns if 'Delta' in c]), use_container_width=True)
                except Exception:
                    st.dataframe(df_merged, use_container_width=True)

                # 3. CHI TIẾT TỪNG MÔ HÌNH
                st.markdown("### Chi Tiết Từng Mô Hình (Ma trận nhầm lẫn)")
                tabs = st.tabs(list(models.keys()))
                for i, (k, v) in enumerate(res.items()):
                    with tabs[i]:
                        st.markdown("##### Phân Tích Phát Hiện")
                        col_cm_m1, col_cm_m2, col_cm_m3, col_cm_m4 = st.columns(4)
                        
                        col_cm_m1.metric("TN (Bình thường đúng)", f"{v['TN']}", delta="Gói tin an toàn", delta_color="normal")
                        col_cm_m2.metric("TP (Tấn công đúng)", f"{v['TP']}", delta="Phát hiện thành công", delta_color="inverse")
                        col_cm_m3.metric("FP (Báo động giả)", f"{v['FP']}", delta="Lỗi cảnh báo sai", delta_color="inverse")
                        col_cm_m4.metric("FN (Bỏ sót tấn công)", f"{v['FN']}", delta="Lỗi nguy hiểm", delta_color="inverse")
                        
                        st.markdown("##### Thống kê Metric và Biểu đồ")
                        c1, c2 = st.columns([1, 2])
                        c1.metric("Accuracy", f"{v['Accuracy']:.2%}")
                        c1.metric("F1-Score", f"{v['F1-Score']:.2%}")
                        c1.metric("Recall", f"{v['Recall']:.2%}")
                        
                        fig_cm = px.imshow(v['CM'], text_auto=True, aspect="equal", color_continuous_scale='Greys',
                                           x=['Normal', 'Anomaly'], y=['Normal', 'Anomaly'])
                        fig_cm.update_layout(height=350, width=350, margin=dict(l=0,r=0,t=0,b=0))
                        fig_cm.update_coloraxes(showscale=False)
                        # FIX: Thêm key duy nhất cho mỗi biểu đồ trong vòng lặp (QUAN TRỌNG ĐỂ FIX LỖI CRASH)
                        c2.plotly_chart(fig_cm, key=f"bench_cm_chart_{i}")
        else:
            st.error("⚠️ Lỗi: Đảm bảo file Ground Truth chứa cột nhãn (class/Label) và số dòng khớp với file Features.")


# --- MODE 3: DASHBOARD (CẬP NHẬT CHI TIẾT) ---
elif mode == "3. Dashboard":
    st.subheader(f"Dashboard Hiệu suất Huấn luyện ({dataset})")
    
    # Sắp xếp theo F1-Score giảm dần
    df_base = pd.DataFrame([{'Model': m, **p} for m, p in base_perf.items()]).sort_values('F1-Score', ascending=False)
    best_model_name = df_base.iloc[0]['Model']
    best_model_data = df_base.iloc[0]
    metric_cols_dash = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    st.markdown(f"#### Mô hình Tốt nhất: **{best_model_name}**")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{best_model_data['Accuracy']:.2%}")
    m2.metric("Precision", f"{best_model_data['Precision']:.2%}")
    m3.metric("Recall (Độ nhạy)", f"{best_model_data['Recall']:.2%}")
    m4.metric("F1-Score", f"{best_model_data['F1-Score']:.2%}")
    
    st.markdown("---")

    # --- DANH SÁCH XẾP HẠNG CHI TIẾT (TEXT THUẦN TÚY - KHÔNG ICON) ---
    st.markdown("#### Bảng Xếp Hạng Chi Tiết (Theo F1-Score)")
    st.info("Xếp hạng dựa trên chỉ số F1-Score (Độ cân bằng giữa Precision và Recall).")

    for i in range(len(df_base)):
        row = df_base.iloc[i]
        rank = i + 1
        model_name = row['Model']
        stats = f"F1-Score: **{row['F1-Score']:.4f}** | Accuracy: {row['Accuracy']:.4f} | Recall: {row['Recall']:.4f}"
        
        # Hiển thị dạng text đơn giản "Top 1:", "Top 2:"...
        st.markdown(f"**Top {rank}: {model_name}**")
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;👉 {stats}") 
        st.write("") 
    
    st.markdown("---")
    # ----------------------------------------------------

    st.markdown("#### Biểu đồ So sánh Tổng quan")
    
    df_long = df_base.melt(id_vars='Model', var_name='Metric', value_name='Value')
    # FIX: Thêm key duy nhất
    st.plotly_chart(px.bar(df_long, x='Model', y='Value', color='Metric', barmode='group', height=400, 
                           color_discrete_sequence=[RED_MAIN, BLACK_MAIN, GRAY_SUB, '#CCCCCC']), key="dash_bar_chart")
    
    c1, c2 = st.columns(2)
    with c1:
        fig_r = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for i, m in enumerate(df_base['Model'].head(3)):
            v = df_base[df_base['Model']==m][metrics].values.flatten().tolist()
            color = RED_MAIN if i == 0 else (BLACK_MAIN if i == 1 else GRAY_SUB)
            dash = 'solid' if i == 0 else ('dash' if i == 1 else 'dot')
            
            fig_r.add_trace(go.Scatterpolar(r=v+[v[0]], theta=metrics+[metrics[0]], name=m, 
                                            line=dict(color=color, width=2, dash=dash), fill='toself', opacity=0.1))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.5, 1], gridcolor=GRAY_SUB, linecolor=BLACK_MAIN, tickfont=dict(color=BLACK_MAIN)),
                angularaxis=dict(tickfont=dict(color=BLACK_MAIN))
            ), 
            height=400, title="Radar Chart (Top 3)", font=dict(color=BLACK_MAIN)
        )
        # FIX: Thêm key duy nhất
        st.plotly_chart(fig_r, key="dash_radar_chart")
        
    with c2:
        fig_l = px.line(df_long, x='Model', y='Value', color='Metric', markers=True, 
                        color_discrete_sequence=[RED_MAIN, BLACK_MAIN, GRAY_SUB, '#CCCCCC'], title="Trend")
        fig_l.update_layout(height=400)
        # FIX: Thêm key duy nhất
        st.plotly_chart(fig_l, key="dash_line_chart")

    st.markdown("---")
    st.markdown("### Performance Table")
    df_styled_dash = df_base.style.highlight_max(axis=0, color=RED_MAIN, subset=metric_cols_dash)
    st.dataframe(df_styled_dash, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Cơ Chế Hoạt Động Của Các Thuật Toán Phân Loại")
    
    tabs = st.tabs(list(models.keys()))
    
    # Nội dung mô tả dựa trên logic training
    model_visuals = {
        "Random Forest": {
            "title": "Rừng Ngẫu Nhiên (Random Forest)",
            "concept": "Tổ hợp nhiều cây quyết định độc lập, kết quả là **phiếu bầu đa số**. Giúp giảm thiểu lỗi Overfitting và tăng tính ổn định của mô hình. ",
        },
        "k-NN": {
            "title": "k-Hàng Xóm Gần Nhất (k-NN)",
            "concept": "Phân loại dựa trên **khoảng cách**. Điểm dữ liệu mới được gán nhãn theo lớp chiếm đa số của **k** điểm gần nhất. Phù hợp cho dữ liệu có ranh giới quyết định phức tạp. ",
        },
        "SVM": {
            "title": "Máy Vector Hỗ Trợ (SVM)",
            "concept": "Tìm **Siêu mặt phẳng (Hyperplane)** với **Margin** lớn nhất để phân tách hai lớp dữ liệu. Chỉ các điểm gần ranh giới (Support Vectors) mới ảnh hưởng đến việc phân loại. ",
        },
        "Decision Tree": {
            "title": "Cây Quyết Định (Decision Tree)",
            "concept": "Cấu trúc dạng cây phân nhánh (Flowchart) sử dụng **quy tắc IF-THEN** tuần tự để đưa ra quyết định, dễ giải thích và trực quan nhất. ",
        },
        "Logistic Regression": {
            "title": "Hồi Quy Logistic (Logistic Regression)",
            "concept": "Là một mô hình tuyến tính, sử dụng hàm **Sigmoid** để ước tính xác suất. Đường phân chia quyết định (Decision Boundary) là **tuyến tính (một đường thẳng)**. ",
        }
    }
    
    for i, model_name in enumerate(models.keys()):
        with tabs[i]:
            visual = model_visuals.get(model_name)
            if visual:
                st.markdown(f"#### {visual['title']}")
                st.write(visual['concept'])
            else:
                st.write(f"Mô tả cho {model_name} đang được cập nhật.")