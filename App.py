import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
import io

# ==========================================
# CẤU HÌNH TRANG WEB ĐẦU TIÊN
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát Hiện Gian Lận & Quản Lý Rủi Ro Tín Dụng",
    page_icon="🛡️"
)

# ==========================================
# HÀM NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU DÙNG CHUNG (CACHE)
# ==========================================
@st.cache_data
def load_and_preprocess_data(file_bytes, file_name):
    """
    Nạp dữ liệu từ bytes và thực hiện xử lý làm sạch cơ bản ban đầu
    """
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        
        # Điền giá trị thiếu bằng trung vị (median) nếu có phát sinh
        df = df.fillna(df.median(numeric_only=True))
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# ==========================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu mẫu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện (CSV/Excel)", 
        type=["csv", "xlsx"],
        help="Chọn file dữ liệu gốc chứa các biến chỉ số X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    st.divider()
    st.subheader("🤖 Tham số mô hình AI")
    st.caption("Thuật toán: Random Forest Classifier")
    
    # Siêu tham số cấu hình mô hình
    n_estimators = st.slider(
        "Số lượng cây quyết định (n_estimators)", 
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây phân loại trong phân cụm rừng ngẫu nhiên."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa của cây (max_depth)", 
        min_value=2, max_value=30, value=10, step=1,
        help="Độ sâu tối đa của các nhánh cây để kiểm soát overfitting."
    )
    
    min_samples_split = st.slider(
        "Mẫu tối thiểu để phân tách nhánh", 
        min_value=2, max_value=10, value=2, step=1,
        help="Số lượng mẫu tối thiểu cần thiết để phân tách một nút nội bộ."
    )
    
    # Tham số nâng cao đặt trong Expander
    with st.expander("⚙️ Tham số nâng cao"):
        criterion = st.selectbox(
            "Tiêu chí đo lường chất lượng phân tách", 
            options=["gini", "entropy", "log_loss"], index=0,
            help="Hàm đo lường chất lượng của một phép tách."
        )
        random_state = st.number_input(
            "Trạng thái ngẫu nhiên (random_state)", 
            value=42, step=1,
            help="Mã số cố định sự ngẫu nhiên giúp tái hiện kết quả chính xác trong các lần chạy."
        )
        test_size = st.slider(
            "Tỷ lệ dữ liệu kiểm tra (Test Size)", 
            min_value=0.1, max_value=0.5, value=0.3, step=0.05,
            help="Tỷ lệ chia tập dữ liệu ra làm phần kiểm định độc lập."
        )

    st.divider()
    # Nút bấm kích hoạt huấn luyện duy nhất
    btn_train = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Hệ thống Phát Hiện Gian Lận & Dự Báo Rủi Ro Tín Dụng")
st.caption(
    "Ứng dụng hỗ trợ phân tích định lượng dữ liệu khách hàng, dự báo tự động trạng thái rủi ro vỡ nợ (Default) "
    "dựa trên nền tảng Học máy mô hình hóa từ dữ liệu lịch sử tài chính doanh nghiệp/cá nhân."
)

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên tệp dữ liệu ở vùng **Cấu hình** (Sidebar bên trái) để bắt đầu sử dụng ứng dụng.")
    st.stop()

# Đã tải file thành công, tiến hành đọc dữ liệu qua bộ nhớ cache
file_bytes = uploaded_file.read()
df_main = load_and_preprocess_data(file_bytes, uploaded_file.name)

if df_main is None:
    st.error("Không thể xử lý định dạng tệp đã tải lên. Hãy đảm bảo đó là tệp CSV hoặc Excel hợp lệ.")
    st.stop()

st.caption(f"📁 **Đang dùng tệp:** `{uploaded_file.name}` | Quy mô: **{df_main.shape[0]}** dòng và **{df_main.shape[1]}** cột.")
st.divider()

# Định nghĩa các biến cấu trúc mô hình suy từ dữ liệu thực tế
feature_cols = [c for c in df_main.columns if c.startswith('X_')]
target_col = 'default'

if len(feature_cols) == 0 or target_col not in df_main.columns:
    st.error(f"Cấu trúc tệp dữ liệu không hợp lệ. Tệp bắt buộc phải chứa các cột tính năng `X_1`, `X_2`,... và cột mục tiêu `{target_col}`.")
    st.stop()

# ==========================================
# KHỐI HUẤN LUYỆN (XỬ LÝ KHI BẤM NÚT TRÊN SIDEBAR)
# ==========================================
if btn_train:
    with st.spinner("Đang tiến hành chuẩn hóa dữ liệu và huấn luyện mô hình Random Forest..."):
        X = df_main[feature_cols]
        y = df_main[target_col]
        
        # Chia tập dữ liệu huấn luyện và kiểm thử
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Khởi tạo và áp dụng chuẩn hóa tính năng
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Huấn luyện mô hình
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion,
            random_state=random_state
        )
        model.fit(X_train_scaled, y_train)
        
        # Dự đoán kết quả phục vụ đánh giá chỉ tiêu
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Lưu trữ trạng thái vào session_state bền vững
        st.session_state['trained_model'] = model
        st.session_state['data_scaler'] = scaler
        st.session_state['feature_columns'] = feature_cols
        st.session_state['evaluation_metrics'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'y_test': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist() if y_prob is not None else [],
            'feature_importances': model.feature_importances_.tolist()
        }
    st.success("🎉 Huấn luyện mô hình hoàn tất! Xem kết quả chi tiết ở các Tab bên dưới.")

# ==========================================
# KHỞI TẠO HỆ THỐNG PHÂN CHIA CÁC TAB CHỨC NĂNG
# ==========================================
tab_overview, tab_viz, tab_metrics, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện & Kiểm định", 
    "🔮 Sử dụng mô hình dự báo"
])

# ------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ------------------------------------------
with tab_overview:
    st.subheader("🔍 Phân tích cấu trúc phân phối dữ liệu thô")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Tổng số dòng dữ liệu (Rows)", f"{df_main.shape[0]:,}")
    with col_m2:
        st.metric("Tổng số cột đặc trưng (Columns)", f"{df_main.shape[1]}")
    with col_m3:
        file_mb = len(file_bytes) / (1024 * 1024)
        st.metric("Dung lượng tệp xử lý", f"{file_mb:.2f} MB")
        
    st.markdown("##### 📄 Xem trước 5 dòng dữ liệu đầu tiên (Head)")
    st.dataframe(df_main.head(5), use_container_width=True)
    
    st.markdown("##### 📈 Bảng mô tả thống kê các biến đặc trưng trong mô hình")
    # Chỉ hiển thị các biến đưa vào mô hình để tránh loãng thông tin
    model_vars = feature_cols + [target_col] if target_col in df_main.columns else feature_cols
    st.dataframe(df_main[model_vars].describe().T, use_container_width=True)

# ------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------
with tab_viz:
    st.subheader("📊 Phân tích biểu đồ trực quan hóa")
    
    # Biểu đồ phân phối biến mục tiêu quan trọng hàng đầu
    st.markdown("##### 🎯 Phân phối của Biến mục tiêu rủi ro vỡ nợ (default)")
    target_counts = df_main[target_col].value_counts().reset_index()
    target_counts.columns = ['Trạng thái (default)', 'Số lượng khách hàng']
    target_counts['Trạng thái (default)'] = target_counts['Trạng thái (default)'].map({0: '0 - Bình thường', 1: '1 - Rủi ro/Gian lận'})
    
    fig_target = px.bar(
        target_counts, x='Trạng thái (default)', y='Số lượng khách hàng',
        color='Trạng thái (default)', text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=350
    )
    fig_target.update_layout(showlegend=False)
    st.plotly_chart(fig_target, use_container_width=True)
    
    st.divider()
    st.markdown("##### 🔍 Khám phá sâu phân phối các biến chỉ số đặc trưng tài chính (X)")
    
    # Cho phép người dùng tùy chọn đa biến linh hoạt nếu có quá nhiều đặc trưng
    selected_features = st.multiselect(
        "Chọn các biến chỉ số chỉ định để trực quan vẽ biểu đồ (Mặc định chọn 4 biến đầu tiên):",
        options=feature_cols,
        default=feature_cols[:min(4, len(feature_cols))]
    )
    
    if len(selected_features) > 0:
        # Tổ chức lưới hiển thị 2x2 cân đối dựa trên danh sách chọn
        rows = (len(selected_features) + 1) // 2
        for r in range(rows):
            cols_grid = st.columns(2)
            for c in range(2):
                idx = r * 2 + c
                if idx < len(selected_features):
                    feat = selected_features[idx]
                    with cols_grid[c]:
                        fig_hist = px.histogram(
                            df_main, x=feat, color=target_col,
                            marginal="box", barmode="overlay",
                            title=f"Biểu đồ phân phối tần suất biến {feat}",
                            labels={target_col: 'Trạng thái default'},
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            height=350
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một biến chỉ số để hiển thị biểu đồ phân tích.")

# ------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ------------------------------------------
with tab_metrics:
    st.subheader("🎯 Đánh giá hiệu năng và các chỉ số đo lường kiểm định AI")
    
    # Kiểm tra trạng thái đồng bộ dữ liệu tập huấn luyện
    if 'evaluation_metrics' not in st.session_state:
        st.info("💡 Chưa tìm thấy kết quả huấn luyện. Vui lòng bấm vào nút **[🚀 Huấn luyện mô hình]** ở Sidebar để tính toán.")
    else:
        metrics = st.session_state['evaluation_metrics']
        
        # Hiển thị 4 chỉ số cốt lõi dạng thẻ điểm số
        st.markdown("##### 📌 Các chỉ số đo lường hiệu năng cốt lõi (Phân loại Nhị phân):")
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        with c_m1:
            st.metric("Độ chính xác tổng thể (Accuracy)", f"{metrics['accuracy']:.4f}")
        with c_m2:
            st.metric("Độ chính xác mô hình (Precision)", f"{metrics['precision']:.4f}")
        with c_m3:
            st.metric("Tỷ lệ bắt sót thực tế (Recall)", f"{metrics['recall']:.4f}")
        with c_m4:
            st.metric("Chỉ số F1-Score cân bằng", f"{metrics['f1']:.4f}")
            
        st.divider()
        
        col_g1, col_g2 = st.columns(2)
        
        with col_g1:
            st.markdown("##### 🧮 Ma trận nhầm lẫn (Confusion Matrix)")
            y_t = np.array(metrics['y_test'])
            y_p = np.array(metrics['y_pred'])
            cm = confusion_matrix(y_t, y_p)
            
            # Trực quan hóa ma trận nhầm lẫn bằng Heatmap của Plotly
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                x=['0 - Bình thường', '1 - Rủi ro'],
                y=['0 - Bình thường', '1 - Rủi ro'],
                color_continuous_scale='Blues',
                height=380
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_g2:
            st.markdown("##### 📈 Đường cong đặc tính hoạt động máy thu (ROC Curve)")
            if len(metrics['y_prob']) > 0:
                fpr, tpr, thresholds = roc_curve(metrics['y_test'], metrics['y_prob'])
                roc_auc = auc(fpr, tpr)
                
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.4f})', line=dict(color='darkorange', width=2)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Dự đoán ngẫu nhiên', line=dict(color='navy', width=2, dash='dash')))
                fig_roc.update_layout(
                    xaxis_title='Tỷ lệ Dương tính giả (False Positive Rate)',
                    yaxis_title='Tỷ lệ Dương tính thật (True Positive Rate)',
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=380,
                    legend=dict(x=0.5, y=0.1)
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Mô hình không cung cấp thông tin phân phối xác suất dự báo.")
                
        st.divider()
        st.markdown("##### 🪵 Xếp hạng thứ tự mức độ quan trọng của các biến tính năng (Feature Importances)")
        feat_imp_df = pd.DataFrame({
            'Chỉ số tính năng': st.session_state['feature_columns'],
            'Mức độ đóng góp': metrics['feature_importances']
        }).sort_values(by='Mức độ đóng góp', ascending=True)
        
        fig_imp = px.bar(
            feat_imp_df, x='Mức độ đóng góp', y='Chỉ số tính năng', orientation='h',
            title='Mức độ ảnh hưởng của các biến X lên quyết định phân loại rủi ro',
            color='Mức độ đóng góp', color_continuous_scale='Viridis',
            height=450
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH DỰ BÁO
# ------------------------------------------
with tab_inference:
    st.subheader("🔮 Chấm điểm tín dụng & Dự đoán rủi ro gian lận khách hàng mới")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Vui lòng kích hoạt huấn luyện mô hình thành công tại Sidebar trước khi thực hiện dự báo dữ liệu mới.")
        st.stop()
        
    model = st.session_state['trained_model']
    scaler = st.session_state['data_scaler']
    f_cols = st.session_state['feature_columns']
    
    mode = st.radio(
        "Lựa chọn phương thức nhập dữ liệu đầu vào khách hàng:",
        options=["✍️ Nhập thủ công từng trường", "📂 Tải tệp dữ liệu danh sách loạt lớn (X_test)"],
        horizontal=True
    )
    
    # CHẾ ĐỘ 1: NHẬP THỦ CÔNG TỪNG TRƯỜNG DỮ LIỆU QUA FORM
    if mode == "✍️ Nhập thủ công từng trường":
        st.markdown("##### Nhập thông số định lượng các chỉ tiêu tài chính:")
        
        with st.form("single_inference_form"):
            col_inf_grid = st.columns(3)
            input_data = {}
            
            # Tự động duyệt qua toàn bộ danh sách 14 biến để sinh widget nhập liệu động phù hợp khoảng dữ liệu gốc
            for idx, col in enumerate(f_cols):
                col_pos = idx % 3
                with col_inf_grid[col_pos]:
                    min_val = float(df_main[col].min())
                    max_val = float(df_main[col].max())
                    med_val = float(df_main[col].median())
                    
                    input_data[col] = st.number_input(
                        f"Chỉ số {col}", 
                        min_value=min_val - abs(min_val)*2, 
                        max_value=max_val + abs(max_val)*2, 
                        value=med_val,
                        format="%.6f",
                        help=f"Khoảng giá trị trong tập mẫu: [{min_val:.4f} đến {max_val:.4f}]"
                    )
            
            submit_pred = st.form_submit_button("🔍 Tiến hành phân tích dự báo", type="primary", use_container_width=True)
            
        if submit_pred:
            # Chuyển đổi dữ liệu và chuẩn hóa qua Scaler đã học tập từ trước
            df_inf = pd.DataFrame([input_data])[f_cols]
            df_inf_scaled = scaler.transform(df_inf)
            
            prediction = model.predict(df_inf_scaled)[0]
            prob = model.predict_proba(df_inf_scaled)[0] if hasattr(model, "predict_proba") else None
            
            st.divider()
            st.markdown("#### Đánh giá kết quả kết luận phân tích từ Hệ thống AI:")
            if prediction == 1:
                st.error(f"🚨 **CẢNH BÁO NGUY CƠ CAO (RỦI RO):** Hệ thống phân loại đối tượng này thuộc nhóm có nguy cơ xảy ra gian lận hoặc biến động vỡ nợ.")
            else:
                st.success(f"✅ **AN TOÀN (BÌNH THƯỜNG):** Đối tượng ghi nhận có chỉ số hoạt động bình thường, nằm trong ngưỡng kiểm soát rủi ro an toàn.")
                
            if prob is not None:
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    st.metric("Xác suất an toàn (An toàn)", f"{prob[0]*100:.2f}%")
                with c_p2:
                    st.metric("Xác suất rủi ro (Default)", f"{prob[1]*100:.2f}%", delta=f"{prob[1]*100:.2f}%", delta_color="inverse")

    # CHẾ ĐỘ 2: TẢI TỆP HÀNG LOẠT (X_TEST)
    else:
        st.markdown("##### Tải lên tệp danh sách tổng hợp chứa các biến mới cần phân tích dự báo hàng loạt")
        st.caption("⚠️ Yêu cầu: Định dạng tệp Excel/CSV bắt buộc phải bao gồm đầy đủ tên các cột: `X_1`, `X_2`, ..., `X_14`.")
        
        bulk_file = st.file_uploader("Chọn tệp dữ liệu cần chấm điểm (X_new)", type=["csv", "xlsx"], key="bulk_uploader")
        
        if bulk_file is not None:
            # Đọc dữ liệu tập danh sách mới cần dự báo
            if bulk_file.name.endswith('.csv'):
                df_bulk = pd.read_csv(bulk_file)
            else:
                df_bulk = pd.read_excel(bulk_file)
                
            # Kiểm tra xem tệp có chứa đủ các thuộc tính cột bắt buộc không
            missing_cols = [c for c in f_cols if c not in df_bulk.columns]
            
            if len(missing_cols) > 0:
                st.error(f"Tệp tải lên thiếu các cột biến đặc trưng sau: {missing_cols}. Vui lòng chuẩn hóa lại cấu trúc tệp dữ liệu.")
            else:
                # Trích xuất đúng tập thuộc tính
                X_bulk = df_bulk[f_cols].fillna(df_main[f_cols].median())
                X_bulk_scaled = scaler.transform(X_bulk)
                
                # Thực hiện chấm điểm hàng loạt đồng thời
                bulk_preds = model.predict(X_bulk_scaled)
                
                # Thêm thông tin kết quả vào dataframe hiển thị dữ liệu gốc
                df_result = df_bulk.copy()
                df_result['Kết_Quả_Dự_Báo_Default'] = bulk_preds
                if hasattr(model, "predict_proba"):
                    bulk_probs = model.predict_proba(X_bulk_scaled)[:, 1]
                    df_result['Xác_Suất_Rủi_Ro_Default'] = bulk_probs
                
                st.success(f"Đã thực hiện phân tích tự động thành công cho toàn bộ {df_result.shape[0]} đối tượng hàng loạt.")
                
                # Biểu đồ tóm tắt phân bổ kết quả phân loại mới
                summary_new = df_result['Kết_Quả_Dự_Báo_Default'].value_counts().reset_index()
                summary_new.columns = ['Mã phân loại', 'Số lượng khách hàng']
                summary_new['Mã phân loại'] = summary_new['Mã phân loại'].map({0: '0 - An toàn', 1: '1 - Rủi ro cao'})
                
                fig_new_summary = px.pie(summary_new, values='Số lượng khách hàng', names='Mã phân loại', title='Tỷ lệ cơ cấu phân phối rủi ro danh sách mới', hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig_new_summary, use_container_width=True)
                
                st.markdown("##### 📑 Bảng chi tiết kết quả tích hợp phân tích trực quan:")
                st.dataframe(df_result, use_container_width=True)
                
                # Xuất file kết quả đầu ra cho quản lý tải về máy
                csv_data = df_result.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 Tải xuống tệp báo cáo kết quả dự báo (CSV)",
                    data=csv_data,
                    file_name="Ket_qua_du_bao_rui_ro_gian_lan.csv",
                    mime="text/csv",
                    use_container_width=True
                )
