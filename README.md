# 🛡️ Hệ thống Phát Hiện Gian Lận & Dự Báo Rủi Rơ Tín Dụng (Streamlit App)

Ứng dụng web được phát triển trên nền tảng **Streamlit** kết hợp mô hình học máy **Random Forest Classifier** nhằm tự động hóa quy trình phân tích, chấm điểm tín dụng và cảnh báo nguy cơ gian lận tài chính hoặc vỡ nợ của đối tượng khách hàng dựa trên chuỗi dữ liệu định lượng.

## ✨ Các tính năng chính của ứng dụng
1. **Cấu hình động tham số AI:** Điều chỉnh linh hoạt các cấu hình siêu tham số của thuật toán `RandomForest` (Số cây quyết định, Độ sâu tối đa, Tỷ lệ phân chia tập Test...) trực tiếp tại khu vực Sidebar.
2. **Tổng quan dữ liệu:** Phân tích nhanh số lượng bản ghi, xem trước dữ liệu thô và thống kê mô tả phân vị các biến chỉ số đầu vào.
3. **Trực quan hóa đồ thị nâng cao:** Vẽ biểu đồ phân bổ lớp mục tiêu và biểu đồ phân phối tần suất tương tác đa biến trực quan thông qua thư viện `Plotly`.
4. **Kiểm định mô hình trực quan:** Hiển thị tự động các chỉ số đánh giá độ chính xác (Accuracy, Precision, Recall, F1-Score), trực quan hóa Ma trận nhầm lẫn (Confusion Matrix), đường cong ROC-AUC cùng biểu đồ xếp hạng mức độ quan trọng của các biến tính năng.
5. **Công cụ dự báo thông minh:**
   - *Chế độ nhập thủ công:* Nhập nhanh thông số tài chính của 1 khách hàng cụ thể để nhận kết quả phân loại đi kèm xác suất chi tiết tức thì.
   - *Chế độ dự báo hàng loạt:* Tải tệp danh sách mới gồm nhiều dòng (file `X_new`), hệ thống tự động xử lý chấm điểm đồng loạt và cung cấp tính năng tải tệp báo cáo định dạng kết quả đầu ra.

## ⚙️ Hướng dẫn cài đặt và vận hành hệ thống

### 1. Chuẩn bị môi trường máy tính
Đảm bảo máy tính của bạn đã được cài đặt sẵn môi trường **Python** (Phiên bản khuyến nghị `>= 3.9`).

### 2. Cài đặt các thư viện phụ thuộc bắt buộc
Di chuyển thư mục terminal đến nơi chứa bộ 3 file này và chạy dòng lệnh sau:
```bash
pip install -r requirements.txt
