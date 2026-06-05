# 📘 Lightweight-DMS: Master Member Guide

Chào mừng bạn tham gia dự án! Đây là hướng dẫn giúp bạn bắt đầu code và đóng góp mà không gặp khó khăn, ngay cả khi bạn mới dùng GitHub lần đầu.

---

## 🚀 1. Hướng dẫn nhanh cho người mới (Git Workflow)

Để không làm hỏng code của nhau, chúng ta sử dụng quy trình **"Nhánh tính năng" (Feature Branch)**. Hãy làm theo các bước sau:

1.  **Tạo nhánh mới để làm việc**: (Thay `ID-task` bằng mã số Issue bạn làm, ví dụ: `17-duration-logic`)
    ```powershell
    git checkout main
    git pull origin main
    git checkout -b feature/ID-task
    ```
2.  **Code và Lưu lại**:
    ```powershell
    git add .
    git commit -m "feat: mô tả ngắn gọn việc bạn đã làm"
    ```
3.  **Đẩy code lên GitHub**:
    ```powershell
    git push origin feature/ID-task
    ```
4.  **Tạo Pull Request (PR)**: Lên trang GitHub của dự án, bạn sẽ thấy nút "Compare & pull request". Nhấn vào đó và nhờ Manager review.

---

## 📂 2. Cấu trúc thư mục bạn cần biết

*   **`src/`**: Nơi chứa tất cả các file xử lý logic mới (Tạo file của bạn ở đây).
*   **`frame/csv/`**: Nơi chứa dữ liệu đầu vào (`features_summary.csv`).
*   **`main.py`**: File chạy chính. Sau khi code xong, bạn phải đăng ký script của mình vào đây.

---

## 🛠️ 3. Chi tiết kỹ thuật theo Stage

### **Giai đoạn 1: Vision Specialist**
*   **Nhiệm vụ**: Trích xuất EAR/MAR và Pose.
*   **Lưu ý**: Luôn sử dụng `cv2.solvePnP` và lưu kết quả vào `landmarks_full.csv`.

### **Giai đoạn 2 & 3: Feature Engineer**
*   **Nhiệm vụ #21 (Duration Logic)**: Tạo file `src/duration_logic.py`.
    *   `blink`: Mắt đóng <= 2 khung hình.
    *   `micro_sleep`: Mắt đóng >= 4 khung hình liên tiếp.
*   **Nhiệm vụ #4 (Aggregation)**: Tạo file `src/stats_aggregation.py`.
    *   Tính trung bình/độ lệch chuẩn trong cửa sổ 60 giây (240 dòng).

### **Giai đoạn 4: ML Lead**
*   **Nhiệm vụ #15 (Training)**: Tạo file `src/train_behavioral.py`.
    *   Mục tiêu: Dự đoán nhãn 0, 5, 10 từ các chỉ số thống kê.

---

## ✅ 4. Checklist "Định nghĩa Hoàn thành"
Trước khi báo cáo xong việc, hãy đảm bảo:
- [ ] Code của bạn không dùng đường dẫn tuyệt đối (C:\Users\...). Hãy dùng `PROJECT_ROOT` từ `core_config.py`.
- [ ] Script của bạn đã được thêm vào danh sách `steps` trong `main.py`.
- [ ] Dữ liệu kết quả đã được lưu/ghi đè vào `frame/csv/features_summary.csv`.
- [ ] Bạn đã chạy thử `python main.py` và không có lỗi.

---

## 🆘 Cần hỗ trợ?
*   Đọc kỹ file **`METHODOLOGY.md`** để hiểu công thức toán học.
*   Nhắn tin cho Manager nếu gặp lỗi xung đột code (Conflict).
