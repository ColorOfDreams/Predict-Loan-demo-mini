# Predict Loan Paid Back

## 1. Giới thiệu đề tài

Trong lĩnh vực tài chính – ngân hàng, việc dự đoán khả năng khách hàng có hoàn trả khoản vay đúng hạn hay không là một bài toán quan trọng, ảnh hưởng trực tiếp đến quản trị rủi ro tín dụng và lợi nhuận của tổ chức cho vay.

Đề tài **Predict Loan Paid Back** tập trung xây dựng mô hình học máy nhằm dự đoán khả năng **loan_paid_back** của khách hàng dựa trên các thông tin tài chính, nhân khẩu học và đặc trưng khoản vay.

### Mục tiêu
- Phân tích dữ liệu cho vay
- Xây dựng pipeline tiền xử lý và đặc trưng hóa dữ liệu
- Huấn luyện mô hình dự đoán khả năng hoàn trả khoản vay
- Đánh giá mô hình bằng các metric phù hợp
- Thực hiện inference trên dữ liệu mới

---

## 2. Dataset

### Nguồn dữ liệu
Dataset được lấy từ cuộc thi Kaggle:

**Playground Series - Season 5, Episode 11**  
https://www.kaggle.com/competitions/playground-series-s5e11

Do giới hạn về dung lượng và bản quyền, dataset **không được upload trực tiếp** lên repository này.

### Hướng dẫn tải dữ liệu
1. Truy cập link Kaggle ở trên
2. Đăng nhập tài khoản Kaggle
3. Chọn tab **Data**
4. Tải các file dữ liệu (.csv)
5. Giải nén và đặt các file CSV vào thư mục `data/`

### Mô tả dữ liệu (tổng quan)
Dataset bao gồm các nhóm thuộc tính chính:
- Thông tin nhân khẩu học khách hàng
- Thông tin tài chính (thu nhập, tín dụng, lãi suất…)
- Thông tin khoản vay
- Nhãn mục tiêu: `loan_paid_back` (binary)

---

## 3. Pipeline xử lý

Pipeline của bài toán được xây dựng theo các bước sau:

1. **Tiền xử lý dữ liệu**
   - Xử lý giá trị thiếu
   - Chuẩn hóa / mã hóa biến số và biến phân loại
   - Xây dựng các biến tỷ lệ (ratio features)

2. **Feature Engineering**
   - Các đặc trưng tài chính có ý nghĩa rủi ro
   - Biến đổi WOE (Weight of Evidence) cho một số thuộc tính

3. **Huấn luyện mô hình**
   - Chia tập train / validation
   - Huấn luyện mô hình học máy

4. **Đánh giá**
   - So sánh hiệu năng mô hình
   - Phân tích ROC – AUC

5. **Inference**
   - Áp dụng model đã huấn luyện lên dữ liệu mới
   - Sinh xác suất hoàn trả khoản vay

---

## 4. Mô hình sử dụng

Các mô hình được xem xét trong bài toán bao gồm:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting / LightGBM

Trong đó, **LightGBM** được lựa chọn làm mô hình chính nhờ:
- Hiệu năng tốt trên dữ liệu tabular
- Khả năng xử lý feature phức tạp
- Thời gian huấn luyện nhanh
- Khả năng kiểm soát overfitting tốt

---

## 5. Kết quả

Mô hình được đánh giá bằng các metric phù hợp với bài toán phân loại:

- ROC – AUC
- Confusion Matrix
- Precision / Recall
- Accuracy

Kết quả cho thấy mô hình đạt hiệu năng tốt trong việc phân biệt khách hàng có khả năng hoàn trả và không hoàn trả khoản vay.

(Các kết quả chi tiết được trình bày trong notebook và báo cáo.)

---

## 6. Hướng dẫn chạy

### 6.1 Cài môi trường

```bash
pip install -r requirements.txt
