# Báo cáo thẩm định tính mới
Tìm kiếm thực hiện 2026-11-24; nguồn mở được và nguồn chỉ thấy trong kết quả được phân biệt trong `novelty_search_raw.md` (đọc tệp đó để kiểm từng dòng). Không chứng minh được "không tồn tại"; **không tuyên bố "đầu tiên trên thế giới"**.

## Bảng so sánh công trình gần nhất
| Công trình | Làm gì (đã xác minh) | Khác BUS-VERIFY |
|---|---|---|
| Chapagai et al. 2026 (arXiv 2611.05011) | Đặc trưng "anatomical enrichment" từ saliency + margin + bất đồng TTA → logistic để dự đoán lỗi; X-quang ngực, dịch chuyển bệnh viện | Cùng cấu trúc bộ phát hiện; dùng saliency, không dùng can thiệp phản thực theo vùng; kết quả âm (+≤0.011 AUROC) |
| Kalita & Singh 2026 (arXiv 2610.36640) | Ổn định nhãn dưới 6 nhiễu nhẹ làm điểm tin cậy; kém MSP trên dữ liệu sạch | Chỉ nhiễu toàn ảnh, không theo vùng |
| DEGRE (AAAI), ConfidNet (NeurIPS 2019) | Bộ phát hiện học trên confidence + ensemble | Không dùng sensitivity theo vùng |
| Lei et al. 2024 (arXiv 2401.03664) | Attribution kiểu RISE + độ tin cậy cho siêu âm vú, BUSI | Không có mô hình lỗi học OOF, không kiểm soát vùng đối chứng |
| Thomas et al. 2026; Lenis et al.; Gurram 2026 | Tỉ lệ blur tiền cảnh/nền; thay vùng; xóa tổn thương vs đối chứng cùng hình | Là công cụ đánh giá/giải thích, không là đặc trưng dự đoán lỗi |
| Bernhardt et al. TMLR 2022 | Không điểm tin cậy nào luôn hơn softmax trên 6 bộ y khoa | Là tiền đề phải thắng: V4 đặt softmax/B1b làm baseline |
| Traub et al. NeurIPS 2024; Jaeger et al. ICLR 2023 | Phê bình AURC; đề xuất AUGRC | V4 báo cả AURC và AUGRC |
| Poolpol 2026; Rong et al. 2022 (ROAD) | Điền giá trị cố định tạo artefact OOD | V4 có đối chứng cùng diện tích/ngẫu nhiên, QA seam, phân tích OOD |

## Phân loại đóng góp (mức tuyên bố tối đa được phép)
- **Thuật toán (yếu nhất):** kết hợp sensitivity phản thực theo vùng *từ mask dự đoán* (tổn thương / lân cận / nền, đã chuẩn hóa bằng đối chứng cùng diện tích và ngẫu nhiên) làm đặc trưng cho bộ phát hiện lỗi OOF. Thành phần đã có riêng lẻ; ghép nối cho BUS chưa tìm thấy, **không khẳng định chưa ai làm**.
- **Quy trình:** không dùng GT mask khi suy luận, OOF + nhãn "mô hình sai", chính sách cảnh báo khóa sha256, đánh giá ở tỉ lệ cảnh báo cố định và FN alert recall. Đóng góp đánh giá/quy trình.
- **Thực nghiệm (có thể bảo vệ nhất):** liệu sensitivity phản thực theo vùng có thêm thông tin ngoài margin/entropy/TTA cho BUS và có chuyển sang BUSI không. Kết quả âm cũng hợp lệ (hai công trình gần nhất báo âm). **Hiện NOT RUN, chưa có kết quả.**

## Tuyên bố bị cấm
"Đầu tiên thế giới", "SOTA", "đáng tin cậy lâm sàng", "XAI chứng minh mô hình đúng".
