# BreastVision-XAI V4 — BUS-VERIFY: báo cáo nghiên cứu
> **Trạng thái kết quả V4: NOT RUN.** Báo cáo mô tả thiết kế, giao thức cố định trước và kết quả V3 đã có. Mục 7 sẽ được điền bằng `artifacts/v4/` sau khi chạy notebook; không có con số V4 nào được bịa.

## 1. Vấn đề
Mô hình phân loại siêu âm vú có thể sai với độ tin cậy cao. Câu hỏi: *bằng chứng theo vùng của chính ảnh (tổn thương/lân cận/nền) có giúp dự đoán khi nào mô hình chính sai, ngoài margin/entropy/ổn định TTA không?*

## 2. Bối cảnh và tính mới
Xem `02_NOVELTY_REPORT.md`. Tuyên bố tối đa: đóng góp quy trình + câu hỏi thực nghiệm; thuật toán chỉ là ghép nối thành phần đã có.

## 3. Phương pháp
- **Mô hình nền:** X6 (khai báo trước; chọn một phần dựa trên test V3 — không độc lập).
- **Vùng:** từ mask DỰ ĐOÁN (largest component, fill holes); lân cận = vành 6–20 px; nền = ngoài vành + lề 4 px. GT mask chỉ ở thí nghiệm oracle tách riêng.
- **Can thiệp:** local blur (σ 3, 7), intensity replacement (0.6, 1.0), permutation (0.5, 1.0); đối chứng cùng diện tích/vị trí/cường độ và ngẫu nhiên; QA (seam ratio, thay đổi pixel).
- **Đặc trưng:** |Δlogit| theo vùng, hướng, flip rate, nhất quán giữa can thiệp, tương phản tổn thương–nền, `cf_specificity` = độ nhạy vùng đích trừ đối chứng, faithfulness của evidence.
- **Bộ phát hiện lỗi:** logistic L2 (impute, signed-log, chuẩn hóa), C chọn bằng CV lồng theo bệnh nhân, hiệu chuẩn Platt/isotonic; nhãn = mô hình chính sai trên dự đoán OOF. Risk **không** là P(ác tính).
- **Chính sách:** ngưỡng từ phân vị risk OOF của dev (20% xem xét, 10% nguy cơ cao), khóa sha256; cổng phạm vi (mask rỗng/bất thường, Mahalanobis thống kê ảnh). 4 trạng thái: không cảnh báo / nguy cơ sai cao / cần xem xét / ngoài phạm vi.
- **Thiết kế:** B0 ngẫu nhiên, B1 max-softmax, B1b khoảng cách tới ngưỡng, B2 calibrated (cùng thứ hạng B1), B3 bất định (ensemble khác seed cùng fold), B4 học trên confidence, B5 chỉ CF, B6 bất định+ổn định không CF, P1 đầy đủ, ablation bỏ từng nhóm. So sánh ở CÙNG tỉ lệ cảnh báo.

## 4. Thống kê
Bootstrap cụm theo bệnh nhân (B=2000), kiểm định ghép cặp, Holm trên toàn bộ so sánh, effect size, Wilson CI, n tối thiểu phát hiện được. Endpoint chính: Error AUPRC và FN alert recall @20%; so sánh chính P1 vs B1/B3/B4/B6. Phụ: Error AUROC, AURC, AUGRC, ECE/Brier của risk.

## 5. Dữ liệu
BUS-BRA 1875 ảnh/1064 bệnh nhân (dev = train+val; test giữ lại **đã khảo sát ở V3**). BUSI 647 → 645 ảnh (loại trùng với BUS-BRA), không có patient ID (CI mức ảnh, lạc quan).

## 6. Kết quả V3 đã xác minh (tham chiếu, không phải V4)
| Mô hình | AUROC nội bộ (BUS-BRA, mean±sd, 3 seed) | AUROC ngoài BUSI [CI95] | Sens/Spec BUSI (ngưỡng khóa) |
|---|---|---|---|
| X6 | 0.930 ± 0.012 | 0.886 [0.856, 0.911] | 0.813 / 0.782 |
| V3_full | 0.920 ± 0.010 | 0.881 [0.846, 0.908] | 0.895 / 0.587 |
| X0 | 0.920 ± 0.011 | 0.866 [0.833, 0.896] | 0.856 / 0.615 |
V3_full không hơn X6/X0 có ý nghĩa về AUROC (CI hiệu chứa 0). Dice phân vùng BUSI ≈ 0.65–0.69 (kém nội bộ ≈ 0.90) → dịch chuyển miền, quan trọng cho cổng phạm vi.

## 7. Kết quả V4
**NOT RUN.** Bảng sẽ gồm: nested-CV dev, test giữ lại, BUSI, ablation, robustness, oracle, hiệu chuẩn risk, đếm trạng thái.

## 8. Giới hạn đã biết
Cross-fit (risk học từ mô hình fold, áp dụng mô hình cuối); ~7 FN ở test → CI rất rộng; một nguồn dữ liệu huấn luyện; can thiệp có thể tạo artefact OOD (đối chứng giảm nhưng không loại hẳn); cổng phạm vi là heuristic; kết quả âm là khả dĩ và hợp lệ.

## 9. Đạo đức/an toàn
Công cụ nghiên cứu, không phải thiết bị y khoa; "không cảnh báo" ≠ an toàn; không lưu ảnh; không dùng GT mask khi suy luận.
