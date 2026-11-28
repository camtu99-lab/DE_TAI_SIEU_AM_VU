# Tài liệu bảo vệ
## A. Poster (nội dung)
**Tiêu đề:** BUS-VERIFY — kiểm chứng bằng chứng phản thực và phát hiện nguy cơ dự đoán sai cho AI siêu âm vú. **Câu hỏi:** bằng chứng theo vùng có dự đoán được lỗi của mô hình? **Phương pháp:** can thiệp tổn thương/lân cận/nền từ mask dự đoán + đối chứng; logistic L2 trên dự đoán OOF; chính sách cảnh báo khóa trước. **V3 (đã xác minh):** X6 AUROC 0.930 nội bộ, 0.886 BUSI; V3_full không hơn X6. **V4:** thiết kế xong, kết quả NOT RUN. **Giới hạn:** ~7 FN ở test, BUSI không có patient ID. **Không phải thiết bị y khoa.**

## B. Kịch bản 7–10 phút
1. (1') Vấn đề: sai mà vẫn tự tin; mục tiêu là *biết khi nào không nên tin*.
2. (1') Kiểm toán V3: cái đúng, cái cần sửa (B1≡B2, ngưỡng 0.176, test đã khảo sát).
3. (1,5') Tính mới trung thực: ghép nối, không "đầu tiên"; hai công trình gần nhất báo kết quả âm.
4. (2') Phương pháp: vùng từ mask dự đoán, đối chứng cùng diện tích, đặc trưng, OOF, chính sách khóa.
5. (1') Giao thức: B0–B6, cùng tỉ lệ cảnh báo, endpoint chính, Holm.
6. (1') Demo app 8 trang; thiếu artifact → NOT RUN, không giả lập.
7. (1') Trung thực: kết quả V4 chưa chạy; cách chạy; kịch bản kết quả âm vẫn đóng góp.
8. (0,5') Kết luận + giới hạn.

## C. 30 câu hỏi khó
1. *Tính mới là gì?* Ghép nối có kiểm soát (vùng dự đoán, đối chứng, OOF) + giao thức; thành phần đã có (Chapagai, Kalita, Thomas). Không khẳng định "đầu tiên".
2. *Sao không dùng softmax?* Có: B1/B1b là baseline; cần P1 thắng có ý nghĩa sau Holm, nếu không thì kết luận âm.
3. *B1 khác B2?* Cùng thứ hạng (T đơn điệu) → không phải tín hiệu độc lập.
4. *Vì sao B1b?* Ngưỡng Youden 0.176 ≠ 0.5; ranh giới thật là logit(thr).
5. *Dùng GT mask?* Không khi suy luận; chỉ oracle, báo cáo riêng.
6. *Rò rỉ test?* Test không dùng để huấn luyện/chọn đặc trưng/C/ngưỡng; có `assert_no_test`. Nhưng X6 được chọn sau khi thấy V3 → khai báo.
7. *Test có "chưa từng thấy"?* Không; gọi là "tập giữ lại đã khảo sát".
8. *Vì sao OOF?* Nếu huấn luyện bộ phát hiện trên dự đoán huấn luyện của mô hình chính, lỗi gần như bằng 0 → không học được.
9. *Cross-fit lệch?* Có: risk học từ mô hình fold (80% dev), áp lên mô hình cuối; kiểm bằng test giữ lại; là giới hạn khai báo.
10. *Ensemble?* Chỉ cho B3; app chỉ có một checkpoint nên P1 không dùng ens_*.
11. *Can thiệp tạo OOD?* Có thể (Poolpol, Rong); dùng đối chứng cùng diện tích/ngẫu nhiên, QA seam, phân tích OOD; không loại hẳn.
12. *Sao logistic?* Ít ca lỗi (vài trăm), diễn giải được (đóng góp tuyến tính chính xác); GBM chỉ khi ≥150 lỗi.
13. *Risk có là xác suất ác tính?* Không: P(mô hình sai).
14. *Chọn ngưỡng cảnh báo?* Phân vị risk OOF của dev; khóa sha256.
15. *FN quá ít?* ~7 ở test: CI rộng; báo cáo n tối thiểu phát hiện; endpoint chính dựa trên dev nested-CV nhiều ca hơn.
16. *BUSI CI?* Không patient ID → CI ảnh lạc quan; có 12 nhóm gần trùng.
17. *Đa so sánh?* Holm cho mọi so sánh ghép cặp.
18. *AURC?* Báo cả AUGRC (Traub 2024).
19. *Vì sao X6?* V3_full không hơn X6; đơn giản hơn; nhưng chọn có dùng test V3 (khai báo).
20. *Có kết quả V4 chưa?* Chưa — NOT RUN; notebook sẵn sàng.
21. *Nếu P1 không thắng B1b?* Báo cáo kết quả âm; đóng góp là bằng chứng thực nghiệm âm với giao thức chặt.
22. *Dịch chuyển miền?* Dice BUSI 0.65–0.69 vs 0.90 nội bộ; cổng phạm vi heuristic, không bảo đảm.
23. *"Không cảnh báo" an toàn?* Không.
24. *Quyền riêng tư?* Không lưu, không gọi ngoài.
25. *Y khoa?* Không phải thiết bị chẩn đoán.
26. *Reproducibility?* Seed, config cố định trước, sha256 manifest/policy, notebook = mã nguồn (test đồng bộ).
27. *Robustness label-preserving vs counterfactual?* Tách cột `label_preserving`; occlusion tổn thương là can thiệp, không kỳ vọng bất biến.
28. *Faithfulness?* deletion/insertion + đối chứng; kết quả phụ thuộc kiểu che (Poolpol).
29. *Tổng quát hóa sang máy khác?* Chưa chứng minh (BUS-BRA 4 máy một nguồn).
30. *Điểm tự chấm?* Xem 06: tối đa ~ 60–70 trước khi có kết quả; >90 không thể biện minh khi NOT RUN.
