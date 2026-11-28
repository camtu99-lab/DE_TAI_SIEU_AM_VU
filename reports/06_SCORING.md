# Chấm điểm độc lập (100 điểm) — theo bằng chứng hiện có
Nguyên tắc: không cộng điểm cho thứ chưa chạy. Điểm tối đa có thể đạt SAU khi chạy thật nằm ở cột phải.

| Tiêu chí | Tối đa | Hiện tại | Lý do | Sau khi chạy thật (kỳ vọng tối đa) |
|---|---|---|---|---|
| Tính mới | 15 | 7 | Thẩm định có nguồn; tuyên bố khiêm tốn; chưa có bằng chứng thực nghiệm | 10–12 (nếu P1 thắng) hoặc 8–10 (kết quả âm tốt) |
| Cơ sở lý thuyết | 10 | 7 | OOF, B1≡B2, thr_margin, vùng/đối chứng có lập luận | 8 |
| Phương pháp | 20 | 14 | Đầy đủ, có kiểm thử logic; chưa kiểm chứng trên dữ liệu thật | 17 |
| Thiết kế thực nghiệm | 15 | 11 | Cố định trước, nested CV, Holm; test đã khảo sát; ít FN | 12 |
| Độ tin cậy | 15 | 6 | Chốt chặn rò rỉ, khóa ngưỡng; nhưng chưa có số liệu V4 | 11 |
| Chất lượng XAI | 10 | 4 | Can thiệp có đối chứng; chưa đo faithfulness thật | 7 |
| Tái lập | 5 | 4 | Config, seed, notebook=mã, 112 test pass | 5 |
| Bảo vệ/ứng dụng | 10 | 6 | App 8 trang, tài liệu, Q&A; app thật chưa chạy | 8 |
| **Tổng** | **100** | **≈ 59** | | **≈ 78–83** |

**Kết luận trung thực:** mục tiêu >90 không đạt và không thể tuyên bố; trần thực tế sau khi chạy thật nằm quanh 80 vì: một nguồn dữ liệu, ít FN, test đã khảo sát, không có thử nghiệm đa trung tâm/lâm sàng. Điểm này là tự chấm, chưa có người phản biện độc lập.
