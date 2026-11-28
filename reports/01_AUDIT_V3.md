# Báo cáo kiểm toán BreastVision-XAI V3 → V4
Nguồn: notebook V3, `outputs.7z`, `figures.7z` (đã giải nén, số liệu trong `artifacts/v3_verified/`). Phân loại theo bằng chứng, không theo ý định.

## 1. Phân loại thành phần
| Thành phần V3 | Phân loại | Bằng chứng / ghi chú |
|---|---|---|
| Kiểm toán dữ liệu BUS-BRA (1875 ảnh, 1064 bệnh nhân, 0 lỗi mask/trùng) | Đã cài đặt và có kiểm chứng | `data_audit.json`; V4 kiểm lại bằng `tests/test_real_manifest.py` (sha256 manifest, ghép ảnh–mask, không rò rỉ bệnh nhân) |
| Chia train/val/test theo bệnh nhân + manifest | Đã cài đặt và có kiểm chứng | manifest sha256 a5b0f502…; V4 tái dùng nguyên |
| Huấn luyện X0/X3/X4/X6/V3_* nhiều seed, resume | Đã cài đặt và có kiểm chứng (kết quả đã chạy) | `internal_results_*.csv` |
| Kết quả nội bộ: X6 AUROC 0.930±0.012; V3_full 0.920±0.010 | Có kiểm chứng | V3_full **không** hơn X6/X0 có ý nghĩa (`paired_comparisons.csv`: CI AUROC chứa 0) |
| T-scaling + ngưỡng Youden khóa trước khi đọc BUSI | Có kiểm chứng | `pipeline_lock_primary.json` (T=1.259, thr=0.176) |
| Đánh giá ngoài BUSI (645 ảnh sau loại trùng) | Có kiểm chứng nhưng CI mức ảnh | BUSI không có patient ID, 12 nhóm gần trùng, 1 xung đột nhãn |
| Selective prediction bằng max-prob | Có kiểm chứng nhưng **cần sửa** | Với ngưỡng 0.176, độ tin cậy max-prob đo khoảng cách tới 0.5 chứ không tới ranh giới quyết định → V4 thêm B1b (`thr_margin`) |
| "Uncertainty" entropy/margin/max-softmax | Cần sửa (trùng lặp) | Ba đại lượng cùng thứ hạng với bài toán nhị phân (B1≡B2) → V4 gộp, ghi rõ, không tính như tín hiệu độc lập |
| Grad-CAM / evidence map như XAI chính | Đã cài đặt nhưng chưa kiểm chứng faithfulness | Chỉ có alignment với mask; không có thử can thiệp → V4 thay bằng kiểm chứng phản thực |
| Thông điệp "XAI đáng tin" | Cần loại bỏ | Chưa có bằng chứng XAI giúp phát hiện lỗi |
| Cảnh báo lỗi học được (failure-risk) | **Chưa cài đặt** ở V3 | Là đóng góp V4 |
| Counterfactual region verification | **Chưa cài đặt** ở V3 | Là đóng góp V4 |
| Robustness lab | **Chưa cài đặt** ở V3 (chỉ có TTA nhẹ) | V4 |
| Ứng dụng Streamlit V3 | Đã cài đặt, chưa kiểm chứng | không đọc artifact thật; V4 viết lại |
| Gọi tập test cũ là "chưa từng thấy" | Cần loại bỏ | Test BUS-BRA/BUSI đã được dùng chọn X6 làm mô hình nền: V4 gọi là "tập giữ lại đã khảo sát" |

## 2. Rủi ro khoa học được tìm thấy và cách xử lý trong V4
1. **Rò rỉ chọn lựa:** lựa chọn X6 làm mô hình nền dựa một phần trên test V3 → khai báo trong `configs/bus_verify.json`; kết quả xác nhận chính chuyển sang nested-CV trên dev.
2. **Baseline yếu:** V3 chỉ có max-prob. V4 thêm B0–B6 (+B1b), cùng tỉ lệ cảnh báo.
3. **Ngưỡng ≠ 0.5:** xem trên.
4. **Công suất thống kê:** ~7 ca FN ở test → FN alert recall có CI rất rộng; báo cáo kèm n nhỏ nhất phát hiện được (`statistical_tests`).
5. **BUSI:** không nhóm theo bệnh nhân, CI lạc quan; ghi trong mọi bảng.
6. **Không có torch/GPU/dữ liệu ở nơi phát triển V4:** mọi kết quả thực nghiệm V4 là NOT RUN.

## 3. Giữ lại từ V3
Module `bus_core*`, `models_xai`, `losses_xai`, `data_audit`, `evaluation_xai` (nguyên văn, `src/v3/SHA256SUMS.txt`), manifest chia dữ liệu, kết quả tham chiếu.
