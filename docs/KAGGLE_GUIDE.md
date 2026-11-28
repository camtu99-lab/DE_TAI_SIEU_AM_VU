# Hướng dẫn Kaggle (GPU)
1. Tạo Notebook mới, Import `BreastVision_XAI_V4_BUS_VERIFY_Kaggle.ipynb`. Accelerator: GPU (T4/P100). Internet có thể bật để tải trọng số pretrained (nếu tắt, backbone khởi tạo ngẫu nhiên và cell huấn luyện in CẢNH BÁO).
2. Add Input: dataset BUS-BRA (có `Images/`, `Masks/`, CSV) và BUSI (có `benign/`, `malignant/`; thư mục `normal` bị loại).
3. Chạy trước `RUN_MODE="smoke_test"` (dữ liệu tổng hợp, vài phút) để kiểm tra mã. Số liệu smoke KHÔNG có giá trị.
4. Đặt `RUN_MODE="full_experiment"` rồi Run All. Chi phí ước tính: 5 fold×3 seed + 3 mô hình cuối = 18 lần huấn luyện X6 (~40 epoch, có early stopping) và trích đặc trưng counterfactual cho toàn bộ dev/test/BUSI. Ước tính nhiều giờ — **chưa đo thực tế**. Notebook resume được: checkpoint (`checkpoints/`) và đặc trưng (`features/`) được lưu; nếu phiên bị ngắt, Run All lại.
5. Lấy kết quả: `outputs/*.csv|json` → chép vào `artifacts/v4/`; `deploy/` (model.pt, risk/, alert_policy.locked.json, deploy_config.json) → thư mục `deploy/` của app.
6. Nếu manifest chia dữ liệu không khớp V3 (`manifest sha256 khớp V3: False`), tập test KHÔNG còn là test V3: ghi rõ trong báo cáo.
Giao thức cố định trước nằm ở `bsv4/configs/bus_verify.json`; đổi sau khi thấy kết quả là vi phạm giao thức.
