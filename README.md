# BreastVision-XAI V3 - Streamlit demo

    pip install -r requirements.txt
    streamlit run app.py

## Hai chế độ
- **DEMO** (mặc định khi chưa có checkpoint): chỉ cần numpy/opencv/matplotlib. Vùng viền xanh tách bằng ngưỡng Otsu,
  bản đồ nhiệt chỉ là làm mờ mask -> KHÔNG phải kết quả mô hình. Dùng để xem giao diện (ảnh mẫu `bus_sample.png` là ảnh tổng hợp).
- **MÔ HÌNH**: chạy mục 27 của `BreastVision_XAI_V3.ipynb`, chép vào cùng thư mục với `app.py`:
  `deploy_config.json`, file checkpoint (.pt) được ghi trong config, và thư mục `results/` (tuỳ chọn).
  App tự chuyển chế độ và gọi `analyze_image_v2` (hiệu chuẩn T, ngưỡng khóa, từ chối dự đoán, evidence map/Grad-CAM).

Các file `bus_core.py, bus_core_v2.py, models_xai.py, ...` là module xuất từ notebook, cần nằm cạnh `app.py`.
