# BreastVision-XAI V3: web demo Streamlit

Công cụ hỗ trợ NGHIÊN CỨU, không phải thiết bị chẩn đoán y tế.

## Cần chuẩn bị (từ notebook)
Chạy `BreastVision_XAI_V3.ipynb` đến **mục 27** (Streamlit export). Notebook tạo thư mục `deploy_xai/` gồm:
- `V3_full_s42_best.pt` (checkpoint; tên có thể khác theo seed) 
- `deploy_config.json` (temperature, ngưỡng phân loại, ngưỡng từ chối đã khóa)
- `results/` (CSV, PNG, báo cáo)

Chép **3 thứ trên** vào thư mục này (cạnh `app.py`), giữ nguyên tên file:

```
BreastVision_XAI_V3_streamlit_demo/
├── app.py                  <- bản mới (thay cho app.py của notebook)
├── bus_core.py, bus_core_v2.py, bus_trust.py, data_audit.py,
│   evaluation_xai.py, losses_xai.py, models_xai.py   <- giống hệt notebook
├── deploy_config.json      <- từ deploy_xai/
├── V3_full_s42_best.pt     <- từ deploy_xai/ (đúng tên trong deploy_config.json)
├── results/                <- từ deploy_xai/results/ (tùy chọn)
├── samples/                <- ảnh mẫu (tùy chọn, ảnh được phép dùng và không có định danh)
└── requirements.txt
```

Hoặc để nguyên thư mục `deploy_xai/` ở đâu đó rồi đặt biến môi trường `BV_MODEL_DIR=/duong/dan/deploy_xai`.

## Chạy
```
pip install -r requirements.txt
streamlit run app.py
```

## Tính năng
- Nhãn benign/malignant theo ngưỡng đã khóa, xác suất đã hiệu chuẩn, **từ chối dự đoán** khi độ tin cậy thấp.
- Mask, bounding box, Grad-CAM, evidence map, vùng quanh tổn thương, vùng hậu âm BI-RADS.
- **Phân rã đóng góp theo vùng** vào logit (tổng khớp chính xác với logit của mô hình).
- Kiểm tra độ tin cậy giải thích: evidence collapse, tỉ lệ evidence/Grad-CAM nằm trong mask, ổn định khi lật ảnh.
- Cảnh báo ảnh quá nhỏ hoặc nhiều màu (ngoài miền B-mode); banner đỏ nếu checkpoint là smoke test.
- Tải hình tổng hợp PNG và báo cáo JSON (không chứa ảnh). Ảnh chỉ xử lý trong bộ nhớ, không lưu.

## Triển khai online
- Streamlit Community Cloud: đẩy repo lên GitHub (file > 100 MB cần Git LFS), `requirements.txt` đã trỏ tới torch CPU.
- Hugging Face Spaces (SDK Streamlit) phù hợp hơn nếu checkpoint lớn.
- Không đưa ảnh bệnh nhân thật lên dịch vụ công cộng.
