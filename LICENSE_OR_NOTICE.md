# Thông báo

**BreastVision-XAI V3 — web demo.** Sản phẩm nghiên cứu phục vụ đề tài KHKT. **Không phải thiết bị chẩn đoán y tế**, chưa được kiểm định lâm sàng, không dùng cho quyết định lâm sàng.

- Mã web (app.py, pages/, src/, tests/) và các tệp trong `breastvision_core/` (trích nguyên văn từ notebook BreastVision_XAI_V3) thuộc tác giả đề tài. **Tác giả cần tự chọn và ghi giấy phép** trước khi công bố công khai.
- Checkpoint `model/V3_full_s44_best.pt` được huấn luyện trên dữ liệu công khai (BUS-BRA; đánh giá ngoài BUSI). Người dùng phải tuân thủ điều khoản/giấy phép của các bộ dữ liệu gốc. Gói này **không phân phối ảnh của các bộ dữ liệu đó**.
- Thư viện bên thứ ba (Streamlit, PyTorch, torchvision, OpenCV, NumPy, pandas, SciPy, scikit-learn, Matplotlib, Pillow, pytest) theo giấy phép riêng của từng thư viện.
- `results/` là kết quả do notebook tạo ra; `results/figures/` chỉ gồm hình tổng hợp (reliability, learning curve), không chứa ảnh bệnh nhân.
