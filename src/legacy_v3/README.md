# legacy_v3 — mã V3 giữ nguyên (đã chạy trên Kaggle, 25/25 test PASS theo artifacts/v3_verified/test_results.json)

Các module này được giữ **nguyên văn** từ BreastVision_XAI_V3.ipynb để checkpoint V3 nạp lại được 100% (tên lớp/tham số không đổi).
Chúng dùng import phẳng (`import bus_core`), nên `src/models.py` thêm thư mục này vào `sys.path`.
Không sửa các file này trừ khi chạy lại bộ test tests/legacy_v3 trên môi trường có PyTorch.
