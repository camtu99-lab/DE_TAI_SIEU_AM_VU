# Báo cáo kiểm thử
Chạy: `python -m pytest -q tests` (Python 3.13, không có torch/streamlit/plotly, không có dữ liệu BUS-BRA/BUSI).
**Kết quả thật: 112 passed, 1 skipped.** (Skipped = `tests/test_torch_parity.py`: cần torch → **NOT RUN**.)

| Yêu cầu | Tệp kiểm thử | Trạng thái |
|---|---|---|
| Mapping nhãn, split theo bệnh nhân, rò rỉ, trùng lặp, ghép ảnh–mask (manifest thật 1875 dòng) | test_real_manifest, test_data_integrity | PASS (trên manifest, không trên ảnh) |
| Tiền xử lý parity | test_preprocessing_parity | PASS |
| Vùng phản thực khớp diện tích, mask rỗng, ảnh không hợp lệ | test_counterfactual_verifier, test_pipeline_e2e | PASS |
| Risk model thiếu, khóa ngưỡng, sha256, chốt chặn test | test_failure_risk_engine | PASS |
| Hiệu chuẩn, bootstrap, Holm, công suất | test_metrics_and_stats | PASS |
| Giao thức benchmark, baseline, ablation | test_benchmark_protocol | PASS (dữ liệu tổng hợp) |
| Robustness lab | test_robustness_lab | PASS |
| Hỗ trợ app, xuất báo cáo | test_app_support | PASS |
| Streamlit khởi động / 7 trang | test_streamlit_pages_smoke | PASS **chỉ với stub** — streamlit thật **NOT RUN** |
| Notebook nhúng đúng mã `src/` | test_notebook_sync | PASS |
| Suy luận CPU, tương thích checkpoint, parity notebook–web có torch | test_torch_parity | **NOT RUN** (skip) |
| Notebook chạy end-to-end (smoke/full) | — | **NOT RUN** |

Lưu ý: test pass trên dữ liệu tổng hợp chứng minh tính đúng của mã, không chứng minh hiệu quả khoa học.
