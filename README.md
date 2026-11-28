# BreastVision-XAI V4 — BUS-VERIFY
**Counterfactual Evidence Verification and Failure-Risk Detection for Breast Ultrasound AI.**
Công cụ NGHIÊN CỨU/GIÁO DỤC — KHÔNG phải thiết bị chẩn đoán y khoa.

## Trạng thái trung thực (đọc trước)
| Hạng mục | Trạng thái |
|---|---|
| Mã nguồn `src/`, app Streamlit 8 trang, 120+ kiểm thử logic | Có; 111 pass + 1 skip (cần torch) trong môi trường không có torch |
| Notebook Kaggle `notebooks/BreastVision_XAI_V4_BUS_VERIFY_Kaggle.ipynb` | Sinh tự động, kiểm cú pháp. **NOT RUN** (chưa chạy GPU/dữ liệu thật) |
| Checkpoint V4, risk model, chính sách khóa, kết quả thực nghiệm V4 | **KHÔNG CÓ** — chưa huấn luyện; app hiển thị NOT RUN, không giả lập |
| Kết quả V3 (tham chiếu, đã chạy thật) | `artifacts/v3_reference/`, `artifacts/v3_verified/` |
| Chạy app Streamlit thật, parity notebook↔web với torch | **NOT RUN** (không có streamlit/torch ở nơi phát triển); có test logic bằng stub |

## Cấu trúc
`src/` lõi (pipeline, counterfactual_verifier, failure_risk, benchmark, robustness, …) · `src/v3/` module V3 giữ nguyên (sha256 trong SHA256SUMS.txt) · `app.py` + `pages/` · `configs/bus_verify.json` (cố định trước) · `notebooks/` · `tests/` · `reports/` · `docs/` · `artifacts/`.

## Chạy
- Huấn luyện + thực nghiệm: `docs/KAGGLE_GUIDE.md`. Demo CPU: `docs/WINDOWS_CPU_GUIDE.md`.
- `python -m pytest -q tests` · `streamlit run app.py`.
- Dựng lại notebook sau khi sửa `src/`: `python notebooks/build_notebook.py` (test `test_notebook_sync` bảo đảm notebook = mã nguồn).

## Quyền riêng tư
Không lưu ảnh tải lên, không gọi dịch vụ ngoài. Mọi xử lý trong bộ nhớ.
