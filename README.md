# BreastVision-XAI V3 — Web demo Streamlit

*Explainable Multi-task AI for Breast Ultrasound Research*

> **RESEARCH DEMO — NOT FOR CLINICAL DIAGNOSIS.** Ứng dụng minh họa nghiên cứu cho đề tài KHKT. Mọi kết quả là *dự đoán của mô hình*, cần chuyên gia đánh giá.

## 1. Mục đích
Chạy **suy luận thật** từ checkpoint `V3_full` (seed 44) đã đóng băng và tái sử dụng đúng mã tiền xử lý/suy luận gốc của notebook, để minh họa: dự đoán benign/malignant (hiệu chuẩn + ngưỡng khóa + từ chối), phân vùng dự đoán, Grad-CAM, evidence map, phân rã logit theo vùng, thí nghiệm can thiệp vùng ảnh và dashboard kết quả thật.

## 2. Cảnh báo nghiên cứu
Không phải thiết bị y tế; chưa kiểm định lâm sàng; không thay thế bác sĩ; heatmap không phải bản đồ bệnh lý. Xem trang *About Limits* trong ứng dụng.

## 3. Cấu trúc repo
```
breastvision_streamlit/
├── app.py                     # Trang chủ: health check + Presentation Mode
├── pages/                     # 1_Analyze, 2_Perturbation_Lab, 3_Research_Dashboard, 4_Error_Cases, 5_Methodology, 6_About_Limits
├── src/                       # config, security, checkpoint, preprocessing, inference, visualization,
│                              # perturbation, research_results, report_export, demo_cases, failure_gate, ui
├── breastvision_core/         # MÃ GỐC từ notebook, giữ nguyên văn (bus_core, bus_core_v2, models_xai, losses_xai,
│                              #  evaluation_xai, bus_trust, data_audit, tests_xai)
├── model/                     # V3_full_s44_best.pt, deploy_config.json, pipeline_lock_primary.json
├── results/                   # CSV/JSON/MD kết quả thật + figures/ (reliability, learning curve)
├── assets/demo_cases/         # (trống) ca mẫu ẩn danh tùy chọn — xem mục 11
├── tests/                     # pytest (62 test)
└── .streamlit/config.toml, requirements.txt, run_windows.bat, run_windows.ps1
```
Nguyên tắc: **UI ≠ inference ≠ dữ liệu kết quả ≠ artifact mô hình.** Trang chỉ gọi `src/`; `src/inference.py` chỉ gọi helper gốc trong `breastvision_core/`.

## 4. Yêu cầu Python
Python 3.10–3.12 (đã chạy với 3.12). Môi trường đã kiểm thử: Python 3.12.16, torch 2.14.1 (CPU), torchvision 0.29.2, numpy 2.5.5, streamlit 1.67.0, pandas 3.1.1, opencv-python-headless 5.1.0, matplotlib 3.11.2, scikit-learn 1.9.1, scipy 1.18.1, pytest 9.2.0.

## 5. Cài đặt Windows
```
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
streamlit run app.py
```
Hoặc nháy đúp `run_windows.bat` (hoặc `.\run_windows.ps1`). Không cần Docker.

## 6. Chạy CPU offline
Chỉ cần cài thư viện một lần (cần Internet lúc cài). Sau đó **không cần Internet**: app không tải trọng số ImageNet (`pretrained=False`, trọng số lấy từ checkpoint), không gọi torchvision download/HuggingFace/Kaggle và không có telemetry (`gatherUsageStats=false`).

## 7. Chạy GPU (tùy chọn)
Cài bản torch có CUDA phù hợp; app tự dùng `cuda` nếu khả dụng, ngược lại `cpu`. Kết quả số có thể lệch ở mức float32 so với CPU.

## 8. Kiểm tra checkpoint
Khi khởi động app kiểm tra: tồn tại → kích thước hợp lý → **SHA-256 khớp `pipeline_lock_primary.json`** → `deploy_config.json` khớp lock → metadata/kiến trúc khớp (strict load) → label mapping. Sai bất kỳ mục nào ⇒ **khóa chức năng phân tích (fail-closed)**, không fallback sang model khác.
Hash thực tế đã xác minh: `f586d7649e86c1146c64c4a579a8fb0f9a719feb06e84e9df1e0231d12c31dfe`.
Tham số khóa (đọc từ config, **không hard-code**): V3_full, seed 44, img_size 256, T = 1.259194134200016, ngưỡng phân loại = 0.17555833710788551, ngưỡng từ chối = 0.6931583152688364, seg_threshold 0.5, min_area 98, TTA lật ngang bật.
Kiểm tra thủ công: `python -c "from src.security import sha256_file; from src import config; print(sha256_file(config.MODEL_DIR/'V3_full_s44_best.pt'))"`.
> Checkpoint là artifact **tin cậy** đóng gói cùng repo (nạp bằng `torch_load` gốc của dự án, cần thiết cho định dạng checkpoint) nên hash-verify trước khi nạp. **Không** cho người dùng upload checkpoint.

## 9. Chạy tests
```
pytest -q            # 62 test: parity, calibration, ngưỡng khóa, perturbation, dashboard, privacy, CPU smoke...
```
Parity test so sánh backend với `C2.analyze_image_v2` và `C2.predict_arrays` của notebook trên cùng ảnh/checkpoint (xác suất thô, xác suất hiệu chuẩn, nhãn, abstain, mask, evidence, Grad-CAM). Dung sai: ≤1e-6 trên xác suất/seg_prob (sai khác 1 ULP float32 giữa hai lượt chạy CPU).

## 10. Cấu trúc dữ liệu kết quả (`results/`)
`internal_results_per_seed.csv`, `internal_results_mean_sd.csv`, `paired_comparisons.csv`, `calibration.csv`, `selective_prediction.csv`, `xai_evaluation.csv` (`model, group, metric, value`), `external_busi_results.json`, `research_report.md`. Loader xác thực cột; thiếu cột/JSON hỏng ⇒ thông báo lỗi rõ ràng. Dashboard **không** nhập tay số liệu. Bảng dự đoán theo ca (tùy chọn): xem trang *Error Cases*.
Không đưa `busbra_split_manifest.csv` vào repo (chứa patient ID và đường dẫn).

## 11. Thêm ca demo ẩn danh
Tạo `assets/demo_cases/manifest.json`:
```json
{"cases": [{"id": "demo-01", "file": "demo01.png", "source": "tên bộ dữ liệu công khai + trích dẫn", "license": "giấy phép", "description": "mẫu dữ liệu nghiên cứu", "anonymized": true}]}
```
Có thể thêm `gt_mask`, `gt_label` (chỉ hiện khi bật “Chế độ nghiên cứu — hiển thị ground truth”). Ảnh phải không có thông tin định danh và được phép phân phối. Không có manifest ⇒ tính năng tự tắt.

## 12. Thêm checkpoint X6 đúng cách
Hiện **không có checkpoint X6** ⇒ chỉ có “X6 — kết quả thực nghiệm lưu trữ” từ CSV. Để bật live: (1) đặt checkpoint + `deploy_config`/lock riêng cho X6 trong thư mục `model/x6/`; (2) xác minh SHA-256 và metadata kiến trúc; (3) xác minh preprocessing; (4) viết và chạy parity test như `tests/test_inference_parity.py`; (5) chỉ khi pass mới thêm selector mô hình. Không tự chọn checkpoint thay thế.

## 13. Vì sao BUS-VERIFY bị tắt
Theo kiểm toán, failure detector chưa có artifact huấn luyện/kiểm định hợp lệ (mô hình, feature schema, version/hash, ngưỡng từ validation, báo cáo đánh giá, test không dùng GT, xác nhận không fit trên locked test). `src/failure_gate.py` kiểm tra đủ các artifact này; thiếu ⇒ hiển thị “chưa khả dụng”. App **không** tạo điểm rủi ro bằng random, 1−confidence, entropy, Δlogit hay abstain score.

## 14. Privacy
Ảnh upload chỉ xử lý trong RAM (`cv2.imdecode`), không `save()`, không tempfile, không database, không analytics, không gọi API ngoài, không log bytes/UploadedFile. Báo cáo ca (`BreastVision_XAI_research_case_<uuid>.json`) tạo trong RAM, không chứa tên tệp gốc/EXIF/bytes/đường dẫn. Test tĩnh (`tests/test_privacy.py`) chặn import mạng/lưu trữ và lệnh ghi file trong mã ứng dụng. Nếu triển khai public: không tải dữ liệu có thông tin nhận dạng cá nhân.

## 15. Troubleshooting
| Triệu chứng | Cách xử lý |
|---|---|
| “SHA-256 không khớp” | Checkpoint hỏng/bị thay; chép lại đúng file từ `deploy_xai/`. |
| “Không tìm thấy checkpoint” | Đảm bảo `model/V3_full_s44_best.pt` có mặt (file lớn; Git LFS nếu dùng GitHub). |
| Chậm | CPU ~1–2 s/ảnh; Grad-CAM mặc định tắt (tick khi cần). |
| `ModuleNotFoundError: bus_core` | Chạy từ thư mục repo (`streamlit run app.py`); `breastvision_core/` được thêm vào `sys.path` bởi `src/config.py`. |
| Hết RAM | Dùng ảnh nhỏ hơn; đóng ứng dụng khác. |
| Cổng 8501 bận | `streamlit run app.py --server.port 8502`. |

## 16. Triển khai tùy chọn
**Chạy local offline là phương án chuẩn cho buổi bảo vệ.** Triển khai public (Streamlit Community Cloud/Hugging Face Spaces) chỉ là tùy chọn; checkpoint ~65 MB cần Git LFS khi >100 MB, và nền tảng miễn phí có thể thiếu RAM/CPU. Không thay checkpoint bằng model nhỏ hơn trừ khi ghi rõ đó là model khác. Không bật lưu trữ ảnh/analytics.
