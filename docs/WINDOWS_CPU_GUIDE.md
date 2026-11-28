# Demo trên Windows (CPU)
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
set BV_DEPLOY_DIR=deploy        (tùy chọn)
streamlit run app.py
```
Cần sẵn `deploy/` (từ notebook Kaggle) và `artifacts/v4/` để có risk score và trang kết quả. Thiếu → app hiển thị "NOT RUN"/"Risk TẮT" kèm lý do; KHÔNG sinh điểm giả. Suy luận 1 ảnh kèm counterfactual trên CPU: chưa đo thời gian thực tế (NOT RUN); ước tính vài chục giây — cần đo khi chạy thật.
