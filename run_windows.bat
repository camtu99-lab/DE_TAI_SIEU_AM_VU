@echo off
REM BreastVision-XAI V3 — chạy local offline trên Windows (CPU)
cd /d "%~dp0"
if not exist .venv (
  echo [1/3] Tao moi truong ao .venv ...
  py -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
echo [2/3] Kiem tra checkpoint + test nhanh (bo qua bang: set SKIP_TESTS=1)
if not "%SKIP_TESTS%"=="1" python -m pytest -q tests\test_checkpoint_integrity.py tests\test_label_mapping.py
echo [3/3] Khoi dong ung dung tai http://localhost:8501
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
