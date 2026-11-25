# BreastVision-XAI V3 — chạy local offline trên Windows (CPU)
$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
if (-not (Test-Path .venv)) {
    Write-Host '[1/3] Tạo môi trường ảo .venv ...'
    py -m venv .venv
    & .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
} else {
    & .\.venv\Scripts\Activate.ps1
}
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
if ($env:SKIP_TESTS -ne '1') {
    Write-Host '[2/3] Kiểm tra checkpoint + test nhanh'
    python -m pytest -q tests/test_checkpoint_integrity.py tests/test_label_mapping.py
}
Write-Host '[3/3] Khởi động ứng dụng tại http://localhost:8501'
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
