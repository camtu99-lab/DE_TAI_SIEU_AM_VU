"""Hằng số, đường dẫn và cầu nối import tới mã nguồn gốc của notebook (breastvision_core/).

Mã gốc dùng import phẳng (`import bus_core as core`), nên thư mục breastvision_core được thêm vào sys.path.
File trong breastvision_core/ được giữ NGUYÊN VĂN từ gói deploy_xai để bảo đảm tái lập inference.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'model'
CORE_DIR = ROOT / 'breastvision_core'
RESULTS_DIR = ROOT / 'results'
ASSETS_DIR = ROOT / 'assets'
DEMO_DIR = ASSETS_DIR / 'demo_cases'
FAILURE_DETECTOR_DIR = MODEL_DIR / 'failure_detector'

APP_NAME = 'BreastVision-XAI V3'
APP_SUBTITLE = 'Explainable Multi-task AI for Breast Ultrasound Research'
APP_VERSION = '1.0.0'
BADGE = 'RESEARCH DEMO — NOT FOR CLINICAL DIAGNOSIS'

# Mapping nhãn giữ nguyên với notebook/data_audit (BUS-BRA: 1 if Pathology startswith 'm'; BUSI: int(c == 'malignant')).
CLASS_NAMES = ('benign', 'malignant')
LABEL_BENIGN = 0
LABEL_MALIGNANT = 1
CLASS_VI = {'benign': 'BENIGN', 'malignant': 'MALIGNANT'}

ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.bmp')
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MIN_SIDE_PX = 64
MAX_SIDE_PX = 6000
MAX_PIXELS = 36_000_000
CKPT_MIN_BYTES = 1 * 1024 * 1024
CKPT_MAX_BYTES = 600 * 1024 * 1024

REQUIRED_CONFIG_FIELDS = {
    'experiment': str, 'checkpoint': str, 'img_size': int, 'temperature': float, 'threshold': float,
    'abstain_threshold': float, 'seg_threshold': float, 'min_area': int, 'run_mode': str,
    'pretrained_loaded': bool, 'seed': int, 'tta': bool,
}

# BUS-VERIFY (failure-risk detector) mặc định TẮT; chỉ bật khi artifact đầy đủ (src/failure_gate.py).
ENABLE_FAILURE_DETECTOR = False

# Perturbation
RING_PX = 16               # = ring_cells(2) x stride(8) của notebook
FAR_BACKGROUND_GAP_PX = 32
MIN_REGION_PIXELS = 50


def ensure_core_on_path() -> None:
    p = str(CORE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


ensure_core_on_path()
