import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402,F401  (thêm breastvision_core vào sys.path)


@pytest.fixture(scope='session')
def rt():
    from src.checkpoint import build_runtime
    r = build_runtime(device=torch.device('cpu'))
    assert r.ok, [(x.name, x.status, x.detail) for x in r.rows]
    return r


def make_synthetic(h=300, w=400, seed=0):
    """Ảnh B-mode tổng hợp xác định (speckle + vùng tối elip). Chỉ dùng cho test kỹ thuật, không phải dữ liệu y khoa."""
    rng = np.random.default_rng(seed)
    g = rng.normal(90, 25, (h, w)).clip(0, 255).astype(np.uint8)
    cv2.ellipse(g, (w // 2, h // 2 - 10), (60, 35), 10, 0, 360, 40, -1)
    return np.stack([g, g, g], -1)


@pytest.fixture(scope='session')
def synth_rgb():
    return make_synthetic()


@pytest.fixture(scope='session')
def synth_result(rt, synth_rgb):
    from src import inference as INF
    return INF.infer_image(rt, synth_rgb, with_gradcam=True)
