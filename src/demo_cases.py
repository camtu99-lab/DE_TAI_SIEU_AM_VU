"""Ca mẫu ẩn danh: chỉ bật nếu assets/demo_cases/manifest.json tồn tại, hợp lệ, có nguồn rõ ràng. Không tạo ca giả."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import config
from src.security import UserFacingError, safe_child

import bus_core as core   # noqa: E402


@dataclass
class DemoCase:
    id: str
    path: Path
    source: str
    license: str
    description: str
    gt_label: str | None = None
    gt_mask_path: Path | None = None


def list_demo_cases(demo_dir: Path | None = None) -> tuple[list[DemoCase], str]:
    demo_dir = Path(demo_dir or config.DEMO_DIR)
    mf = demo_dir / 'manifest.json'
    if not mf.exists():
        return [], 'Chưa có ca mẫu ẩn danh (assets/demo_cases/manifest.json không tồn tại).'
    try:
        data = json.loads(mf.read_text(encoding='utf-8'))
        items = data['cases']
    except (OSError, ValueError, KeyError, TypeError):
        return [], 'manifest.json của ca mẫu không hợp lệ; tính năng bị tắt.'
    cases = []
    for it in items:
        try:
            if not it.get('anonymized') or not it.get('source') or not it.get('id'):
                continue
            f = safe_child(demo_dir, it['file'], config.ALLOWED_EXT)
            gm = safe_child(demo_dir, it['gt_mask'], config.ALLOWED_EXT) if it.get('gt_mask') else None
            if f.exists():
                cases.append(DemoCase(str(it['id']), f, it['source'], it.get('license', 'không rõ'), it.get('description', 'mẫu dữ liệu nghiên cứu'),
                                      it.get('gt_label'), gm if (gm and gm.exists()) else None))
        except (UserFacingError, AttributeError, TypeError):
            continue
    return cases, ('' if cases else 'manifest.json không có ca hợp lệ (cần anonymized=true, source, file tồn tại).')


def dice_vs_gt(pred_mask: np.ndarray, gt_mask_bytes_rgb: np.ndarray, size: int) -> float:
    """Chỉ dùng ở chế độ nghiên cứu cho ca demo có GT; KHÔNG dùng trong inference."""
    gt = core.preprocess_mask((gt_mask_bytes_rgb[..., 0] > 127).astype(np.uint8), size) > 0
    p = pred_mask > 0
    return float(2 * (p & gt).sum() / (p.sum() + gt.sum() + 1e-9))
