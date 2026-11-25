"""Hiển thị: overlay, contour, bounding box, heatmap, biểu đồ. Dùng helper overlay gốc của bus_core."""
from __future__ import annotations

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import bus_core as core   # noqa: E402

PALETTE = {'neutral': '#495057', 'info': '#0b7285', 'warning': '#e67700', 'error': '#c92a2a', 'pos': '#c2255c', 'neg': '#1c7ed6'}
REGION_COLORS = {'lesion': (230, 70, 70), 'ring': (240, 180, 40), 'posterior': (60, 130, 220), 'background': (120, 120, 120)}


def norm01(a) -> np.ndarray:
    a = np.asarray(a, np.float32)
    return a / (a.max() + 1e-12)


def contour_overlay(img: np.ndarray, mask: np.ndarray, color=(255, 255, 0), thickness: int = 1) -> np.ndarray:
    out = img.copy()
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thickness)
    return out


def seg_panel(img, mask, boxes, alpha: float, show_boxes: bool = True, show_contour: bool = True) -> np.ndarray:
    ov = core.overlay_mask(img, mask, alpha=alpha) if mask.any() else img.copy()
    if show_contour and mask.any():
        ov = contour_overlay(ov, mask)
    if show_boxes and boxes:
        ov = core.draw_boxes(ov, boxes)
    return ov


def bbox_only_panel(img, boxes) -> np.ndarray:
    return core.draw_boxes(img, boxes) if boxes else img.copy()


def heat_panel(img, heat, alpha: float) -> np.ndarray:
    return core.overlay_cam(img, norm01(heat), alpha=alpha)


def region_overlay(img: np.ndarray, regions: dict, alpha: float = 0.45, selected: dict | None = None) -> np.ndarray:
    out = img.astype(np.float32).copy()
    for name, reg in regions.items():
        m = (selected[name] if selected and name in selected else reg)
        col = np.array(REGION_COLORS[name], np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * col
    return out.clip(0, 255).astype(np.uint8)


def analysis_panels(res, alpha: float, show_boxes: bool = True, show_contour: bool = True, original_rgb=None):
    """Danh sách (tiêu đề, ảnh) cho các panel chính."""
    p = []
    if original_rgb is not None:
        p.append(('1. Ảnh gốc', original_rgb))
    p.append(('2. Sau tiền xử lý (letterbox)', res.image))
    p.append(('3. Mask dự đoán', seg_panel(res.image, res.mask, [], alpha, False, True)))
    p.append(('4. Bounding box từ mask dự đoán', bbox_only_panel(res.image, res.boxes)))
    if res.cam is not None:
        p.append(('5. Grad-CAM: vùng kích hoạt ảnh hưởng đến dự đoán', heat_panel(res.image, res.cam, alpha)))
    if res.evidence is not None:
        p.append(('6. Evidence map (nội tại của mô hình)', heat_panel(res.image, res.evidence, alpha)))
    return p


def contrib_figure(c: dict):
    names = list(c['names']) + ['Hằng số (bias)']
    margin = list(c['vals'][:, 1] - c['vals'][:, 0]) + [float(c['bias'][1] - c['bias'][0])]
    total = float(sum(margin))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.3))
    ax.barh(names[::-1], margin[::-1], color=[PALETTE['pos'] if v > 0 else PALETTE['neg'] for v in margin[::-1]])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Đóng góp vào (logit malignant − logit benign), trước hiệu chuẩn')
    ax.set_title(f'Tổng = {total:+.3f}  (khớp logit mô hình; sai số {c["max_abs_err"]:.1e})', fontsize=10)
    fig.tight_layout()
    return fig, margin, total


def model_bar_figure(df, metric: str, title: str, models=None, highlight: str = 'V3_full'):
    """Cột mean ± sd theo mô hình từ internal_results_mean_sd."""
    d = df.copy()
    if models:
        d = d[d['model'].isin(models)]
    mean, sd = d[f'{metric}_mean'].astype(float).values, d[f'{metric}_sd'].astype(float).fillna(0).values
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    cols = [PALETTE['info'] if m == highlight else '#adb5bd' for m in d['model']]
    ax.bar(d['model'], mean, yerr=sd, color=cols, capsize=4)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(metric)
    lo = max(0.0, float(np.nanmin(mean - sd)) - 0.05)
    ax.set_ylim(lo, min(1.0, float(np.nanmax(mean + sd)) + 0.04))
    fig.tight_layout()
    return fig
