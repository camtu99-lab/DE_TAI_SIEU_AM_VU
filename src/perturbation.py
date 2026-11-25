"""Perturbation Lab: can thiệp vùng ảnh dựa trên MASK DỰ ĐOÁN (không GT) để minh họa độ nhạy của mô hình.
Không phải bằng chứng nhân quả y khoa. Mặc định equal-area: mọi vùng bị tác động cùng số pixel."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from src import config
from src.inference import infer_preprocessed
from src.security import UserFacingError

REGION_LABELS = {'lesion': 'Tổn thương (mask dự đoán)', 'ring': 'Vòng quanh tổn thương', 'posterior': 'Vùng hậu âm', 'background': 'Nền xa'}
KINDS = {'shuffle': 'Matched pixel shuffle (hoán vị pixel trong vùng chọn)', 'blur': 'Local blur (làm mờ cục bộ)'}
WARNING_TEXT = ('Perturbation có thể tạo ảnh ngoài phân phối huấn luyện. Độ thay đổi dự đoán phản ánh độ nhạy của mô hình, '
                'không chứng minh đặc điểm đó là nguyên nhân bệnh lý.')


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def build_regions(mask: np.ndarray, post_prob: np.ndarray | None, valid: np.ndarray) -> tuple[dict, dict]:
    """Trả về (regions hợp lệ, lý do vùng không hợp lệ). Các vùng loại trừ nhau và nằm trong vùng ảnh thật (không phải viền letterbox)."""
    valid = valid.astype(bool)
    lesion = (mask > 0) & valid
    if int(lesion.sum()) < config.MIN_REGION_PIXELS:
        raise UserFacingError('perturbation', 'Mask dự đoán rỗng hoặc quá nhỏ: không thể xác định các vùng để can thiệp.', f'lesion_px={int(lesion.sum())}')
    near = _dilate(lesion, config.RING_PX)
    ring = near & ~lesion & valid
    post = None
    if post_prob is not None:
        post = (post_prob >= 0.5) & valid & ~lesion & ~ring
    far = valid & ~_dilate(lesion, config.FAR_BACKGROUND_GAP_PX)
    if post is not None:
        far &= ~post
    cand = {'lesion': lesion, 'ring': ring, 'posterior': post, 'background': far}
    regions, skipped = {}, {}
    for k, v in cand.items():
        if v is None:
            skipped[k] = 'kiến trúc không cung cấp vùng hậu âm'
        elif int(v.sum()) < config.MIN_REGION_PIXELS:
            skipped[k] = f'vùng quá nhỏ ({int(v.sum())} pixel < {config.MIN_REGION_PIXELS})'
        else:
            regions[k] = v
    return regions, skipped


def plan_pixels(regions: dict, fraction: float, seed: int, equal_area: bool = True) -> dict:
    """Chọn ngẫu nhiên (có seed) các pixel bị tác động. equal_area: N = floor(fraction × kích thước vùng nhỏ nhất) cho MỌI vùng."""
    if not 0.0 < fraction <= 1.0:
        raise UserFacingError('perturbation', 'Tỉ lệ can thiệp phải trong (0, 1].')
    if len(regions) < 2:
        raise UserFacingError('perturbation', 'Không đủ vùng hợp lệ (cần ≥ 2) để so sánh.')
    sizes = {k: int(v.sum()) for k, v in regions.items()}
    n_eq = max(1, int(math.floor(fraction * min(sizes.values()))))
    out = {}
    for i, (k, reg) in enumerate(regions.items()):
        n = n_eq if equal_area else max(1, int(math.ceil(fraction * sizes[k])))
        idx = np.flatnonzero(reg.ravel())
        rng = np.random.default_rng([int(seed), i])
        out[k] = np.sort(rng.choice(idx, size=min(n, idx.size), replace=False))
    return out


def apply_perturbation(img: np.ndarray, idx: np.ndarray, kind: str, seed: int, blur_sigma: float = 4.0) -> np.ndarray:
    if kind not in KINDS:
        raise UserFacingError('perturbation', f'Kiểu perturbation không hỗ trợ: {kind}')
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3).copy()
    if kind == 'shuffle':
        rng = np.random.default_rng([int(seed), 9973])
        flat[idx] = flat[idx][rng.permutation(idx.size)]
    else:
        blurred = cv2.GaussianBlur(img, (0, 0), blur_sigma).reshape(-1, 3)
        flat[idx] = blurred[idx]
    return flat.reshape(h, w, 3)


@dataclass
class PerturbationOutcome:
    region: str
    region_size: int
    n_perturbed: int
    orig_logit_malignant: float
    pert_logit_malignant: float
    delta_logit_malignant: float
    orig_p_malignant_cal: float
    pert_p_malignant_cal: float
    orig_label: str
    pert_label: str
    label_changed: bool
    abstain_changed: bool
    image: np.ndarray
    selected: np.ndarray


def run_lab(rt, base, regions: dict, kind: str, fraction: float, seed: int, equal_area: bool = True) -> list[PerturbationOutcome]:
    """base: InferenceResult của ảnh gốc. Mỗi vùng -> một lượt suy luận trên ảnh đã can thiệp (cùng pipeline, TTA, hiệu chuẩn và ngưỡng khóa)."""
    plan = plan_pixels(regions, fraction, seed, equal_area)
    outs = []
    for name, idx in plan.items():
        img_p = apply_perturbation(base.image, idx, kind, seed)
        r = infer_preprocessed(rt, img_p, base.original_shape, want_extras=False)
        sel = np.zeros(base.image.shape[:2], bool)
        sel.ravel()[idx] = True
        z0, z1 = float(base.raw_logits[config.LABEL_MALIGNANT]), float(r.raw_logits[config.LABEL_MALIGNANT])
        outs.append(PerturbationOutcome(name, int(regions[name].sum()), int(idx.size), z0, z1, z1 - z0, base.p_malignant, r.p_malignant, base.label, r.label,
                                        r.label != base.label, r.abstain != base.abstain, img_p, sel))
    return outs
