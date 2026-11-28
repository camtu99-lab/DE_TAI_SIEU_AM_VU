"""Hậu xử lý mặt nạ phân vùng dự đoán và dựng 3 vùng (lesion / perilesional / background). Thuần numpy/cv2.
QUY ƯỚC: mọi hàm ở đây nhận mặt nạ DỰ ĐOÁN. Mặt nạ ground-truth chỉ được dùng trong các hàm có tiền tố `oracle_`."""
from dataclasses import dataclass
import numpy as np
import cv2


def dice(a, b, eps=1e-8):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0                       # cả hai rỗng: trùng khớp tuyệt đối (quy ước, được ghi trong báo cáo)
    return float(2.0 * (a & b).sum() / (s + eps))


def iou(a, b, eps=1e-8):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / (u + eps))


def postprocess_mask(seg_prob, thr=0.5, min_area_frac=0.0015, keep='largest'):
    """seg_prob (H,W) -> mask uint8 + info. keep='largest' giữ thành phần liên thông lớn nhất; 'all' giữ mọi thành phần >= min_area."""
    sp = np.asarray(seg_prob, dtype=np.float32)
    H, W = sp.shape
    m = (sp >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_area = max(1, int(round(min_area_frac * H * W)))
    comps = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    comps.sort(reverse=True)
    out = np.zeros_like(m)
    if comps:
        if keep == 'largest':
            out[lab == comps[0][1]] = 1
        else:
            for _, i in comps:
                out[lab == i] = 1
    info = dict(n_components=len(comps), area_frac=float(out.mean()), empty=not out.any(), min_area_px=min_area,
                n_raw_components=int(n - 1))
    return out, info


@dataclass
class Regions:
    lesion: np.ndarray
    peri: np.ndarray
    background: np.ndarray
    valid: np.ndarray
    ok: bool
    reason: str = ''


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def build_regions(pred_mask, valid=None, ring_px=12, min_lesion_px=16):
    """Ba vùng rời nhau, hợp = vùng nội dung hợp lệ: lesion = mask dự đoán ∩ valid; peri = giãn(lesion, ring_px) \\ lesion ∩ valid;
    background = valid \\ (lesion ∪ peri). Mặt nạ rỗng/quá nhỏ -> ok=False (KHÔNG tự bịa vùng)."""
    pm = np.asarray(pred_mask) > 0
    H, W = pm.shape
    v = np.ones((H, W), bool) if valid is None else np.asarray(valid, bool)
    les = pm & v
    if les.sum() < min_lesion_px:
        z = np.zeros((H, W), bool)
        return Regions(z, z, v & ~z, v, False, 'empty_mask' if les.sum() == 0 else 'mask_too_small')
    dil = cv2.dilate(les.astype(np.uint8), _disk(int(ring_px))) > 0
    peri = dil & ~les & v
    bg = v & ~dil
    return Regions(les, peri, bg, v, True, '')


def regions_are_disjoint(r: Regions):
    a, b, c = r.lesion, r.peri, r.background
    return not ((a & b).any() or (a & c).any() or (b & c).any())


def mask_plausibility(info, min_frac=0.0015, max_frac=0.60, max_components=3):
    """Cờ 'ngoài phạm vi đã kiểm định' theo hình thái mặt nạ: rỗng, quá lớn, quá nhiều mảnh."""
    reasons = []
    if info['empty']:
        reasons.append('mask_empty')
    if info['area_frac'] > max_frac:
        reasons.append('mask_too_large')
    if info['n_components'] > max_components:
        reasons.append('mask_fragmented')
    return reasons


def oracle_mask_agreement(pred_mask, gt_mask):
    """ORACLE (chỉ dùng khi đánh giá offline, báo cáo riêng): Dice/IoU giữa mask dự đoán và GT."""
    return dict(dice=dice(pred_mask, gt_mask), iou=iou(pred_mask, gt_mask))
