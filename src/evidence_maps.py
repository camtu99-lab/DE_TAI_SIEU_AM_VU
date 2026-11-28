"""Tiện ích bản đồ bằng chứng (numpy). Evidence do mô hình V3 tạo (lưới h×w, tổng = 1); Grad-CAM nằm ở bus_core_v2 (cần torch)."""
import numpy as np
import cv2


def upsample_evidence(E, size_hw):
    E = np.asarray(E, np.float32)
    out = cv2.resize(E, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
    s = out.sum()
    return out / s if s > 0 else out


def evidence_mass_in(E, region, valid=None):
    """Tỉ lệ khối lượng evidence nằm trong `region` (mặt nạ DỰ ĐOÁN khi suy luận)."""
    E = np.asarray(E, np.float64)
    if valid is not None:
        E = E * valid
    s = E.sum()
    return float((E * region).sum() / s) if s > 0 else float('nan')


def evidence_entropy_norm(E):
    p = np.asarray(E, np.float64).ravel()
    p = p / max(p.sum(), 1e-12)
    h = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(h / np.log(p.size))


def cosine(a, b, eps=1e-8):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def oracle_localisation(E, gt_mask, valid=None):
    """ORACLE (đánh giá offline, báo cáo tách riêng): khối lượng evidence trong GT lesion và 'area-adjusted' (0 = ngẫu nhiên)."""
    gt = np.asarray(gt_mask) > 0
    mass = evidence_mass_in(E, gt, valid)
    area = float(gt.mean())
    return dict(mass_in_gt=mass, area_adjusted=(mass - area) / max(1 - area, 1e-8))
