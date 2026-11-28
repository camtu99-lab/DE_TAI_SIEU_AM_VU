# losses_xai.py - loss phân vùng/phân loại, boundary loss, ràng buộc giải thích không gian, counterfactual faithfulness, nhất quán và hình học
import math
import torch
import torch.nn.functional as F

EPS = 1e-8


# ============================ Phân vùng / phân loại ============================
def dice_loss_from_logits(logits, target, eps=1.0):
    p = torch.sigmoid(logits.float()).flatten(1)
    t = target.float().flatten(1)
    inter = (p * t).sum(1)
    return 1 - ((2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def seg_loss(seg_logits, masks):
    assert seg_logits.shape == masks.shape, f'seg {tuple(seg_logits.shape)} != mask {tuple(masks.shape)}'
    return F.binary_cross_entropy_with_logits(seg_logits.float(), masks.float()) + dice_loss_from_logits(seg_logits, masks)


def boundary_mask(masks):
    # Đường biên hình thái học (dilate - erode) trên mask GT, dùng để tăng trọng số loss tại ranh giới tổn thương
    m = masks.float()
    eroded = 1.0 - dilate(1.0 - m, 1)
    return (dilate(m, 1) - eroded).clamp(0, 1)


def boundary_loss(seg_logits, masks, edge_weight=3.0):
    assert seg_logits.shape == masks.shape
    m = masks.float()
    edge = boundary_mask(m)
    w = 1.0 + edge_weight * edge
    bce = F.binary_cross_entropy_with_logits(seg_logits.float(), m, weight=w)
    p = torch.sigmoid(seg_logits.float())
    inter = (p * m * edge).sum((1, 2, 3))
    den = (p * edge).sum((1, 2, 3)) + (m * edge).sum((1, 2, 3))
    edge_dice = 1.0 - ((2.0 * inter + 1.0) / (den + 1.0)).mean()
    return 0.5 * bce + 0.5 * edge_dice


def cls_loss(logits, labels, class_weights=None, label_smoothing=0.0):
    return F.cross_entropy(logits.float(), labels, weight=class_weights, label_smoothing=float(label_smoothing))


# ============================ ROI: tổn thương, vòng quanh và vùng hậu âm BI-RADS ============================
def downsample_mask(mask, size):
    # mask (B,1,H,W) -> (B,1,h,w) trung bình diện tích, giá trị trong [0,1]
    return F.adaptive_avg_pool2d(mask.float(), size)


def dilate(m, k):
    # giãn nở bằng max-pool (k = số ô đặc trưng); khả vi theo đoạn, dùng cho mask mềm
    return F.max_pool2d(m, kernel_size=2 * k + 1, stride=1, padding=k)


def lesion_ring(m, k):
    # vòng quanh tổn thương: dilate_k(m) - m, nằm trong [0,1]
    return (dilate(m, k) - m).clamp(0, 1)


def lesion_plus_context(m, k):
    return dilate(m, k).clamp(0, 1)


def posterior_zone(m, depth_cells=6):
    # Vùng hậu âm phía dưới tổn thương theo trục dọc chùm tia siêu âm (BI-RADS posterior acoustic shadowing/enhancement).
    # Chỉ phụ thuộc mask dự đoán m (B,1,h,w), hoàn toàn không dùng GT mask lúc suy luận; tách rời với m.
    if depth_cells <= 0:
        return torch.zeros_like(m)
    k_h = 2 * depth_cells + 1
    ker = m.new_zeros((1, 1, k_h, 3))
    ker[0, 0, :depth_cells, :] = 1.0
    raw = F.conv2d(m.float(), ker, padding=(depth_cells, 1))
    return (raw.clamp(0.0, 1.0) * (1.0 - m.float())).clamp(0.0, 1.0)


def normalize_evidence(E):
    # chuẩn hóa tổng = 1 trên không gian: loại bỏ phụ thuộc vào thang đo tuyệt đối
    return E / (E.sum((1, 2, 3), keepdim=True) + EPS)


# ============================ Spatial explanation constraint ============================
def explanation_loss(E, gt_mask, mode='lesion_context', ring_cells=2, delta=0.5, density_margin=1.0, min_roi=1e-3, max_roi=0.9):
    En = normalize_evidence(E.float())
    m = downsample_mask(gt_mask, E.shape[-2:])
    roi = lesion_plus_context(m, ring_cells)
    a_l, a_r = m.mean((1, 2, 3)), roi.mean((1, 2, 3))
    M_l, M_r = (En * m).sum((1, 2, 3)), (En * roi).sum((1, 2, 3))
    if mode == 'none':
        per = torch.zeros_like(M_l)
    elif mode == 'mask_only':
        per = 1 - M_l
    elif mode == 'lesion_context':
        per = F.relu((a_r + delta * (1 - a_r)) - M_r)
    elif mode == 'density':
        rho_in, rho_out = M_r / (a_r + EPS), (1 - M_r).clamp(min=0) / (1 - a_r + EPS)
        per = F.relu(torch.log(rho_out + EPS) - torch.log(rho_in + EPS) + math.log(1 + density_margin))
    else:
        raise ValueError('expl mode không hợp lệ: ' + str(mode))
    valid = ((a_r > min_roi) & (a_r < max_roi)).float()
    loss = (per * valid).sum() / valid.sum().clamp(min=1.0)
    return loss, dict(M_l=M_l.detach(), M_r=M_r.detach(), a_l=a_l.detach(), a_r=a_r.detach(), valid=valid.detach())


# ============================ Counterfactual faithfulness loss (V3) ============================
def counterfactual_faithfulness_loss(logits_full, logits_drop_hi, logits_drop_lo, margin=0.15, low_tol=0.05, conf_thr=0.6):
    # Tối ưu trực tiếp tính trung thực nhân quả: che vùng evidence cao nhất phải làm tụt xác suất lớp dự đoán >= margin,
    # trong khi che vùng evidence thấp nhất chỉ gây thay đổi nhỏ <= low_tol (trên các mẫu có độ tự tin >= conf_thr).
    p_full = F.softmax(logits_full.float(), 1)
    p_hi = F.softmax(logits_drop_hi.float(), 1)
    p_lo = F.softmax(logits_drop_lo.float(), 1)
    cls = p_full.argmax(1)
    idx = torch.arange(p_full.size(0), device=p_full.device)
    pc = p_full[idx, cls]
    d_hi = pc - p_hi[idx, cls]
    d_lo = pc - p_lo[idx, cls]
    per = F.relu(margin - d_hi) + F.relu(d_lo.abs() - low_tol)
    valid = (pc >= conf_thr).float()
    return (per * valid).sum() / valid.sum().clamp(min=1.0)


# ============================ Hình học: affine, inverse warp, mặt nạ pixel hợp lệ ============================
def sample_affine(B, device, max_rot_deg=10.0, scale=(0.9, 1.1), max_trans=0.05, flip_prob=0.5, generator=None):
    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(B, device=device, generator=generator)
    ang = u(-max_rot_deg, max_rot_deg) * math.pi / 180
    s = u(scale[0], scale[1])
    fx = torch.where(torch.rand(B, device=device, generator=generator) < flip_prob, -torch.ones(B, device=device), torch.ones(B, device=device))
    tx, ty = u(-max_trans, max_trans) * 2, u(-max_trans, max_trans) * 2
    c, sn = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = c / s * fx, -sn / s, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sn / s * fx, c / s, ty
    return theta


def hflip_theta(B, device):
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 1, 1] = -1.0, 1.0
    return theta


def invert_theta(theta):
    B = theta.shape[0]
    A = torch.cat([theta, torch.tensor([0., 0., 1.], device=theta.device).view(1, 1, 3).expand(B, 1, 3)], 1)
    return torch.linalg.inv(A)[:, :2, :]


def warp(t, theta, mode='bilinear', padding_mode='zeros'):
    grid = F.affine_grid(theta, list(t.shape), align_corners=False)
    return F.grid_sample(t, grid, mode=mode, padding_mode=padding_mode, align_corners=False)


def inverse_warp(t_aug, theta, mode='bilinear'):
    return warp(t_aug, invert_theta(theta), mode=mode)


def valid_pixel_mask(theta, size_hw, device=None, thr=0.999):
    ones = torch.ones(theta.shape[0], 1, size_hw[0], size_hw[1], device=theta.device)
    real_content = (warp(ones, theta) > thr).float()
    return (inverse_warp(real_content, theta) > thr).float()


def intensity_jitter(x, contrast=(0.9, 1.1), brightness=0.1, generator=None):
    B = x.shape[0]
    c = contrast[0] + (contrast[1] - contrast[0]) * torch.rand(B, 1, 1, 1, device=x.device, generator=generator)
    b = (torch.rand(B, 1, 1, 1, device=x.device, generator=generator) * 2 - 1) * brightness
    return x * c + b


def apply_label_preserving(x, theta, jitter=True):
    xa = warp(x, theta)
    return intensity_jitter(xa) if jitter else xa


# ============================ Explanation consistency ============================
def prediction_consistency(logits1, logits2):
    p1, p2 = F.softmax(logits1.float(), 1), F.softmax(logits2.float(), 1)
    l1, l2 = F.log_softmax(logits1.float(), 1), F.log_softmax(logits2.float(), 1)
    kl12 = (p1 * (l1 - l2)).sum(1)
    kl21 = (p2 * (l2 - l1)).sum(1)
    return (0.5 * (kl12 + kl21)).mean()


def evidence_alignment(E1, E2, theta, min_valid_frac=0.25):
    E1n, E2n = normalize_evidence(E1.float()), normalize_evidence(E2.float())
    back = inverse_warp(E2n, theta)
    V = valid_pixel_mask(theta, E1.shape[-2:])
    a, b = (E1n * V).flatten(1), (back * V).flatten(1)
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + EPS)
    ok = (V.mean((1, 2, 3)) >= min_valid_frac).float()
    return cos, ok, back, V


def evidence_consistency(E1, E2, theta):
    cos, ok, _, _ = evidence_alignment(E1, E2, theta)
    return ((1 - cos) * ok).sum() / ok.sum().clamp(min=1.0)


# ============================ Tổng loss ============================
class LossCfg:
    def __init__(self, lam_seg=1.0, lam_cls=1.0, lam_exp=0.5, expl_mode='none', ring_cells=2, delta=0.5, density_margin=1.0,
                 lam_cons_pred=0.0, lam_cons_ev=0.0, label_smoothing=0.0, lam_edge=0.0, lam_faith=0.0, faith_q=0.2, faith_margin=0.15):
        self.__dict__.update(locals()); self.__dict__.pop('self')


def total_loss(out, masks, labels, cfg, class_weights=None, out_aug=None, theta=None, logits_drop_hi=None, logits_drop_lo=None):
    parts = {}
    if out.get('seg') is not None and cfg.lam_seg > 0:
        parts['seg'] = cfg.lam_seg * seg_loss(out['seg'], masks)
    if out.get('seg') is not None and getattr(cfg, 'lam_edge', 0.0) > 0:
        parts['edge'] = cfg.lam_edge * boundary_loss(out['seg'], masks)
    if out.get('logits') is not None and cfg.lam_cls > 0:
        parts['cls'] = cfg.lam_cls * cls_loss(out['logits'], labels, class_weights, getattr(cfg, 'label_smoothing', 0.0))
    if cfg.expl_mode != 'none' and out.get('evidence') is not None and cfg.lam_exp > 0:
        parts['exp'] = cfg.lam_exp * explanation_loss(out['evidence'], masks, cfg.expl_mode, cfg.ring_cells, cfg.delta, cfg.density_margin)[0]
    if out_aug is not None and theta is not None:
        if cfg.lam_cons_pred > 0 and out.get('logits') is not None:
            parts['cons_pred'] = cfg.lam_cons_pred * prediction_consistency(out['logits'], out_aug['logits'])
        if cfg.lam_cons_ev > 0 and out.get('evidence') is not None and out_aug.get('evidence') is not None:
            parts['cons_ev'] = cfg.lam_cons_ev * evidence_consistency(out['evidence'], out_aug['evidence'], theta)
    if getattr(cfg, 'lam_faith', 0.0) > 0 and logits_drop_hi is not None and logits_drop_lo is not None and out.get('logits') is not None:
        parts['faith'] = cfg.lam_faith * counterfactual_faithfulness_loss(
            out['logits'], logits_drop_hi, logits_drop_lo, margin=getattr(cfg, 'faith_margin', 0.15)
        )
    total = sum(parts.values()) if parts else torch.zeros((), requires_grad=True)
    return total, {k: float(v.detach()) for k, v in parts.items()}
