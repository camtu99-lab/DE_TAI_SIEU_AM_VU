"""Suy luận V3 tái sử dụng đúng forward path/helper gốc (bus_core, bus_core_v2, bus_trust, evaluation_xai).

Quy tắc:
- Không huấn luyện, không fit calibration, không tối ưu ngưỡng, không dùng ground-truth (hàm KHÔNG nhận mask/nhãn).
- Inference chuẩn dùng torch.no_grad(); Grad-CAM tách riêng (enable_grad) và lỗi Grad-CAM không làm hỏng kết quả chính.
- T, ngưỡng phân loại, ngưỡng từ chối lấy từ deploy_config (đã khóa trên validation).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from src import config
from src.preprocessing import preprocess, to_tensor
from src.security import UserFacingError

import bus_core as core            # noqa: E402
import bus_core_v2 as C2           # noqa: E402
import bus_trust as T              # noqa: E402
import evaluation_xai as EV        # noqa: E402

DECOMP_TOL = 1e-3


@dataclass
class InferenceResult:
    image: np.ndarray
    original_shape: tuple
    raw_logits: np.ndarray                 # logit thô, ảnh gốc (không lật)
    raw_probs: np.ndarray                  # softmax thô (TTA nếu bật)
    p_cal: np.ndarray                      # softmax sau temperature scaling
    p_malignant: float
    label: str
    label_index: int
    confidence: float
    abstain: bool
    seg_prob: np.ndarray
    mask: np.ndarray
    boxes: list
    mask_fraction: float
    evidence: np.ndarray | None = None
    evidence_grid: np.ndarray | None = None
    ring_prob: np.ndarray | None = None
    post_prob: np.ndarray | None = None
    cam: np.ndarray | None = None
    cam_error: str | None = None
    contrib: dict | None = None
    decomposition_note: str = ''
    flip_abs_diff: float | None = None
    collapse: dict | None = None
    warnings: list = field(default_factory=list)
    timing_s: float = 0.0


def calibrate(raw_probs, temperature: float) -> np.ndarray:
    """Đúng công thức notebook: softmax(log(clip(p_raw)) / T) qua bus_trust.softmax."""
    return T.softmax(np.log(np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-12, 1)), float(temperature))


def decide(p_cal, threshold: float, abstain_threshold: float):
    """Quyết định theo NGƯỠNG ĐÃ KHÓA (không phải 0.5). Trả về (label, label_index, confidence, abstain)."""
    p_mal = float(p_cal[config.LABEL_MALIGNANT])
    label = config.CLASS_NAMES[config.LABEL_MALIGNANT] if p_mal >= float(threshold) else config.CLASS_NAMES[config.LABEL_BENIGN]
    conf = float(np.max(p_cal))
    return label, config.CLASS_NAMES.index(label), conf, bool(conf < float(abstain_threshold))


def _region_contributions(model, o, device):
    """Phân rã logit theo vùng cho fusion concat: z = Σ_i W_i·proj_i(r_i) + b (chính xác)."""
    mc = getattr(model, 'cfg', None)
    if mc is None or getattr(model, 'fusion', None) is None or getattr(model.fusion, 'mode', None) != 'concat' or not o.get('regions'):
        return None, 'Kiến trúc không hỗ trợ phân rã logit theo vùng.'
    names = ['Toàn ảnh (evidence)']
    if mc.use_lesion:
        names.append('Tổn thương (mask dự đoán)')
    if mc.use_peri:
        names.append('Quanh tổn thương')
    if mc.use_posterior:
        names.append('Hậu âm BI-RADS')
    D = mc.dim
    W, b = model.head[1].weight, model.head[1].bias
    vals = []
    for i, r in enumerate(o['regions']):
        v = model.fusion.proj[i](r)
        vals.append((v @ W[:, i * D:(i + 1) * D].T)[0].float().cpu().numpy())
    vals = np.array(vals)
    bias = b.float().cpu().numpy()
    logits = o['logits'][0].float().cpu().numpy()
    err = float(np.abs(vals.sum(0) + bias - logits).max())
    # đối chiếu chéo với hàm gốc decompose_logits nếu có
    if hasattr(model, 'decompose_logits'):
        try:
            Cc, rest, lg = model.decompose_logits(o['_x'])
            err2 = float((Cc.sum((2, 3)) + rest - lg).abs().max())
            err = max(err, err2)
        except (RuntimeError, ValueError, KeyError, AttributeError, TypeError):
            return None, 'decompose_logits gốc lỗi; ẩn phân rã.'
    if err > DECOMP_TOL or len(names) != len(vals):
        return None, f'Tổng đóng góp lệch logit {err:.2e} > {DECOMP_TOL}; ẩn phân rã để tránh suy diễn.'
    return dict(names=names, vals=vals, bias=bias, logits=logits, max_abs_err=err), ''


def infer_preprocessed(rt, img: np.ndarray, original_shape=None, want_extras: bool = True) -> InferenceResult:
    """Suy luận trên ảnh ĐÃ tiền xử lý (uint8 size×size×3). Không nhận ground truth."""
    t0 = time.perf_counter()
    cfg, model, device = rt.cfg, rt.model, rt.device
    size = int(cfg['img_size'])
    x = to_tensor(img, device)
    use_tta = bool(cfg['tta'])
    model.eval()
    try:
        with torch.no_grad():
            o = model(x)
            need_flip = use_tta or want_extras
            of = model(x.flip(-1)) if need_flip else None
            if use_tta:
                raw_t = 0.5 * (F.softmax(o['logits'].float(), 1) + F.softmax(of['logits'].float(), 1))
                sp_t = 0.5 * (torch.sigmoid(o['seg'].float()) + torch.sigmoid(of['seg'].float()).flip(-1))
            else:
                raw_t = F.softmax(o['logits'].float(), 1)
                sp_t = torch.sigmoid(o['seg'].float())
            raw = raw_t[0].cpu().numpy()
            logits = o['logits'][0].float().cpu().numpy()
            if not (np.isfinite(raw).all() and np.isfinite(logits).all() and torch.isfinite(sp_t).all()):
                raise UserFacingError('inference', 'Mô hình trả về giá trị NaN/Inf cho ảnh này; không đưa ra dự đoán.', 'non-finite logits/probs')
            cal = calibrate(raw, cfg['temperature'])
            if not np.isfinite(cal).all():
                raise UserFacingError('inference', 'Xác suất hiệu chuẩn không hữu hạn; không đưa ra dự đoán.', 'non-finite calibrated probs')
            label, lab_idx, conf, abst = decide(cal, cfg['threshold'], cfg['abstain_threshold'])
            sp = sp_t[0, 0].cpu().numpy()
            mask = (sp >= float(cfg['seg_threshold'])).astype(np.uint8)
            boxes = core.mask_to_boxes(mask, int(cfg['min_area']))
            res = InferenceResult(image=img, original_shape=tuple(original_shape or img.shape[:2]), raw_logits=logits, raw_probs=raw, p_cal=cal,
                                  p_malignant=float(cal[config.LABEL_MALIGNANT]), label=label, label_index=lab_idx, confidence=conf, abstain=abst,
                                  seg_prob=sp, mask=mask, boxes=boxes, mask_fraction=float(mask.mean()))
            if o.get('evidence') is not None:
                E = o['evidence'].float()
                if use_tta:
                    E = 0.5 * (E + of['evidence'].float().flip(-1))
                res.evidence_grid = E[0, 0].cpu().numpy()
                res.evidence = F.interpolate(E / (E.sum() + 1e-12), size=(size, size), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            if want_extras:
                for key, attr in (('ring', 'ring_prob'), ('post', 'post_prob')):
                    if o.get(key) is not None:
                        setattr(res, attr, F.interpolate(o[key].float(), size=(size, size), mode='bilinear', align_corners=False)[0, 0].cpu().numpy())
                if o.get('evidence') is not None:
                    res.collapse = EV.evidence_collapse_report(o['evidence'].float().cpu())
                pm_o = float(F.softmax(o['logits'].float(), 1)[0, 1])
                pm_f = float(F.softmax(of['logits'].float(), 1)[0, 1])
                res.flip_abs_diff = abs(pm_o - pm_f)
                o['_x'] = x
                res.contrib, res.decomposition_note = _region_contributions(model, o, device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            raise UserFacingError('inference', 'Hết bộ nhớ khi suy luận. Hãy dùng ảnh nhỏ hơn hoặc đóng ứng dụng khác.', f'{type(e).__name__}') from e
        raise UserFacingError('inference', 'Lỗi khi chạy mô hình.', f'{type(e).__name__}: {str(e)[:200]}') from e
    if res.mask_fraction == 0.0:
        res.warnings.append('Mask dự đoán rỗng: mô hình không tìm thấy vùng nghi ngờ; các vùng lesion/quanh/hậu âm không xác định.')
    elif res.mask_fraction > 0.6:
        res.warnings.append('Mask dự đoán chiếm >60% ảnh: phân vùng có thể không đáng tin cậy.')
    res.timing_s = time.perf_counter() - t0
    return res


def infer_image(rt, rgb: np.ndarray, with_gradcam: bool = False, want_extras: bool = True) -> InferenceResult:
    """Suy luận từ RGB uint8 gốc. KHÔNG có tham số mask/nhãn ground truth."""
    img = preprocess(rgb, int(rt.cfg['img_size']))
    res = infer_preprocessed(rt, img, rgb.shape[:2], want_extras=want_extras)
    if with_gradcam:
        compute_gradcam(rt, res)
    return res


def compute_gradcam(rt, res: InferenceResult) -> InferenceResult:
    """Grad-CAM tùy chọn (enable_grad) cho lớp argmax của xác suất hiệu chuẩn, đúng như analyze_image_v2. Lỗi -> cam_error, không raise."""
    x = to_tensor(res.image, rt.device)
    try:
        cam, _ = C2.gradcam_dict(rt.model, x, class_idx=int(np.argmax(res.p_cal)))
        if not np.isfinite(cam).all():
            raise ValueError('Grad-CAM chứa NaN/Inf')
        res.cam, res.cam_error = cam, None
    except (RuntimeError, ValueError, AttributeError, KeyError, TypeError) as e:
        res.cam, res.cam_error = None, f'{type(e).__name__}: {str(e)[:160]}'
    finally:
        rt.model.zero_grad(set_to_none=True)
        rt.model.eval()
    return res
