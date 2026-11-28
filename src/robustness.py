"""Robustness Lab: nhiễu/biến đổi ảnh có kiểm soát, tách LABEL-PRESERVING và COUNTERFACTUAL. Thuần numpy/cv2.

label_preserving = True : biến đổi không đổi thông tin bệnh lý (kỳ vọng nhãn giữ nguyên) — dùng đo độ ổn định.
label_preserving = False: can thiệp có thể xoá/đổi thông tin bệnh lý (ví dụ che lên tổn thương) — là phép THỬ phản thực, không đo "ổn định".
Che cục bộ ở NỀN (xa tổn thương) được xem là label-preserving; che đè lên mask dự đoán là counterfactual."""
import numpy as np
import cv2

CORRUPTIONS = {
    #  tên                : (mức 1..5 -> tham số)
    'speckle':   dict(label_preserving=True,  levels=[0.05, 0.10, 0.15, 0.20, 0.30], desc='nhiễu nhân (multiplicative) Gauss'),
    'gaussian':  dict(label_preserving=True,  levels=[3, 6, 10, 15, 25], desc='nhiễu cộng Gauss, sigma theo thang 0-255'),
    'brightness': dict(label_preserving=True, levels=[-40, -20, 20, 40, 60], desc='dịch độ sáng (cộng hằng số)'),
    'contrast':  dict(label_preserving=True,  levels=[0.5, 0.7, 0.85, 1.2, 1.5], desc='nhân tương phản quanh trung bình'),
    'blur':      dict(label_preserving=True,  levels=[0.7, 1.2, 1.8, 2.5, 3.5], desc='làm mờ Gauss, sigma px'),
    'geometric': dict(label_preserving=True,  levels=[2, 4, 6, 8, 12], desc='xoay (độ) kèm co giãn/dịch nhỏ'),
    'hflip':     dict(label_preserving=True,  levels=[1], desc='lật ngang'),
    'occlusion_background': dict(label_preserving=True, levels=[0.02, 0.04, 0.08, 0.12, 0.2], desc='che ô vuông ở NỀN (tỉ lệ diện tích)'),
    'occlusion_lesion': dict(label_preserving=False, levels=[0.25, 0.5, 0.75, 1.0], desc='COUNTERFACTUAL: che một phần mask dự đoán (tỉ lệ diện tích tổn thương)'),
}


def _affine_matrix(h, w, angle, scale, tx, ty):
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += tx * w
    M[1, 2] += ty * h
    return M


def apply_corruption(img, name, level, rng=None, pred_mask=None, valid=None):
    """Trả (img_out uint8, M 2x3 hoặc None). M != None khi biến đổi hình học (để inverse-warp mặt nạ/evidence)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    spec = CORRUPTIONS[name]
    p = spec['levels'][int(level) - 1] if name != 'hflip' else None
    x = img.astype(np.float32)
    H, W = img.shape[:2]
    M = None
    if name == 'speckle':
        x = x * (1.0 + p * rng.standard_normal(x.shape[:2])[..., None])
    elif name == 'gaussian':
        x = x + p * rng.standard_normal(x.shape)
    elif name == 'brightness':
        x = x + p
    elif name == 'contrast':
        m = x.mean()
        x = (x - m) * p + m
    elif name == 'blur':
        x = cv2.GaussianBlur(x, (0, 0), p)
    elif name == 'geometric':
        sgn = 1 if rng.random() < 0.5 else -1
        M = _affine_matrix(H, W, sgn * p, 1.0 + sgn * 0.01 * p, 0.005 * p * sgn, -0.004 * p * sgn)
        x = cv2.warpAffine(x, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    elif name == 'hflip':
        x = x[:, ::-1].copy()
        M = np.array([[-1, 0, W - 1], [0, 1, 0]], np.float64)
    elif name == 'occlusion_background':
        side = max(4, int(np.sqrt(p * H * W)))
        forb = np.zeros((H, W), bool) if pred_mask is None else (cv2.dilate((pred_mask > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        ok = False
        for _ in range(200):
            y0, x0 = int(rng.integers(0, H - side + 1)), int(rng.integers(0, W - side + 1))
            box = np.zeros((H, W), bool); box[y0:y0 + side, x0:x0 + side] = True
            if not (box & forb).any() and (valid is None or not (box & ~valid).any()):
                ok = True; break
        if not ok:
            raise ValueError(f'occlusion_background mức {level}: không đặt được ô che {side}px tránh vùng tổn thương/ngoài nội dung (giảm mức hoặc bỏ phép này cho ảnh này)')
        x[box] = float(x[valid].mean()) if valid is not None and valid.any() else float(x.mean())
    elif name == 'occlusion_lesion':
        if pred_mask is None or not (pred_mask > 0).any():
            raise ValueError('occlusion_lesion cần mặt nạ dự đoán không rỗng')
        ys, xs = np.nonzero(pred_mask > 0)
        k = max(1, int(p * len(ys)))
        order = np.argsort(xs + 1e-3 * ys)          # che phần bên trái của tổn thương trước (xác định, tái lập)
        sel = order[:k]
        fill = float(x[valid].mean()) if valid is not None and valid.any() else float(x.mean())
        x[ys[sel], xs[sel]] = fill
    else:
        raise ValueError(name)
    return np.clip(x, 0, 255).astype(np.uint8), M


def inverse_warp_map(arr, M, out_hw):
    """Đưa bản đồ (mask/evidence) của ảnh đã biến đổi về hệ toạ độ ảnh gốc."""
    if M is None:
        return arr
    Mi = cv2.invertAffineTransform(np.asarray(M, np.float64))
    return cv2.warpAffine(np.asarray(arr, np.float32), Mi, (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def scale_affine_to_grid(M, s):
    """Chuyển affine (toạ độ pixel ảnh HxW) sang lưới thấp hơn có tỉ lệ s = h/H (quy ước tâm pixel của OpenCV):
    A giữ nguyên, t' = s*t + 0.5*(1-s)*(A@1 - 1). Kiểm chứng với lật ngang: ra đúng t' = w-1."""
    M = np.asarray(M, np.float64).copy()
    A = M[:, :2]
    one = np.ones(2)
    M[:, 2] = s * M[:, 2] + 0.5 * (1 - s) * (A @ one - one)
    return M


# 5 góc nhìn xác định (không ngẫu nhiên) dùng cho đặc trưng ổn định/bất đồng. Tất cả label-preserving.
STABILITY_VIEWS = (
    dict(name='hflip', kind='hflip'),
    dict(name='aff+', kind='affine', angle=4.0, scale=1.03, tx=0.02, ty=0.0),
    dict(name='aff-', kind='affine', angle=-4.0, scale=0.97, tx=-0.02, ty=0.01),
    dict(name='contrast', kind='contrast', gain=1.10),
    dict(name='bright', kind='brightness', shift=12.0),
)


def make_view(img, spec):
    H, W = img.shape[:2]
    k = spec['kind']
    if k == 'hflip':
        return img[:, ::-1].copy(), np.array([[-1, 0, W - 1], [0, 1, 0]], np.float64)
    if k == 'affine':
        M = _affine_matrix(H, W, spec['angle'], spec['scale'], spec['tx'], spec['ty'])
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), M
    x = img.astype(np.float32)
    if k == 'contrast':
        m = x.mean(); x = (x - m) * spec['gain'] + m
    elif k == 'brightness':
        x = x + spec['shift']
    return np.clip(x, 0, 255).astype(np.uint8), None


def stability_features(img, predict_fn, T=1.0, thr=0.5, pred_mask=None, seg_thr=0.5, valid=None):
    """Chạy ảnh gốc + STABILITY_VIEWS qua mô hình; trả đặc trưng bất định/ổn định cho MỘT ảnh và đầu ra gốc.
    Trả dict(features=..., base=dict(logits, seg_prob, evidence))."""
    from .uncertainty import softmax, js_disagreement, margin, view_stats
    from .segmentation import dice, postprocess_mask
    from .evidence_maps import cosine
    views = [make_view(img, s) for s in STABILITY_VIEWS]
    batch = np.stack([img] + [v[0] for v in views], 0)
    out = predict_fn(batch)
    L = np.asarray(out['logits'], np.float64)
    P = softmax(L, T)                                # (S+1, 2)
    z = margin(L, T)
    zthr = float(np.log(thr / (1 - thr)))
    vs = view_stats(z[:, None], zthr)
    feats = dict(view_js=float(js_disagreement(P[:, None, :])[0]), view_pred_flip=float(vs['view_pred_flip'][0]),
                 view_dz_sd=float(vs['view_dz_sd'][0]), view_dz_absmax=float(vs['view_dz_absmax'][0]))
    feats['prediction_stability'] = 1.0 - feats['view_pred_flip']
    nanf = float('nan')
    feats['mask_consistency'] = nanf
    feats['evidence_stability'] = nanf
    seg = out.get('seg_prob')
    if seg is not None:
        m0 = postprocess_mask(seg[0], seg_thr)[0]
        ds = []
        for i, (_, M) in enumerate(views, start=1):
            mi = postprocess_mask(inverse_warp_map(seg[i], M, seg[i].shape), seg_thr)[0]
            if M is not None:                       # chỉ so trong vùng giao hợp lệ của phép warp
                vm = inverse_warp_map(np.ones_like(seg[i], np.float32), M, seg[i].shape) > 0.99
                a, b = (m0 > 0) & vm, (mi > 0) & vm
                ds.append(dice(a, b))
            else:
                ds.append(dice(m0, mi))
        feats['mask_consistency'] = float(np.mean(ds))
    ev = out.get('evidence')
    if ev is not None:
        cs = []
        for i, (_, M) in enumerate(views, start=1):
            e0 = np.asarray(ev[0], np.float32)
            ei = ev[i]
            if M is not None:                       # evidence ở lưới thấp: warp tương ứng bằng M co theo tỉ lệ
                ei = inverse_warp_map(ev[i], scale_affine_to_grid(M, ev[i].shape[0] / img.shape[0]), ev[i].shape)
            cs.append(cosine(e0, ei))
        feats['evidence_stability'] = float(np.mean(cs))
    return dict(features=feats, base=dict(logits=L[0], seg_prob=None if seg is None else seg[0], evidence=None if ev is None else ev[0]), probs=P, z=z)


def run_robustness_lab(images, predict_fn, corruptions=None, levels=(1, 3, 5), seed=0, T=1.0, thr=0.5, seg_thr=0.5, min_area_frac=0.0015, risk_fn=None):
    """Chạy các biến đổi có kiểm soát trên danh sách ảnh đã tiền xử lý. Mỗi dòng: độ ổn định phân loại (|Δp|, flip), phân vùng (Dice với mask sạch,
    sau inverse-warp nếu hình học), evidence (cosine), risk (Δ nếu có risk_fn: img -> float). `label_preserving` tách rõ biến đổi bảo toàn nhãn
    khỏi can thiệp phản thực (occlusion_lesion). Ca không thực hiện được (không đặt được ô che, mask rỗng) được GHI `status` chứ không bị bỏ."""
    import pandas as pd
    from .uncertainty import softmax, margin
    from .segmentation import postprocess_mask, dice
    from .evidence_maps import cosine
    names = list(corruptions or CORRUPTIONS)
    rows = []
    for i, img in enumerate(images):
        valid = (img.max(2) != 0)
        o = predict_fn(img[None])
        p0 = float(softmax(o['logits'], T)[0, 1]); lab0 = p0 >= thr
        m0 = postprocess_mask(o['seg_prob'][0], seg_thr, min_area_frac)[0] if o.get('seg_prob') is not None else None
        e0 = None if o.get('evidence') is None else np.asarray(o['evidence'][0], np.float32)
        r0 = None if risk_fn is None else float(risk_fn(img))
        for name in names:
            for lv in levels:
                if lv > len(CORRUPTIONS[name]['levels']):
                    continue
                row = dict(image=i, corruption=name, level=int(lv), label_preserving=CORRUPTIONS[name]['label_preserving'], p0=p0, status='ok')
                rng = np.random.default_rng([seed, i, lv, __import__('zlib').crc32(name.encode()) % 10007])
                try:
                    out, M = apply_corruption(img, name, lv, rng, pred_mask=m0, valid=valid)
                except ValueError as e:
                    row['status'] = f'skipped: {e}'[:120]; rows.append(row); continue
                o1 = predict_fn(out[None])
                p1 = float(softmax(o1['logits'], T)[0, 1])
                row.update(p1=p1, abs_dp=abs(p1 - p0), flip=int((p1 >= thr) != lab0), mean_abs_pixel_change=float(np.abs(out.astype(np.float32) - img).mean()))
                if m0 is not None and o1.get('seg_prob') is not None:
                    sp = inverse_warp_map(o1['seg_prob'][0], M, o1['seg_prob'][0].shape)
                    m1 = postprocess_mask(sp, seg_thr, min_area_frac)[0]
                    if M is not None:
                        vm = inverse_warp_map(np.ones_like(sp, np.float32), M, sp.shape) > 0.99
                        row['seg_dice'] = dice((m0 > 0) & vm, (m1 > 0) & vm)
                    else:
                        row['seg_dice'] = dice(m0, m1)
                if e0 is not None and o1.get('evidence') is not None:
                    e1 = o1['evidence'][0]
                    if M is not None:
                        e1 = inverse_warp_map(e1, scale_affine_to_grid(M, e1.shape[0] / img.shape[0]), e1.shape)
                    row['evidence_cosine'] = cosine(e0, e1)
                if r0 is not None:
                    r1 = float(risk_fn(out)); row['risk0'], row['risk1'], row['abs_drisk'] = r0, r1, abs(r1 - r0)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_robustness(df):
    """Tóm tắt theo (corruption, level): flip rate, |Δp| trung bình, Dice, evidence cosine, |Δrisk|; tách label_preserving."""
    d = df[df['status'] == 'ok']
    agg = {c: 'mean' for c in ('abs_dp', 'flip', 'seg_dice', 'evidence_cosine', 'abs_drisk', 'mean_abs_pixel_change') if c in d}
    g = d.groupby(['label_preserving', 'corruption', 'level']).agg(n=('image', 'count'), **{k: (k, v) for k, v in agg.items()}).reset_index()
    sk = df[df['status'] != 'ok'].groupby(['corruption', 'level']).size().rename('n_skipped').reset_index()
    return g.merge(sk, on=['corruption', 'level'], how='left').fillna({'n_skipped': 0})
