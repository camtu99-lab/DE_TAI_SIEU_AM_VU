"""Counterfactual Evidence Verification Engine (BUS-VERIFY).

Ý tưởng: với một ảnh và mặt nạ phân vùng DỰ ĐOÁN, ta can thiệp có kiểm soát lên ba vùng (lesion / perilesional / background)
và đo mô hình phản ứng ra sao. Mô hình "dựa vào bằng chứng đúng chỗ" khi logit nhạy với can thiệp lên tổn thương nhiều hơn
so với can thiệp lên vùng ĐỐI CHỨNG cùng diện tích, cùng loại phép biến đổi, cường độ trung bình gần nhau, độ sâu gần nhau.

Ràng buộc khoa học (được kiểm thử):
  * Không dùng ground-truth mask ở suy luận: hàm công khai chỉ nhận `pred_mask`; không có tham số GT.
  * Phép biến đổi gây dịch chuyển phân phối (OOD): mỗi can thiệp ghi bản QA (diện tích, thay đổi pixel, 'đường nối', kết cấu) để phân tích
    xem độ nhạy là do bằng chứng hay do artefact; đối chứng matched-area/random là thước đo nền của artefact.
  * Thuần numpy/cv2. Mô hình được truyền vào dưới dạng `predict_fn(images uint8 (N,H,W,3)) -> dict(logits=(N,2), [seg_prob], [evidence])`.
"""
from dataclasses import dataclass, field, asdict
import numpy as np
import cv2

from .segmentation import build_regions, Regions

OPS = ('blur', 'replace', 'permute')
EPS = 1e-8


@dataclass
class CFConfig:
    ring_px: int = 12                  # độ rộng vùng quanh tổn thương (ở ảnh 256 px)
    blur_sigma: float = 6.0
    feather_px: float = 0.0            # 0 = biên cứng (mặc định, để diện tích kiểm soát chính xác)
    n_matched: int = 3                 # số bản đối chứng matched-area cho mỗi (vùng, phép biến đổi)
    n_random: int = 3                  # số đối chứng ngẫu nhiên (vị trí và diện tích ngẫu nhiên quanh diện tích tổn thương)
    n_candidates: int = 40             # số ứng viên vị trí khi khớp cường độ
    control_depth_tol: float = 0.10    # |dy| tối đa (tỉ lệ chiều cao) cho đối chứng 'cùng độ sâu'; None = bất kỳ
    min_region_px: int = 64
    ops: tuple = OPS
    ev_q: float = 0.20                 # tỉ lệ diện tích 'evidence cao nhất' bị can thiệp
    z_floor: float = 0.05              # sàn độ lệch chuẩn khi tính z-score (đơn vị logit)
    seed: int = 0


# ------------------------------------------------------------------ biến đổi vùng
def _alpha(region, feather):
    a = region.astype(np.float32)
    if feather and feather > 0:
        a = np.maximum(cv2.GaussianBlur(a, (0, 0), float(feather)), a)
    return a[..., None]


def apply_op(img, region, op, rng, ref=None, cfg=None):
    """Áp dụng một phép biến đổi lên `region` (bool HxW). Trả ảnh uint8.
    blur   : làm mờ Gauss (loại tần số cao, giữ cường độ trung bình).
    replace: thay bằng cường độ lấy mẫu có hoàn lại từ vùng tham chiếu `ref` (phá cấu trúc, cường độ khớp vùng tham chiếu).
    permute: hoán vị pixel trong vùng (giữ NGUYÊN histogram của vùng, phá cấu trúc không gian)."""
    cfg = cfg or CFConfig()
    src = img.astype(np.float32)
    out = src.copy()
    idx = np.where(region)
    n = len(idx[0])
    if n == 0:
        return img.copy()
    if op == 'blur':
        pert = cv2.GaussianBlur(src, (0, 0), cfg.blur_sigma)
        vals = pert[idx]
    elif op == 'replace':
        if ref is None or not np.asarray(ref).any():
            ref = ~region
        if not np.asarray(ref).any():
            return img.copy()
        ridx = np.where(ref)
        pick = rng.integers(0, len(ridx[0]), size=n)
        g = src[ridx[0][pick], ridx[1][pick]]                       # (n,3) giữ nguyên bộ ba kênh của pixel tham chiếu
        vals = g
    elif op == 'permute':
        perm = rng.permutation(n)
        vals = src[idx][perm]
    else:
        raise ValueError(f'op không hợp lệ: {op}')
    if cfg.feather_px and cfg.feather_px > 0 and op == 'blur':
        pert_full = cv2.GaussianBlur(src, (0, 0), cfg.blur_sigma)
        a = _alpha(region, cfg.feather_px)
        out = a * pert_full + (1 - a) * src
    else:
        out[idx] = vals
    return np.clip(out, 0, 255).astype(np.uint8)


def perturbation_qa(img, pert, region):
    """Thống kê QA của một can thiệp: diện tích, thay đổi pixel, 'đường nối' (gradient dọc biên), kết cấu trong vùng."""
    a, b = img.astype(np.float32).mean(2), pert.astype(np.float32).mean(2)
    reg = region
    area = float(reg.mean())
    chg = float(np.abs(b - a)[reg].mean()) if reg.any() else 0.0
    ring = cv2.dilate(reg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    band = ring & ~(cv2.erode(reg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)

    def gmag(x):
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx * gx + gy * gy)
    ga, gb = gmag(a), gmag(b)
    seam = float(gb[band].mean() / (ga[band].mean() + EPS)) if band.any() else 1.0
    tex = float(b[reg].std() / (a[reg].std() + EPS)) if reg.sum() > 1 else 1.0
    return dict(area_frac=area, mean_abs_change=chg, seam_ratio=seam, texture_ratio=tex)


# ------------------------------------------------------------------ vùng đối chứng
def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return ys.min(), ys.max(), xs.min(), xs.max()


def _shift(mask, dy, dx):
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(mask)
    out[ys + dy, xs + dx] = True
    return out


def _disk_mask(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def matched_area_controls(region, forbidden, valid, gray, rng, k=3, n_candidates=40, depth_tol=0.10):
    """Tìm k vùng đối chứng: CÙNG HÌNH DẠNG và CÙNG DIỆN TÍCH (dịch tịnh tiến nguyên), nằm hoàn toàn trong `valid`, không giao `forbidden`,
    ưu tiên cường độ trung bình gần vùng gốc nhất và (nếu depth_tol) cùng độ sâu. Trả (list mask, list thông tin)."""
    H, W = region.shape
    y0, y1, x0, x1 = _bbox(region)
    target = float(gray[region].mean())
    cands = []
    tries = 0
    while len(cands) < n_candidates and tries < n_candidates * 25:
        tries += 1
        dx = int(rng.integers(-x0, W - x1))
        if depth_tol is None:
            dy = int(rng.integers(-y0, H - y1))
        else:
            lo, hi = max(-y0, -int(depth_tol * H)), min(H - 1 - y1, int(depth_tol * H))
            if lo > hi:
                continue
            dy = int(rng.integers(lo, hi + 1))
        if abs(dx) + abs(dy) < 3:
            continue
        m = _shift(region, dy, dx)
        if (m & forbidden).any() or (m & ~valid).any():
            continue
        cands.append((abs(float(gray[m].mean()) - target), dy, dx, m))
    cands.sort(key=lambda t: t[0])
    chosen, info, used = [], [], []
    for gap, dy, dx, m in cands:
        if any((m & u).any() for u in used):
            continue
        chosen.append(m); used.append(m)
        info.append(dict(kind='matched', dy=dy, dx=dx, intensity_gap=gap, area_px=int(m.sum())))
        if len(chosen) >= k:
            break
    return chosen, info


def random_controls(area_px, forbidden, valid, gray, rng, k=3, area_jitter=(0.5, 1.5)):
    """Đối chứng ngẫu nhiên: đĩa ở vị trí ngẫu nhiên (bất kỳ độ sâu) với diện tích ~ U(0.5,1.5)×diện tích tổn thương."""
    H, W = valid.shape
    out, info = [], []
    tries = 0
    while len(out) < k and tries < 400:
        tries += 1
        a = area_px * float(rng.uniform(*area_jitter))
        r = max(3.0, float(np.sqrt(a / np.pi)))
        cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        m = _disk_mask((H, W), cy, cx, r)
        if m.sum() < 16 or (m & forbidden).any() or (m & ~valid).any():
            continue
        out.append(m)
        info.append(dict(kind='random', cy=cy, cx=cx, area_px=int(m.sum()), intensity_gap=abs(float(gray[m].mean()) - float(gray[valid].mean()))))
    return out, info


# ------------------------------------------------------------------ lõi
def _z(logits, T):
    l = np.asarray(logits, np.float64)
    return (l[:, 1] - l[:, 0]) / float(T)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def evidence_top_mask(evidence, valid, q, out_hw):
    """Mặt nạ q% diện tích có bằng chứng cao nhất (evidence của CHÍNH mô hình, không dùng GT)."""
    E = cv2.resize(np.asarray(evidence, np.float32), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    vals = E[valid]
    if vals.size == 0:
        return np.zeros(out_hw, bool)
    k = max(1, int(round(q * valid.sum())))
    ys, xs = np.nonzero(valid)
    order = np.argsort(-E[ys, xs], kind='stable')[:k]            # chọn ĐÚNG k pixel (tie xác định theo vị trí): diện tích kiểm soát chính xác
    out = np.zeros(out_hw, bool)
    out[ys[order], xs[order]] = True
    return out


def verify_image(img, pred_mask, predict_fn, T=1.0, thr=0.5, cfg=None, valid=None, evidence=None, rng=None, return_records=False):
    """Chạy toàn bộ kiểm chứng phản thực cho MỘT ảnh. Chỉ nhận mặt nạ dự đoán.
    Trả dict đặc trưng (float; NaN nếu không áp dụng được) và `status` ('ok'|'empty_mask'|...)."""
    cfg = cfg or CFConfig()
    rng = rng if rng is not None else np.random.default_rng(cfg.seed)
    H, W = img.shape[:2]
    reg = build_regions(pred_mask, valid, cfg.ring_px)
    nan = float('nan')
    keys = []
    for r in ('lesion', 'peri', 'bg'):
        keys += [f'{r}_sens', f'{r}_dir', f'{r}_dp', f'{r}_flip', f'{r}_consistency']
    keys += ['matched_les_sens', 'matched_peri_sens', 'random_sens', 'cf_specificity', 'cf_ratio', 'cf_z', 'lesion_bg_contrast',
             'lesion_peri_contrast', 'cf_flip_rate', 'cf_consistency', 'ev_drop_top', 'ev_drop_random', 'ev_faithfulness', 'n_controls_ok',
             'mean_control_intensity_gap']
    feats = {k: nan for k in keys}
    if not reg.ok:
        return dict(status=reg.reason, features=feats, records=[])
    gray = img.astype(np.float32).mean(2)
    valid_m = reg.valid
    forbidden = (cv2.dilate(reg.lesion.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.ring_px + 1,) * 2)) > 0)
    ref_les = reg.peri if reg.peri.sum() >= 16 else reg.background
    refs = {'lesion': ref_les, 'peri': reg.background if reg.background.any() else ref_les, 'bg': reg.peri if reg.peri.any() else reg.lesion}
    regions = {'lesion': reg.lesion, 'peri': reg.peri, 'bg': reg.background}

    items = []                                 # (group, region_name, op, kind, mask)
    batch = [img]
    qa = []
    for rn, m in regions.items():
        if m.sum() < cfg.min_region_px:
            continue
        for op in cfg.ops:
            p = apply_op(img, m, op, rng, ref=refs[rn], cfg=cfg)
            items.append(('region', rn, op, 'target', m)); batch.append(p)
            qa.append(dict(group='region', region=rn, op=op, kind='target', **perturbation_qa(img, p, m)))
    n_ctrl = 0
    ctrl_gaps = []
    for rn in ('lesion', 'peri'):
        m = regions[rn]
        if m.sum() < cfg.min_region_px:
            continue
        ctrls, info = matched_area_controls(m, forbidden, valid_m, gray, rng, cfg.n_matched, cfg.n_candidates, cfg.control_depth_tol)
        if len(ctrls) < cfg.n_matched:       # nới ràng buộc độ sâu (ghi nhận trong n_controls_ok)
            more, info2 = matched_area_controls(m, forbidden, valid_m, gray, rng, cfg.n_matched - len(ctrls), cfg.n_candidates, None)
            ctrls += more; info += info2
        for cm, ci in zip(ctrls, info):
            ctrl_gaps.append(ci['intensity_gap']); n_ctrl += 1
            for op in cfg.ops:
                p = apply_op(img, cm, op, rng, ref=refs[rn], cfg=cfg)
                items.append(('matched', rn, op, 'matched', cm)); batch.append(p)
                qa.append(dict(group='matched', region=rn, op=op, kind='matched', **perturbation_qa(img, p, cm), **{k: v for k, v in ci.items() if k in ('dy', 'dx', 'intensity_gap')}))
    rc, rinfo = random_controls(int(reg.lesion.sum()), forbidden, valid_m, gray, rng, cfg.n_random)
    for cm, ci in zip(rc, rinfo):
        for op in cfg.ops:
            p = apply_op(img, cm, op, rng, ref=refs['lesion'], cfg=cfg)
            items.append(('random', 'lesion', op, 'random', cm)); batch.append(p)
            qa.append(dict(group='random', region='lesion', op=op, kind='random', **perturbation_qa(img, p, cm)))
    # can thiệp theo evidence của chính mô hình
    ev_items = []
    if evidence is not None:
        top = evidence_top_mask(evidence, valid_m, cfg.ev_q, (H, W))
        if top.sum() >= cfg.min_region_px:
            p = apply_op(img, top, 'replace', rng, ref=~top & valid_m, cfg=cfg)
            ev_items.append(('ev_top', len(batch))); batch.append(p)
            rr, _ = random_controls(int(top.sum()), top, valid_m, gray, rng, cfg.n_random, area_jitter=(1.0, 1.0))
            for cm in rr:
                p = apply_op(img, cm, 'replace', rng, ref=~cm & valid_m, cfg=cfg)
                ev_items.append(('ev_rand', len(batch))); batch.append(p)

    out = predict_fn(np.stack(batch, 0))
    z_all = _z(out['logits'], T)
    z0 = float(z_all[0]); p0 = float(_sig(z0)); y0 = int(p0 >= thr); s0 = 1.0 if z0 >= 0 else -1.0
    zp = z_all[1:1 + len(items)]
    recs = []
    for it, z in zip(items, zp):
        dz = float(z - z0)
        recs.append(dict(group=it[0], region=it[1], op=it[2], dz=dz, absdz=abs(dz), dir=s0 * dz, dp=float(_sig(z) - p0),
                         flip=int((_sig(z) >= thr) != bool(y0)), area_px=int(it[4].sum())))
    for r, q in zip(recs, qa):
        r.update({k: v for k, v in q.items() if k in ('mean_abs_change', 'seam_ratio', 'texture_ratio', 'intensity_gap')})

    def agg(group, region, field_, fn=np.mean):
        v = [r[field_] for r in recs if r['group'] == group and r['region'] == region]
        return float(fn(v)) if v else nan

    for rn in ('lesion', 'peri', 'bg'):
        v = [r for r in recs if r['group'] == 'region' and r['region'] == rn]
        if not v:
            continue
        feats[f'{rn}_sens'] = float(np.mean([r['absdz'] for r in v]))
        feats[f'{rn}_dir'] = float(np.mean([r['dir'] for r in v]))
        feats[f'{rn}_dp'] = float(np.mean([abs(r['dp']) for r in v]))
        feats[f'{rn}_flip'] = float(np.mean([r['flip'] for r in v]))
        sg = np.sign([r['dz'] for r in v])
        feats[f'{rn}_consistency'] = float(abs(sg.mean()))
    feats['matched_les_sens'] = agg('matched', 'lesion', 'absdz')
    feats['matched_peri_sens'] = agg('matched', 'peri', 'absdz')
    feats['random_sens'] = agg('random', 'lesion', 'absdz')
    ctrl_vals = np.array([r['absdz'] for r in recs if r['group'] == 'matched' and r['region'] == 'lesion'], float)
    if np.isfinite(feats['lesion_sens']) and ctrl_vals.size:
        mu, sd = float(ctrl_vals.mean()), float(ctrl_vals.std())
        feats['cf_specificity'] = feats['lesion_sens'] - mu
        feats['cf_ratio'] = feats['lesion_sens'] / (mu + 1e-2)
        feats['cf_z'] = (feats['lesion_sens'] - mu) / max(sd, cfg.z_floor)
    if np.isfinite(feats['lesion_sens']) and np.isfinite(feats['bg_sens']):
        feats['lesion_bg_contrast'] = feats['lesion_sens'] - feats['bg_sens']
    if np.isfinite(feats['lesion_sens']) and np.isfinite(feats['peri_sens']):
        feats['lesion_peri_contrast'] = feats['lesion_sens'] - feats['peri_sens']
    tgt = [r for r in recs if r['group'] == 'region']
    if tgt:
        feats['cf_flip_rate'] = float(np.mean([r['flip'] for r in tgt]))
        feats['cf_consistency'] = float(abs(np.sign([r['dz'] for r in tgt]).mean()))
    if ev_items:
        zt = [z_all[i] for k, i in ev_items if k == 'ev_top']
        zr = [z_all[i] for k, i in ev_items if k == 'ev_rand']
        if zt:
            feats['ev_drop_top'] = float(s0 * (z0 - zt[0]))             # >0: bỏ evidence cao làm GIẢM độ tin vào lớp dự đoán
        if zr:
            feats['ev_drop_random'] = float(np.mean([s0 * (z0 - z) for z in zr]))
        if zt and zr:
            feats['ev_faithfulness'] = feats['ev_drop_top'] - feats['ev_drop_random']
    feats['n_controls_ok'] = float(n_ctrl)
    feats['mean_control_intensity_gap'] = float(np.mean(ctrl_gaps)) if ctrl_gaps else nan
    res = dict(status='ok', features=feats, z0=z0, p0=p0, pred=y0)
    if return_records:
        res['records'] = recs
    return res


# ------------------------------------------------------------------ phép thử shortcut (thăm dò, NGOÀI P1 chính)
def annotation_mask(img, valid=None, bright=235, max_comp_px=200):
    """Phát hiện chữ/dấu đo (caliper) kiểu BUS-BRA: thành phần sáng nhỏ trong vùng hợp lệ. Heuristic, KHÔNG phải nhãn chuẩn."""
    g = img.max(2)
    m = ((g >= bright) & (valid if valid is not None else True)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] <= max_comp_px:
            out[lab == i] = 1
    return cv2.dilate(out, np.ones((5, 5), np.uint8)) > 0


def shortcut_probe(img, valid, predict_fn, T=1.0, thr=0.5):
    """Độ nhạy của logit khi (a) lấp dải letterbox bằng phản chiếu nội dung, (b) xoá chữ/dấu đo bằng inpainting.
    Phản ánh nghi vấn Grad-CAM của V3 dồn vào dải đen/mép ảnh. Trả {pad_sens, annot_sens, annot_frac}."""
    batch, kinds = [img], []
    pad = ~valid
    if pad.any():
        ys, xs = np.nonzero(valid)
        crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        top, left = ys.min(), xs.min()
        bottom, right = img.shape[0] - ys.max() - 1, img.shape[1] - xs.max() - 1
        filled = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        if filled.shape[:2] == img.shape[:2]:
            batch.append(filled); kinds.append('pad')
    am = annotation_mask(img, valid)
    if am.any():
        batch.append(cv2.inpaint(img, am.astype(np.uint8), 3, cv2.INPAINT_TELEA)); kinds.append('annot')
    res = dict(pad_sens=float('nan'), annot_sens=float('nan'), annot_frac=float(am.mean()), pad_frac=float(pad.mean()))
    if len(batch) == 1:
        return res
    z = _z(predict_fn(np.stack(batch, 0))['logits'], T)
    for k, zz in zip(kinds, z[1:]):
        res[f'{k}_sens'] = float(abs(zz - z[0]))
    return res
