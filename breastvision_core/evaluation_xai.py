# evaluation_xai.py - đánh giá phân loại/phân vùng, XAI (định vị, trung thực, nhất quán), hiệu chuẩn, chọn lọc dự đoán, thống kê, khóa pipeline
import os, json, time, hashlib
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage

import bus_trust as T
import losses_xai as LX
import bus_core as core
from data_audit import sha256_file


# ============================ Phân loại (lớp 1 = malignant) ============================
def classification_metrics(y, p_mal, threshold=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score
    y, p_mal = np.asarray(y), np.asarray(p_mal, float)
    pred = (p_mal >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    two = len(np.unique(y)) == 2
    return dict(auroc=float(roc_auc_score(y, p_mal)) if two else float('nan'), auprc=float(average_precision_score(y, p_mal)) if two else float('nan'),
                sensitivity_malignant=float(sens), specificity=float(spec), balanced_accuracy=float(np.nanmean([sens, spec])),
                f1=float(f1_score(y, pred, zero_division=0)), accuracy=float((pred == y).mean()), threshold=float(threshold))


def select_threshold_youden(y, p_mal):
    # CHỈ gọi trên validation của BUS-BRA
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p_mal)
    return float(thr[int(np.argmax(tpr - fpr))])


# ============================ Phân vùng ============================
def hd95(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float('nan')
    ps = pred ^ ndimage.binary_erosion(pred); gs = gt ^ ndimage.binary_erosion(gt)
    d_to_gt, d_to_pred = ndimage.distance_transform_edt(~gt), ndimage.distance_transform_edt(~pred)
    return float(np.percentile(np.concatenate([d_to_gt[ps], d_to_pred[gs]]), 95))


def seg_metrics(seg_prob, gt, thr=0.5, with_hd95=True):
    pb, gb = seg_prob.astype(np.float32) >= thr, gt > 0
    inter = (pb & gb).sum((1, 2)).astype(float); ps, gs = pb.sum((1, 2)).astype(float), gb.sum((1, 2)).astype(float)
    has = gs > 0
    out = dict(dice=2 * inter / (ps + gs + 1e-7), iou=inter / (ps + gs - inter + 1e-7), has_lesion=has)
    out['hd95'] = np.array([hd95(p, g) for p, g in zip(pb, gb)]) if with_hd95 else None
    hd = out['hd95'][has] if with_hd95 else np.array([])
    ok_hd = np.isfinite(hd)
    out['summary'] = dict(dice=float(out['dice'][has].mean()), iou=float(out['iou'][has].mean()),
                          hd95=float(hd[ok_hd].mean()) if ok_hd.any() else float('nan'),
                          n_hd95_undefined=int((~ok_hd).sum()) if with_hd95 else 0)          # HD95 không xác định khi mask dự đoán rỗng
    return out


# ============================ XAI: định vị ============================
def upsample_evidence(E, size):
    E = torch.as_tensor(E, dtype=torch.float32)
    if E.ndim == 3:
        E = E[:, None]
    U = F.interpolate(E, size=size, mode='bilinear', align_corners=False).clamp(min=0)
    return (U / (U.sum((1, 2, 3), keepdim=True) + 1e-12))[:, 0].numpy()


def localisation_report(E, gt, ring_px=8, min_area=1e-3, max_area=0.9):
    # Relevance Mass (trong tổn thương và trong tổn thương+vòng), Pointing Game, area-adjusted attribution.
    # Đây là đo ĐỊNH VỊ so với mask; KHÔNG chứng minh tính trung thực (faithfulness).
    U = upsample_evidence(E, gt.shape[-2:])
    rm, rmc, pg, aaa, aaac = [], [], [], [], []
    k = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    for u, g in zip(U, gt):
        g = (g > 0).astype(np.uint8)
        a = g.mean()
        if not (min_area < a < max_area):
            continue
        roi = cv2.dilate(g, k)
        mass_l, mass_r, a_r = float((u * g).sum()), float((u * roi).sum()), float(roi.mean())
        rm.append(mass_l); rmc.append(mass_r)
        aaa.append((mass_l - a) / (1 - a)); aaac.append((mass_r - a_r) / (1 - a_r) if a_r < 1 else float('nan'))
        pg.append(float(roi[np.unravel_index(np.argmax(u), u.shape)] > 0))
    f = lambda v: float(np.nanmean(v)) if len(v) else float('nan')
    return dict(n=len(rm), relevance_mass_lesion=f(rm), relevance_mass_lesion_context=f(rmc), pointing_game=f(pg),
                area_adjusted_lesion=f(aaa), area_adjusted_lesion_context=f(aaac))


# ============================ XAI: trung thực (faithfulness) ============================
def _auc(q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    return float(np.sum((q[1:] - q[:-1]) * (v[1:] + v[:-1]) / 2))


@torch.no_grad()
def _prob_of(model, x, cls):
    return float(F.softmax(model(x)['logits'].float(), 1)[0, cls])


@torch.no_grad()
def deletion_insertion_curves(model, x, rank_map, cls, steps=10):
    # rank_map (h,w): điểm quan trọng; xoá/chèn dần các ô quan trọng nhất; baseline = 0 (ảnh trung bình sau chuẩn hóa)
    h, w = rank_map.shape
    order = torch.as_tensor(rank_map).flatten().argsort(descending=True)
    base = torch.zeros_like(x)
    dele, ins = [], []
    for s in range(steps + 1):
        k = int(round(h * w * s / steps))
        cm = torch.zeros(h * w); cm[order[:k]] = 1
        pm = F.interpolate(cm.view(1, 1, h, w).to(x.device), size=x.shape[-2:], mode='nearest')
        dele.append(_prob_of(model, x * (1 - pm) + base * pm, cls))
        ins.append(_prob_of(model, base * (1 - pm) + x * pm, cls))
    q = np.linspace(0, 1, steps + 1)
    return q, np.array(dele), np.array(ins)


def faithfulness_report(model, X, indices, device, steps=8, max_images=40, seed=0, use_cam=True):
    # So sánh xếp hạng theo evidence / Grad-CAM với xếp hạng NGẪU NHIÊN. Deletion AUC thấp hơn và Insertion AUC cao hơn ngẫu nhiên
    # là bằng chứng (chưa phải chứng minh) rằng vùng được xếp hạng ảnh hưởng thật tới dự đoán.
    from bus_core_v2 import gradcam_dict
    rng = np.random.RandomState(seed)
    model.eval()
    rows = []
    for k in indices[:max_images]:
        x = core.normalize_to_tensor(X[k]).unsqueeze(0).to(device)
        with torch.no_grad():
            o = model(x)
        cls = int(o['logits'].argmax(1))
        h, w = (o['evidence'].shape[-2:] if o.get('evidence') is not None else (16, 16))
        maps = {'random': rng.rand(h, w)}
        if o.get('evidence') is not None:
            maps['evidence'] = o['evidence'][0, 0].float().cpu().numpy()
        if use_cam:
            cam, _ = gradcam_dict(model, x, class_idx=cls)
            maps['gradcam'] = cv2.resize(cam, (w, h), interpolation=cv2.INTER_AREA)
        row = {}
        for nm, mp in maps.items():
            q, d, i = deletion_insertion_curves(model, x, mp, cls, steps)
            row[f'deletion_auc_{nm}'], row[f'insertion_auc_{nm}'] = _auc(q, d), _auc(q, i)
        rows.append(row)
    keys = rows[0].keys() if rows else []
    return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {'n': len(rows)}


@torch.no_grad()
def evidence_intervention(model, X, indices, device, q=0.2, max_images=40, seed=0):
    # Can thiệp trực tiếp vào evidence: loại bỏ q% ô có evidence lớn nhất (ev_mask) và đo giảm xác suất lớp dự đoán, so với loại ô ngẫu nhiên cùng số lượng.
    if getattr(getattr(model, 'cfg', None), 'evidence_mode', 'none') not in ('ewp', 'egr'):
        return dict(applicable=False)
    g = torch.Generator().manual_seed(seed)
    model.eval()
    d_top, d_rand = [], []
    for k in indices[:max_images]:
        x = core.normalize_to_tensor(X[k]).unsqueeze(0).to(device)
        o = model(x)
        cls = int(o['logits'].argmax(1)); p0 = float(F.softmax(o['logits'].float(), 1)[0, cls])
        E = o['evidence']; n = E.numel(); kk = max(1, int(q * n))
        top = E.flatten().topk(kk).indices.cpu()
        rnd = torch.randperm(n, generator=g)[:kk]
        for idxs, store in ((top, d_top), (rnd, d_rand)):
            mk = torch.ones(n); mk[idxs] = 0
            p1 = float(F.softmax(model(x, ev_mask=mk.view_as(E).to(device))['logits'].float(), 1)[0, cls])
            store.append(p0 - p1)
    return dict(applicable=True, q=q, n=len(d_top), mean_drop_top_evidence=float(np.mean(d_top)), mean_drop_random=float(np.mean(d_rand)))


# ============================ Phát hiện evidence collapse và kiểm tra nhiễu nền ============================
def evidence_collapse_report(E, uniform_entropy=0.98, ess_uniform=0.9, ess_peaked_cells=2.0):
    E = torch.as_tensor(E, dtype=torch.float32)
    if E.ndim == 4:
        E = E[:, 0]
    N, h, w = E.shape
    P = E.flatten(1).clamp(min=0)
    P = P / (P.sum(1, keepdim=True) + 1e-12)
    H = -(P * torch.log(P + 1e-12)).sum(1) / np.log(h * w)           # entropy chuẩn hóa: 1 = đồng đều
    ess = 1.0 / ((P ** 2).sum(1) + 1e-12)                            # số ô hiệu dụng
    uniform = (H > uniform_entropy) & (ess / (h * w) > ess_uniform)
    peaked = ess < ess_peaked_cells
    return dict(n=N, mean_norm_entropy=float(H.mean()), mean_ess_fraction=float((ess / (h * w)).mean()), uniform_collapse_rate=float(uniform.float().mean()),
                peaked_collapse_rate=float(peaked.float().mean()), collapse_flag=bool(uniform.float().mean() > 0.5 or peaked.float().mean() > 0.5))


@torch.no_grad()
def background_perturbation(model, X, M, indices, device, ring_px=8, max_images=30, seed=0):
    # Xáo trộn pixel NGOÀI vùng (tổn thương + vòng) -> kỳ vọng dự đoán/evidence ít đổi nếu mô hình dựa vào tổn thương.
    # Xáo trộn pixel TRONG tổn thương (đối chứng) -> kỳ vọng thay đổi lớn hơn. Đây là kiểm tra mô tả, không phải chứng minh.
    rng = np.random.RandomState(seed)
    kern = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
    model.eval()
    d_bg, d_les, mass_b, mass_a = [], [], [], []
    for k in indices[:max_images]:
        g = (M[k] > 0).astype(np.uint8)
        if not (0.001 < g.mean() < 0.9):
            continue
        roi = cv2.dilate(g, kern).astype(bool)
        x = core.normalize_to_tensor(X[k]).unsqueeze(0).to(device)
        o = model(x)
        cls = int(o['logits'].argmax(1)); p0 = float(F.softmax(o['logits'].float(), 1)[0, cls])

        def shuffled(region):
            xs = x.clone()
            ii = np.flatnonzero(region.ravel())
            perm = rng.permutation(ii)
            flat = x.view(1, 3, -1)
            xs.view(1, 3, -1)[:, :, torch.as_tensor(ii)] = flat[:, :, torch.as_tensor(perm)]
            return xs
        ob = model(shuffled(~roi)); ol = model(shuffled(g.astype(bool)))
        d_bg.append(abs(p0 - float(F.softmax(ob['logits'].float(), 1)[0, cls]))); d_les.append(abs(p0 - float(F.softmax(ol['logits'].float(), 1)[0, cls])))
        if o.get('evidence') is not None:
            mk = F.interpolate(torch.as_tensor(roi[None, None].astype(np.float32)), size=o['evidence'].shape[-2:], mode='nearest').to(device)
            En = LX.normalize_evidence(o['evidence'].float()); Eb = LX.normalize_evidence(ob['evidence'].float())
            mass_b.append(float((En * mk).sum())); mass_a.append(float((Eb * mk).sum()))
    f = lambda v: float(np.mean(v)) if len(v) else float('nan')
    return dict(n=len(d_bg), mean_abs_dprob_background_shuffle=f(d_bg), mean_abs_dprob_lesion_shuffle=f(d_les),
                evidence_mass_in_roi_before=f(mass_b), evidence_mass_in_roi_after_background_shuffle=f(mass_a))


# ============================ XAI: nhất quán ============================
@torch.no_grad()
def explanation_consistency_eval(model, X, indices, device, batch_size=16, seed=0, hflip_only=False):
    # prediction consistency và evidence consistency được TÁCH RIÊNG; evidence được inverse-warp và chỉ tính trên pixel hợp lệ
    g = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    agree, dprob, cos_all = [], [], []
    for i in range(0, len(indices), batch_size):
        idx = indices[i:i + batch_size]
        x = torch.stack([core.normalize_to_tensor(X[k]) for k in idx]).to(device)
        theta = LX.hflip_theta(len(idx), device) if hflip_only else LX.sample_affine(len(idx), device, generator=g)
        o1, o2 = model(x), model(LX.apply_label_preserving(x, theta))
        p1, p2 = F.softmax(o1['logits'].float(), 1), F.softmax(o2['logits'].float(), 1)
        agree += (p1.argmax(1) == p2.argmax(1)).float().cpu().tolist()
        dprob += (p1[:, 1] - p2[:, 1]).abs().cpu().tolist()
        if o1.get('evidence') is not None:
            cos, ok, _, _ = LX.evidence_alignment(o1['evidence'], o2['evidence'], theta)
            cos_all += cos[ok > 0].cpu().tolist()
    return dict(prediction_agreement=float(np.mean(agree)), mean_abs_dprob_malignant=float(np.mean(dprob)),
                evidence_cosine_after_inverse_warp=float(np.mean(cos_all)) if cos_all else float('nan'), n=len(agree))


# ============================ Hiệu chuẩn, chọn lọc dự đoán ============================
def calibration_report(prob_val, y_val, prob_test, y_test):
    # T chọn trên validation, áp dụng cho test (hoặc cho tập ngoài)
    lg_v, lg_t = np.log(np.clip(prob_val, 1e-12, 1)), np.log(np.clip(prob_test, 1e-12, 1))
    Tm = T.fit_temperature(lg_v, y_val)
    rows = []
    for tag, tt in (('trước', 1.0), ('sau', Tm)):
        p = T.softmax(lg_t, tt)
        rows.append(dict(hiệu_chuẩn=tag, T=tt, NLL=T.nll(p, y_test), Brier=T.brier(p, y_test), ECE=T.ece(p, y_test)))
    return Tm, rows


def selective_report(prob_val, prob_test, y_test, coverages=(0.9, 0.8, 0.7), malignant=1):
    # độ tin cậy = max xác suất; ngưỡng chọn trên validation theo coverage mục tiêu
    cv, ct = prob_val.max(1), prob_test.max(1)
    pred = prob_test.argmax(1)
    wrong = pred != y_test
    cov, risk, aurc = T.risk_coverage(-ct, wrong)
    out = dict(aurc=float(aurc), rows=[])
    for c in coverages:
        thr = float(np.quantile(cv, 1 - c))
        acc = ct >= thr
        mal = y_test == malignant
        out['rows'].append(dict(coverage_target=c, coverage_actual=float(acc.mean()), selective_risk=float(wrong[acc].mean()) if acc.any() else float('nan'),
                                abstain_threshold=thr, sensitivity_malignant_accepted=float((pred[acc & mal] == malignant).mean()) if (acc & mal).any() else float('nan'),
                                malignant_abstained=float((~acc[mal]).mean()) if mal.any() else float('nan')))
    return out


# ============================ Thống kê ============================
def bootstrap_ci(fn, n, B=1000, seed=0):
    rng = np.random.RandomState(seed); v = []
    for _ in range(B):
        x = fn(rng.randint(0, n, n))
        if np.isfinite(x):
            v.append(x)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 20 else (float('nan'), float('nan'))


def paired_bootstrap_diff(fa, fb, n, B=1000, seed=0):
    rng = np.random.RandomState(seed); d = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        x = fa(i) - fb(i)
        if np.isfinite(x):
            d.append(x)
    return (float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) > 20 else (float('nan'),) * 3


def summarize_seeds(values):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    return dict(n_seeds=int(len(v)), mean=float(v.mean()) if len(v) else float('nan'), sd=float(v.std(ddof=1)) if len(v) > 1 else float('nan'), values=v.tolist())


# ============================ Khóa pipeline và đánh giá ngoài ============================
class LockError(RuntimeError):
    pass


class ExternalEvalLock:
    # Trình tự: (1) chọn mô hình/ngưỡng/temperature CHỈ trên BUS-BRA -> lock(); (2) evaluate_external() kiểm tra khóa rồi mới đọc BUSI;
    # (3) sau khi đánh giá, khóa bị đánh dấu 'đã dùng' và KHÔNG cho khóa lại hoặc đánh giá lại (một lần).
    def __init__(self, path):
        self.path = path

    def read(self):
        if not os.path.exists(self.path):
            raise LockError('Chưa có pipeline lock: không được truy cập dữ liệu ngoài (BUSI).')
        return json.load(open(self.path, encoding='utf-8'))

    def lock(self, ckpt_path, config, temperature, threshold, abstain_threshold, notes=''):
        if os.path.exists(self.path) and self.read().get('external_evaluated'):
            raise LockError('Pipeline đã được dùng để đánh giá ngoài; không được khóa lại.')
        cfg_sha = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        json.dump(dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), ckpt=os.path.basename(ckpt_path), ckpt_sha256=sha256_file(ckpt_path), config=config,
                       config_sha256=cfg_sha, temperature=float(temperature), threshold=float(threshold), abstain_threshold=float(abstain_threshold),
                       notes=notes, external_evaluated=False), open(self.path, 'w', encoding='utf-8'), indent=1, default=str)

    def verify(self, ckpt_path):
        L = self.read()
        if sha256_file(ckpt_path) != L['ckpt_sha256']:
            raise LockError('Checkpoint đã thay đổi sau khi khóa (sha256 không khớp).')
        return L

    def guard_access(self, ckpt_path):
        L = self.verify(ckpt_path)
        if L.get('external_evaluated'):
            raise LockError('Đánh giá ngoài chỉ được thực hiện MỘT lần với pipeline đã khóa.')
        return L

    def mark_evaluated(self, summary):
        L = self.read()
        L['external_evaluated'] = True; L['external_summary'] = summary; L['evaluated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        json.dump(L, open(self.path, 'w', encoding='utf-8'), indent=1, default=str)


def evaluate_external(lock, ckpt_path, load_external_fn, predict_fn):
    # load_external_fn(): đọc BUSI (chỉ được gọi SAU khi qua guard). predict_fn(model_data) -> dict(prob, ...)
    L = lock.guard_access(ckpt_path)
    data = load_external_fn()
    out = predict_fn(data, L)
    lock.mark_evaluated({k: v for k, v in out.items() if isinstance(v, (int, float, str, dict, list))})
    return out
