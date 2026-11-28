"""Chỉ số đánh giá (thuần numpy/sklearn). Đối chiếu với V3 (bus_trust) trong tests/."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from .segmentation import dice, iou
from . import selective_prediction as SP
from . import statistical_tests as ST

EPS = 1e-12


def safe_auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float('nan')


def safe_auprc(y, s):
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if y.sum() > 0 else float('nan')


def classification_metrics(y, p_mal, thr=0.5):
    y = np.asarray(y).astype(int); p = np.asarray(p_mal, float)
    pred = (p >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    return dict(n=len(y), accuracy=float((pred == y).mean()), auroc=safe_auroc(y, p), auprc=safe_auprc(y, p),
                f1=float(f1_score(y, pred, zero_division=0)), sensitivity=sens, specificity=spec,
                balanced_accuracy=float(np.nanmean([sens, spec])), tp=tp, tn=tn, fp=fp, fn=fn, threshold=float(thr))


def brier_binary(p_mal, y):
    return float(((np.asarray(p_mal, float) - np.asarray(y)) ** 2).mean())


def nll_binary(p_mal, y):
    p = np.clip(np.asarray(p_mal, float), EPS, 1 - EPS); y = np.asarray(y)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ece_binary(p_mal, y, n_bins=10):
    """ECE theo confidence của lớp dự đoán — cùng định nghĩa bus_trust.ece (V3) để so sánh được; có test tương đương."""
    p = np.asarray(p_mal, float); y = np.asarray(y)
    P = np.stack([1 - p, p], 1)
    conf, pred = P.max(1), P.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def deletion_insertion_auc(img, predict_fn, rank_map, cls_sign=1.0, steps=10, baseline='blur', T=1.0):
    """Deletion/Insertion AUC trên xác suất lớp malignant (dùng margin logit/T -> sigmoid). rank_map (H,W): điểm cao = quan trọng.
    cls_sign=+1: theo dõi P(malignant); -1: theo dõi P(benign). Trả (deletion_auc, insertion_auc). Deletion thấp & insertion cao là tốt."""
    import cv2
    H, W = img.shape[:2]
    base = cv2.GaussianBlur(img, (0, 0), 8) if baseline == 'blur' else np.full_like(img, int(img.mean()))
    order = np.argsort(-np.asarray(rank_map, np.float64).ravel(), kind='stable')
    n = order.size
    fr = np.linspace(0, 1, steps + 1)
    dele, ins = [], []
    for f in fr:
        k = int(round(f * n))
        m = np.zeros(n, bool); m[order[:k]] = True; m = m.reshape(H, W)[..., None]
        dele.append(np.where(m, base, img)); ins.append(np.where(m, img, base))
    out = predict_fn(np.stack(dele + ins, 0))['logits']
    z = (out[:, 1] - out[:, 0]) / T * cls_sign
    p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    return float(np.trapz(p[:len(fr)], fr)), float(np.trapz(p[len(fr):], fr))


def error_detection_table(scores, wrong, is_fn, groups, rates=(0.1, 0.2, 0.3), B=1000, seed=0, primary_rate=0.2, ci=True):
    """scores: dict tên -> mảng điểm (cao = nguy cơ sai). Trả DataFrame: Error AUROC/AUPRC (CI cụm), FN Alert Recall tại mỗi tỉ lệ cảnh báo, AURC, AUGRC.
    So sánh CÙNG tỉ lệ cảnh báo: mỗi phương pháp cảnh báo đúng ceil(rate*n) ca điểm cao nhất (tie ngẫu nhiên có seed)."""
    wrong = np.asarray(wrong).astype(int); is_fn = np.asarray(is_fn).astype(bool)
    rows = []
    for name, s in scores.items():
        s = np.asarray(s, float)
        row = dict(method=name, n=len(s), n_errors=int(wrong.sum()), n_fn=int(is_fn.sum()), error_prevalence=float(wrong.mean()),
                   error_auroc=safe_auroc(wrong, s), error_auprc=safe_auprc(wrong, s), aurc=SP.aurc(s, wrong), augrc=SP.augrc(s, wrong))
        for r in rates:
            a = SP.alert_set(s, r, rng=np.random.default_rng(seed))
            m = SP.alert_metrics(a, wrong, is_fn)
            row[f'fn_alert_recall@{int(r*100)}'] = m['fn_alert_recall']; row[f'error_recall@{int(r*100)}'] = m['error_recall']
            row[f'fn_caught@{int(r*100)}'] = m['fn_caught']
        if ci:
            e = ST.bootstrap_ci(lambda i: safe_auprc(wrong[i], s[i]), groups, B=B, seed=seed)
            row['error_auprc_lo'], row['error_auprc_hi'] = e['lo'], e['hi']
            def fnr(i, s=s):
                a = SP.alert_set(s[i], primary_rate, rng=np.random.default_rng(seed))
                nf = is_fn[i].sum()
                return float((a & is_fn[i]).sum() / nf) if nf else float('nan')
            f = ST.bootstrap_ci(fnr, groups, B=B, seed=seed)
            row[f'fn_alert_recall@{int(primary_rate*100)}_lo'], row[f'fn_alert_recall@{int(primary_rate*100)}_hi'] = f['lo'], f['hi']
        rows.append(row)
    return pd.DataFrame(rows)


def ood_confound_analysis(records):
    """Phân tích gây nhiễu OOD của can thiệp: nếu |Δlogit| của đối chứng tương quan mạnh với 'thay đổi pixel'/'seam', thì độ nhạy của
    can thiệp một phần là artefact. records: DataFrame từ verify_image(return_records=True) gộp nhiều ảnh (cột group,absdz,mean_abs_change,seam_ratio).
    Trả Spearman(|Δz|, mean_abs_change) và Spearman(|Δz|, seam_ratio) theo nhóm, và so sánh target vs matched sau khi chia tầng theo thay đổi pixel."""
    from scipy.stats import spearmanr
    out = {}
    for g, d in records.groupby('group'):
        if len(d) > 10:
            out[f'{g}_spearman_absdz_vs_pixelchange'] = float(spearmanr(d['absdz'], d['mean_abs_change'])[0])
            out[f'{g}_spearman_absdz_vs_seam'] = float(spearmanr(d['absdz'], d['seam_ratio'])[0])
    t, m = records[(records.group == 'region') & (records.region == 'lesion')], records[(records.group == 'matched') & (records.region == 'lesion')]
    if len(t) > 20 and len(m) > 20:
        q = np.quantile(pd.concat([t['mean_abs_change'], m['mean_abs_change']]), [0, .25, .5, .75, 1.0])
        diffs = []
        for lo, hi in zip(q[:-1], q[1:]):
            a = t[(t.mean_abs_change >= lo) & (t.mean_abs_change <= hi)]['absdz']
            b = m[(m.mean_abs_change >= lo) & (m.mean_abs_change <= hi)]['absdz']
            if len(a) > 5 and len(b) > 5:
                diffs.append(float(a.mean() - b.mean()))
        out['lesion_minus_matched_absdz_within_pixelchange_strata'] = diffs
    return out
