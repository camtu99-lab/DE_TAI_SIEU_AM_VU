"""Logic của ứng dụng Streamlit, KHÔNG import streamlit/plotly (để kiểm thử được). Mọi chức năng dựa trên artifact thật; thiếu -> trả lý do, không giả lập."""
import os, json, glob
import numpy as np
import pandas as pd
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, 'artifacts')
V3_REF = os.path.join(ART, 'v3_reference')
V4_DIR = os.path.join(ART, 'v4')
DEPLOY = os.environ.get('BV_DEPLOY_DIR', os.path.join(ROOT, 'deploy'))
SAMPLES = os.path.join(ROOT, 'assets', 'samples')
VERSION = 'BreastVision-XAI V4 — BUS-VERIFY'

DISCLAIMER = ('Đây là công cụ NGHIÊN CỨU/GIÁO DỤC, KHÔNG phải thiết bị chẩn đoán y khoa và không thay thế đánh giá của bác sĩ. '
              'Xác suất ác tính, risk score và cảnh báo chỉ có giá trị trong phạm vi dữ liệu và thực nghiệm đã báo cáo. '
              'Ảnh tải lên chỉ được xử lý trong bộ nhớ phiên làm việc, không lưu và không gửi ra dịch vụ bên ngoài.')
RISK_LIMITS = ('Risk score ước lượng khả năng MÔ HÌNH CHÍNH DỰ ĐOÁN SAI, KHÔNG phải xác suất ác tính. "Không phát cảnh báo" không có nghĩa là an toàn. '
               'Cảnh báo được huấn luyện trên dữ liệu BUS-BRA (một nguồn, 4 máy) và chưa chứng minh tổng quát hóa cho thiết bị/quần thể khác; '
               'ngưỡng đã khóa từ dữ liệu phát triển. Với số ca false-negative nhỏ, khả năng bắt FN có khoảng tin cậy rất rộng (xem trang Scientific Results).')

# tên artifact V4 mà notebook Kaggle sinh ra (trang web đọc trực tiếp)
V4_FILES = dict(
    manifest='RUN_MANIFEST.json',
    dev_table='v4_dev_nested_cv_table.csv', dev_pairs='v4_dev_paired_comparisons.csv',
    test_table='v4_busbra_test_table.csv', test_pairs='v4_busbra_test_paired_comparisons.csv',
    busi_table='v4_busi_external_table.csv', busi_pairs='v4_busi_external_paired_comparisons.csv',
    ablation='v4_ablation_dev.csv', robustness='v4_robustness_summary.csv', oracle='v4_oracle_experiment.csv',
    pred_dev='v4_predictions_dev_oof.csv', pred_test='v4_predictions_busbra_test.csv', pred_busi='v4_predictions_busi.csv',
    risk_coverage='v4_risk_coverage_curves.csv', classification='v4_classification_metrics.csv', status_table='v4_alert_status_counts.csv',
)


def load_table(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def load_json(path):
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else None


def v4_status():
    """{khóa: (có_không, đường_dẫn)} — dùng để hiển thị NOT RUN trung thực."""
    out = {k: (os.path.exists(os.path.join(V4_DIR, f)), os.path.join(V4_DIR, f)) for k, f in V4_FILES.items()}
    out['deploy_bundle'] = (os.path.exists(os.path.join(DEPLOY, 'deploy_config.json')), os.path.join(DEPLOY, 'deploy_config.json'))
    return out


def v4_has_results():
    s = v4_status()
    return s['manifest'][0] and (s['dev_table'][0] or s['test_table'][0])


def list_samples():
    return sorted(glob.glob(os.path.join(SAMPLES, '*.png')) + glob.glob(os.path.join(SAMPLES, '*.jpg')))


def load_runtime(deploy_dir=None):
    """Trả (bundle, predictor, thông_báo[]). Thiếu torch/checkpoint/artifact -> None + lý do cụ thể (không giả lập)."""
    deploy_dir = deploy_dir or DEPLOY
    msgs, bundle, pred = [], None, None
    try:
        from .pipeline import Bundle
        from .failure_risk import RiskModelMissing
        bundle = Bundle(deploy_dir)
        if bundle.risk_error:
            msgs.append(f'Risk/cảnh báo TẮT: {bundle.risk_error}')
    except Exception as e:                      # lỗi hiển thị cho người dùng, không bị nuốt
        msgs.append(f'Không nạp được gói triển khai ({type(e).__name__}): {e}')
        return None, None, msgs
    ck = bundle.cfg.get('checkpoint')
    if not ck:
        msgs.append('deploy_config.json không khai báo checkpoint.')
        return bundle, None, msgs
    path = ck if os.path.isabs(ck) else os.path.join(deploy_dir, ck)
    if not os.path.exists(path):
        alt = os.path.join(ROOT, 'checkpoints', os.path.basename(ck))
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        msgs.append(f'Chưa có checkpoint: {ck}. Huấn luyện bằng notebook Kaggle rồi chép vào thư mục checkpoints/.')
        return bundle, None, msgs
    try:
        import torch
    except ImportError:
        msgs.append('PyTorch chưa được cài: cài bản CPU để chạy suy luận (xem README).')
        return bundle, None, msgs
    from .inference import load_predictor
    pred = load_predictor(path, torch.device('cpu'))
    return bundle, pred, msgs


# ------------------------------------------------------------------ can thiệp tương tác (phạm vi hợp lệ)
RANGES = {'blur': (0.5, 12.0), 'replace': (0.1, 1.0), 'permute': (0.1, 1.0)}
REGION_KEYS = {'lesion': 'Tổn thương (mask dự đoán)', 'peri': 'Lân cận tổn thương', 'bg': 'Nền'}


def interactive_intervention(img, regions, region_key, op, strength, seed=0):
    """Một can thiệp do người dùng chọn. blur: sigma (px). replace/permute: tỉ lệ pixel của vùng bị can thiệp (0.1–1). Ngoài phạm vi -> ValueError.
    Trả (ảnh_sau, mask_vùng_thực_sự_bị_đổi)."""
    from . import counterfactual_verifier as CF
    if op not in RANGES:
        raise ValueError(f'op không hợp lệ: {op}')
    lo, hi = RANGES[op]
    if not (lo <= float(strength) <= hi):
        raise ValueError(f'mức {strength} ngoài phạm vi hợp lệ [{lo}, {hi}] của {op}')
    region = {'lesion': regions.lesion, 'peri': regions.peri, 'bg': regions.background}[region_key]
    if region.sum() < 16:
        raise ValueError(f'vùng {region_key} rỗng/quá nhỏ: không thể can thiệp')
    rng = np.random.default_rng(seed)
    ref = (regions.background if region_key != 'bg' else regions.peri | regions.lesion)
    if op == 'blur':
        return CF.apply_op(img, region, 'blur', rng, cfg=CF.CFConfig(blur_sigma=float(strength))), region
    sub = region.copy()
    if strength < 1.0:
        idx = np.flatnonzero(region.ravel())
        keep = rng.choice(idx, size=max(1, int(round(strength * len(idx)))), replace=False)
        sub = np.zeros(region.size, bool); sub[keep] = True; sub = sub.reshape(region.shape)
    return CF.apply_op(img, sub, op, rng, ref=ref), sub


def logit_prob_change(predict_fn, img0, img1, T, thr):
    from .uncertainty import margin, softmax
    o = predict_fn(np.stack([img0, img1]))
    z = margin(o['logits'], T); p = softmax(o['logits'], T)[:, 1]
    return dict(z0=float(z[0]), z1=float(z[1]), dz=float(z[1] - z[0]), p0=float(p[0]), p1=float(p[1]), dp=float(p[1] - p[0]),
                flip=bool((p[0] >= thr) != (p[1] >= thr)))


# ------------------------------------------------------------------ hiển thị
def overlay_mask(img, mask, color=(255, 64, 64), alpha=0.4):
    out = img.astype(np.float32).copy()
    m = np.asarray(mask) > 0
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.drawContours(out.astype(np.uint8), cnt, -1, color, 1)
    return out


def heat_overlay(img, heat, alpha=0.45):
    h = np.asarray(heat, np.float32)
    if h.shape != img.shape[:2]:
        h = cv2.resize(h, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    cm = cv2.cvtColor(cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (np.clip((1 - alpha) * img + alpha * cm, 0, 255)).astype(np.uint8)


def case_filter(df, kind, conf_thr=0.9):
    """Lọc ca: TP/TN/FP/FN theo nhãn thật & dự đoán; 'high_conf_error' = sai với độ tin cậy (đã hiệu chuẩn) >= conf_thr."""
    y, pred = df['y'].astype(int), df['pred'].astype(int)
    m = {'TP': (y == 1) & (pred == 1), 'TN': (y == 0) & (pred == 0), 'FP': (y == 0) & (pred == 1), 'FN': (y == 1) & (pred == 0)}
    if kind == 'high_conf_error':
        conf = np.maximum(df['p_mal'], 1 - df['p_mal'])
        return df[(y != pred) & (conf >= conf_thr)]
    return df[m[kind]] if kind in m else df


PRED_COLUMNS = ['id', 'y', 'p_mal', 'pred', 'wrong', 'is_fn', 'risk', 'status']


def validate_prediction_table(df):
    miss = [c for c in PRED_COLUMNS if c not in df]
    if miss:
        raise ValueError(f'Bảng dự đoán thiếu cột {miss}')
    return True
