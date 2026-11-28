# bus_trust.py - BUS-Trust: hiệu chuẩn, độ bất định, chọn lọc dự đoán, nhất quán đa mô hình, báo cáo PDF
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

EPS = 1e-12


# ------------------------- Xác suất, hiệu chuẩn -------------------------
def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def entropy(p):
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(-1)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0)).mean())


def brier(p, y):
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean())


def ece(p, y, n_bins=10):
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def fit_temperature(logits, y):
    # tìm T > 0 cực tiểu NLL trên tập validation (tìm lưới thô rồi tinh)
    grid = np.exp(np.linspace(np.log(0.2), np.log(10.0), 200))
    losses = [nll(softmax(logits, t), y) for t in grid]
    i = int(np.argmin(losses))
    fine = np.linspace(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)], 100)
    return float(fine[int(np.argmin([nll(softmax(logits, t), y) for t in fine]))])


def mutual_information(P):
    # P: (S, N, K) các mẫu xác suất (MC dropout) -> thông tin tương hỗ (độ bất định tri thức)
    return entropy(P.mean(0)) - entropy(P).mean(0)


# ------------------------- Chọn lọc dự đoán -------------------------
def risk_coverage(unc, wrong):
    order = np.argsort(unc, kind='stable')
    w = np.asarray(wrong, float)[order]
    n = len(w)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(w) / np.arange(1, n + 1)
    return cov, risk, float(risk.mean())        # (coverage, risk, AURC)


def safe_auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    if len(np.unique(y)) < 2 or not np.isfinite(s).all():
        return float('nan')
    return float(roc_auc_score(y, s))


def accept_by_val_quantile(val_unc, test_unc, coverage):
    thr = float(np.quantile(val_unc, coverage))   # ngưỡng chọn CHỈ trên validation
    return thr, np.asarray(test_unc) <= thr


# ------------------------- Nhất quán phân loại - phân vùng - Grad-CAM -------------------------
def seg_evidence(seg_prob, thr=0.5):
    sp = np.asarray(seg_prob, np.float32)
    if sp.ndim == 2:
        sp = sp[None]
    area = (sp >= thr).mean((1, 2))
    peak = sp.reshape(len(sp), -1).max(1)
    return np.stack([peak, np.log10(area + 1e-4)], 1)


def cam_alignment(cam, mask):
    # tỉ lệ năng lượng Grad-CAM nằm trong mask dự đoán; NaN nếu mask rỗng
    cam = np.asarray(cam, np.float32)
    m = np.asarray(mask) > 0
    if cam.ndim == 2:
        cam, m = cam[None], m[None]
    tot = cam.reshape(len(cam), -1).sum(1) + 1e-8
    inside = (cam * m).reshape(len(cam), -1).sum(1) / tot
    area = m.reshape(len(m), -1).mean(1)
    return np.where(area > 0, inside, np.nan), area


class Standardizer:
    def fit(self, F):
        F = np.asarray(F, float)
        self.mean = np.nanmean(F, 0)
        self.mean = np.where(np.isfinite(self.mean), self.mean, 0.0)
        Fi = np.where(np.isfinite(F), F, self.mean)
        self.std = Fi.std(0) + 1e-8
        return self

    def transform(self, F):
        F = np.asarray(F, float)
        return (np.where(np.isfinite(F), F, self.mean) - self.mean) / self.std


class TumorPresenceMapper:
    # ánh xạ bằng chứng phân vùng -> P(ảnh có khối u); học trên validation để cùng thang xác suất với P(tumor) của bộ phân loại
    def fit(self, ev, has_tumor):
        self.sc = Standardizer().fit(ev)
        self.lr = LogisticRegression(C=1.0).fit(self.sc.transform(ev), has_tumor.astype(int))
        return self

    def predict(self, ev):
        return self.lr.predict_proba(self.sc.transform(ev))[:, 1]

    def to_dict(self):
        return dict(mean=self.sc.mean.tolist(), std=self.sc.std.tolist(), coef=self.lr.coef_[0].tolist(), intercept=float(self.lr.intercept_[0]))

    @staticmethod
    def from_dict(d):
        m = TumorPresenceMapper()
        m.sc = Standardizer(); m.sc.mean, m.sc.std = np.array(d['mean']), np.array(d['std'])
        m.coef, m.intercept = np.array(d['coef']), d['intercept']
        m.predict = lambda ev: 1 / (1 + np.exp(-(((np.asarray(ev) - m.sc.mean) / m.sc.std) @ m.coef + m.intercept)))
        return m


class ErrorPredictor:
    # hồi quy logistic chuẩn hóa, ít đặc trưng, regularization mạnh; học trên validation để dự đoán 'dự đoán của E0 sai'
    def __init__(self, C=0.3):
        self.C = C

    def fit(self, F, wrong):
        self.sc = Standardizer().fit(F)
        self.lr = LogisticRegression(C=self.C, class_weight='balanced', max_iter=1000).fit(self.sc.transform(F), np.asarray(wrong).astype(int))
        return self

    def score(self, F):
        return self.lr.decision_function(self.sc.transform(F))

    def to_dict(self):
        return dict(mean=self.sc.mean.tolist(), std=self.sc.std.tolist(), coef=self.lr.coef_[0].tolist(), intercept=float(self.lr.intercept_[0]), C=self.C)

    @staticmethod
    def from_dict(d):
        p = ErrorPredictor(d.get('C', 0.3))
        mean, std, coef, b = np.array(d['mean']), np.array(d['std']), np.array(d['coef']), d['intercept']
        p.score = lambda F: ((np.where(np.isfinite(np.asarray(F, float)), np.asarray(F, float), mean) - mean) / std) @ coef + b
        return p


# ------------------------- Phân tích một ảnh (dùng trong ứng dụng) -------------------------
def analyze_trust(res, cfg):
    # res: kết quả core.predict_array (cần probs, mask, seg_prob, cam); cfg: trust_config.json đã nạp
    names = cfg['class_names']
    probs = np.clip(np.asarray(res['probs'], float), EPS, 1.0)
    p_cal = softmax(np.log(probs), cfg['temperature'])
    if p_cal.ndim == 2:
        p_cal = p_cal[0]
    out = dict(p_cal=p_cal.tolist(), pred_idx=int(np.argmax(p_cal)), entropy=float(entropy(p_cal)), reasons=[])
    ni = cfg.get('normal_idx')
    ev = seg_evidence(res['seg_prob'], cfg['seg_threshold'])
    if ni is not None and cfg.get('mapper'):
        mapper = TumorPresenceMapper.from_dict(cfg['mapper'])
        p_seg = float(mapper.predict(ev)[0])
        p_cls = float(1.0 - p_cal[ni])
        out.update(p_tumor_cls=p_cls, p_tumor_seg=p_seg, dcs=abs(p_cls - p_seg))
    else:
        out.update(p_tumor_cls=float('nan'), p_tumor_seg=float('nan'), dcs=0.0)
    inside = float('nan')
    if res.get('cam') is not None:
        inside = float(cam_alignment(res['cam'], res['mask'])[0][0])
    out['cam_inside'] = inside
    out['mask_area'] = float((np.asarray(res['mask']) > 0).mean())
    F = np.array([[out['entropy'], out['dcs'], inside]])
    pred = ErrorPredictor.from_dict(cfg['predictor'])
    out['risk_score'] = float(pred.score(F)[0])
    th = cfg['thresholds']
    if out['entropy'] > th['entropy']:
        out['reasons'].append('Độ bất định của dự đoán cao (entropy vượt ngưỡng chọn trên tập validation).')
    if out['dcs'] > th['dcs']:
        out['reasons'].append('Kết quả phân loại và phân vùng chưa nhất quán về việc ảnh có khối u hay không.')
    if np.isfinite(inside) and inside < th['cam_inside']:
        out['reasons'].append('Vùng Grad-CAM ít trùng với vùng được phân vùng.')
    out['flag'] = bool(out['risk_score'] >= th['risk_score'])
    out['pred_name'] = names[out['pred_idx']]
    return out


# ------------------------- Báo cáo PDF (không cần thư viện ngoài) -------------------------
def make_pdf_report(path, filename, res, class_names, trust, meta_text, fig=None):
    import matplotlib
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages(path) as pdf:
        f = plt.figure(figsize=(8.27, 11.69))
        f.text(0.07, 0.95, 'BÁO CÁO PHÂN TÍCH ẢNH SIÊU ÂM VÚ (hỗ trợ nghiên cứu)', fontsize=14, weight='bold')
        lines = [f'Tệp ảnh: {filename}', f'Thời gian: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', meta_text, '',
                 'Xác suất (đã hiệu chuẩn bằng temperature scaling):']
        for c, p in zip(class_names, trust['p_cal']):
            lines.append(f'   - {c}: {p * 100:.1f}%')
        lines += ['', f'Lớp dự đoán: {trust["pred_name"]}', f'Entropy dự đoán: {trust["entropy"]:.3f}',
                  f'Tỉ lệ diện tích vùng phân vùng: {trust["mask_area"] * 100:.2f}%',
                  f'Chênh lệch P(có khối u) giữa phân loại và phân vùng: {trust["dcs"]:.3f}',
                  f'Độ trùng Grad-CAM với vùng phân vùng: ' + ('không xác định (mask rỗng)' if not np.isfinite(trust['cam_inside']) else f'{trust["cam_inside"]:.3f}'),
                  '', 'TRẠNG THÁI: ' + ('CẦN CHUYÊN GIA XEM XÉT' if trust['flag'] else 'Không có cảnh báo từ hệ thống kiểm tra độ tin cậy')]
        for r in trust['reasons']:
            lines.append('   * ' + r)
        lines += ['', 'GIỚI HẠN SỬ DỤNG:',
                  '   - Đây là công cụ hỗ trợ nghiên cứu, KHÔNG phải kết luận chẩn đoán.',
                  '   - Mô hình huấn luyện trên một bộ dữ liệu công khai nhỏ, chưa kiểm định ngoài.',
                  '   - Không có cảnh báo không có nghĩa là dự đoán đúng; mọi kết quả cần được bác sĩ chuyên khoa đánh giá.']
        y = 0.91
        for ln in lines:
            f.text(0.07, y, ln, fontsize=9.5)
            y -= 0.022
        pdf.savefig(f); plt.close(f)
        if fig is not None:
            pdf.savefig(fig)