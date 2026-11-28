"""Selective prediction / cảnh báo nguy cơ sai. Thuần numpy.
Quy ước: `score` CAO = nguy cơ sai CAO (cảnh báo). `wrong` = 1 nếu mô hình chính dự đoán sai.
`is_fn` = ca ác tính bị dự đoán thành lành tính (false negative) — kết cục lâm sàng nặng nhất."""
import numpy as np

STATUS_NONE = 'Không phát cảnh báo'
STATUS_HIGH = 'Nguy cơ dự đoán sai cao'
STATUS_REVIEW = 'Cần xem xét lại'
STATUS_OOS = 'Đầu vào ngoài phạm vi đã kiểm định'


def _order_by_score(score, rng=None):
    s = np.asarray(score, float)
    if rng is None:
        return np.argsort(-s, kind='stable')
    return np.lexsort((rng.random(len(s)), -s))      # tie ngẫu nhiên có seed


def risk_coverage_curve(score, wrong, rng=None):
    """Nhận ca có risk THẤP nhất trước. Trả coverage (k/n), selective risk (lỗi/k), generalized risk (lỗi nhận/n)."""
    s = np.asarray(score, float); w = np.asarray(wrong, float)
    order = np.argsort(s, kind='stable') if rng is None else np.lexsort((rng.random(len(s)), s))
    cum = np.cumsum(w[order])
    k = np.arange(1, len(s) + 1)
    return k / len(s), cum / k, cum / len(s)


def aurc(score, wrong):
    """AURC = trung bình selective risk qua mọi coverage k/n (Geifman & El-Yaniv)."""
    _, sr, _ = risk_coverage_curve(score, wrong)
    return float(sr.mean())


def augrc(score, wrong):
    """AUGRC (Traub et al., 2024) = trung bình generalized risk = P(được nhận VÀ sai) qua mọi coverage."""
    _, _, gr = risk_coverage_curve(score, wrong)
    return float(gr.mean())


def oracle_aurc(wrong):
    """AURC lý tưởng (mọi ca đúng được nhận trước): cận dưới để diễn giải."""
    w = np.asarray(wrong, float)
    return aurc(w, w)


def alert_set(score, alert_rate, rng=None):
    """Chọn ceil(rate*n) ca có score cao nhất (so sánh CÙNG tỉ lệ cảnh báo giữa các phương pháp)."""
    n = len(score)
    k = int(np.ceil(alert_rate * n - 1e-9))
    a = np.zeros(n, bool)
    a[_order_by_score(score, rng)[:k]] = True
    return a


def alert_metrics(alert, wrong, is_fn):
    alert = np.asarray(alert, bool); wrong = np.asarray(wrong, bool); is_fn = np.asarray(is_fn, bool)
    n = len(alert)
    nw, nf = int(wrong.sum()), int(is_fn.sum())
    out = dict(n=n, alert_rate=float(alert.mean()), n_errors=nw, n_fn=nf,
               error_recall=float((alert & wrong).sum() / nw) if nw else float('nan'),
               error_precision=float((alert & wrong).sum() / alert.sum()) if alert.any() else float('nan'),
               fn_alert_recall=float((alert & is_fn).sum() / nf) if nf else float('nan'),
               fn_caught=int((alert & is_fn).sum()),
               residual_error_rate=float((wrong & ~alert).sum() / max((~alert).sum(), 1)),
               residual_fn=int((is_fn & ~alert).sum()))
    return out


def random_referral_expectation(wrong, is_fn, alert_rate, n_draws=500, seed=0):
    """B0: cảnh báo ngẫu nhiên. Kỳ vọng error_recall = fn_alert_recall = alert_rate; ước lượng Monte-Carlo để kiểm chứng + AUPRC kỳ vọng."""
    rng = np.random.default_rng(seed)
    r, f = [], []
    for _ in range(n_draws):
        a = alert_set(rng.random(len(wrong)), alert_rate)
        m = alert_metrics(a, wrong, is_fn)
        r.append(m['error_recall']); f.append(m['fn_alert_recall'])
    return dict(error_recall=float(np.mean(r)), fn_alert_recall=float(np.nanmean(f)))


def decide_status(risk, t_review, t_high, in_scope=True, oos_reasons=()):
    """Bốn trạng thái. Ngưỡng t_review <= t_high đã KHÓA từ dữ liệu phát triển (không từ test)."""
    if not in_scope:
        return STATUS_OOS
    if risk >= t_high:
        return STATUS_HIGH
    if risk >= t_review:
        return STATUS_REVIEW
    return STATUS_NONE
