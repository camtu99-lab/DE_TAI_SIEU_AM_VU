"""Độ tin cậy, entropy, bất định và bất đồng giữa các góc nhìn. Thuần numpy.
LƯU Ý nhị phân: entropy H(p) là hàm đơn điệu của max(p) = confidence, nên hai điểm số này cho CÙNG thứ hạng (xem test)."""
import numpy as np

EPS = 1e-12


def softmax(z, T=1.0):
    z = np.asarray(z, np.float64) / float(T)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def calibrated_probs(logits, T):
    return softmax(logits, T)


def margin(logits, T=1.0):
    l = np.asarray(logits, np.float64)
    return (l[..., 1] - l[..., 0]) / float(T)


def entropy(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1.0)
    return -(p * np.log(p)).sum(-1)


def confidence(p):
    return np.asarray(p).max(-1)


def js_disagreement(P):
    """Generalized Jensen-Shannon giữa S phân phối (S,N,K) -> (N,). = H(mean) - mean(H). Bằng thông tin tương hỗ (BALD) khi các mẫu là các 'mô hình'.
    Dùng cho bất đồng giữa các góc nhìn TTA (một mô hình) HOẶC giữa các mô hình (ensemble) — nguồn được ghi trong `ens_source`."""
    P = np.asarray(P, np.float64)
    return entropy(P.mean(0)) - entropy(P).mean(0)


def view_stats(z_views, thr_logit=0.0):
    """z_views (S,N): margin logit hiệu chuẩn qua S góc nhìn (hàng 0 = ảnh gốc). Trả độ ổn định dự đoán."""
    z = np.asarray(z_views, np.float64)
    flip = ((z[1:] >= thr_logit) != (z[0:1] >= thr_logit)).mean(0) if len(z) > 1 else np.zeros(z.shape[1])
    return dict(view_pred_flip=flip, view_dz_sd=z.std(0), view_dz_absmax=np.abs(z[1:] - z[0:1]).max(0) if len(z) > 1 else np.zeros(z.shape[1]))


def ensemble_features(member_logits, T=1.0, thr=0.5):
    """member_logits (M,2): logits(hoặc log-prob) của M mô hình THẬT (cùng fold, khác seed; đều chưa thấy ảnh này). Trả đặc trưng bất đồng.
    ens_js = Jensen-Shannon/BALD giữa các thành viên; ens_dz_sd = độ lệch chuẩn margin; ens_flip = tỉ lệ thành viên bất đồng nhãn với trung bình."""
    L = np.asarray(member_logits, np.float64)
    if L.ndim != 2 or L.shape[0] < 2:
        raise ValueError('ensemble cần >= 2 thành viên')
    P = softmax(L, T)                                   # (M,2)
    z = margin(L, T)
    zthr = float(np.log(thr / (1 - thr)))
    pm = P.mean(0)
    lab = (P[:, 1] >= thr)
    return dict(ens_js=float(js_disagreement(P[:, None, :])[0]), ens_dz_sd=float(z.std()), ens_flip=float((lab != (pm[1] >= thr)).mean()),
                ens_p_mal=float(pm[1]))
