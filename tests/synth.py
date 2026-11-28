"""Dữ liệu và mô hình GIẢ chỉ dùng cho KIỂM THỬ ĐƠN VỊ logic. Không bao giờ được dùng để báo cáo kết quả khoa học."""
import numpy as np
import cv2


def make_us_image(rng, size=256, lesion_center=(128, 128), radius=28, contrast=0.5, pad=0, annotation=False):
    """Ảnh 'siêu âm' tổng hợp: nền speckle, một tổn thương tròn tối (contrast thấp=tối), tuỳ chọn dải đen letterbox và dấu đo sáng."""
    base = 110 + 25 * rng.standard_normal((size, size)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 1.2) + 18 * rng.standard_normal((size, size)).astype(np.float32)
    yy, xx = np.ogrid[:size, :size]
    m = (yy - lesion_center[0]) ** 2 + (xx - lesion_center[1]) ** 2 <= radius ** 2
    base[m] *= contrast
    img = np.clip(np.stack([base] * 3, -1), 0, 255).astype(np.uint8)
    if pad:
        img[:, :pad] = 0; img[:, -pad:] = 0
    if annotation:
        cv2.putText(img, '+', (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img, m


def lesion_model(mask_getter=None, scale=0.08):
    """Mô hình giả nhạy với TƯƠNG PHẢN tổn thương: z = scale*(mean quanh - mean trong vùng tối trung tâm). Trả cả seg_prob đúng vị trí."""
    def fn(images):
        L, S = [], []
        for im in images:
            g = im.mean(2)
            c = g[108:148, 108:148].mean()
            ring = np.concatenate([g[40:70, 40:70].ravel(), g[190:220, 190:220].ravel()]).mean()
            z = scale * (ring - c) - 1.0
            L.append([0.0, z]); S.append(((g < 80) & (np.abs(np.arange(256)[:, None] - 128) < 40) & (np.abs(np.arange(256)[None] - 128) < 40)).astype(np.float32))
        return dict(logits=np.array(L), seg_prob=np.stack(S), evidence=None)
    return fn


def background_shortcut_model(scale=0.12):
    """Mô hình giả ĐI TẮT: chỉ nhìn một mảng nền cố định (góc trên-trái) — không dùng tổn thương."""
    def fn(images):
        L = []
        for im in images:
            z = scale * (im.mean(2)[20:80, 20:80].mean() - 110)
            L.append([0.0, z])
        return dict(logits=np.array(L), seg_prob=None, evidence=None)
    return fn


def center_mask(size=256, c=(128, 128), r=28):
    yy, xx = np.ogrid[:size, :size]
    return ((yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= r ** 2).astype(np.uint8)


def make_feature_table(n=1500, seed=0, signal=True, n_groups=900, cf_signal=True):
    """Bảng đặc trưng TỔNG HỢP cho kiểm thử logic bộ phát hiện (KHÔNG phải dữ liệu khoa học).
    Nếu signal: nguy cơ sai tăng khi cf_specificity thấp và bg_sens cao (hiện tượng 'đi tắt') và khi thr_margin nhỏ."""
    import pandas as pd
    from src import failure_risk as FR
    r = np.random.RandomState(seed)
    cols = sorted(set(sum(FR.FEATURE_SETS.values(), [])) - {'cf_missing', 'conf', 'entropy', 'margin_abs', 'ens_js', 'ens_dz_sd', 'ens_flip'})
    X = pd.DataFrame(r.randn(n, len(cols)), columns=cols)
    X['p_mal'] = 1 / (1 + np.exp(-r.randn(n) * 1.5))
    X['thr_margin'] = np.abs(r.randn(n)) * 1.5
    X['p_raw'] = X['p_mal']
    X['view_js'] = np.abs(r.randn(n)) * 0.05
    groups = r.randint(0, n_groups, n)
    lg = -1.6 - 0.6 * X['thr_margin'] * (1 if signal else 0)
    if signal and cf_signal:
        lg = lg - 0.9 * X['cf_specificity'] + 0.7 * X['bg_sens']
    y = (r.rand(n) < 1 / (1 + np.exp(-lg))).astype(int)
    X = FR.add_derived_features(X)
    return X, y, groups
