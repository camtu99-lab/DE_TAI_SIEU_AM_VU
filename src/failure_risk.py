"""Failure-Risk Detection Engine: mô hình phụ dự đoán P(mô hình chính dự đoán SAI | đặc trưng).
Nguyên tắc: (1) nhãn mục tiêu = mô hình chính đúng/sai trên dự đoán OUT-OF-FOLD; (2) KHÔNG test labels trong huấn luyện/chọn đặc trưng/chọn C/
ngưỡng (có chốt chặn `assert_no_test`); (3) logistic regression L2 là mặc định; GBM chỉ so sánh khi đủ số ca lỗi; (4) risk != P(malignant)."""
import os, json, hashlib, time
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss

SCHEMA_VERSION = 'bus-verify-risk-1'

G_CONF = ['p_mal', 'margin_abs', 'entropy', 'conf', 'thr_margin']
G_UNC = ['view_js', 'view_dz_sd', 'ens_js', 'ens_dz_sd', 'ens_flip']      # ens_* chỉ có khi chạy ensemble thật; cột thiếu bị bỏ khi dựng
G_STAB = ['prediction_stability', 'view_dz_absmax', 'mask_consistency', 'evidence_stability']
G_CF = ['lesion_sens', 'peri_sens', 'bg_sens', 'lesion_dir', 'lesion_bg_contrast', 'lesion_peri_contrast', 'cf_specificity', 'cf_z',
        'cf_flip_rate', 'cf_consistency', 'lesion_flip', 'bg_flip', 'ev_faithfulness', 'cf_missing']
G_SHORTCUT = ['pad_sens', 'annot_sens']          # thăm dò, ngoài P1 chính
FEATURE_SETS = {
    'B4_conf_only': G_CONF,
    'B5_cf_only': G_CF,
    'B6_unc_stab_no_cf': G_CONF + G_UNC + G_STAB,
    'P1_full': G_CONF + G_UNC + G_STAB + G_CF,
}
ABLATIONS = {                                    # P1 bỏ từng nhóm đặc trưng (nhóm con theo yêu cầu)
    'P1-lesion_sens': ['lesion_sens', 'lesion_dir', 'lesion_bg_contrast', 'lesion_peri_contrast', 'cf_specificity', 'cf_z', 'lesion_flip'],
    'P1-background_sens': ['bg_sens', 'bg_flip', 'lesion_bg_contrast'],
    'P1-uncertainty': G_UNC,
    'P1-stability': G_STAB,
    'P1-counterfactual': G_CF,
}


class RiskModelMissing(RuntimeError):
    pass


class PolicyTamperedError(RuntimeError):
    pass


class TestLeakageError(RuntimeError):
    __test__ = False


def assert_no_test(split_labels):
    s = set(map(str, np.unique(np.asarray(split_labels))))
    if 'test' in s or 'external' in s:
        raise TestLeakageError(f'Dữ liệu test/external không được dùng để huấn luyện/chọn tham số/ngưỡng (phát hiện nhãn tập: {sorted(s)}).')


def add_derived_features(df):
    """Tính cột phái sinh từ đặc trưng thô: margin_abs, conf, entropy (nhị phân), cf_missing."""
    d = df.copy()
    if 'p_mal' in d:
        p = d['p_mal'].clip(1e-9, 1 - 1e-9)
        d['conf'] = np.maximum(p, 1 - p)
        d['entropy'] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        d['margin_abs'] = np.abs(np.log(p / (1 - p)))
    if 'lesion_sens' in d:
        d['cf_missing'] = d['lesion_sens'].isna().astype(float)
    return d


def dedupe_rank_equivalent(df, names, tol=1e-9):
    """Bỏ đặc trưng có |Spearman| = 1 với đặc trưng đã giữ (nhị phân: conf, entropy, margin_abs đơn điệu với nhau). Trả (giữ, bỏ{tên: tương đương với})."""
    keep, dropped = [], {}
    for n in names:
        if n not in df:
            continue
        v = df[n]
        dup = None
        for k in keep:
            ok = v.notna() & df[k].notna()
            if ok.sum() > 5 and abs(v[ok].rank().corr(df[k][ok].rank()) ) > 1 - tol:
                dup = k; break
        if dup is None:
            keep.append(n)
        else:
            dropped[n] = dup
    return keep, dropped


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_pipeline(C):
    return Pipeline([('imp', SimpleImputer(strategy='median', keep_empty_features=True)), ('slog', FunctionTransformer(_slog)),
                     ('sc', StandardScaler()), ('lr', LogisticRegression(C=C, max_iter=2000))])


def _folds(y, groups, k, seed):
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def oof_risk(X, y, groups, C, k=5, seed=0):
    X = np.asarray(X, float); y = np.asarray(y).astype(int)
    out = np.full(len(y), np.nan)
    for tr, te in _folds(y, groups, k, seed):
        m = make_pipeline(C).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def select_C(X, y, groups, grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0), k=5, seed=0):
    """Chọn C bằng log-loss out-of-fold theo cụm TRÊN DỮ LIỆU PHÁT TRIỂN."""
    res = {}
    for C in grid:
        p = np.clip(oof_risk(X, y, groups, C, k, seed), 1e-6, 1 - 1e-6)
        res[C] = float(log_loss(y, p))
    best = min(res, key=res.get)
    return best, res


class RiskCalibrator:
    def __init__(self, method='auto', n_err=0):
        self.method = ('isotonic' if n_err >= 100 else 'sigmoid') if method == 'auto' else method
        self.iso = None; self.coef = None

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, raw, y):
        if self.method == 'isotonic':
            self.iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(raw, y)
        else:
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw)[:, None], y)
            self.coef = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        return self

    def __call__(self, raw):
        if self.method == 'isotonic':
            return self.iso.predict(np.asarray(raw, float))
        a, b = self.coef
        return 1 / (1 + np.exp(-(a * self._logit(raw) + b)))


class FailureRiskModel:
    def __init__(self, feature_names, seed=0, C=None, k=5):
        self.feature_names = list(feature_names); self.seed = seed; self.C = C; self.k = k
        self.pipe = None; self.calib = None; self.card = {}

    def fit(self, X_df, wrong, groups, split_labels):
        assert_no_test(split_labels)
        X = X_df[self.feature_names].to_numpy(float)
        y = np.asarray(wrong).astype(int)
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            raise ValueError(f'Quá ít ca lỗi/đúng để huấn luyện bộ phát hiện (lỗi={int(y.sum())}, đúng={int(len(y)-y.sum())}).')
        if self.C is None:
            self.C, grid = select_C(X, y, groups, k=self.k, seed=self.seed)
        else:
            grid = {}
        oof = oof_risk(X, y, groups, self.C, self.k, self.seed)
        self.calib = RiskCalibrator('auto', int(y.sum())).fit(oof, y)
        self.pipe = make_pipeline(self.C).fit(X, y)
        self.oof_raw = oof
        self.card = dict(schema=SCHEMA_VERSION, features=self.feature_names, C=float(self.C), C_grid_logloss=grid, n=int(len(y)), n_errors=int(y.sum()),
                         calibrator=self.calib.method, penalty='l2', trained_on_splits=sorted(map(str, np.unique(split_labels))),
                         created=time.strftime('%Y-%m-%d %H:%M:%S'))
        return self

    def _check(self):
        if self.pipe is None:
            raise RiskModelMissing('Mô hình risk chưa huấn luyện/chưa nạp.')

    def predict_raw(self, X_df):
        self._check()
        missing = [c for c in self.feature_names if c not in X_df]
        if missing:
            raise KeyError(f'Thiếu đặc trưng {missing}')
        return self.pipe.predict_proba(X_df[self.feature_names].to_numpy(float))[:, 1]

    def predict_risk(self, X_df):
        return self.calib(self.predict_raw(X_df))

    def contributions(self, x_row):
        """Đóng góp tuyến tính chính xác vào logit (sau impute/log/scale): coef_j * z_j. Tổng + intercept = logit thô."""
        self._check()
        x = np.asarray([x_row[c] for c in self.feature_names], float)[None]
        z = self.pipe[:-1].transform(x)[0]
        coef = self.pipe.named_steps['lr'].coef_[0]
        return pd.Series(coef * z, index=self.feature_names).sort_values(key=np.abs, ascending=False)

    def intercept(self):
        return float(self.pipe.named_steps['lr'].intercept_[0])

    def save(self, directory):
        import joblib
        os.makedirs(directory, exist_ok=True)
        p = os.path.join(directory, 'risk_model.joblib')
        joblib.dump(dict(pipe=self.pipe, calib=self.calib, feature_names=self.feature_names, card=self.card, schema=SCHEMA_VERSION), p)
        card = dict(self.card, sha256=sha256_file(p), coefficients=dict(zip(self.feature_names, map(float, self.pipe.named_steps['lr'].coef_[0]))),
                    intercept=self.intercept())
        json.dump(card, open(os.path.join(directory, 'risk_model_card.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        return p

    @classmethod
    def load(cls, directory):
        import joblib
        p = os.path.join(directory, 'risk_model.joblib')
        if not os.path.exists(p):
            raise RiskModelMissing(f'Không có artifact risk model tại {p}. Chức năng risk score bị tắt (không giả lập).')
        d = joblib.load(p)
        if d.get('schema') != SCHEMA_VERSION:
            raise RiskModelMissing(f"schema không khớp: {d.get('schema')} != {SCHEMA_VERSION}")
        m = cls(d['feature_names']); m.pipe, m.calib, m.card = d['pipe'], d['calib'], d['card']
        cardp = os.path.join(directory, 'risk_model_card.json')
        if os.path.exists(cardp) and json.load(open(cardp, encoding='utf-8')).get('sha256') != sha256_file(p):
            raise RiskModelMissing('SHA256 của risk_model.joblib không khớp model card (artifact bị sửa).')
        return m


def fit_gbm_comparator(X_df, wrong, groups, feature_names, k=5, seed=0, min_errors=150):
    """So sánh gradient boosting CHỈ khi đủ ca lỗi; nếu không trả None kèm lý do (không ép chạy trên dữ liệu quá nhỏ)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = np.asarray(wrong).astype(int)
    if y.sum() < min_errors:
        return None, f'bỏ qua GBM: chỉ {int(y.sum())} ca lỗi < {min_errors}'
    X = X_df[feature_names].to_numpy(float)
    oof = np.full(len(y), np.nan)
    for tr, te in _folds(y, groups, k, seed):
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, 'ok'


# ------------------------------------------------------------------ khóa ngưỡng cảnh báo
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def _canon(d):
    return json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)


def threshold_for_alert_rate(risk_dev, rate):
    """Ngưỡng sao cho ~rate của dữ liệu PHÁT TRIỂN (OOF) bị cảnh báo: phân vị 1-rate."""
    return float(np.quantile(np.asarray(risk_dev, float), 1 - rate))


def lock_policy(path, content, force=False):
    """Ghi chính sách cảnh báo kèm sha256 nội dung. Không ghi đè khi đã tồn tại (trừ force) -> ngưỡng không bị đổi lặng lẽ."""
    if os.path.exists(path) and not force:
        raise PolicyTamperedError(f'{path} đã tồn tại: ngưỡng đã khóa, không được ghi đè.')
    payload = dict(content=content, content_sha256=hashlib.sha256(_canon(content).encode()).hexdigest(), locked_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(payload, open(path, 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=str)
    return payload['content_sha256']


def load_policy(path):
    if not os.path.exists(path):
        raise RiskModelMissing(f'Chưa có chính sách cảnh báo đã khóa: {path}')
    d = json.load(open(path, encoding='utf-8'))
    if hashlib.sha256(_canon(d['content']).encode()).hexdigest() != d['content_sha256']:
        raise PolicyTamperedError('Nội dung ngưỡng bị chỉnh sửa sau khi khóa (sha256 không khớp).')
    return d['content']


# ------------------------------------------------------------------ cổng phạm vi đầu vào (heuristic, cần được kiểm chứng)
def image_stats(img_uint8, valid=None):
    import cv2
    a = np.asarray(img_uint8)
    g = a.mean(2).astype(np.float32)
    v = np.ones(g.shape, bool) if valid is None else valid
    gv = g[v] if v.any() else g.ravel()
    lap = cv2.Laplacian(g, cv2.CV_32F).var()
    chroma = float(np.abs(a[..., 0].astype(np.float32) - a[..., 1]).mean() + np.abs(a[..., 1].astype(np.float32) - a[..., 2]).mean())
    return np.array([gv.mean(), gv.std(), float(v.mean()), np.log1p(lap), float((gv >= 250).mean()), float((gv <= 5).mean()), chroma], float)


class ImageStatGate:
    """Mahalanobis (co rút) trên thống kê ảnh, ngưỡng = phân vị q trên ảnh phát triển. Chỉ phát hiện bất thường thô (không phải ảnh siêu âm,
    cường độ/độ nét cực đoan) — KHÔNG đảm bảo phát hiện mọi dịch chuyển miền."""
    def __init__(self, q=0.995, shrink=0.1):
        self.q, self.shrink = q, shrink

    def fit(self, stats_dev):
        S = np.asarray(stats_dev, float)
        self.mu = S.mean(0)
        cov = np.cov(S.T) + 1e-6 * np.eye(S.shape[1])
        cov = (1 - self.shrink) * cov + self.shrink * np.diag(np.diag(cov))
        self.prec = np.linalg.inv(cov)
        self.thr = float(np.quantile(self.score(S), self.q))
        return self

    def score(self, S):
        d = np.asarray(S, float) - self.mu
        return np.sqrt(np.einsum('ij,jk,ik->i', d, self.prec, d))

    def in_scope(self, s):
        return bool(self.score(np.asarray(s)[None])[0] <= self.thr)

    def to_dict(self):
        return dict(mu=self.mu.tolist(), prec=self.prec.tolist(), thr=self.thr, q=self.q, shrink=self.shrink)

    @classmethod
    def from_dict(cls, d):
        g = cls(d['q'], d['shrink']); g.mu = np.array(d['mu']); g.prec = np.array(d['prec']); g.thr = d['thr']
        return g
