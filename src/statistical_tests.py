"""Thống kê: bootstrap theo cụm (bệnh nhân), bootstrap ghép cặp, kiểm định hoán vị theo cụm, hiệu chỉnh đa so sánh, effect size, CI Wilson.
Khi KHÔNG có mã bệnh nhân hợp lệ (BUSI), nhóm phải là cụm gần-trùng; nếu không, CI là CI mức ảnh và PHẢI được ghi rõ là lạc quan."""
import numpy as np


def _cluster_index(groups):
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    return members


def cluster_bootstrap_indices(groups, rng):
    """Một mẫu bootstrap: lấy có hoàn lại các CỤM (bệnh nhân), nối mọi ảnh của cụm."""
    members = _cluster_index(groups)
    pick = rng.integers(0, len(members), size=len(members))
    return np.concatenate([members[i] for i in pick])


def bootstrap_ci(metric_fn, groups, B=1000, seed=0, alpha=0.05):
    """CI percentile cho metric_fn(idx) -> float (hoặc NaN nếu không xác định; NaN bị bỏ và đếm lại)."""
    rng = np.random.default_rng(seed)
    base = metric_fn(np.arange(len(groups)))
    vals = []
    for _ in range(B):
        v = metric_fn(cluster_bootstrap_indices(groups, rng))
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if len(vals) < max(20, 0.5 * B):
        return dict(estimate=float(base), lo=float('nan'), hi=float('nan'), n_valid=int(len(vals)))
    return dict(estimate=float(base), lo=float(np.quantile(vals, alpha / 2)), hi=float(np.quantile(vals, 1 - alpha / 2)),
                se=float(vals.std(ddof=1)), n_valid=int(len(vals)))


def paired_cluster_bootstrap(metric_a, metric_b, groups, B=2000, seed=0, alpha=0.05):
    """Δ = metric_a - metric_b trên CÙNG các mẫu bootstrap cụm. p hai phía = 2*min(P(Δ*<=0), P(Δ*>=0)) (bootstrap, xấp xỉ).
    Trả cả effect size chuẩn hoá Δ/SE_boot."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    d0 = metric_a(np.arange(n)) - metric_b(np.arange(n))
    ds = []
    for _ in range(B):
        idx = cluster_bootstrap_indices(groups, rng)
        a, b = metric_a(idx), metric_b(idx)
        if np.isfinite(a) and np.isfinite(b):
            ds.append(a - b)
    ds = np.asarray(ds)
    if len(ds) < 20:
        return dict(delta=float(d0), lo=float('nan'), hi=float('nan'), p=float('nan'), z=float('nan'), n_valid=int(len(ds)))
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    se = ds.std(ddof=1)
    return dict(delta=float(d0), lo=float(np.quantile(ds, alpha / 2)), hi=float(np.quantile(ds, 1 - alpha / 2)), p=float(min(1.0, max(p, 1.0 / (len(ds) + 1)))),
                se=float(se), z=float(d0 / se) if se > 0 else float('nan'), n_valid=int(len(ds)))


def paired_cluster_sign_flip_test(score_a, score_b, metric_of_scores, groups, B=2000, seed=0):
    """Kiểm định hoán vị ghép cặp: với mỗi cụm, hoán đổi ngẫu nhiên (score_a, score_b); p = P(|Δ*| >= |Δ_obs|) có hiệu chỉnh +1.
    H0: hai điểm số có cùng phân phối ghép cặp trong cụm. metric_of_scores(score) -> float."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    members = _cluster_index(groups)
    obs = metric_of_scores(a) - metric_of_scores(b)
    cnt = 0
    for _ in range(B):
        flip = rng.random(len(members)) < 0.5
        aa, bb = a.copy(), b.copy()
        for f, m in zip(flip, members):
            if f:
                aa[m], bb[m] = b[m], a[m]
        d = metric_of_scores(aa) - metric_of_scores(bb)
        cnt += abs(d) >= abs(obs) - 1e-12
    return dict(delta=float(obs), p=float((cnt + 1) / (B + 1)))


def holm_bonferroni(pvals, alpha=0.05):
    """Holm step-down. Trả (p_adj, reject)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj, adj < alpha


def benjamini_hochberg(pvals, alpha=0.05):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = prev
    return adj, adj < alpha


def wilson_ci(k, n, alpha=0.05):
    """CI Wilson cho tỉ lệ (dùng cho FN Alert Recall khi n_FN nhỏ)."""
    if n == 0:
        return float('nan'), float('nan')
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0, c - h)), float(min(1, c + h))


def min_detectable_n(p_null, p_alt, alpha=0.05, power=0.8):
    """Cỡ mẫu FN (xấp xỉ chuẩn, một mẫu) để phân biệt recall p_alt với p_null — dùng để chỉ ra giới hạn công suất của tập test nhỏ."""
    from scipy.stats import norm
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = (za * np.sqrt(p_null * (1 - p_null)) + zb * np.sqrt(p_alt * (1 - p_alt))) ** 2
    return int(np.ceil(num / (p_alt - p_null) ** 2))
