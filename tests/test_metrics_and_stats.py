import numpy as np
import pytest
from itertools import product
from src import selective_prediction as SP, statistical_tests as ST, evaluation as EV, uncertainty as UN
from src._paths import ensure_v3
ensure_v3()
import bus_trust as T3                                    # V3 (không cần torch)


def test_calibration_consistency_with_v3_metrics():
    rng = np.random.default_rng(0)
    p = rng.random(500); y = (rng.random(500) < p).astype(int)
    P = np.stack([1 - p, p], 1)
    assert EV.ece_binary(p, y) == pytest.approx(T3.ece(P, y), abs=1e-12)
    assert EV.brier_binary(p, y) * 2 == pytest.approx(T3.brier(P, y), abs=1e-12)      # V3 Brier cộng hai lớp = 2× Brier nhị phân
    assert EV.nll_binary(p, y) == pytest.approx(T3.nll(P, y), abs=1e-9)
    z = rng.standard_normal((300, 2)) * 3; yy = rng.integers(0, 2, 300)
    t = T3.fit_temperature(z, yy)
    assert T3.nll(T3.softmax(z, t), yy) <= T3.nll(T3.softmax(z, 1.0), yy) + 1e-9


def test_uncertainty_matches_v3_softmax_and_js():
    z = np.random.default_rng(1).standard_normal((50, 2))
    assert np.allclose(UN.softmax(z, 2.0), T3.softmax(z, 2.0))
    P = np.random.default_rng(2).dirichlet([1, 1], size=(4, 30))
    assert np.allclose(UN.js_disagreement(P), T3.mutual_information(P))


def test_binary_entropy_and_confidence_are_rank_equivalent():
    """Lưu ý đề bài: nhị phân thì entropy và max-confidence cho CÙNG thứ hạng -> B1 (conf) và 'entropy' không phải hai điểm số độc lập."""
    p = np.random.default_rng(3).random(1000)
    P = np.stack([1 - p, p], 1)
    from scipy.stats import spearmanr
    assert spearmanr(UN.entropy(P), -UN.confidence(P))[0] == pytest.approx(1.0, abs=1e-12)


def test_temperature_scaling_preserves_confidence_ranking():
    """B1 (softmax thô) và B2 (hiệu chuẩn) có CÙNG thứ hạng -> chỉ khác ở ECE/Brier/NLL và ngưỡng cố định, không khác AUPRC lỗi."""
    z = np.random.default_rng(4).standard_normal((500, 2)) * 2
    from scipy.stats import spearmanr
    assert spearmanr(UN.confidence(UN.softmax(z, 1.0)), UN.confidence(UN.softmax(z, 3.0)))[0] == pytest.approx(1.0, abs=1e-9)


def _brute_aurc(score, wrong):
    o = np.argsort(score, kind='stable'); w = np.asarray(wrong)[o]
    return float(np.mean([w[:k].mean() for k in range(1, len(w) + 1)]))


def _brute_augrc(score, wrong):
    o = np.argsort(score, kind='stable'); w = np.asarray(wrong)[o]; n = len(w)
    return float(np.mean([w[:k].sum() / n for k in range(1, n + 1)]))


def test_aurc_augrc_match_brute_force_and_bounds():
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(20, 80)); s = rng.random(n); w = (rng.random(n) < 0.3).astype(int)
        assert SP.aurc(s, w) == pytest.approx(_brute_aurc(s, w)) and SP.augrc(s, w) == pytest.approx(_brute_augrc(s, w))
        assert SP.oracle_aurc(w) <= SP.aurc(s, w) + 1e-12
    w = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0]); perfect = w.astype(float)
    assert SP.aurc(perfect, w) == pytest.approx(SP.oracle_aurc(w))


def test_same_alert_rate_and_fn_recall():
    wrong = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0]); fn = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    s = np.array([.9, .1, .2, .8, .3, .1, .05, .2, .1, .3])
    a = SP.alert_set(s, 0.2)
    assert a.sum() == 2 and a[0] and a[3]
    m = SP.alert_metrics(a, wrong, fn)
    assert m['fn_alert_recall'] == pytest.approx(0.5) and m['error_recall'] == pytest.approx(2 / 3) and m['alert_rate'] == pytest.approx(0.2)
    r = SP.random_referral_expectation(wrong, fn, 0.2, n_draws=3000)
    assert r['error_recall'] == pytest.approx(0.2, abs=0.03)       # B0: recall kỳ vọng = tỉ lệ cảnh báo


def test_status_decision():
    assert SP.decide_status(0.9, .3, .6) == SP.STATUS_HIGH and SP.decide_status(0.4, .3, .6) == SP.STATUS_REVIEW
    assert SP.decide_status(0.1, .3, .6) == SP.STATUS_NONE and SP.decide_status(0.9, .3, .6, in_scope=False) == SP.STATUS_OOS


def test_cluster_bootstrap_resamples_whole_patients():
    groups = np.repeat(np.arange(20), 3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = ST.cluster_bootstrap_indices(groups, rng)
        c = np.bincount(groups[idx], minlength=20)
        assert set(c % 3) == {0}                                  # mỗi bệnh nhân vào mẫu nguyên khối (bội của 3)


def test_bootstrap_ci_coverage_and_cluster_wider_than_iid():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(60), 4); eff = np.repeat(rng.standard_normal(60), 4)       # tương quan trong cụm
    x = eff + 0.3 * rng.standard_normal(240)
    f = lambda i: float(x[i].mean())
    cl = ST.bootstrap_ci(f, g, B=600, seed=0); iid = ST.bootstrap_ci(f, np.arange(240), B=600, seed=0)
    assert (cl['hi'] - cl['lo']) > 1.3 * (iid['hi'] - iid['lo'])                         # bootstrap ảnh coi như độc lập -> CI quá hẹp


def test_paired_bootstrap_detects_real_difference_and_not_null():
    rng = np.random.default_rng(2)
    n = 300; g = np.arange(n); y = (rng.random(n) < .3).astype(int)
    good = y + 1.0 * rng.standard_normal(n); bad = rng.standard_normal(n); same = good + 0.0
    fa = lambda s: (lambda i: EV.safe_auprc(y[i], s[i]))
    r = ST.paired_cluster_bootstrap(fa(good), fa(bad), g, B=400)
    assert r['delta'] > 0.2 and r['lo'] > 0 and r['p'] < 0.01
    r0 = ST.paired_cluster_bootstrap(fa(good), fa(same), g, B=200)
    assert r0['delta'] == 0 and r0['lo'] <= 0 <= r0['hi']


def test_sign_flip_permutation_null_calibration():
    rng = np.random.default_rng(3); ps = []
    for _ in range(40):
        n = 80; y = (rng.random(n) < .4).astype(int); a, b = rng.standard_normal(n), rng.standard_normal(n)
        ps.append(ST.paired_cluster_sign_flip_test(a, b, lambda s: EV.safe_auroc(y, s), np.arange(n), B=100, seed=0)['p'])
    assert np.mean(np.array(ps) < 0.05) < 0.2                   # dưới H0 tỉ lệ dương tính giả không bị thổi phồng


def test_multiple_comparison_corrections():
    p = np.array([0.001, 0.02, 0.04, 0.3])
    adj, rej = ST.holm_bonferroni(p)
    assert np.allclose(adj, [0.004, 0.06, 0.08, 0.3]) and list(rej) == [True, False, False, False]
    bh, r2 = ST.benjamini_hochberg(p)
    assert np.all(bh >= p - 1e-12) and np.all(bh <= 1)
    assert np.all(adj >= bh - 1e-12)                            # Holm bảo thủ hơn BH (kiểm soát FWER vs FDR)


def test_wilson_and_power_expose_small_test_limit():
    lo, hi = ST.wilson_ci(7, 10)
    assert lo < 0.7 < hi and (hi - lo) > 0.4                    # với 10 ca FN, khoảng tin cậy rộng > 40 điểm %
    assert 10 < ST.min_detectable_n(0.2, 0.5) < 30              # ~17 ca FN để phân biệt recall 50% với cảnh báo ngẫu nhiên 20% (test BUS-BRA chỉ có ~7 FN)
