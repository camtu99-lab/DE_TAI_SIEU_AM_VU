"""Giao thức B0–B6/P1: kiểm thử LOGIC trên bảng tổng hợp (không phải kết quả khoa học)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from src import benchmark as BM, failure_risk as FR, evaluation as EV, selective_prediction as SP
from tests.synth import make_feature_table


def _targets(X, y_unused=None):
    return None


def test_B1_equals_B2_rank_but_B1b_differs_when_threshold_not_half():
    X, y, g = make_feature_table(800, 0)
    thr, T = 0.176, 1.7
    z_raw = np.random.RandomState(1).randn(len(X)) * 2
    X['p_raw'] = 1 / (1 + np.exp(-z_raw)); X['p_mal'] = 1 / (1 + np.exp(-z_raw / T))
    X['thr_margin'] = np.abs(z_raw / T - np.log(thr / (1 - thr)))
    un, src = BM.unlearned_scores(X)
    from scipy.stats import spearmanr
    assert spearmanr(un['B1'], un['B2'])[0] == pytest.approx(1.0)       # T-scaling đơn điệu: cùng thứ hạng
    assert spearmanr(un['B1'], un['B1b'])[0] < 0.9                      # nhưng khoảng cách tới NGƯỠNG thật khác max-softmax
    assert src == 'tta_views' and set(un) == {'B0', 'B1', 'B1b', 'B2', 'B3'}
    wrong = (np.random.RandomState(2).rand(len(X)) < 0.6 * np.exp(-1.2 * X['thr_margin'].to_numpy())).astype(int)   # lỗi tập trung gần NGƯỠNG quyết định
    assert roc_auc_score(wrong, un['B1b']) > roc_auc_score(wrong, un['B1'])   # B1b là thước đo đúng khi thr != 0.5


def test_b3_uses_true_ensemble_when_available():
    X, y, g = make_feature_table(100, 1)
    X['ens_js'] = np.linspace(0, 1, len(X))
    un, src = BM.unlearned_scores(X)
    assert src == 'ensemble' and np.allclose(un['B3'], X['ens_js'])


def test_B0_random_has_chance_performance_and_is_seeded():
    X, y, g = make_feature_table(2000, 2)
    a, _ = BM.unlearned_scores(X, seed=3); b, _ = BM.unlearned_scores(X, seed=3)
    assert np.array_equal(a['B0'], b['B0'])
    assert abs(roc_auc_score(y, a['B0']) - 0.5) < 0.06


def test_feature_sets_drop_missing_columns_and_record_duplicates():
    X, y, g = make_feature_table(500, 3)
    sets, notes = BM.feature_sets(X)
    assert 'ens_js' not in sets['P1_full'] and 'entropy' not in sets['P1_full'] and 'conf' not in sets['P1_full']
    assert notes['P1_full'].keys() >= {'entropy', 'conf'}
    X['ens_js'] = 0.1 * np.abs(np.random.RandomState(0).randn(len(X)))
    assert 'ens_js' not in BM.feature_sets(X)[0]['P1_full']                      # mặc định: tương thích triển khai (1 checkpoint)
    assert 'ens_js' in BM.feature_sets(X, deploy_compatible=False)[0]['P1_full']
    for k, v in sets.items():
        assert 'cf_missing' not in v or k != 'B4_conf_only'


def test_nested_cv_no_leak_with_shuffled_labels_and_finds_signal():
    X, y, g = make_feature_table(1500, 4)
    sets, _ = BM.feature_sets(X)
    sub = {k: sets[k] for k in ('B4_conf_only', 'P1_full')}
    sc, chosen = BM.nested_oof_scores(X, sub, y, g, k=4, seed=0, C_grid=(0.03, 0.3))
    assert roc_auc_score(y, sc['P1_full']) > roc_auc_score(y, sc['B4_conf_only']) > 0.55        # tín hiệu counterfactual được phát hiện
    ys = np.random.RandomState(5).permutation(y)
    sc2, _ = BM.nested_oof_scores(X, sub, ys, g, k=4, seed=0, C_grid=(0.03, 0.3))
    assert abs(roc_auc_score(ys, sc2['P1_full']) - 0.5) < 0.07
    assert all(len(c) == 4 for c in chosen.values())


def test_cf_features_add_nothing_when_no_cf_signal():
    """Âm tính chuẩn: khi nhãn lỗi KHÔNG phụ thuộc đặc trưng CF, P1 không được vượt B6 một cách giả tạo."""
    X, y, g = make_feature_table(2500, 6, cf_signal=False)
    sets, _ = BM.feature_sets(X)
    sc, _ = BM.nested_oof_scores(X, {k: sets[k] for k in ('B6_unc_stab_no_cf', 'P1_full')}, y, g, k=5, seed=1, C_grid=(0.03, 0.3))
    assert roc_auc_score(y, sc['P1_full']) - roc_auc_score(y, sc['B6_unc_stab_no_cf']) < 0.02


def test_paired_comparisons_holm_and_effect():
    X, y, g = make_feature_table(1200, 7)
    sets, _ = BM.feature_sets(X)
    sc, _ = BM.nested_oof_scores(X, {k: sets[k] for k in ('B4_conf_only', 'P1_full')}, y, g, k=4, seed=0, C_grid=(0.3,))
    un, _ = BM.unlearned_scores(X)
    scores = {'P1': sc['P1_full'], 'B4': sc['B4_conf_only'], 'B1': un['B1'], 'B3': un['B3'], 'B6': un['B3']}
    wrong = y; is_fn = (y == 1) & (np.random.RandomState(0).rand(len(y)) < 0.5)
    df = BM.paired_comparisons(scores, wrong, is_fn, g, B=200, seed=0)
    assert len(df) == 8 and {'p_holm', 'effect_z', 'verdict'} <= set(df.columns)
    assert (df.p_holm >= df.p - 1e-12).all()                            # Holm không bao giờ làm giảm p
    r = df[(df.A == 'P1') & (df.B == 'B1') & (df.metric == 'error_auprc')].iloc[0]
    assert r.delta > 0 and r.ci_lo < r.delta < r.ci_hi


def test_alert_policy_uses_dev_only_and_status_counts():
    risk = np.linspace(0, 1, 1000)
    pol = BM.make_alert_policy(risk, 0.2, 0.1)
    assert pol['t_review'] <= pol['t_high'] and pol['t_high'] == pytest.approx(0.9, abs=2e-3) and 'test' in pol['source'] and 'không dùng test' in pol['source']
    counts, st = BM.status_counts(risk, pol)
    assert counts[SP.STATUS_HIGH] == pytest.approx(100, abs=2) and counts[SP.STATUS_REVIEW] == pytest.approx(100, abs=2)
    c2, _ = BM.status_counts(risk, pol, in_scope=np.arange(1000) < 500)
    assert c2[SP.STATUS_OOS] == 500


def test_risk_calibration_report_distinguishes_miscalibration():
    r = np.random.RandomState(0).rand(5000); y = (np.random.RandomState(1).rand(5000) < r).astype(int)
    assert BM.risk_calibration_report(r, y)['ece'] < 0.03
    assert BM.risk_calibration_report(r ** 0.3, y)['ece'] > 0.1


def test_protocol_end_to_end_synthetic():
    X, y, g = make_feature_table(1400, 8)
    dev = X.iloc[:1000].reset_index(drop=True); ev = X.iloc[1000:].reset_index(drop=True)
    yd, ye = y[:1000], y[1000:]; gd, ge = g[:1000], g[1000:]
    fn_d = (yd == 1) & (np.random.RandomState(1).rand(1000) < 0.4); fn_e = (ye == 1) & (np.random.RandomState(2).rand(400) < 0.4)
    out = BM.run_protocol(dev, yd, fn_d, gd, np.array(['train'] * 1000), {'heldout': ev}, {'heldout': ye}, {'heldout': fn_e}, {'heldout': ge},
                          seed=0, k=3, B=60, include_ablations=False)
    t = out['dev_table'].set_index('method')
    assert {'B0', 'B1', 'B1b', 'B2', 'B3', 'B4', 'B5', 'B6', 'P1'} <= set(t.index)
    assert t.loc['P1', 'error_auprc'] > t.loc['B0', 'error_auprc']
    assert out['policy']['t_review'] <= out['policy']['t_high']
    h = out['held_out']['heldout']
    assert h['table'].set_index('method').loc['P1', 'error_auroc'] > 0.55 and 'ece' in h['risk_calibration']
    with pytest.raises(FR.TestLeakageError):                            # dev có nhãn 'test' => bị chặn
        BM.run_protocol(dev, yd, fn_d, gd, np.array(['test'] * 1000), {'h': ev}, {'h': ye}, {'h': fn_e}, {'h': ge}, k=3, B=20, include_ablations=False)
