import json, os
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from src import failure_risk as FR
from tests.synth import make_feature_table

P1 = [c for c in FR.FEATURE_SETS['P1_full'] if not c.startswith('ens_')]      # bảng tổng hợp không có ensemble thật


def _split(n, test=False):
    return np.array(['train'] * n)


def test_detects_planted_signal_and_shuffled_labels_chance():
    X, y, g = make_feature_table(1500, 0)
    oof = FR.oof_risk(X[P1].to_numpy(float), y, g, C=0.3)
    assert roc_auc_score(y, oof) > 0.65
    ys = np.random.RandomState(9).permutation(y)
    oof2 = FR.oof_risk(X[P1].to_numpy(float), ys, g, C=0.3)
    assert abs(roc_auc_score(ys, oof2) - 0.5) < 0.07                    # nhãn xáo trộn -> ~ngẫu nhiên: không có rò rỉ nhãn


def test_test_labels_cannot_be_used_for_training():
    X, y, g = make_feature_table(300, 1)
    with pytest.raises(FR.TestLeakageError):
        FR.FailureRiskModel(P1).fit(X, y, g, np.array(['train'] * 299 + ['test']))
    with pytest.raises(FR.TestLeakageError):
        FR.FailureRiskModel(P1).fit(X, y, g, np.array(['external'] * 300))
    FR.FailureRiskModel(P1, C=0.3).fit(X, y, g, np.array(['train'] * 150 + ['val'] * 150))


def test_too_few_errors_refused_not_faked():
    X, y, g = make_feature_table(300, 2)
    y2 = np.zeros_like(y); y2[:3] = 1
    with pytest.raises(ValueError):
        FR.FailureRiskModel(P1, C=0.3).fit(X, y2, g, _split(300))


def test_artifact_roundtrip_missing_and_tamper(tmp_path):
    X, y, g = make_feature_table(600, 3)
    m = FR.FailureRiskModel(P1, C=0.3).fit(X, y, g, _split(600))
    d = str(tmp_path / 'risk')
    m.save(d)
    m2 = FR.FailureRiskModel.load(d)
    assert np.allclose(m.predict_risk(X), m2.predict_risk(X), atol=1e-12)
    card = json.load(open(os.path.join(d, 'risk_model_card.json')))
    assert card['n'] == 600 and card['trained_on_splits'] == ['train'] and len(card['sha256']) == 64 and set(card['coefficients']) == set(P1)
    with pytest.raises(FR.RiskModelMissing):                            # thiếu artifact => KHÔNG giả lập
        FR.FailureRiskModel.load(str(tmp_path / 'nonexistent'))
    with open(os.path.join(d, 'risk_model.joblib'), 'ab') as f:
        f.write(b'x')
    with pytest.raises(FR.RiskModelMissing):                            # artifact bị sửa => sha256 không khớp
        FR.FailureRiskModel.load(d)
    with pytest.raises(FR.RiskModelMissing):
        FR.FailureRiskModel(P1).predict_risk(X)                          # chưa huấn luyện


def test_contributions_sum_to_logit_and_nan_imputed():
    X, y, g = make_feature_table(600, 4)
    m = FR.FailureRiskModel(P1, C=0.3).fit(X, y, g, _split(600))
    row = X.iloc[0]
    c = m.contributions(row)
    raw = m.predict_raw(X.iloc[[0]])[0]
    assert (c.sum() + m.intercept()) == pytest.approx(np.log(raw / (1 - raw)), abs=1e-8)
    Xn = X.copy(); Xn.loc[:, ['lesion_sens', 'cf_specificity', 'bg_sens']] = np.nan   # mask rỗng -> CF thiếu: vẫn hữu hạn
    assert np.isfinite(m.predict_risk(Xn)).all()
    with pytest.raises(KeyError):
        m.predict_raw(X.drop(columns=['bg_sens']))


def test_risk_is_calibrated_on_heldout():
    X, y, g = make_feature_table(4000, 5, n_groups=3000)
    tr = np.arange(len(y)) < 2800
    m = FR.FailureRiskModel(P1, C=0.3).fit(X[tr], y[tr], g[tr], _split(tr.sum()))
    p = m.predict_risk(X[~tr])
    assert abs(p.mean() - y[~tr].mean()) < 0.05
    hi = p > np.quantile(p, 0.8)
    assert abs(p[hi].mean() - y[~tr][hi].mean()) < 0.12


def test_ablation_definitions_and_feature_groups():
    s = set(P1)
    for name, drop in FR.ABLATIONS.items():
        assert set(drop) & s, name                                       # mỗi ablation thật sự bỏ ít nhất một đặc trưng của P1
    assert set(FR.G_CF) <= s and set(FR.G_CONF) <= s
    assert not (set(FR.FEATURE_SETS['B6_unc_stab_no_cf']) & set(FR.G_CF))      # B6 không có counterfactual
    assert not (set(FR.FEATURE_SETS['B5_cf_only']) & (set(FR.G_CONF) | set(FR.G_UNC) | set(FR.G_STAB)))
    assert not (set(FR.FEATURE_SETS['B4_conf_only']) & set(FR.G_CF))


def test_dedupe_rank_equivalent_binary_confidence_entropy_margin():
    X, y, g = make_feature_table(500, 6)
    keep, dropped = FR.dedupe_rank_equivalent(X, ['margin_abs', 'entropy', 'conf', 'thr_margin'])
    assert keep[0] == 'margin_abs' and set(dropped) == {'entropy', 'conf'}     # thr_margin KHÔNG tương đương khi thr != 0.5


def test_gbm_only_when_enough_errors():
    X, y, g = make_feature_table(400, 7)
    oof, msg = FR.fit_gbm_comparator(X, y, g, P1, min_errors=int(y.sum()) + 1)
    assert oof is None and 'bỏ qua' in msg
    oof, msg = FR.fit_gbm_comparator(X, y, g, P1, min_errors=5)
    assert oof is not None and np.isfinite(oof).all()


def test_policy_lock_cannot_be_overwritten_or_tampered(tmp_path):
    p = str(tmp_path / 'policy.json')
    FR.lock_policy(p, dict(t_review=0.3, t_high=0.5))
    with pytest.raises(FR.PolicyTamperedError):
        FR.lock_policy(p, dict(t_review=0.1, t_high=0.2))
    assert FR.load_policy(p)['t_high'] == 0.5
    d = json.load(open(p)); d['content']['t_high'] = 0.9; json.dump(d, open(p, 'w'))
    with pytest.raises(FR.PolicyTamperedError):
        FR.load_policy(p)
    with pytest.raises(FR.RiskModelMissing):
        FR.load_policy(str(tmp_path / 'none.json'))


def test_threshold_from_dev_quantile_only():
    r = np.linspace(0, 1, 1001)
    assert FR.threshold_for_alert_rate(r, 0.2) == pytest.approx(0.8, abs=1e-3)


def test_image_stat_gate_flags_extreme_input():
    rng = np.random.default_rng(0)
    base = np.stack([np.array([100, 25, 0.9, 5.0, 0.0, 0.02, 1.0]) + rng.standard_normal(7) * 0.1 * np.array([20, 5, 0.05, 0.5, 0.01, 0.01, 0.5]) for _ in range(500)])
    g = FR.ImageStatGate(q=0.99).fit(base)
    assert g.in_scope(base[0]) and not g.in_scope(np.array([250, 1, 0.1, 0.0, 0.9, 0.0, 80.0]))
    g2 = FR.ImageStatGate.from_dict(g.to_dict())
    assert g2.in_scope(base[0]) == g.in_scope(base[0]) and g2.thr == g.thr
