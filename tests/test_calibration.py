import numpy as np

import bus_trust as T
from src import inference as INF


def test_calibration(rt):
    Tt = rt.cfg['temperature']
    raw = np.array([0.3, 0.7])
    expected = T.softmax(np.log(np.clip(raw, 1e-12, 1)), Tt)
    assert np.allclose(INF.calibrate(raw, Tt), expected, atol=1e-15)
    z = np.log(raw) / Tt
    assert np.allclose(INF.calibrate(raw, Tt), np.exp(z) / np.exp(z).sum(), atol=1e-12)


def test_temperature_one_is_identity_and_T_gt_1_softens():
    raw = np.array([0.1, 0.9])
    assert np.allclose(INF.calibrate(raw, 1.0), raw, atol=1e-12)
    assert INF.calibrate(raw, 2.0).max() < raw.max()


def test_result_uses_locked_temperature(rt, synth_result):
    assert np.allclose(synth_result.p_cal, INF.calibrate(synth_result.raw_probs, rt.cfg['temperature']), atol=1e-12)
    assert abs(synth_result.p_cal.sum() - 1) < 1e-9


def test_locked_threshold(rt):
    thr = rt.cfg['threshold']
    assert abs(thr - 0.5) > 0.1
    lab, *_ = INF.decide([1 - (thr + 1e-4), thr + 1e-4], thr, rt.cfg['abstain_threshold'])
    assert lab == 'malignant'
    lab, *_ = INF.decide([1 - (thr - 1e-4), thr - 1e-4], thr, rt.cfg['abstain_threshold'])
    assert lab == 'benign'
    lab, *_ = INF.decide([0.6, 0.4], thr, rt.cfg['abstain_threshold'])      # 0.4 > ngưỡng khóa 0.1756 -> malignant dù < 0.5
    assert lab == 'malignant'


def test_abstain_threshold(rt):
    a = rt.cfg['abstain_threshold']
    assert INF.decide([1 - 0.5, 0.5], rt.cfg['threshold'], a)[3] is True
    assert INF.decide([1 - 0.95, 0.95], rt.cfg['threshold'], a)[3] is False
    just_above = INF.decide([a + 1e-6, 1 - a - 1e-6], rt.cfg['threshold'], a)
    assert just_above[3] is False
    just_below = INF.decide([a - 1e-6, 1 - a + 1e-6], rt.cfg['threshold'], a)
    assert just_below[3] is True


def test_result_decision_consistent_with_config(rt, synth_result):
    lab, idx, conf, ab = INF.decide(synth_result.p_cal, rt.cfg['threshold'], rt.cfg['abstain_threshold'])
    assert (synth_result.label, synth_result.abstain) == (lab, ab)
