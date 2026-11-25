import json
import shutil

import pandas as pd
import pytest

from src import config, failure_gate as FG, research_results as RR
from src.security import UserFacingError


def test_results_real_files():
    assert RR.check_files() == []
    for n in RR.REQUIRED_COLS:
        assert len(RR.load_table(n)) > 0
    assert set(RR.load_external()) >= {'V3_full'}


def test_kpis_are_read_from_files_not_hardcoded():
    ms = pd.read_csv(config.RESULTS_DIR / 'internal_results_mean_sd.csv')
    r = ms[ms.model == 'V3_full'].iloc[0]
    ks = {k['label']: k for k in RR.kpis('V3_full')}
    assert ks['AUROC']['value'] == f"{r.auroc_mean:.3f} ± {r.auroc_sd:.3f}"
    assert ks['Dice']['value'] == f"{r.dice_mean:.3f} ± {r.dice_sd:.3f}"
    ext = json.loads((config.RESULTS_DIR / 'external_busi_results.json').read_text(encoding='utf-8'))['V3_full']
    assert f"{ext['auroc']:.3f}" in ks['AUROC ngoài (BUSI)']['value']
    for k in ks.values():
        assert k['model'] and k['dataset'] and k['source']


def test_kpis_follow_modified_files(tmp_path):
    shutil.copytree(config.RESULTS_DIR, tmp_path / 'r')
    p = tmp_path / 'r' / 'internal_results_mean_sd.csv'
    df = pd.read_csv(p)
    df.loc[df.model == 'V3_full', 'auroc_mean'] = 0.5
    df.to_csv(p, index=False)
    assert RR.kpis('V3_full', tmp_path / 'r')[0]['value'].startswith('0.500')


def test_missing_column_and_malformed_json_raise(tmp_path):
    shutil.copytree(config.RESULTS_DIR, tmp_path / 'r')
    p = tmp_path / 'r' / 'calibration.csv'
    pd.read_csv(p).drop(columns=['ECE']).to_csv(p, index=False)
    with pytest.raises(UserFacingError):
        RR.load_table('calibration.csv', tmp_path / 'r')
    (tmp_path / 'r' / 'external_busi_results.json').write_text('{oops', encoding='utf-8')
    with pytest.raises(UserFacingError):
        RR.load_external(tmp_path / 'r')
    assert 'calibration.csv' in RR.check_files(tmp_path / 'r') and 'external_busi_results.json' in RR.check_files(tmp_path / 'r')


def test_paired_verdict_derived_from_ci():
    assert 'Chưa đủ bằng chứng' in RR.verdict_from_ci(0.0005, -0.017, 0.018, 'V3_full', 'X6')
    assert 'cao hơn' in RR.verdict_from_ci(0.02, 0.01, 0.03, 'A', 'B') and 'A cao hơn B' in RR.verdict_from_ci(0.02, 0.01, 0.03, 'A', 'B')
    assert 'B cao hơn A' in RR.verdict_from_ci(-0.02, -0.03, -0.01, 'A', 'B')
    t = RR.paired_table('V3_full', 'X6')
    r = t[t.metric == 'auroc'].iloc[0]
    if r.ci_low <= 0 <= r.ci_high:
        assert 'Chưa đủ bằng chứng' in r['kết luận (tính từ CI)']


def test_xai_groups_are_separate_and_value_numeric():
    assert RR.load_table('xai_evaluation.csv')['value'].dtype.kind in 'fi'
    seen = set()
    for keys in RR.XAI_GROUPS.values():
        assert not (set(keys) & seen)
        seen |= set(keys)
    assert 'collapse' not in seen


def test_failure_detector_gate(tmp_path):
    g = FG.evaluate_gate(tmp_path)
    assert g.enabled is False and 'chưa khả dụng' in g.message and len(g.missing) == len(FG.REQUIRED_ARTIFACTS)
    assert config.ENABLE_FAILURE_DETECTOR is False
    assert FG.evaluate_gate().enabled is False


def test_failure_detector_gate_stays_off_even_if_files_exist_but_flag_off(tmp_path):
    for n in FG.REQUIRED_ARTIFACTS:
        if n != 'detector_model':
            (tmp_path / n).write_text('{}', encoding='utf-8')
    (tmp_path / 'detector.pt').write_bytes(b'x')
    assert FG.evaluate_gate(tmp_path).enabled is False


def test_predictions_artifact_absent_and_privacy_rules(tmp_path):
    df, msg = RR.load_predictions()
    assert df is None and 'chưa chứa bảng dự đoán theo từng ảnh' in msg
    base = dict(case_id=['a', 'b'], y_true=[1, 0], p_malignant=[0.8, 0.3], pred=[1, 0], split=['test', 'test'], model=['V3_full'] * 2, seed=[44, 44])
    pd.DataFrame(base).to_csv(tmp_path / 'predictions_per_image.csv', index=False)
    d, m = RR.load_predictions(tmp_path)
    assert d is not None and m == ''
    pd.DataFrame(dict(base, patient_id=['p1', 'p2'])).to_csv(tmp_path / 'predictions_per_image.csv', index=False)
    d, m = RR.load_predictions(tmp_path)
    assert d is None and 'định danh' in m
    assert [RR.outcome(1, 0), RR.outcome(0, 1), RR.outcome(1, 1), RR.outcome(0, 0)] == ['FN', 'FP', 'TP', 'TN']


def test_demo_cases_disabled_without_manifest(tmp_path):
    from src.demo_cases import list_demo_cases
    cases, msg = list_demo_cases(tmp_path)
    assert cases == [] and msg
