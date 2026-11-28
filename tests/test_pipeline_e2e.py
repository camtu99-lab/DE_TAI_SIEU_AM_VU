"""Pipeline dùng chung notebook/app, với mô hình GIẢ để kiểm thử cơ chế (không phải kết quả khoa học)."""
import json, os
import numpy as np
import pandas as pd
import pytest
from src import pipeline as PL, failure_risk as FR, preprocessing as PP, selective_prediction as SP, benchmark as BM
from tests.synth import make_us_image, lesion_model, background_shortcut_model


def _imgs(n, seed=0):
    r = np.random.default_rng(seed)
    return [make_us_image(r, contrast=float(r.uniform(0.35, 0.9)), pad=int(r.integers(0, 12)))[0] for _ in range(n)]


def test_extract_features_row_has_all_groups_and_no_gt():
    img = _imgs(1)[0]
    row, extra = PL.extract_features(img, lesion_model(), T=1.0, thr=0.3, seed=1)
    for c in ['p_mal', 'p_raw', 'thr_margin', 'view_js', 'prediction_stability', 'mask_consistency', 'lesion_sens', 'bg_sens', 'cf_status']:
        assert c in row, c
    assert row['cf_status'] == 'ok' and extra['mask'] is not None
    assert row['thr_margin'] == pytest.approx(abs(row['z'] - np.log(0.3 / 0.7)))


def test_no_segmentation_is_reported_not_faked():
    row, extra = PL.extract_features(_imgs(1)[0], background_shortcut_model(), T=1.0, thr=0.5)
    assert row['cf_status'] == 'no_segmentation' and row['mask_empty'] == 1 and not np.isfinite(row.get('lesion_sens', np.nan))


def test_extract_is_deterministic():
    img = _imgs(1)[0]
    a, _ = PL.extract_features(img, lesion_model(), seed=5); b, _ = PL.extract_features(img, lesion_model(), seed=5)
    assert a == b or all((a[k] == b[k]) or (isinstance(a[k], float) and np.isnan(a[k]) and np.isnan(b[k])) for k in a)


def test_members_add_true_ensemble_features():
    img = _imgs(1)[0]
    m2 = lambda ims: dict(logits=np.tile([[0.0, 1.5]], (len(ims), 1)), seg_prob=None, evidence=None)
    row, _ = PL.extract_features(img, lesion_model(), members=[m2], with_cf=False)
    assert {'ens_js', 'ens_dz_sd', 'ens_flip'} <= set(row)


def test_extract_dataframe_and_error_targets():
    imgs = _imgs(6)
    df = PL.extract_dataframe(imgs, [f'i{k}' for k in range(6)], lesion_model(), base_seed=0, thr=0.4)
    assert len(df) == 6 and 'conf' in df and 'entropy' in df and df['id'].tolist()[0] == 'i0'
    t = PL.make_error_targets([1, 0, 1, 0], [0.9, 0.9, 0.1, 0.1], 0.5)
    assert t['wrong'].tolist() == [0, 1, 1, 0] and t['is_fn'].tolist() == [0, 0, 1, 0] and t['is_fp'].tolist() == [0, 1, 0, 0]


def _deploy(tmp, with_risk=True):
    X, y, g = __import__('tests.synth', fromlist=['x']).make_feature_table(700, 0)
    os.makedirs(tmp, exist_ok=True)
    json.dump(dict(img_size=256, temperature=1.0, threshold=0.5, seg_threshold=0.5, min_area_frac=0.0015, seed=0), open(os.path.join(tmp, 'deploy_config.json'), 'w'))
    if with_risk:
        P1 = BM.feature_sets(X)[0]['P1_full']
        m = FR.FailureRiskModel(P1, C=0.3).fit(X, y, g, np.array(['train'] * 700)); m.save(os.path.join(tmp, 'risk'))
        pol = BM.make_alert_policy(m.calib(m.oof_raw))
        FR.lock_policy(os.path.join(tmp, 'alert_policy.locked.json'), pol)
    return tmp


def test_bundle_without_artifacts_disables_risk_with_reason(tmp_path):
    with pytest.raises(FR.RiskModelMissing):
        PL.Bundle(str(tmp_path))                                         # không có deploy_config
    b = PL.Bundle(_deploy(str(tmp_path), with_risk=False))
    assert not b.risk_enabled and 'risk' in b.risk_error.lower()
    out = PL.analyze_array(_imgs(1)[0], lesion_model(), b)
    assert out['risk'] is None and out['status'] is None and 0 <= out['p_malignant'] <= 1            # KHÔNG có điểm giả


def test_bundle_with_artifacts_gives_real_risk_and_status(tmp_path):
    b = PL.Bundle(_deploy(str(tmp_path)))
    assert b.risk_enabled
    out = PL.analyze_array(_imgs(1)[0], lesion_model(), b)
    assert 0 <= out['risk'] <= 1 and out['status'] in (SP.STATUS_NONE, SP.STATUS_HIGH, SP.STATUS_REVIEW, SP.STATUS_OOS)
    assert out['contributions'] is not None and out['inference_seconds'] > 0


def test_tampered_policy_turns_risk_off(tmp_path):
    d = _deploy(str(tmp_path))
    p = os.path.join(d, 'alert_policy.locked.json'); j = json.load(open(p)); j['content']['t_high'] = 0.99; json.dump(j, open(p, 'w'))
    b = PL.Bundle(d)
    assert not b.risk_enabled and 'sha256' in b.risk_error


def test_invalid_image_raises(tmp_path):
    b = PL.Bundle(_deploy(str(tmp_path), with_risk=False))
    for bad in (None, np.zeros((10, 10, 3), np.uint8), np.full((100, 100, 3), 7, np.uint8), np.zeros((50, 50, 3, 1))):
        with pytest.raises(PP.InvalidImageError):
            PL.analyze_array(bad, lesion_model(), b)


def test_empty_mask_gives_out_of_scope_status(tmp_path):
    b = PL.Bundle(_deploy(str(tmp_path)))
    out = PL.analyze_array(_imgs(1)[0], background_shortcut_model(), b)       # không có seg -> không kiểm chứng được
    assert out['status'] == SP.STATUS_OOS and 'no_segmentation' in out['oos_reasons']
