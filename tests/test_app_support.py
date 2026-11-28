import json, os
import numpy as np
import pandas as pd
import pytest
from src import app_support as AS, report_export as RE, segmentation as SG, pipeline as PL
from tests.synth import make_us_image, lesion_model, center_mask


def _setup():
    img, _ = make_us_image(np.random.default_rng(0))
    reg = SG.build_regions(center_mask(), np.ones((256, 256), bool), 12)
    return img, reg


def test_interventions_respect_valid_ranges_and_regions():
    img, reg = _setup()
    out, ch = AS.interactive_intervention(img, reg, 'lesion', 'blur', 4.0)
    assert (out[~ch] == img[~ch]).all() and (out != img).any()                    # chỉ vùng chọn bị đổi
    out, ch = AS.interactive_intervention(img, reg, 'bg', 'permute', 0.5, seed=1)
    assert abs(ch.sum() / reg.background.sum() - 0.5) < 0.02 and (out[~ch] == img[~ch]).all()
    for bad in (('lesion', 'blur', 99.0), ('lesion', 'replace', 1.5), ('lesion', 'permute', 0.0), ('lesion', 'nope', 1.0)):
        with pytest.raises(ValueError):
            AS.interactive_intervention(img, reg, *bad)
    empty = SG.build_regions(np.zeros((256, 256), np.uint8), np.ones((256, 256), bool), 12)
    with pytest.raises(ValueError):
        AS.interactive_intervention(img, empty, 'lesion', 'blur', 3.0)


def test_logit_prob_change_reports_real_model_output():
    img, reg = _setup()
    out, _ = AS.interactive_intervention(img, reg, 'lesion', 'replace', 1.0, seed=0)
    r = AS.logit_prob_change(lesion_model(), img, out, 1.0, 0.5)
    assert r['dz'] == pytest.approx(r['z1'] - r['z0']) and 0 <= r['p0'] <= 1 and abs(r['dz']) > 0


def test_case_filter_and_prediction_schema():
    df = pd.DataFrame(dict(id=range(6), y=[1, 1, 0, 0, 1, 0], pred=[1, 0, 1, 0, 0, 1], p_mal=[.9, .05, .97, .1, .4, .6]))
    assert list(AS.case_filter(df, 'FN').id) == [1, 4] and list(AS.case_filter(df, 'FP').id) == [2, 5]
    assert list(AS.case_filter(df, 'TP').id) == [0] and list(AS.case_filter(df, 'TN').id) == [3]
    assert list(AS.case_filter(df, 'high_conf_error').id) == [1, 2]
    with pytest.raises(ValueError):
        AS.validate_prediction_table(df)


def test_v4_status_reports_not_run_honestly():
    s = AS.v4_status()
    assert 'deploy_bundle' in s and set(AS.V4_FILES) <= set(s)
    if not os.path.exists(os.path.join(AS.V4_DIR, 'RUN_MANIFEST.json')):
        assert not AS.v4_has_results()


def test_runtime_without_deploy_returns_reason_not_fake(tmp_path):
    b, p, msgs = AS.load_runtime(str(tmp_path))
    assert b is None and p is None and msgs and 'gói triển khai' in msgs[0]


def test_report_export_contains_required_sections_and_no_fake_risk():
    img, _ = make_us_image(np.random.default_rng(0))
    rec = dict(images={'Ảnh gốc': img, 'Mask': np.stack([center_mask() * 255] * 3, -1)}, model={'version': 'V4', 'T': 1.2},
               results={'p_malignant': 0.4}, risk=None, limitations=['mẫu nhỏ'], warnings=['không phải chẩn đoán'])
    h = RE.build_report(rec)
    for s in ('KHÔNG phải thiết bị chẩn đoán', 'data:image/png;base64', 'Risk score KHÔNG được tính', 'mẫu nhỏ', 'Giới hạn', 'Mô hình và phiên bản'):
        assert s in h
    rec['risk'] = {'risk': 0.31, 'status': 'Cần xem xét lại'}
    h2 = RE.build_report(rec)
    assert 'Risk score KHÔNG được tính' not in h2 and 'KHÔNG phải xác suất ác tính' in h2
    assert json.loads(RE.build_json(rec))['risk']['risk'] == 0.31 and 'images' not in json.loads(RE.build_json(rec))
    assert '<script' not in RE.build_report(dict(rec, warnings=['<script>alert(1)</script>']))            # thoát HTML
