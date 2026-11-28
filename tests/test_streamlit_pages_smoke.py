"""Smoke test LOGIC các trang bằng streamlit/plotly GIẢ (stub). KHÔNG thay thế việc chạy streamlit thật (xem báo cáo kiểm thử: NOT RUN)."""
import os, sys, runpy, types
from unittest import mock
import numpy as np
import pytest
from PIL import Image
from src import app_support as AS
from tests.synth import make_us_image, lesion_model
from tests.test_pipeline_e2e import _deploy

ROOT = AS.ROOT
PAGES = ['app.py'] + sorted(f'pages/{f}' for f in os.listdir(os.path.join(ROOT, 'pages')) if f.endswith('.py'))


class Stop(Exception):
    pass


def make_st(upload_choice=1):
    st = mock.MagicMock(name='st')
    st.session_state = {}
    st.stop.side_effect = Stop
    st.columns.side_effect = lambda spec, **k: [st] * (spec if isinstance(spec, int) else len(spec))      # cột dùng chung stub để ghi nhận lời gọi
    st.tabs.side_effect = lambda names: [st for _ in names]
    st.radio.side_effect = lambda label, options, **k: list(options)[upload_choice if label == 'Nguồn ảnh' else 0]
    st.selectbox.side_effect = lambda label, options, **k: list(options)[0]
    st.multiselect.side_effect = lambda label, options, default=None, **k: default or list(options)
    st.slider.side_effect = lambda label, lo=None, hi=None, val=None, *a, **k: (val if val is not None else lo)
    st.button.return_value = False
    st.file_uploader.return_value = None
    st.cache_resource = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    return st


class FakeUI:
    """Dùng ui thật nhưng thay setup để trả stub st."""


def run_page(page, st, plot_stub=True):
    mods = {'streamlit': st}
    if plot_stub:
        px, go = mock.MagicMock(), mock.MagicMock()
        mods.update({'plotly': mock.MagicMock(), 'plotly.express': px, 'plotly.graph_objects': go})
    with mock.patch.dict(sys.modules, mods):
        sys.modules.pop('src.ui', None)
        try:
            runpy.run_path(os.path.join(ROOT, page), run_name='__main__')
        except Stop:
            pass
    return st


@pytest.mark.parametrize('page', PAGES)
def test_every_page_runs_without_artifacts_and_reports_not_run(page, tmp_path, monkeypatch):
    monkeypatch.setattr(AS, 'DEPLOY', str(tmp_path / 'none')); monkeypatch.setattr(AS, 'V4_DIR', str(tmp_path / 'v4'))
    st = run_page(page, make_st())
    calls = ' '.join(str(c) for c in st.markdown.call_args_list)
    assert st.markdown.called                                                   # tiêu đề/hero luôn được vẽ
    if page != 'app.py' and 'Research_Report' not in page and 'Counterfactual' not in page and 'Robustness' not in page:
        assert 'NOT RUN' in calls or 'KHÔNG' in calls or st.info.called or st.warning.called or st.error.called


def _runtime(tmp_path, monkeypatch, risk=True):
    d = _deploy(str(tmp_path / 'deploy'), with_risk=risk)
    from src import pipeline as PL
    bundle = PL.Bundle(d)
    pred = lesion_model()
    wrapper = types.SimpleNamespace(__call__=pred)

    class P:
        def __call__(self, ims): return pred(ims)
        def gradcam(self, img): return np.random.default_rng(0).random((256, 256)).astype(np.float32)
    monkeypatch.setattr(AS, 'load_runtime', lambda deploy_dir=None: (bundle, P(), []))
    img = make_us_image(np.random.default_rng(1))[0]
    p = tmp_path / 's.png'; Image.fromarray(img).save(p)
    monkeypatch.setattr(AS, 'list_samples', lambda: [str(p)])
    return bundle


def test_full_flow_with_stub_model_all_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(AS, 'V4_DIR', str(tmp_path / 'v4'))
    _runtime(tmp_path, monkeypatch, risk=True)
    st = make_st(upload_choice=1)
    sess = {}
    for page in PAGES:
        st = make_st(upload_choice=1); st.session_state = sess
        run_page(page, st)
    assert 'analysis' in sess and sess['analysis']['risk'] is not None
    assert sess['analysis']['status'] in ('Không phát cảnh báo', 'Nguy cơ dự đoán sai cao', 'Cần xem xét lại', 'Đầu vào ngoài phạm vi đã kiểm định')


def test_risk_page_refuses_when_artifacts_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(AS, 'V4_DIR', str(tmp_path / 'v4'))
    _runtime(tmp_path, monkeypatch, risk=False)
    st = make_st(); run_page('pages/3_Failure_Risk_Detection.py', st)
    txt = ' '.join(str(c) for c in st.markdown.call_args_list)
    assert 'NOT RUN' in txt and 'TẮT' in txt
    assert not any('Risk score (xác suất' in str(c) for c in st.metric.call_args_list)       # không hiển thị số risk


def test_failure_case_explorer_reads_real_csv(tmp_path, monkeypatch):
    import pandas as pd
    v4 = tmp_path / 'v4'; v4.mkdir()
    pd.DataFrame(dict(id=range(5), y=[1, 1, 0, 0, 1], p_mal=[.9, .1, .95, .1, .3], pred=[1, 0, 1, 0, 0], wrong=[0, 1, 1, 0, 1], is_fn=[0, 1, 0, 0, 1],
                      risk=[.1, .6, .5, .05, .4], status=['Không phát cảnh báo', 'Nguy cơ dự đoán sai cao', 'Cần xem xét lại', 'Không phát cảnh báo', 'Cần xem xét lại'])).to_csv(v4 / AS.V4_FILES['pred_dev'], index=False)
    monkeypatch.setattr(AS, 'V4_DIR', str(v4))
    st = make_st(); run_page('pages/6_Failure_Case_Explorer.py', st)
    assert st.dataframe.called and any('Số ca' in str(c) for c in st.metric.call_args_list)


def test_cf_and_robustness_pages_reach_real_computation(tmp_path, monkeypatch):
    monkeypatch.setattr(AS, 'V4_DIR', str(tmp_path / 'v4'))
    _runtime(tmp_path, monkeypatch, risk=True)
    sess = {}
    st = make_st(1); st.session_state = sess; run_page('pages/1_Ultrasound_Analysis.py', st)
    for page, label in (('pages/2_Counterfactual_Verification.py', 'Δ logit'), ('pages/4_Robustness_Lab.py', '|Δ P(ác tính)|'), ('pages/3_Failure_Risk_Detection.py', 'Ngưỡng "Cần xem xét lại" (khóa)')):
        st = make_st(1); st.session_state = sess; run_page(page, st)
        assert any(label in str(c) for c in st.metric.call_args_list), page
