import time

import cv2
import numpy as np
import torch

from src import checkpoint as ck, inference as INF, preprocessing as PP
from conftest import make_synthetic


def test_cpu_smoke():
    t0 = time.perf_counter()
    rt = ck.build_runtime(device=torch.device('cpu'))
    assert rt.ok and rt.device.type == 'cpu'
    rgb = make_synthetic()
    ok, buf = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    res = INF.infer_image(rt, PP.decode_image(buf.tobytes()), with_gradcam=True)
    assert np.isfinite(res.raw_logits).all() and np.isfinite(res.p_cal).all()
    assert abs(res.p_cal.sum() - 1) < 1e-9 and res.label in ('benign', 'malignant')
    assert res.mask.shape == (256, 256) and res.evidence.shape == (256, 256) and res.cam.shape == (256, 256)
    assert time.perf_counter() - t0 < 120


def test_streamlit_pages_load_without_exception():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    for f in ['app.py', 'pages/3_Research_Dashboard.py', 'pages/4_Error_Cases.py', 'pages/5_Methodology.py', 'pages/6_About_Limits.py', 'pages/1_Analyze.py']:
        at = AppTest.from_file(str(root / f), default_timeout=180)
        at.run()
        assert not at.exception, (f, [e.value for e in at.exception])


def test_perturbation_page_runs_with_session_result(rt, synth_result):
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    at = AppTest.from_file(str(root / 'pages/2_Perturbation_Lab.py'), default_timeout=240)
    at.session_state['current_result'] = synth_result
    at.run()
    assert not at.exception, [e.value for e in at.exception]
