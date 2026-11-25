"""PARITY: backend Streamlit == inference gốc của notebook (C2.analyze_image_v2 / C2.predict_arrays) trên cùng ảnh + checkpoint."""
import numpy as np
import pytest
import torch

import bus_core as core
import bus_core_v2 as C2
from src import inference as INF
from conftest import make_synthetic

IMAGES = [make_synthetic(300, 400, 0), make_synthetic(256, 256, 5), make_synthetic(480, 320, 9)]


@pytest.mark.parametrize('k', range(len(IMAGES)))
def test_inference_parity(rt, k):
    rgb = IMAGES[k]
    orig = C2.analyze_image_v2(rgb, rt.model, rt.cfg, rt.device)
    mine = INF.infer_image(rt, rgb, with_gradcam=True)
    assert np.allclose(mine.raw_probs, orig['p_raw'], rtol=0, atol=1e-6)
    assert np.allclose(mine.p_cal, orig['p_cal'], rtol=0, atol=1e-7)       # dung sai ~1 ULP float32 giữa hai lượt chạy CPU
    assert mine.label == orig['label'] and mine.abstain == orig['abstain']
    assert abs(mine.confidence - orig['confidence']) < 1e-7
    assert np.array_equal(mine.mask, orig['mask'])
    assert np.allclose(mine.seg_prob, orig['seg_prob'], atol=1e-6)
    assert mine.boxes == orig['boxes']
    assert np.array_equal(mine.image, orig['image'])
    assert np.allclose(mine.evidence, orig['evidence'], atol=1e-7)
    assert mine.cam is not None and np.allclose(mine.cam, orig['cam'], atol=1e-4)


def test_parity_with_predict_arrays_validation_path(rt):
    """Đường dự đoán dùng khi huấn luyện/đánh giá (batch + TTA)."""
    rgb = IMAGES[0]
    img = core.preprocess_image(rgb, int(rt.cfg['img_size']))
    ref = C2.predict_arrays(rt.model, [img], [0], rt.device, tta=bool(rt.cfg['tta']))
    mine = INF.infer_image(rt, rgb)
    assert np.allclose(mine.raw_probs, ref['prob'][0], atol=1e-6)
    assert np.allclose(mine.seg_prob, ref['seg_prob'][0].astype(np.float32), atol=2e-3)
    ev = mine.evidence_grid / (mine.evidence_grid.sum() + 1e-8)
    assert np.allclose(ev, ref['evidence'][0], atol=1e-6)


def test_raw_logits_equal_model_forward(rt):
    rgb = IMAGES[1]
    img = core.preprocess_image(rgb, int(rt.cfg['img_size']))
    x = core.normalize_to_tensor(img).unsqueeze(0).to(rt.device)
    with torch.no_grad():
        lg = rt.model(x)['logits'][0].float().cpu().numpy()
    assert np.allclose(INF.infer_image(rt, rgb).raw_logits, lg, atol=1e-6)


def test_deterministic_repeat(rt):
    a, b = INF.infer_image(rt, IMAGES[0]), INF.infer_image(rt, IMAGES[0])
    assert np.array_equal(a.raw_logits, b.raw_logits) and np.array_equal(a.mask, b.mask)


def test_decomposition_sums_to_logit(synth_result):
    c = synth_result.contrib
    assert c is not None and c['max_abs_err'] < INF.DECOMP_TOL
    assert np.allclose(c['vals'].sum(0) + c['bias'], c['logits'], atol=INF.DECOMP_TOL)
    assert np.allclose(c['logits'], synth_result.raw_logits, atol=1e-6)


def test_no_training_state_after_inference(rt, synth_rgb):
    INF.infer_image(rt, synth_rgb, with_gradcam=True)
    assert rt.model.training is False
    assert all(p.grad is None for p in rt.model.parameters())     # Grad-CAM không để lại gradient tham số


def test_gradcam_optional(rt, synth_rgb, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('mô phỏng lỗi Grad-CAM')
    monkeypatch.setattr(INF.C2, 'gradcam_dict', boom)
    res = INF.infer_image(rt, synth_rgb, with_gradcam=True)
    assert res.cam is None and 'RuntimeError' in res.cam_error
    assert res.label in ('benign', 'malignant') and np.isfinite(res.p_cal).all()


def test_gradcam_not_run_unless_requested(rt, synth_rgb):
    assert INF.infer_image(rt, synth_rgb, with_gradcam=False).cam is None
