"""Parity notebook<->web và tương thích checkpoint: CẦN torch. Không có torch -> SKIP (ghi 'NOT RUN' trong báo cáo kiểm thử, không tính là pass)."""
import os, numpy as np, pytest
torch = pytest.importorskip('torch')


def test_predictor_cpu_inference_shapes_and_determinism():
    from src import inference as INF
    from src._paths import ensure_v3
    ensure_v3()
    import models_xai as MX
    model, _ = MX.build_experiment('X6', pretrained=False)
    p = INF.TorchPredictor(model, torch.device('cpu'))
    x = (np.random.RandomState(0).rand(2, 64, 64, 3) * 255).astype(np.uint8)
    a, b = p(x), p(x)
    assert a['logits'].shape == (2, 2) and np.allclose(a['logits'], b['logits'], atol=1e-5)
    assert a['seg_prob'].shape[0] == 2 and a['evidence'].shape[0] == 2


def test_checkpoint_roundtrip_matches_in_memory_model(tmp_path):
    from src import inference as INF
    from src._paths import ensure_v3
    ensure_v3()
    import models_xai as MX, bus_core_v2 as C2
    model, cfgd = MX.build_experiment('X6', pretrained=False)
    meta = dict(experiment='X6', num_classes=2, dim=128, base=32, img_size=64, model_cfg=cfgd)
    path = str(tmp_path / 'm.pt'); C2.save_model_checkpoint(path, model, meta)
    x = (np.random.RandomState(1).rand(2, 64, 64, 3) * 255).astype(np.uint8)
    ref = INF.TorchPredictor(model, torch.device('cpu'))(x)['logits']
    got = INF.load_predictor(path, torch.device('cpu'))(x)['logits']
    assert np.allclose(ref, got, atol=1e-4)
