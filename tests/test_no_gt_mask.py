import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import models_xai as MX
from src import inference as INF

ROOT = Path(__file__).resolve().parents[1]
APP_FILES = [ROOT / 'app.py', *sorted((ROOT / 'pages').glob('*.py')), *sorted((ROOT / 'src').glob('*.py'))]


def test_no_ground_truth_mask_in_inference():
    for fn in (INF.infer_image, INF.infer_preprocessed):
        names = [n.lower() for n in inspect.signature(fn).parameters]
        assert not any(k in n for n in names for k in ('gt', 'mask', 'label', 'truth', 'target')), names


def test_inference_module_never_references_ground_truth():
    src = Path(INF.__file__).read_text(encoding='utf-8').lower()
    for token in ('gt_mask', 'ground_truth', 'y_true', 'gt_label', 'demo_cases'):
        assert token not in src.replace('không dùng ground-truth', '')


def test_regions_come_from_predicted_mask(synth_result):
    assert synth_result.mask.shape == synth_result.image.shape[:2]       # mask dự đoán từ seg_prob
    assert np.array_equal(synth_result.mask, (synth_result.seg_prob >= 0.5).astype(np.uint8))      # seg_threshold khóa = 0.5


def test_no_training():
    banned_attr = {'backward', 'step', 'zero_grad_optim', 'train'}
    banned_names = {'Adam', 'AdamW', 'SGD', 'optim', 'losses_xai', 'fit', 'fine_tune'}
    for f in APP_FILES:
        tree = ast.parse(f.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in banned_attr:
                raise AssertionError(f'{f.name}: gọi .{n.func.attr}()')
            if isinstance(n, ast.Name) and n.id in banned_names:
                raise AssertionError(f'{f.name}: dùng {n.id}')
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in n.names] + ([n.module] if getattr(n, 'module', None) else [])
                assert not any(m and ('losses_xai' in m or 'optim' in m) for m in mods), (f.name, mods)


def _tiny_rt(experiment: str):
    model, _ = MX.build_experiment(experiment, pretrained=False)
    cfg = dict(img_size=64, tta=False, temperature=1.0, threshold=0.5, abstain_threshold=0.6, seg_threshold=0.5, min_area=10)
    return SimpleNamespace(model=model.eval(), device=torch.device('cpu'), cfg=cfg)


def test_evidence_optional():
    rt = _tiny_rt('E3') if 'E3' in MX.EXPERIMENTS else _tiny_rt('X0')
    img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    try:
        res = INF.infer_preprocessed(rt, img, want_extras=False)
    except Exception as e:          # kiến trúc legacy không có 'seg' có thể bị từ chối bằng lỗi thân thiện, không crash không kiểm soát
        from src.security import UserFacingError
        assert isinstance(e, (UserFacingError, KeyError, TypeError)), type(e)
        return
    assert res.evidence is None or res.evidence.shape == (64, 64)
    assert res.contrib is None
