"""Preprocessing parity: các hàm V4 numpy trùng KHÔNG SAI MỘT PIXEL với bản V3 trong src/v3/bus_core.py (trích bằng AST, không cần torch)."""
import ast, os
import numpy as np
import cv2
import pytest
from src import preprocessing as PP
from src.counterfactual_verifier import CFConfig

V3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'v3', 'bus_core.py')


def _load_v3_funcs():
    tree = ast.parse(open(V3, encoding='utf-8').read())
    want = {'ensure_rgb_uint8', 'letterbox_square', 'preprocess_image', 'preprocess_mask'}
    mod = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want], type_ignores=[])
    ns = dict(np=np, cv2=cv2)
    exec(compile(mod, V3, 'exec'), ns)
    return ns


def test_preprocess_parity_with_v3_random_shapes():
    ns = _load_v3_funcs()
    rng = np.random.default_rng(0)
    for shape in [(353, 274, 3), (200, 300, 3), (480, 640, 3), (100, 100, 3), (300, 200), (256, 256, 4)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        for size in (64, 256):
            assert np.array_equal(PP.preprocess_image(img, size), ns['preprocess_image'](img, size))
        m = (rng.random(shape[:2]) > 0.7).astype(np.uint8)
        assert np.array_equal(PP.preprocess_mask(m, 256), ns['preprocess_mask'](m, 256))
    f = rng.random((50, 60)).astype(np.float32)
    assert np.array_equal(PP.ensure_rgb_uint8(f), ns['ensure_rgb_uint8'](f))


def test_normalize_matches_v3_formula():
    """normalize_to_tensor của V3: (x/255 - mean)/std, CHW. Kiểm tra công thức bằng AST (nguồn) và tính toán độc lập."""
    src = open(V3, encoding='utf-8').read()
    assert 'IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]' in src and 'IMAGENET_STD = np.array([0.229, 0.224, 0.225]' in src
    img = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    ref = ((img.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1)
    assert np.allclose(PP.normalize_np(img), ref)


def test_invalid_images_raise_clear_errors():
    for bad in (None, np.zeros((5, 5, 3), np.uint8), np.zeros((100,), np.uint8), np.ones((64, 64, 3), np.uint8) * 7,
                np.full((64, 64, 3), np.nan, np.float32), np.zeros((64, 64, 7), np.uint8)):
        with pytest.raises(PP.InvalidImageError):
            PP.validate_image(bad)
    PP.validate_image(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))


def test_valid_content_mask_finds_letterbox():
    img = np.full((256, 256, 3), 100, np.uint8); img[:, :30] = 0; img[:, -30:] = 0
    v = PP.valid_content_mask(img)
    assert v[:, 30:-30].all() and not v[:, :30].any() and not v[:, -30:].any()
