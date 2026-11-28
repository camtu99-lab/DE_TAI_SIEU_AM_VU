import cv2
import numpy as np
import pytest

import bus_core as core
from src import preprocessing as PP
from src.security import UserFacingError
from conftest import make_synthetic


def _encode(rgb, ext, **kw):
    ok, buf = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), **kw)
    assert ok
    return buf.tobytes()


def test_decode_matches_notebook_imread(tmp_path):
    rgb = make_synthetic()
    p = tmp_path / 'a.png'
    p.write_bytes(_encode(rgb, '.png'))
    ref = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)     # đường đọc ảnh của notebook
    assert np.array_equal(PP.decode_image(p.read_bytes()), ref)


def test_preprocessing_parity(rt):
    rgb = make_synthetic(240, 330, seed=3)
    size = int(rt.cfg['img_size'])
    a, b = PP.preprocess(rgb, size), core.preprocess_image(rgb, size)
    assert np.array_equal(a, b) and a.dtype == np.uint8 and a.shape == (size, size, 3)
    t = PP.to_tensor(a, rt.device)
    assert t.shape == (1, 3, size, size) and str(t.dtype) == 'torch.float32'
    assert np.array_equal(t[0].numpy(), core.normalize_to_tensor(b).numpy())


def test_grayscale_rgba_jpeg_are_handled():
    rgb = make_synthetic()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ok, g_png = cv2.imencode('.png', gray)
    assert PP.decode_image(g_png.tobytes()).shape == (300, 400, 3)
    rgba = np.dstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), np.full(gray.shape, 200, np.uint8)])
    ok, a_png = cv2.imencode('.png', rgba)
    assert PP.decode_image(a_png.tobytes()).shape == (300, 400, 3)
    assert PP.decode_image(_encode(rgb, '.jpg', params=[cv2.IMWRITE_JPEG_QUALITY, 90])).shape == (300, 400, 3)


def test_bad_inputs_raise_friendly_errors():
    for data in (b'', b'not an image', b'\x89PNG\r\n\x1a\n' + b'0' * 50):
        with pytest.raises(UserFacingError):
            PP.decode_image(data)
    tiny = make_synthetic(40, 40)
    with pytest.raises(UserFacingError):
        PP.decode_image(_encode(tiny, '.png'))
