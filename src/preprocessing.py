"""Giải mã + tiền xử lý ảnh bằng ĐÚNG helper gốc của notebook (bus_core): decode RGB, letterbox, resize, normalize.
Mọi thao tác chỉ trong RAM; không ghi đĩa."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from src import config
from src.security import UserFacingError

import bus_core as core   # noqa: E402


def decode_image(data: bytes) -> np.ndarray:
    """bytes -> RGB uint8 (H,W,3). cv2.IMREAD_COLOR giống cv2.imread của notebook (bỏ alpha, gray->3 kênh; EXIF không được xuất ra)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        raise UserFacingError('upload', 'Tệp rỗng.')
    try:
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error as e:
        raise UserFacingError('upload', 'Không giải mã được ảnh (tệp hỏng).', f'cv2.error: {str(e)[:120]}') from e
    if bgr is None:
        raise UserFacingError('upload', 'Không giải mã được ảnh (tệp hỏng hoặc không phải ảnh).')
    h, w = bgr.shape[:2]
    if min(h, w) < config.MIN_SIDE_PX:
        raise UserFacingError('upload', f'Ảnh quá nhỏ ({w}×{h}px); cần cạnh ngắn ≥ {config.MIN_SIDE_PX}px.')
    if max(h, w) > config.MAX_SIDE_PX or h * w > config.MAX_PIXELS:
        raise UserFacingError('upload', f'Ảnh quá lớn ({w}×{h}px).')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_stats(rgb: np.ndarray) -> dict:
    chroma = float((rgb.max(2).astype(np.int16) - rgb.min(2).astype(np.int16)).mean())
    return dict(height=int(rgb.shape[0]), width=int(rgb.shape[1]), chroma=chroma, mean_intensity=float(rgb.mean()))


def preprocess(rgb: np.ndarray, size: int) -> np.ndarray:
    return core.preprocess_image(rgb, size)


def to_tensor(img: np.ndarray, device) -> torch.Tensor:
    return core.normalize_to_tensor(img).unsqueeze(0).to(device)


def valid_region(rgb_shape: tuple[int, int], size: int) -> np.ndarray:
    """Vùng ảnh thật (không phải viền letterbox) sau tiền xử lý, bằng cách letterbox một mask toàn 1 qua helper gốc."""
    ones = np.ones(rgb_shape[:2], dtype=np.uint8)
    return core.preprocess_mask(ones, size) > 0


def preprocessing_config(cfg: dict) -> dict:
    return dict(decode='cv2.imdecode(IMREAD_COLOR)->RGB', resize='bus_core.preprocess_image (letterbox vuông + resize)', img_size=int(cfg['img_size']),
                normalize='bus_core.normalize_to_tensor', tta_hflip=bool(cfg['tta']), seg_threshold=float(cfg['seg_threshold']), min_area=int(cfg['min_area']))
