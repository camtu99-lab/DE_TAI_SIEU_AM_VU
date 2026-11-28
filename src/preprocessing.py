"""Tiền xử lý thuần numpy/cv2 (KHÔNG cần torch). Bản sao hành vi của bus_core V3; tests/test_preprocessing_parity.py
đối chiếu từng hàm với mã V3 (trích bằng AST) để bảo đảm tương đương từng pixel."""
import numpy as np
import cv2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InvalidImageError(ValueError):
    """Ảnh đầu vào không hợp lệ (None, sai chiều, quá nhỏ, chứa NaN/Inf, hoặc đồng nhất)."""


def validate_image(img, min_side=32):
    """Kiểm tra ảnh. Ném InvalidImageError với thông báo rõ ràng; KHÔNG nuốt lỗi."""
    if img is None:
        raise InvalidImageError('ảnh là None')
    a = np.asarray(img)
    if a.ndim not in (2, 3):
        raise InvalidImageError(f'ảnh phải có 2 hoặc 3 chiều, nhận {a.ndim}')
    if a.ndim == 3 and a.shape[2] not in (1, 3, 4):
        raise InvalidImageError(f'số kênh không hỗ trợ: {a.shape[2]}')
    if min(a.shape[0], a.shape[1]) < min_side:
        raise InvalidImageError(f'ảnh quá nhỏ {a.shape[:2]} (< {min_side}px)')
    if a.dtype.kind == 'f' and not np.isfinite(a).all():
        raise InvalidImageError('ảnh chứa NaN/Inf')
    if a.size and float(a.max()) == float(a.min()):
        raise InvalidImageError('ảnh đồng nhất (một giá trị duy nhất)')
    return a


def ensure_rgb_uint8(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.dtype != np.uint8:
        mx = float(img.max()) if img.size else 1.0
        img = (img.astype(np.float32) / (mx if mx > 0 else 1.0) * 255.0).clip(0, 255).astype(np.uint8)
    return img


def letterbox_square(arr, pad_value=0):
    h, w = arr.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    canvas = np.full((s, s) + arr.shape[2:], pad_value, dtype=arr.dtype)
    canvas[top:top + h, left:left + w] = arr
    return canvas


def preprocess_image(img_rgb, size):
    sq = letterbox_square(ensure_rgb_uint8(img_rgb))
    interp = cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(sq, (size, size), interpolation=interp)


def preprocess_mask(mask_bin, size):
    sq = letterbox_square((np.asarray(mask_bin) > 0).astype(np.uint8) * 255)
    interp = cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_LINEAR
    r = cv2.resize(sq, (size, size), interpolation=interp)
    return (r > 127).astype(np.uint8)


def normalize_np(img_uint8):
    """Tương đương bus_core.normalize_to_tensor nhưng trả ndarray CHW float32."""
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def valid_content_mask(img_uint8, pad_value=0):
    """Mặt nạ vùng có nội dung (loại dải đen letterbox ở biên: cột/hàng liên tiếp từ biên có toàn bộ pixel == pad_value).
    Ảnh siêu âm thật có thể có biên đen tự nhiên; khi đó vùng này cũng được coi là 'không phải mô' — là cố ý (xem tài liệu shortcut)."""
    a = np.asarray(img_uint8)
    g = a.max(axis=2) if a.ndim == 3 else a
    nz = g != pad_value
    H, W = g.shape
    if not nz.any():
        return np.zeros((H, W), bool)
    cols = np.where(nz.any(0))[0]
    rows = np.where(nz.any(1))[0]
    m = np.zeros((H, W), bool)
    m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] = True
    return m
