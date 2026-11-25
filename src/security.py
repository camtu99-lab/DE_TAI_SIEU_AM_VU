"""An toàn tệp/ảnh: hash, đường dẫn, kiểm tra upload, lỗi thân thiện. Không ghi ảnh ra đĩa."""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from src import config


class UserFacingError(Exception):
    """Lỗi có thông điệp tiếng Việt cho người dùng; `technical` chỉ chứa loại/thông điệp lỗi, KHÔNG chứa dữ liệu ảnh."""

    def __init__(self, component: str, message_vi: str, technical: str = ''):
        super().__init__(message_vi)
        self.component = component
        self.message_vi = message_vi
        self.technical = technical


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def safe_child(base: Path, name: str, allowed_suffix: tuple[str, ...] | None = None) -> Path:
    """Trả về base/name nếu name là tên tệp thuần (không traversal, không tuyệt đối) và nằm trong base."""
    if not isinstance(name, str) or not name or name != Path(name).name or name in ('.', '..') or '\\' in name or '/' in name:
        raise UserFacingError('checkpoint', 'Tên tệp trong cấu hình không hợp lệ (nghi ngờ path traversal).', f'name={name!r}')
    p = (base / name).resolve()
    if base.resolve() not in p.parents:
        raise UserFacingError('checkpoint', 'Đường dẫn tệp nằm ngoài thư mục model được phép.', f'path={p.name}')
    if allowed_suffix and p.suffix.lower() not in allowed_suffix:
        raise UserFacingError('checkpoint', f'Phần mở rộng tệp không được phép: {p.suffix}', '')
    return p


def validate_upload(name: str, data: bytes) -> None:
    """Kiểm tra đuôi tệp và kích thước; không dùng tên tệp cho bất kỳ mục đích lưu trữ nào."""
    ext = Path(name or '').suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise UserFacingError('upload', f'Định dạng tệp không được hỗ trợ ({ext or "không có đuôi"}). Chỉ nhận: ' + ', '.join(config.ALLOWED_EXT))
    if len(data) == 0:
        raise UserFacingError('upload', 'Tệp rỗng.')
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise UserFacingError('upload', f'Tệp quá lớn (>{config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB).')


def new_case_id() -> str:
    return uuid.uuid4().hex
