"""Cổng artifact cho BUS-VERIFY failure-risk detector.

Theo kiểm toán, detector CHƯA hoàn thành => mặc định TẮT. Module này KHÔNG tạo điểm rủi ro nào
(không random, không heuristic 1-confidence/entropy/Δlogit/abstain-score đổi tên).
Chỉ kiểm tra sự tồn tại đủ artifact; ngay cả khi đủ, tích hợp suy luận cần được rà soát thủ công.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src import config

REQUIRED_ARTIFACTS = {
    'detector_model': 'detector.pt (mô hình failure detector đã huấn luyện)',
    'feature_schema.json': 'cấu hình feature schema',
    'version.json': 'version + hash artifact',
    'calibration_threshold.json': 'calibration/threshold chọn trên development/validation',
    'evaluation_report.json': 'báo cáo đánh giá + metric error-detection',
    'no_gt_leak_test.json': 'kết quả test: inference không dùng GT label/mask',
    'no_locked_test_fit.json': 'xác nhận artifact chưa được fit trên locked test',
}
MESSAGE_DISABLED = 'BUS-VERIFY failure-risk detector: chưa khả dụng trong bản thực nghiệm đã kiểm toán.'


@dataclass
class FailureGate:
    enabled: bool
    missing: list[str] = field(default_factory=list)
    message: str = MESSAGE_DISABLED


def evaluate_gate(base=None) -> FailureGate:
    base = base or config.FAILURE_DETECTOR_DIR
    missing = []
    for fname, desc in REQUIRED_ARTIFACTS.items():
        p = base / fname
        if fname == 'detector_model':
            if not any((base / n).exists() for n in ('detector.pt', 'detector.pkl', 'detector.joblib')):
                missing.append(desc)
        elif not p.exists():
            missing.append(desc)
        else:
            try:
                json.loads(p.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                missing.append(desc + ' (JSON không hợp lệ)')
    if missing or not config.ENABLE_FAILURE_DETECTOR:
        return FailureGate(False, missing)
    # Đủ artifact và cờ được bật thủ công: vẫn cần mô-đun suy luận được rà soát riêng; app KHÔNG tự suy diễn điểm rủi ro.
    return FailureGate(False, [], 'Artifact BUS-VERIFY có mặt nhưng chưa có mô-đun suy luận được kiểm toán; vẫn tắt.')
