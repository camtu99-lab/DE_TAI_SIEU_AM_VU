"""Nạp cấu hình khóa + checkpoint tin cậy (hash-verified) và kiểm tra sức khỏe khi khởi động. Fail-closed."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src import config
from src.security import UserFacingError, safe_child, sha256_file

import bus_core as core            # noqa: E402  (breastvision_core)
import bus_core_v2 as C2           # noqa: E402
import models_xai as MX            # noqa: E402

LOCK_NAME = 'pipeline_lock_primary.json'
CONFIG_NAME = 'deploy_config.json'
_FLOAT_KEYS = ('temperature', 'threshold', 'abstain_threshold', 'seg_threshold')


@dataclass
class HealthRow:
    name: str
    status: str            # OK | FAIL | WARN | DISABLED
    detail: str = ''


@dataclass
class Runtime:
    ok: bool
    rows: list[HealthRow] = field(default_factory=list)
    model: Any = None
    device: Any = None
    cfg: dict | None = None
    lock: dict | None = None
    meta: dict | None = None
    sha256: str | None = None
    evidence_ok: bool = False
    gradcam_ok: bool = False
    error: UserFacingError | None = None

    @property
    def device_name(self) -> str:
        return str(self.device) if self.device is not None else 'n/a'


def _read_json(path: Path, component: str) -> dict:
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except FileNotFoundError as e:
        raise UserFacingError(component, f'Không tìm thấy tệp {path.name}.', type(e).__name__) from e
    except (OSError, ValueError) as e:
        raise UserFacingError(component, f'Không đọc được {path.name} (JSON hỏng hoặc không truy cập được).', f'{type(e).__name__}: {e}') from e
    if not isinstance(d, dict):
        raise UserFacingError(component, f'{path.name} có cấu trúc không hợp lệ.', 'not a JSON object')
    return d


def load_deploy_config(model_dir: Path | None = None) -> dict:
    """Đọc deploy_config.json thật và kiểm tra schema/kiểu/miền giá trị. Không đặt giá trị mặc định cho tham số khóa."""
    model_dir = Path(model_dir or config.MODEL_DIR)
    cfg = _read_json(model_dir / CONFIG_NAME, 'config')
    for k, typ in config.REQUIRED_CONFIG_FIELDS.items():
        if k not in cfg:
            raise UserFacingError('config', f'deploy_config.json thiếu trường bắt buộc: {k}.', f'missing={k}')
        v = cfg[k]
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) if typ is float else (isinstance(v, bool) if typ is bool else isinstance(v, typ) and not isinstance(v, bool))
        if not ok:
            raise UserFacingError('config', f'Trường {k} trong deploy_config.json có kiểu không hợp lệ.', f'{k}={v!r}')
    for k in _FLOAT_KEYS:
        if not math.isfinite(float(cfg[k])):
            raise UserFacingError('config', f'Trường {k} không phải số hữu hạn.', f'{k}={cfg[k]!r}')
    if float(cfg['temperature']) <= 0:
        raise UserFacingError('config', 'temperature phải > 0.', '')
    for k in ('threshold', 'abstain_threshold', 'seg_threshold'):
        if not 0.0 < float(cfg[k]) <= 1.0:
            raise UserFacingError('config', f'{k} phải nằm trong (0, 1].', f'{k}={cfg[k]}')
    if cfg['img_size'] < 32 or cfg['img_size'] > 1024:
        raise UserFacingError('config', 'img_size ngoài miền hợp lý.', f'img_size={cfg["img_size"]}')
    return cfg


def load_lock(model_dir: Path | None = None) -> dict:
    model_dir = Path(model_dir or config.MODEL_DIR)
    lock = _read_json(model_dir / LOCK_NAME, 'config')
    for k in ('ckpt', 'ckpt_sha256', 'config', 'temperature', 'threshold', 'abstain_threshold'):
        if k not in lock:
            raise UserFacingError('config', f'{LOCK_NAME} thiếu trường {k}.', f'missing={k}')
    return lock


def cross_check(cfg: dict, lock: dict) -> None:
    """deploy_config.json phải khớp pipeline lock (nguồn sự thật đã khóa từ validation)."""
    lc = lock['config']
    pairs = [('experiment', cfg['experiment'], lc.get('experiment')), ('seed', cfg['seed'], lc.get('seed')), ('img_size', cfg['img_size'], lc.get('img_size')),
             ('seg_threshold', cfg['seg_threshold'], lc.get('seg_threshold')), ('min_area', cfg['min_area'], lc.get('min_area')),
             ('run_mode', cfg['run_mode'], lc.get('run_mode')), ('checkpoint', cfg['checkpoint'], lock['ckpt'])]
    for k in ('temperature', 'threshold', 'abstain_threshold'):
        pairs.append((k, cfg[k], lock[k]))
    for name, a, b in pairs:
        same = math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-15) if isinstance(a, float) or isinstance(b, float) else a == b
        if not same:
            raise UserFacingError('config', f'deploy_config.json mâu thuẫn với pipeline lock ở trường "{name}". Ứng dụng dừng (fail-closed).', f'{name}: config={a!r} lock={b!r}')


def verify_checkpoint(path: Path, expected_sha: str) -> str:
    if not path.exists():
        raise UserFacingError('checkpoint', f'Không tìm thấy checkpoint {path.name} trong thư mục model/.', 'FileNotFoundError')
    size = path.stat().st_size
    if not config.CKPT_MIN_BYTES <= size <= config.CKPT_MAX_BYTES:
        raise UserFacingError('checkpoint', 'Kích thước checkpoint không hợp lý.', f'size={size}')
    got = sha256_file(path)
    if got.lower() != str(expected_sha).lower():
        raise UserFacingError('checkpoint', 'SHA-256 của checkpoint KHÔNG khớp pipeline lock. Ứng dụng từ chối nạp mô hình.', f'expected={expected_sha} got={got}')
    return got


def select_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device='cuda')
            return torch.device('cuda')
        except (RuntimeError, AssertionError):
            return torch.device('cpu')
    return torch.device('cpu')


def load_trusted_model(path: Path, cfg: dict, device: torch.device):
    """Nạp checkpoint đã xác minh hash: torch_load gốc (map CPU) -> build đúng kiến trúc (không tải ImageNet) -> strict load -> eval."""
    ck = core.torch_load(str(path), torch.device('cpu'))
    if not isinstance(ck, dict) or 'model' not in ck or 'meta' not in ck:
        raise UserFacingError('architecture', 'Định dạng checkpoint không đúng (thiếu model/meta).', '')
    meta = ck['meta']
    if meta.get('experiment') != cfg['experiment'] or int(meta.get('img_size', -1)) != int(cfg['img_size']) or int(meta.get('seed', -1)) != int(cfg['seed']):
        raise UserFacingError('architecture', 'Metadata checkpoint không khớp deploy_config (experiment/img_size/seed).', f'meta={ {k: meta.get(k) for k in ("experiment", "img_size", "seed")} }')
    if int(meta.get('num_classes', 2)) != 2:
        raise UserFacingError('architecture', 'Checkpoint không phải bài toán 2 lớp benign/malignant.', '')
    model, model_cfg = MX.build_experiment(meta['experiment'], num_classes=2, pretrained=False, dim=meta.get('dim', 128), base=meta.get('base', 32))
    want = {k: v for k, v in (meta.get('model_cfg') or {}).items() if k != 'pretrained'}
    have = {k: v for k, v in model_cfg.items() if k != 'pretrained'}
    if want and want != have:
        raise UserFacingError('architecture', 'Cấu hình kiến trúc trong checkpoint khác với kiến trúc dựng từ mã nguồn.', f'diff={ {k: (want.get(k), have.get(k)) for k in set(want) | set(have) if want.get(k) != have.get(k)} }')
    try:
        model.load_state_dict(ck['model'])          # strict=True
    except RuntimeError as e:
        raise UserFacingError('architecture', 'Trọng số checkpoint không khớp kiến trúc mô hình.', f'{type(e).__name__}: {str(e)[:300]}') from e
    return model.to(device).eval(), meta


def _smoke(model, device, size: int):
    """Một lượt forward + Grad-CAM trên ảnh tổng hợp để kiểm tra khả năng chạy (không dùng dữ liệu người dùng)."""
    x = torch.zeros(1, 3, size, size, device=device)
    with torch.no_grad():
        o = model(x)
    finite = bool(torch.isfinite(o['logits']).all())
    evidence_ok = o.get('evidence') is not None
    gradcam_ok, gc_err = False, ''
    try:
        cam, _ = C2.gradcam_dict(model, x, class_idx=0)
        gradcam_ok = bool(np.isfinite(cam).all())
    except (RuntimeError, ValueError, AttributeError, KeyError, TypeError) as e:
        gc_err = f'{type(e).__name__}: {str(e)[:120]}'
    finally:
        model.zero_grad(set_to_none=True)
    return finite, evidence_ok, gradcam_ok, gc_err


def build_runtime(model_dir: Path | None = None, results_check=None, failure_gate=None, device: torch.device | None = None) -> Runtime:
    model_dir = Path(model_dir or config.MODEL_DIR)
    rt = Runtime(ok=False)
    if results_check is None:
        from src.research_results import check_files as results_check
    if failure_gate is None:
        from src.failure_gate import evaluate_gate as failure_gate
    R = {n: HealthRow(n, 'FAIL', 'chưa kiểm tra') for n in (
        'Checkpoint', 'Checkpoint SHA256', 'Model architecture', 'Label mapping', 'Deploy config', 'Calibration parameters',
        'Research result files', 'Grad-CAM', 'Evidence map', 'BUS-VERIFY detector', 'CPU inference')}

    def upd(name, status, detail=''):
        R[name] = HealthRow(name, status, detail)

    # Label mapping (hằng số + kiểm tra trong test so với data_audit.py)
    ok_lab = config.CLASS_NAMES[config.LABEL_BENIGN] == 'benign' and config.CLASS_NAMES[config.LABEL_MALIGNANT] == 'malignant' and (config.LABEL_BENIGN, config.LABEL_MALIGNANT) == (0, 1)
    upd('Label mapping', 'OK' if ok_lab else 'FAIL', 'benign=0, malignant=1')

    if results_check is not None:
        miss = results_check()
        upd('Research result files', 'OK' if not miss else 'WARN', 'đầy đủ' if not miss else 'thiếu/lỗi: ' + ', '.join(miss))
    if failure_gate is not None:
        g = failure_gate()
        upd('BUS-VERIFY detector', 'DISABLED' if not g.enabled else 'OK', g.message)

    try:
        cfg = load_deploy_config(model_dir)
        lock = load_lock(model_dir)
        cross_check(cfg, lock)
        upd('Deploy config', 'OK', f"{cfg['experiment']} seed {cfg['seed']}, khớp pipeline lock")
        upd('Calibration parameters', 'OK', f"T={cfg['temperature']:.6f}, ngưỡng={cfg['threshold']:.6f}, từ chối<{cfg['abstain_threshold']:.6f}")
        rt.cfg, rt.lock = cfg, lock
        ckpt_path = safe_child(model_dir, cfg['checkpoint'], ('.pt',))
        upd('Checkpoint', 'OK' if ckpt_path.exists() else 'FAIL', ckpt_path.name)
        rt.sha256 = verify_checkpoint(ckpt_path, lock['ckpt_sha256'])
        upd('Checkpoint SHA256', 'OK', rt.sha256[:16] + '…')
        rt.device = device or select_device()
        rt.model, rt.meta = load_trusted_model(ckpt_path, cfg, rt.device)
        upd('Model architecture', 'OK', f"{rt.meta['experiment']} (EfficientNet-B3 + decoder + đa vùng), strict load")
        finite, rt.evidence_ok, rt.gradcam_ok, gc_err = _smoke(rt.model, rt.device, int(cfg['img_size']))
        upd('CPU inference' if rt.device.type == 'cpu' else 'CPU inference', 'OK' if finite else 'FAIL', f'thiết bị: {rt.device}')
        upd('Evidence map', 'OK' if rt.evidence_ok else 'WARN', 'có' if rt.evidence_ok else 'kiến trúc không trả evidence')
        upd('Grad-CAM', 'OK' if rt.gradcam_ok else 'WARN', 'hoạt động' if rt.gradcam_ok else f'không hoạt động ({gc_err})')
        rt.ok = finite and ok_lab
    except UserFacingError as e:
        rt.error = e
        comp = {'checkpoint': ('Checkpoint', 'Checkpoint SHA256'), 'architecture': ('Model architecture',), 'config': ('Deploy config', 'Calibration parameters')}.get(e.component, ())
        for n in comp:
            if R[n].status != 'OK':
                upd(n, 'FAIL', e.message_vi)
        if e.component == 'checkpoint' and R['Checkpoint'].status == 'FAIL':
            upd('Checkpoint SHA256', 'FAIL', 'không thể kiểm tra')
    except (RuntimeError, OSError, KeyError, ValueError) as e:        # lỗi nạp bất ngờ: vẫn fail-closed
        rt.error = UserFacingError('architecture', 'Không nạp được mô hình (lỗi không xác định).', f'{type(e).__name__}: {str(e)[:200]}')
        upd('Model architecture', 'FAIL', rt.error.message_vi)
    for n in ('Model architecture', 'CPU inference', 'Evidence map', 'Grad-CAM'):
        if rt.model is None and R[n].detail == 'chưa kiểm tra':
            upd(n, 'FAIL' if n in ('Model architecture', 'CPU inference') else 'WARN', 'không khả dụng vì mô hình chưa nạp')
    rt.rows = [R[n] for n in R]
    return rt
