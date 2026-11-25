import ast
import builtins
import json
from pathlib import Path

import cv2

from src import inference as INF, report_export as RE
from src.security import validate_upload, UserFacingError
import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / 'app.py', *sorted((ROOT / 'pages').glob('*.py')), *sorted((ROOT / 'src').glob('*.py'))]
BANNED_MODULES = {'requests', 'urllib', 'urllib3', 'http', 'httpx', 'socket', 'tempfile', 'sqlite3', 'ftplib', 'smtplib', 'openai', 'google', 'kaggle', 'huggingface_hub', 'boto3', 'shutil'}
BANNED_CALLS = {'save', 'imwrite', 'write_bytes', 'write_text', 'savefig', 'to_csv', 'to_pickle'}


def test_static_no_network_storage_or_image_saving():
    for f in FILES:
        tree = ast.parse(f.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert not {a.name.split('.')[0] for a in n.names} & BANNED_MODULES, (f.name, [a.name for a in n.names])
            if isinstance(n, ast.ImportFrom) and n.module:
                assert n.module.split('.')[0] not in BANNED_MODULES, (f.name, n.module)
            if isinstance(n, ast.Call):
                fn = n.func
                if isinstance(fn, ast.Attribute) and fn.attr in BANNED_CALLS:
                    # savefig vào BytesIO và to_csv() không có đường dẫn (trả về chuỗi) được phép trong export
                    if f.name == 'report_export.py' and fn.attr == 'savefig':
                        continue
                    if fn.attr == 'to_csv' and not n.args and not any(k.arg == 'path_or_buf' for k in n.keywords):
                        continue      # to_csv() không có đường dẫn trả về chuỗi trong RAM
                    raise AssertionError(f'{f.name}: gọi .{fn.attr}()')
                if isinstance(fn, ast.Name) and fn.id == 'open':
                    mode = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else ''
                    assert not any(c in str(mode) for c in 'wax+'), f'{f.name}: open() ghi tệp'


def test_streamlit_config_disables_telemetry():
    cfg = (ROOT / '.streamlit' / 'config.toml').read_text(encoding='utf-8')
    assert 'gatherUsageStats = false' in cfg


def test_privacy_no_save(rt, synth_rgb, monkeypatch):
    before = {p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    writes = []
    real_open = builtins.open

    def spy(file, mode='r', *a, **k):
        if any(c in str(mode) for c in 'wax+'):
            writes.append((str(file), mode))
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, 'open', spy)
    monkeypatch.setattr(cv2, 'imwrite', lambda *a, **k: writes.append(('imwrite', a[0])))
    res = INF.infer_image(rt, synth_rgb, with_gradcam=True)
    report = RE.build_case_report(rt, res)
    RE.case_report_bytes(report)
    monkeypatch.undo()
    after = {p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    assert writes == [] and before == after


def test_case_report_has_no_identifying_content(rt, synth_result):
    rep = RE.build_case_report(rt, synth_result)
    name, payload = RE.case_report_bytes(rep)
    assert name.startswith('BreastVision_XAI_research_case_') and name.endswith('.json') and 'medical' not in name
    txt = payload.decode('utf-8').lower()
    for bad in ('exif', 'filename', 'file_name', 'patient', '/home/', 'c:\\', 'uploads/'):
        assert bad not in txt, bad
    d = json.loads(payload)
    assert d['model']['checkpoint_sha256'] == rt.sha256 and d['label_mapping'] == {'benign': 0, 'malignant': 1}
    assert d['locked_parameters']['temperature'] == rt.cfg['temperature']
    assert 'không phải thiết bị chẩn đoán' in d['disclaimer']
    # không chứa mảng ảnh
    assert len(payload) < 20_000


def test_validate_upload_rules():
    validate_upload('a.PNG', b'x')
    for name, data in (('a.gif', b'x'), ('a.png', b''), ('noext', b'x'), ('a.png', b'x' * (16 * 1024 * 1024))):
        with pytest.raises(UserFacingError):
            validate_upload(name, data)
