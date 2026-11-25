import json
import shutil

import pytest
import torch

from src import config
from src import checkpoint as ck
from src.security import UserFacingError, safe_child, sha256_file


def test_checkpoint_sha256(rt):
    lock = json.loads((config.MODEL_DIR / 'pipeline_lock_primary.json').read_text(encoding='utf-8'))
    p = config.MODEL_DIR / rt.cfg['checkpoint']
    assert sha256_file(p) == lock['ckpt_sha256'] == rt.sha256


def test_config_matches_lock_and_fields(rt):
    lock = ck.load_lock()
    ck.cross_check(rt.cfg, lock)       # không raise
    for k in config.REQUIRED_CONFIG_FIELDS:
        assert k in rt.cfg
    assert rt.cfg['experiment'] == 'V3_full' and rt.cfg['run_mode'] == 'full_experiment'


def test_cross_check_fails_closed_on_changed_temperature(rt):
    bad = dict(rt.cfg, temperature=rt.cfg['temperature'] * 1.01)
    with pytest.raises(UserFacingError):
        ck.cross_check(bad, ck.load_lock())
    bad = dict(rt.cfg, threshold=0.5)
    with pytest.raises(UserFacingError):
        ck.cross_check(bad, ck.load_lock())


def test_tampered_checkpoint_rejected(tmp_path, rt):
    src = config.MODEL_DIR / rt.cfg['checkpoint']
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    with open(dst, 'r+b') as f:
        f.seek(-1, 2)
        last = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([last[0] ^ 0xFF]))
    with pytest.raises(UserFacingError) as ei:
        ck.verify_checkpoint(dst, rt.sha256)
    assert 'SHA-256' in ei.value.message_vi


def test_missing_checkpoint_and_config_errors(tmp_path):
    rtx = ck.build_runtime(model_dir=tmp_path, device=torch.device('cpu'))
    assert not rtx.ok and rtx.model is None
    names = {r.name: r.status for r in rtx.rows}
    assert names['Deploy config'] == 'FAIL'


def test_config_schema_validation(tmp_path):
    (tmp_path / 'deploy_config.json').write_text(json.dumps({'experiment': 'V3_full'}), encoding='utf-8')
    with pytest.raises(UserFacingError):
        ck.load_deploy_config(tmp_path)
    (tmp_path / 'deploy_config.json').write_text('{not json', encoding='utf-8')
    with pytest.raises(UserFacingError):
        ck.load_deploy_config(tmp_path)


def test_path_traversal_rejected():
    for bad in ('../x.pt', '/etc/passwd', 'a/b.pt', '..', '', 'x\\y.pt'):
        with pytest.raises(UserFacingError):
            safe_child(config.MODEL_DIR, bad)
    with pytest.raises(UserFacingError):
        safe_child(config.MODEL_DIR, 'model.exe', ('.pt',))


def test_model_load_cpu(rt):
    assert rt.device.type == 'cpu'
    assert all(p.device.type == 'cpu' for p in rt.model.parameters())
    assert rt.model.training is False


def test_label_and_health_rows(rt):
    rows = {r.name: r for r in rt.rows}
    for n in ('Checkpoint', 'Checkpoint SHA256', 'Model architecture', 'Label mapping', 'Deploy config', 'Calibration parameters', 'CPU inference'):
        assert rows[n].status == 'OK', (n, rows[n].detail)
    assert rows['BUS-VERIFY detector'].status == 'DISABLED'
