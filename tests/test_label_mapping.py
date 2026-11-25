import re
from pathlib import Path

from src import config

CORE = Path(__file__).resolve().parents[1] / 'breastvision_core'


def test_label_mapping():
    assert config.LABEL_BENIGN == 0 and config.LABEL_MALIGNANT == 1
    assert config.CLASS_NAMES == ('benign', 'malignant')
    assert config.CLASS_NAMES.index('malignant') == 1


def test_label_mapping_matches_training_source():
    """Đối chiếu với data_audit.py gốc: BUS-BRA 1 nếu Pathology bắt đầu bằng 'm'; BUSI int(c == 'malignant')."""
    src = (CORE / 'data_audit.py').read_text(encoding='utf-8')
    assert re.search(r"label = 1 if lab\.startswith\('m'\) else \(0 if lab\.startswith\('b'\) else None\)", src)
    assert "label=int(c == 'malignant')" in src


def test_decision_uses_malignant_index(rt):
    from src.inference import decide
    lab, idx, conf, ab = decide([0.1, 0.9], rt.cfg['threshold'], rt.cfg['abstain_threshold'])
    assert (lab, idx) == ('malignant', 1)
    lab, idx, *_ = decide([0.99, 0.01], rt.cfg['threshold'], rt.cfg['abstain_threshold'])
    assert (lab, idx) == ('benign', 0)
