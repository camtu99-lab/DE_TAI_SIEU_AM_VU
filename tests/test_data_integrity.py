"""Kiểm thử dữ liệu trên dữ liệu TỔNG HỢP của V3 (make_synthetic_*): kiểm tra MÃ, không phải kết quả khoa học."""
import os, shutil, tempfile
import numpy as np
import pandas as pd
import cv2
import pytest
from src import data_audit as DA


@pytest.fixture(scope='module')
def synth():
    d = tempfile.mkdtemp()
    root = DA.make_synthetic_busbra(os.path.join(d, 'bb'), n_patients=40, seed=0)
    df, info = DA.index_busbra(root)
    a, rep = DA.audit_dataframe(df)
    clean, drop = DA.drop_invalid(a)
    yield dict(dir=d, root=root, df=df, clean=clean, split=DA.grouped_split(clean, seed=1))
    shutil.rmtree(d, ignore_errors=True)


def test_label_mapping_benign0_malignant1(synth):
    assert DA.check_label_mapping(synth['clean'])
    assert DA.LABEL_MAP == {'benign': 0, 'malignant': 1}
    bad = synth['clean'].copy(); bad.loc[bad.index[0], 'label'] = 1 - bad.loc[bad.index[0], 'label']
    with pytest.raises(ValueError):
        DA.check_label_mapping(bad)


def test_busi_binary_excludes_normal(synth):
    root = DA.make_synthetic_busi(os.path.join(synth['dir'], 'busi'), n=20, seed=5)
    b, info = DA.index_busi_binary(root)
    assert info['n_normal_excluded'] > 0 and set(b['label']) <= {0, 1} and (b['pathology'] != 'normal').all()
    assert DA.check_label_mapping(b)


def test_patient_level_split_no_leakage(synth):
    s = synth['split']
    rep = DA.leakage_report(s)
    assert rep['ok'] and rep['patient_id_across_splits'] == 0 and rep['group_across_splits'] == 0
    pats = {k: set(g['patient_id']) for k, g in s.groupby('split')}
    assert not (pats['train'] & pats['val']) and not (pats['train'] & pats['test']) and not (pats['val'] & pats['test'])


def test_leakage_detector_actually_detects(synth):
    s = synth['split'].copy()
    victim = s[s.split == 'train'].iloc[0]['patient_id']
    other = s[s.split == 'test'].index[0]
    s.loc[other, 'patient_id'] = victim                      # cố ý nhồi cùng bệnh nhân vào test
    assert DA.leakage_report(s)['patient_id_across_splits'] >= 1 and not DA.leakage_report(s)['ok']


def test_duplicate_images_detected(tmp_path):
    root = DA.make_synthetic_busbra(str(tmp_path / 'bb'), n_patients=10, seed=3)
    df, _ = DA.index_busbra(root)
    src = df.iloc[0]
    imgdir = os.path.dirname(src.image_path)
    dup = os.path.join(imgdir, 'dupe_copy.png')
    shutil.copy(src.image_path, dup)
    row = src.copy(); row['image_path'] = dup; row['patient_id'] = 'other_patient'
    df2 = pd.concat([df, row.to_frame().T], ignore_index=True)
    a, rep = DA.audit_dataframe(df2)
    assert rep['n_exact_duplicate_images'] >= 1


def test_image_mask_pairing(synth):
    for _, r in synth['clean'].head(20).iterrows():
        im, mk = DA.read_rgb(r.image_path), DA.read_mask(r.mask_path)
        assert im.shape[:2] == mk.shape[:2] and mk.sum() > 0
        assert os.path.basename(r.image_path).replace('bus_', '') .split('.')[0] == os.path.basename(r.mask_path).replace('mask_', '').split('.')[0]


def test_dev_folds_never_use_test_and_have_no_group_leak(synth):
    s = synth['split']
    f = DA.dev_folds(s, k=3, seed=0)
    rep = DA.oof_leakage_report(s, f)
    assert rep['ok'] and rep['test_in_dev_folds'] == 0
    assert set(f[s['split'].values == 'test']) == {-1}
    assert set(f[s['split'].values != 'test']) == {0, 1, 2}
