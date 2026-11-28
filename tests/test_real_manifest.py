"""Kiểm tra DỮ LIỆU THẬT: manifest BUS-BRA do V3 xuất (1875 ảnh, 1064 bệnh nhân). Đây là kiểm toán dữ liệu thật, không phải dữ liệu giả."""
import os
import numpy as np
import pandas as pd
import pytest
from src import data_audit as DA

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(HERE, 'artifacts', 'busbra_split_manifest.csv')


@pytest.fixture(scope='module')
def man():
    return pd.read_csv(MAN)


def test_manifest_hash_matches_recorded():
    r = DA.verify_manifest_sha(MAN, MAN + '.sha256.json')
    assert r['ok'] and r['n_recorded'] == 1875


def test_label_mapping_benign0_malignant1(man):
    assert DA.check_label_mapping(man) is True
    assert set(man.loc[man.pathology == 'benign', 'label']) == {0} and set(man.loc[man.pathology == 'malignant', 'label']) == {1}
    bad = man.copy(); bad.loc[0, 'label'] = 1 - bad.loc[0, 'label']
    with pytest.raises(ValueError):
        DA.check_label_mapping(bad)


def test_patient_level_split_no_leakage(man):
    rep = DA.leakage_report(man)
    assert rep['ok'], rep
    assert rep['patient_id_across_splits'] == 0 and rep['group_across_splits'] == 0 and rep['sha256_across_splits'] == 0
    assert man['sha256'].is_unique


def test_leakage_detector_catches_injected_leak(man):
    bad = man.copy()
    pid = bad.loc[bad.split == 'train', 'patient_id'].iloc[0]
    k = bad[bad.split == 'test'].index[0]
    bad.loc[k, 'patient_id'] = pid
    assert not DA.leakage_report(bad)['ok']
    dup = man.copy(); i, j = dup.index[dup.split == 'train'][0], dup.index[dup.split == 'test'][0]
    dup.loc[j, 'sha256'] = dup.loc[i, 'sha256']                       # ảnh trùng chính xác nằm ở 2 tập
    assert DA.leakage_report(dup)['sha256_across_splits'] >= 1


def test_image_mask_pairing_and_audit_summary(man):
    r = DA.check_pairing(man)
    assert r['ok'] and r['image_unique'] and r['mask_unique'], r
    a = DA.audit_manifest(man)
    assert a['n'] == 1875 and a['n_patients'] == 1064 and a['empty_or_bad_masks'] == 0 and a['exact_duplicate_sha256'] == 0 and a['issues'] == 0
    ps = {r['split']: r for r in a['per_split']}
    assert (ps['train']['n'], ps['val']['n'], ps['test']['n']) == (1336, 267, 272)
    assert ps['test']['n_malignant'] == 73                              # số ca ác tính test: giới hạn công suất cho FN (xem báo cáo)
    broken = man.copy(); broken.loc[3, 'mask_path'] = broken.loc[4, 'mask_path']
    assert not DA.check_pairing(broken)['ok']


@pytest.mark.parametrize('seed', [0, 1, 42])
def test_dev_folds_patient_disjoint_stratified_and_exclude_test(man, seed):
    fold = DA.dev_folds(man, 5, seed)
    rep = DA.oof_leakage_report(man, fold)
    assert rep['ok'], rep
    assert set(fold[fold >= 0]) == set(range(5)) and (fold >= 0).sum() == 1603
    assert (fold[man.split.values == 'test'] == -1).all()                # fold KHÔNG chạm tập test cũ
    rates = list(DA.fold_malignant_rates(man, fold).values())
    assert max(rates) - min(rates) < 0.06
    assert not (set(man.patient_id[fold >= 0]) & set(man.patient_id[man.split == 'test']))


def test_fold_checker_catches_leak(man):
    fold = DA.dev_folds(man, 5, 0)
    bad = man.copy()
    a, b = np.where(fold == 0)[0][0], np.where(fold == 1)[0][0]
    bad.loc[b, 'patient_id'] = bad.loc[a, 'patient_id']
    assert not DA.oof_leakage_report(bad, fold)['ok']
    fold2 = fold.copy(); fold2[np.where(man.split.values == 'test')[0][0]] = 0
    assert not DA.oof_leakage_report(man, fold2)['ok']


def test_test_set_has_few_fn_so_power_is_limited(man):
    """Ràng buộc thống kê THẬT: 73 ca ác tính trong test; ở sensitivity ~0.9 chỉ ~7 FN => FN Alert Recall trên test không đủ công suất."""
    from src import statistical_tests as ST
    nm = int((man[man.split == 'test'].label == 1).sum())
    n_fn = round(nm * 0.1)
    lo, hi = ST.wilson_ci(n_fn // 2, n_fn)
    assert (hi - lo) > 0.5
    assert ST.min_detectable_n(0.2, 0.5) > n_fn
