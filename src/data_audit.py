"""Kiểm toán dữ liệu: dùng lại data_audit V3 (đã chạy thật trên BUS-BRA/BUSI) + bổ sung K-fold theo nhóm trên dữ liệu phát triển (OOF)."""
import numpy as np
import pandas as pd
from ._paths import ensure_v3
ensure_v3()
from data_audit import *  # noqa: F401,F403  (V3, giữ nguyên)
import data_audit as _V3  # noqa: F401

LABEL_MAP = {'benign': 0, 'malignant': 1}


def check_label_mapping(df):
    # benign = 0, malignant = 1; báo lỗi rõ ràng nếu có dòng sai
    bad = []
    for p, l in zip(df['pathology'].astype(str).str.lower(), df['label']):
        key = 'malignant' if p.startswith('m') else ('benign' if p.startswith('b') else None)
        if key is None or LABEL_MAP[key] != int(l):
            bad.append((p, l))
    if bad:
        raise ValueError(f'Mapping nhãn sai/lạ ở {len(bad)} dòng, ví dụ {bad[:3]}')
    return True


def dev_folds(df_split, k=5, seed=42):
    # gán fold 0..k-1 cho dữ liệu phát triển (train+val), nhóm theo cột 'group' (bệnh nhân + ảnh trùng); test = -1
    from sklearn.model_selection import StratifiedGroupKFold
    df = df_split.reset_index(drop=True)
    dev = np.where(df['split'].values != 'test')[0]
    fold = np.full(len(df), -1)
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(sgk.split(dev, df['label'].values[dev], df['group'].values[dev])):
        fold[dev[te]] = f
    return fold


def oof_leakage_report(df_split, fold):
    df = df_split.reset_index(drop=True)
    rep = {}
    for col in ('group', 'patient_id', 'sha256'):
        if col not in df:
            continue
        d = df[fold >= 0]
        rep[f'{col}_across_folds'] = int((d.assign(f=fold[fold >= 0]).groupby(col)['f'].nunique() > 1).sum())
    rep['test_in_dev_folds'] = int(((df['split'] == 'test') & (fold >= 0)).sum())
    rep['ok'] = all(v == 0 for k, v in rep.items() if k != 'ok')
    return rep


# ------------------------------------------------------------------ kiểm toán manifest (chạy được không cần ảnh/torch)
def verify_manifest_sha(manifest_path, record_path):
    import hashlib, json
    got = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
    rec = json.load(open(record_path, encoding='utf-8'))
    return dict(ok=got == rec['sha256'], sha256=got, recorded=rec['sha256'], n_recorded=int(rec.get('n', -1)))


def check_pairing(df):
    """Ghép ảnh–mask theo tên tệp (bus_XXXX-l.png <-> mask_XXXX-l.png). Trả danh sách cặp không khớp."""
    import os, re
    bi = df['image_path'].map(lambda p: re.sub(r'^bus_', '', os.path.splitext(os.path.basename(p))[0]))
    bm = df['mask_path'].map(lambda p: re.sub(r'^mask_', '', os.path.splitext(os.path.basename(p))[0]))
    bad = np.where(bi.values != bm.values)[0]
    return dict(ok=len(bad) == 0, n_mismatch=int(len(bad)), examples=df['image_path'].iloc[bad[:3]].tolist(),
                image_unique=bool(df['image_path'].is_unique), mask_unique=bool(df['mask_path'].is_unique))


def audit_manifest(df):
    per = [dict(split=k, n=int(len(g)), n_patients=int(g['patient_id'].nunique()), malignant_rate=float(g['label'].mean()), n_malignant=int(g['label'].sum()))
           for k, g in df.groupby('split')]
    return dict(n=int(len(df)), n_patients=int(df['patient_id'].nunique()), exact_duplicate_sha256=int(df['sha256'].duplicated().sum()),
                empty_or_bad_masks=int(((df['mask_ratio'] <= 0) | df['mask_ratio'].isna()).sum()), per_split=per,
                issues=int((df['issue'].fillna('') != '').sum()))


def fold_malignant_rates(df_split, fold):
    d = df_split.reset_index(drop=True)
    return {int(f): float(d['label'].values[fold == f].mean()) for f in sorted(set(fold[fold >= 0]))}
