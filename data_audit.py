# data_audit.py - khám phá dataset BUS-BRA / BUSI, kiểm toán dữ liệu, chia theo bệnh nhân, kiểm tra rò rỉ, manifest
import os, re, json, hashlib
import numpy as np
import pandas as pd
import cv2

IMG_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
MASK_RE = re.compile(r'^(.*)_mask(?:_\d+)?$', re.IGNORECASE)


# ============================ Tiện ích ============================
def read_rgb(path):
    b = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if b is None else cv2.cvtColor(b, cv2.COLOR_BGR2RGB)


def read_mask(path, hw=None):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if hw is not None and m.shape != tuple(hw):
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    mx = int(m.max())
    return (m > mx // 2).astype(np.uint8) if mx > 0 else np.zeros_like(m)     # chịu được mask 0/1, 0/255, hay nhãn 0/k


def read_union_mask(paths, hw):
    u = np.zeros(hw, np.uint8)
    for p in paths:
        m = read_mask(p, hw)
        if m is not None:
            u |= m
    return u


def dhash256(img_rgb):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    s = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA)
    return (s[:, 1:] > s[:, :-1]).flatten().astype(np.uint8)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _key(name):
    s = os.path.splitext(os.path.basename(str(name)))[0].lower()
    s = re.sub(r'^(bus|mask|img|image)[_\-]?', '', s)
    s = re.sub(r'[_\-]?mask$', '', s)
    return s


# ============================ Khám phá BUS-BRA ============================
COLS = dict(image=['ID', 'id', 'Image', 'image', 'Filename', 'filename', 'File', 'file', 'Image_ID'],
            patient=['Case', 'case', 'Patient', 'patient', 'PatientID', 'Patient_ID', 'patient_id', 'Case_ID'],
            label=['Pathology', 'pathology', 'Label', 'label', 'Class', 'class'],
            birads=['BIRADS', 'BI-RADS', 'birads', 'BI_RADS'],
            device=['Device', 'device', 'Scanner', 'scanner'],
            fold=['Fold', 'fold', 'Split', 'split', 'Partition'])


def _pick(df, cands, override=None):
    if override:
        if override not in df.columns:
            raise KeyError(f'Cột {override!r} không có trong CSV. Các cột hiện có: {list(df.columns)}')
        return override
    for c in cands:
        if c in df.columns:
            return c
    return None


def find_busbra_candidates(base):
    out = []
    for dp, dns, fs in os.walk(base):
        low = {d.lower(): d for d in dns}
        if 'images' in low and 'masks' in low:
            out.append(dp)
            dns[:] = []
    return out


def find_csvs(root):
    return [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f.lower().endswith('.csv')]


def index_busbra(root, csv_path=None, col_overrides=None, allow_missing_patient=False):
    # Dựng bảng (ảnh, mask, bệnh nhân, nhãn). Cấu trúc CSV của BUS-BRA chưa được xác minh trực tiếp bởi tác giả code:
    # hàm dò cột theo danh sách tên phổ biến; nếu không khớp sẽ DỪNG kèm danh sách cột để bạn đặt col_overrides.
    col_overrides = col_overrides or {}
    sub = {d.lower(): d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    if 'images' not in sub or 'masks' not in sub:
        raise FileNotFoundError(f'{root} phải có thư mục Images và Masks. Hiện có: {sorted(os.listdir(root))}')
    img_dir, msk_dir = os.path.join(root, sub['images']), os.path.join(root, sub['masks'])
    parent = os.path.dirname(os.path.abspath(root))
    pool = find_csvs(root) + [os.path.join(parent, f) for f in os.listdir(parent) if f.lower().endswith('.csv')]       # CSV có thể nằm ở thư mục cha
    csvs = [csv_path] if csv_path else [c for c in pool if 'fold' not in os.path.basename(c).lower() and 'split' not in os.path.basename(c).lower()]
    if not csvs:
        raise FileNotFoundError('Không tìm thấy file CSV thông tin trong ' + root)
    csv = None
    for c in csvs:
        d = pd.read_csv(c)
        if _pick(d, COLS['label'], col_overrides.get('label')):
            csv, meta = c, d
            break
    if csv is None:
        raise KeyError('Không CSV nào có cột nhãn (Pathology). Hãy đặt col_overrides={"label": ...}. CSV tìm thấy: ' + str(csvs))
    c_img, c_pat, c_lab = (_pick(meta, COLS[k], col_overrides.get(k)) for k in ('image', 'patient', 'label'))
    if c_img is None:
        raise KeyError(f'Không xác định được cột tên ảnh. Cột hiện có: {list(meta.columns)}; đặt col_overrides={{"image": ...}}')
    if c_pat is None and not allow_missing_patient:
        raise KeyError(f'Không có cột mã bệnh nhân (ví dụ Case). Cột hiện có: {list(meta.columns)}. Đặt col_overrides={{"patient": ...}} '
                       'hoặc allow_missing_patient=True (KHÔNG khuyến nghị: mất khả năng chia theo bệnh nhân).')
    imgs = {_key(f): os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in IMG_EXT}
    msks = {_key(f): os.path.join(msk_dir, f) for f in os.listdir(msk_dir) if os.path.splitext(f)[1].lower() in IMG_EXT}
    rows, missing = [], []
    for _, r in meta.iterrows():
        k = _key(r[c_img])
        if k not in imgs or k not in msks:
            missing.append(str(r[c_img])); continue
        lab = str(r[c_lab]).strip().lower()
        label = 1 if lab.startswith('m') else (0 if lab.startswith('b') else None)
        if label is None:
            missing.append(f'{r[c_img]} (nhãn lạ: {lab})'); continue
        rows.append(dict(image_path=imgs[k], mask_path=msks[k], patient_id=str(r[c_pat]) if c_pat else f'img_{k}', label=label, pathology=lab,
                         birads=r[_pick(meta, COLS['birads'])] if _pick(meta, COLS['birads']) else None,
                         device=r[_pick(meta, COLS['device'])] if _pick(meta, COLS['device']) else None,
                         official_fold=r[_pick(meta, COLS['fold'])] if _pick(meta, COLS['fold']) else None, source='BUS-BRA'))
    df = pd.DataFrame(rows)
    info = dict(csv=csv, columns=dict(image=c_img, patient=c_pat, label=c_lab), n_csv_rows=len(meta), n_indexed=len(df), n_unpaired=len(missing), unpaired_examples=missing[:10])
    if len(df) == 0:
        raise RuntimeError(f'Không ghép được ảnh–mask nào. Ví dụ tên trong CSV: {meta[c_img].head(3).tolist()}; ví dụ tên ảnh: {list(imgs)[:3]}; mask: {list(msks)[:3]}')
    return df, info


# ============================ Khám phá BUSI (chỉ benign/malignant) ============================
def find_busi_candidates(base):
    out = []
    for dp, dns, _ in os.walk(base):
        low = {d.lower() for d in dns}
        if 'benign' in low and 'malignant' in low:
            out.append(dp); dns[:] = []
    return out


def index_busi_binary(root):
    # KHÔNG chuyển normal thành benign: ảnh normal bị loại khỏi tập đánh giá ngoài.
    rows, n_normal = [], 0
    for d in sorted(os.listdir(root)):
        c = d.lower()
        full = os.path.join(root, d)
        if not os.path.isdir(full) or c not in ('benign', 'malignant', 'normal'):
            continue
        imgs, masks = {}, {}
        for f in sorted(os.listdir(full)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMG_EXT:
                continue
            m = MASK_RE.match(stem)
            (masks.setdefault(m.group(1), []).append(os.path.join(full, f)) if m else imgs.__setitem__(stem, os.path.join(full, f)))
        if c == 'normal':
            n_normal += len(imgs); continue
        for stem, p in imgs.items():
            rows.append(dict(image_path=p, mask_paths=sorted(masks.get(stem, [])), patient_id=f'busi_{c}_{stem}', label=int(c == 'malignant'),
                             pathology=c, source='BUSI'))
    return pd.DataFrame(rows), dict(n_normal_excluded=n_normal)


# ============================ Kiểm toán ============================
def audit_dataframe(df, mask_col='mask_path', dup_hamming=8, read_pixels=True):
    # Trả (df_đã_gắn_cột, báo_cáo). Phát hiện: ảnh/mask lỗi, mask rỗng, lệch kích thước, trùng tuyệt đối, gần trùng, nhãn mâu thuẫn.
    df = df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        raise ValueError('audit_dataframe nhận bảng RỖNG: bước khám phá dataset không tìm thấy ảnh nào. Kiểm tra đường dẫn và tên cột.')
    md5, dh, ok, hh, ww, area, msz, issue = [], [], [], [], [], [], [], []
    for r in df.itertuples(index=False):
        iss = []
        img = read_rgb(r.image_path)
        if img is None:
            md5.append(''); dh.append(np.zeros(256, np.uint8)); hh.append(0); ww.append(0); area.append(0); msz.append(False); issue.append('anh_loi'); continue
        mp = getattr(r, mask_col)
        paths = mp if isinstance(mp, (list, tuple)) else [mp]
        m = np.zeros(img.shape[:2], np.uint8)
        mism = False
        for p in paths:
            raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if p else None
            if raw is None:
                iss.append('mask_loi_hoac_thieu'); continue
            if raw.shape != img.shape[:2]:
                mism = True
            m |= read_mask(p, img.shape[:2])
        if m.sum() == 0:
            iss.append('mask_rong')
        md5.append(sha256_file(r.image_path)); dh.append(dhash256(img)); hh.append(img.shape[0]); ww.append(img.shape[1])
        area.append(float(m.mean())); msz.append(mism); issue.append(';'.join(iss))
    df['sha256'], df['h'], df['w'], df['mask_ratio'], df['mask_size_mismatch'], df['issue'] = md5, hh, ww, area, msz, issue
    bits = np.stack(dh).astype(np.int32)
    ham = bits @ (1 - bits).T + (1 - bits) @ bits.T
    uf = _UF(n)
    ii, jj = np.where(np.triu(ham <= dup_hamming, k=1))
    for a, b in zip(ii, jj):
        if df.loc[a, 'sha256'] != '' and df.loc[b, 'sha256'] != '':
            uf.union(int(a), int(b))
    df['dup_group'] = [uf.find(i) for i in range(n)]
    gs = df.groupby('dup_group').size()
    exact = df.groupby('sha256').filter(lambda g: len(g) > 1 and g.name != '')
    conflict_groups = df.groupby('dup_group')['label'].nunique()
    rep = dict(n=n, n_bad_image=int((df['issue'].str.contains('anh_loi')).sum()), n_bad_or_missing_mask=int(df['issue'].str.contains('mask_loi').sum()),
               n_empty_mask=int(df['issue'].str.contains('mask_rong').sum()), n_mask_size_mismatch=int(df['mask_size_mismatch'].sum()),
               n_exact_duplicate_images=int(len(exact)), n_neardup_groups=int((gs > 1).sum()), n_images_in_neardup_groups=int(gs[gs > 1].sum()),
               n_label_conflict_groups=int((conflict_groups > 1).sum()), class_counts=df['label'].value_counts().sort_index().to_dict(),
               n_patients=int(df['patient_id'].nunique()) if 'patient_id' in df else None)
    df['_dhash'] = [d.tobytes() for d in dh]
    return df, rep


def drop_invalid(df):
    # loại ảnh/mask lỗi, mask rỗng, nhãn mâu thuẫn trong cùng nhóm trùng; giữ 1 bản cho trùng tuyệt đối cùng nhãn
    bad = df['issue'] != ''
    conflict = df.groupby('dup_group')['label'].transform('nunique') > 1
    keep = ~bad & ~conflict
    out = df[keep].copy()
    if len(out) == 0:
        raise RuntimeError('Sau khi lọc không còn ảnh nào hợp lệ. Số lượng theo lý do: ' + str(df['issue'].replace('', 'ok').value_counts().to_dict())
                           + ' | nhãn mâu thuẫn trong nhóm trùng: ' + str(int(conflict.sum())) + '. Xem hướng dẫn xử lý trong README.')
    out = out.drop_duplicates(subset='sha256', keep='first')
    return out.reset_index(drop=True), dict(dropped_invalid=int(bad.sum()), dropped_label_conflict=int((conflict & ~bad).sum()), dropped_exact_dups=int(keep.sum() - len(out)))


# ============================ Gom nhóm và chia theo bệnh nhân ============================
def build_groups(df):
    # nhóm = hợp của (cùng bệnh nhân) và (ảnh trùng/gần trùng)
    n = len(df)
    uf = _UF(n)
    for col in ('patient_id', 'dup_group'):
        first = {}
        for i, v in enumerate(df[col].values):
            if v in first:
                uf.union(first[v], i)
            else:
                first[v] = i
    return np.array([uf.find(i) for i in range(n)])


def grouped_split(df, seed=42, val_frac=0.15, test_frac=0.15, official_test_fold=None):
    # Chia theo nhóm (bệnh nhân + ảnh trùng), phân tầng theo nhãn. Nếu official_test_fold được cho và cột official_fold có giá trị,
    # test = fold đó nhưng validation vẫn tái tạo từ bệnh nhân KHÔNG thuộc test (tránh lệch bệnh nhân giữa train/val được cung cấp).
    from sklearn.model_selection import StratifiedGroupKFold
    df = df.reset_index(drop=True).copy()
    df['group'] = build_groups(df)
    idx = np.arange(len(df))
    y, g = df['label'].values, df['group'].values
    if official_test_fold is not None and df['official_fold'].notna().any():
        test = np.where(df['official_fold'].astype(str).values == str(official_test_fold))[0]
        tg = set(g[test])
        test = np.array([i for i in idx if g[i] in tg])          # kéo cả nhóm vào test (bệnh nhân không tách đôi)
        dev = np.array([i for i in idx if g[i] not in tg])
    else:
        k1 = max(2, int(round(1 / test_frac)))
        dev, test = next(StratifiedGroupKFold(n_splits=k1, shuffle=True, random_state=seed).split(idx, y, g))
    k2 = max(2, int(round(1 / (val_frac / (1 - test_frac)))))
    tr_rel, va_rel = next(StratifiedGroupKFold(n_splits=k2, shuffle=True, random_state=seed).split(dev, y[dev], g[dev]))
    df['split'] = ''
    df.loc[dev[tr_rel], 'split'] = 'train'; df.loc[dev[va_rel], 'split'] = 'val'; df.loc[test, 'split'] = 'test'
    return df


def leakage_report(df, dup_hamming=8):
    # kiểm tra: nhóm, bệnh nhân, sha256, và cặp gần trùng (dHash) nằm ở các tập khác nhau
    rep = {}
    for col in ('group', 'patient_id', 'sha256'):
        if col in df:
            s = df.groupby(col)['split'].nunique()
            rep[f'{col}_across_splits'] = int((s > 1).sum())
    if '_dhash' in df:
        bits = np.stack([np.frombuffer(b, np.uint8) for b in df['_dhash']]).astype(np.int32)
        ham = bits @ (1 - bits).T + (1 - bits) @ bits.T
        sp = df['split'].values
        i, j = np.where(np.triu(ham <= dup_hamming, k=1))
        rep['neardup_pairs_across_splits'] = int(sum(sp[a] != sp[b] for a, b in zip(i, j)))
    rep['ok'] = all(v == 0 for k, v in rep.items() if k != 'ok')
    return rep


def cross_dataset_duplicates(dfA, dfB, dup_hamming=8):
    # kiểm tra ảnh trùng/gần trùng giữa hai dataset (ví dụ BUS-BRA và BUSI)
    if len(dfA) == 0 or len(dfB) == 0:
        raise ValueError(f'So trùng giữa hai dataset cần cả hai bảng khác rỗng (A={len(dfA)}, B={len(dfB)}). Bảng rỗng nghĩa là dataset đó bị loại hết ở bước audit.')
    out = dict(sha256_shared=int(len(set(dfA['sha256']) & set(dfB['sha256']))))
    A = np.stack([np.frombuffer(b, np.uint8) for b in dfA['_dhash']]).astype(np.int32)
    B = np.stack([np.frombuffer(b, np.uint8) for b in dfB['_dhash']]).astype(np.int32)
    ham = A @ (1 - B).T + (1 - A) @ B.T
    out['neardup_pairs'] = int((ham <= dup_hamming).sum())
    out['ok'] = out['sha256_shared'] == 0 and out['neardup_pairs'] == 0
    return out


def cross_dataset_duplicate_rows(dfA, dfB, dup_hamming=8):
    # mặt nạ boolean trên các dòng của dfB là bản sao/gần trùng của ảnh nào đó trong dfA (dùng để loại khỏi tập đánh giá ngoài)
    if len(dfA) == 0 or len(dfB) == 0:
        raise ValueError(f'So trùng giữa hai dataset cần cả hai bảng khác rỗng (A={len(dfA)}, B={len(dfB)}). Bảng rỗng nghĩa là dataset đó bị loại hết ở bước audit.')
    A = np.stack([np.frombuffer(b, np.uint8) for b in dfA['_dhash']]).astype(np.int32)
    B = np.stack([np.frombuffer(b, np.uint8) for b in dfB['_dhash']]).astype(np.int32)
    ham = A @ (1 - B).T + (1 - A) @ B.T
    near = (ham <= dup_hamming).any(0)
    exact = dfB['sha256'].isin(set(dfA['sha256'])).values
    return near | exact


def write_manifest(df, path):
    cols = [c for c in df.columns if not c.startswith('_')]
    d = df[cols].copy()
    for c in d.columns:
        if d[c].map(lambda v: isinstance(v, (list, tuple))).any():
            d[c] = d[c].map(lambda v: ';'.join(v) if isinstance(v, (list, tuple)) else v)
    d.to_csv(path, index=False)
    h = sha256_file(path)
    json.dump(dict(manifest=os.path.basename(path), sha256=h, n=len(d)), open(path + '.sha256.json', 'w'))
    return h


# ============================ Dữ liệu TỔNG HỢP (chỉ để kiểm thử code; KHÔNG phải dữ liệu y tế) ============================
def _synth_image(rng, malignant, size=(190, 250), seed_bg=None):
    h, w = size
    bg_rng = np.random.RandomState(seed_bg) if seed_bg is not None else rng
    img = cv2.GaussianBlur((bg_rng.rand(h, w) * 90).astype(np.float32), (0, 0), 2.0) + 50
    mask = np.zeros((h, w), np.uint8)
    cx, cy = rng.randint(60, w - 60), rng.randint(50, h - 50)
    ax, ay = rng.randint(14, 36), rng.randint(10, 26)
    cv2.ellipse(mask, (cx, cy), (ax, ay), int(rng.randint(0, 90)), 0, 360, 255, -1)
    img = img + (mask > 0) * (55 if malignant else -30) + rng.rand(h, w) * 8
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), mask


def make_synthetic_busbra(root, n_patients=30, seed=0):
    # CẤU TRÚC GIẢ ĐỊNH giống BUS-BRA (Images/, Masks/, bus_data.csv với ID, Case, Pathology). Dùng để kiểm thử mã, không đại diện dữ liệu thật.
    rng = np.random.RandomState(seed)
    os.makedirs(os.path.join(root, 'Images'), exist_ok=True); os.makedirs(os.path.join(root, 'Masks'), exist_ok=True)
    rows, k = [], 0
    for p in range(n_patients):
        mal = p % 3 == 0
        for v in range(rng.randint(1, 4)):
            k += 1
            img, m = _synth_image(rng, mal, seed_bg=p)
            name = f'bus_{k:04d}-l'
            cv2.imwrite(os.path.join(root, 'Images', name + '.png'), img); cv2.imwrite(os.path.join(root, 'Masks', name.replace('bus_', 'mask_') + '.png'), m)
            rows.append(dict(ID=name, Case=f'C{p:04d}', Pathology='malignant' if mal else 'benign', BIRADS=4 + int(mal), Device='synthetic'))
    pd.DataFrame(rows).to_csv(os.path.join(root, 'bus_data.csv'), index=False)
    open(os.path.join(root, 'SYNTHETIC_README.txt'), 'w').write('Dữ liệu tổng hợp chỉ để kiểm thử mã. KHÔNG phải ảnh y tế.')
    return root


def make_synthetic_busi(root, n=24, seed=1, with_normal=True):
    rng = np.random.RandomState(seed)
    for c, cnt in (('benign', n), ('malignant', n // 2), ('normal', 6 if with_normal else 0)):
        d = os.path.join(root, c); os.makedirs(d, exist_ok=True)
        for i in range(1, cnt + 1):
            img, m = _synth_image(rng, c == 'malignant')
            if c == 'normal':
                m[:] = 0
            cv2.imwrite(f'{d}/{c} ({i}).png', img); cv2.imwrite(f'{d}/{c} ({i})_mask.png', m)
    open(os.path.join(root, 'SYNTHETIC_README.txt'), 'w').write('Dữ liệu tổng hợp chỉ để kiểm thử mã.')
    return root
