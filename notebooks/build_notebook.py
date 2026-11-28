"""Dựng BreastVision_XAI_V4_BUS_VERIFY_Kaggle.ipynb từ chính mã nguồn trong src/ (các cell %%writefile = nội dung tệp thật => notebook và web CÙNG một mã)."""
import os, json, glob, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []
def md(t): cells.append(dict(cell_type='markdown', metadata={}, source=t.strip('\n').splitlines(True)))
def code(t): cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=t.strip('\n').splitlines(True)))

md("""
# BreastVision-XAI V4 — BUS-VERIFY (Kaggle)
**Counterfactual Evidence Verification & Failure-Risk Detection cho siêu âm vú.** Công cụ NGHIÊN CỨU, không phải thiết bị y khoa.

> **TRẠNG THÁI: NOT RUN.** Notebook này được sinh tự động và đã kiểm cú pháp, nhưng CHƯA được chạy trên GPU/dữ liệu thật trong môi trường phát triển (không có torch, BUS-BRA, BUSI). Mọi con số chỉ có giá trị sau khi bạn chạy `RUN_MODE="full_experiment"`. `smoke_test` dùng dữ liệu tổng hợp: số liệu KHÔNG có giá trị khoa học.

**Cách dùng:** Add Input: BUS-BRA và BUSI -> bật GPU -> Run All. Quy trình (cố định trước trong `configs/bus_verify.json`):
1. Kiểm toán dữ liệu + chia theo bệnh nhân (khớp manifest V3).
2. Huấn luyện X6: 5 fold OOF x seeds trên dữ liệu phát triển (train+val), cộng mô hình cuối x seeds (resume được).
3. Fit T-scaling và ngưỡng Youden CHỈ trên dự đoán OOF của dữ liệu phát triển.
4. Trích đặc trưng (độ tin cậy, bất định, ổn định, counterfactual trên mask DỰ ĐOÁN).
5. Giao thức benchmark: nested CV trên dev (xác nhận chính) -> fit trên dev -> BUS-BRA test (ĐÃ khảo sát ở V3, không gọi là chưa từng thấy) và BUSI.
6. Khóa risk model + chính sách cảnh báo (sha256), robustness lab, thí nghiệm ORACLE tách riêng, ghi artifact cho web app.

Hạn chế đã biết: cross-fit (risk học từ mô hình fold, áp dụng cho mô hình cuối); ensemble chỉ cho baseline B3; BUSI không có patient ID; số FN ở test rất nhỏ nên CI rộng.
""")

code("""
import os
for d in ['bsv4/src/v3', 'bsv4/configs', 'bsv4/artifacts']:
    os.makedirs(d, exist_ok=True)
""")

def add_file(rel, src_path):
    txt = open(src_path, encoding='utf-8').read()
    assert '\n%%' not in txt
    code(f"%%writefile bsv4/{rel}\n" + txt)

md("## 1. Ghi mã nguồn gói BUS-VERIFY (nội dung = src/ của ứng dụng)")
for f in sorted(glob.glob(os.path.join(ROOT, 'src', '*.py'))):
    n = os.path.basename(f)
    if n in ('ui.py', 'report_export.py', 'app_support.py'):
        continue                                                   # chỉ giao diện web
    add_file(f'src/{n}', f)
for f in sorted(glob.glob(os.path.join(ROOT, 'src', 'v3', '*.py'))):
    add_file(f'src/v3/{os.path.basename(f)}', f)
add_file('configs/bus_verify.json', os.path.join(ROOT, 'configs', 'bus_verify.json'))
add_file('artifacts/busbra_split_manifest.sha256.json', os.path.join(ROOT, 'artifacts', 'busbra_split_manifest.csv.sha256.json'))

md("## 2. Cấu hình (cố định trước)")
code('''
import os, sys, json, time, copy, gc, hashlib, importlib
import numpy as np, pandas as pd
RUN_MODE = "full_experiment"          # "smoke_test" (dữ liệu tổng hợp, ~vài phút) | "full_experiment"
BUSBRA_ROOT, BUSI_ROOT = None, None
BUSBRA_COL_OVERRIDES = {}
OFFICIAL_TEST_FOLD = None
SPLIT_SEED = 42
CFG = json.load(open('bsv4/configs/bus_verify.json', encoding='utf-8'))
PRE = CFG['preregistered']
BASE = PRE['base_model']                          # 'X6' — khai báo trước
INPUT_DIR = '/kaggle/input'
WORK_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('./working')
SMOKE = RUN_MODE == 'smoke_test'
assert RUN_MODE in ('smoke_test', 'full_experiment')
WORK_DIR = os.path.join(WORK_DIR, RUN_MODE)
if SMOKE:
    IMG_SIZE, EPOCHS, PATIENCE, BATCH, SEEDS, K, B, ROB_N, NUM_WORKERS, PRETRAINED = 64, 2, 3, 4, [42, 43], 3, 50, 4, 0, False
else:
    IMG_SIZE, EPOCHS, PATIENCE, BATCH, SEEDS, K, B, ROB_N, NUM_WORKERS, PRETRAINED = CFG['preprocessing']['img_size'], 40, 10, 16, PRE['seeds'], PRE['oof_folds'], PRE['n_bootstrap'], 30, 2, True
SEG_THR, MIN_AREA_FRAC = CFG['preprocessing']['seg_threshold'], CFG['preprocessing']['min_area_frac']
RATES, PRIMARY_RATE, HIGH_RATE = PRE['alert_rates'], PRE['primary_alert_rate'], PRE['high_risk_alert_rate']
CKPT_DIR, OUT_DIR, FEAT_DIR, DEPLOY_DIR = (os.path.join(WORK_DIR, d) for d in ('checkpoints', 'outputs', 'features', 'deploy'))
for d in (WORK_DIR, CKPT_DIR, OUT_DIR, FEAT_DIR, DEPLOY_DIR):
    os.makedirs(d, exist_ok=True)
print(f'RUN_MODE={RUN_MODE} IMG={IMG_SIZE} EPOCHS={EPOCHS} SEEDS={SEEDS} K={K} B={B}')
if SMOKE:
    print('!' * 70 + '\\n SMOKE TEST: dữ liệu TỔNG HỢP. MỌI SỐ LIỆU KHÔNG CÓ GIÁ TRỊ KHOA HỌC.\\n' + '!' * 70)
''')

code('''
sys.path.insert(0, os.path.abspath('bsv4'))
for m in [m for m in list(sys.modules) if m.split('.')[0] in ('src', 'bus_core', 'bus_trust', 'losses_xai', 'models_xai', 'data_audit', 'bus_core_v2', 'evaluation_xai')]:
    sys.modules.pop(m, None)
importlib.invalidate_caches()
import torch
from src import data_audit as DA, pipeline as PL, benchmark as BM, failure_risk as FR, inference as INF, preprocessing as PP, segmentation as SG
from src import selective_prediction as SP, evaluation as EV, robustness as RB, counterfactual_verifier as CF, uncertainty as UN
import bus_core as core, bus_core_v2 as C2, models_xai as MX, bus_trust as TRUST
DEVICE = C2.get_device()
TRAIN_CFG = C2.TrainCfg(epochs=EPOCHS, patience=PATIENCE, batch_size=BATCH, amp=DEVICE.type == 'cuda', num_workers=NUM_WORKERS)
C2.seed_everything(SPLIT_SEED)
print('device', DEVICE, '| torch', torch.__version__)
''')

md("## 3. Dữ liệu: tìm, kiểm toán, chia theo bệnh nhân, khớp manifest V3")
code('''
if SMOKE:
    BUSBRA_ROOT = DA.make_synthetic_busbra(os.path.join(WORK_DIR, 'synthetic_busbra'), n_patients=40, seed=0)
    BUSI_ROOT = DA.make_synthetic_busi(os.path.join(WORK_DIR, 'synthetic_busi'), n=24, seed=5)
else:
    def one(c, what):
        if len(c) != 1:
            raise FileNotFoundError(f'{what}: tìm thấy {len(c)} ứng viên {c}. Đặt đường dẫn thủ công.')
        return c[0]
    BUSBRA_ROOT = BUSBRA_ROOT or one(DA.find_busbra_candidates(INPUT_DIR), 'BUS-BRA')
    BUSI_ROOT = BUSI_ROOT or one(DA.find_busi_candidates(INPUT_DIR), 'BUSI')
df_raw, info = DA.index_busbra(BUSBRA_ROOT, col_overrides=BUSBRA_COL_OVERRIDES)
df_a, rep = DA.audit_dataframe(df_raw)
df_clean, drop = DA.drop_invalid(df_a)
busi_raw, busi_info = DA.index_busi_binary(BUSI_ROOT)
busi_a, busi_rep = DA.audit_dataframe(busi_raw, mask_col='mask_paths')
busi_clean, _ = DA.drop_invalid(busi_a)
dup = DA.cross_dataset_duplicate_rows(df_clean, busi_clean)
busi_clean = busi_clean[~dup].reset_index(drop=True)
assert DA.cross_dataset_duplicates(df_clean, busi_clean)['ok']
df = DA.grouped_split(df_clean, seed=SPLIT_SEED, official_test_fold=OFFICIAL_TEST_FOLD)
leak = DA.leakage_report(df); assert leak['ok'], leak
DA.check_label_mapping(df)
df = df.reset_index(drop=True)
print('pairing:', DA.check_pairing(df))
mp = DA.write_manifest(df, os.path.join(OUT_DIR, 'busbra_split_manifest.csv'))
rec = json.load(open('bsv4/artifacts/busbra_split_manifest.sha256.json'))
MANIFEST_MATCH_V3 = (mp == rec['sha256'])
print('manifest sha256 khớp V3:', MANIFEST_MATCH_V3, '(nếu False: chia khác V3 -> phải khai báo trong báo cáo; tập test cũ KHÔNG còn là test V3)')
X, M, y = C2.load_arrays(df, IMG_SIZE)
Xb, Mb, yb = C2.load_arrays(busi_clean, IMG_SIZE, mask_col='mask_paths')
split = df['split'].values
dev_idx = np.where(split != 'test')[0]; test_idx = np.where(split == 'test')[0]
fold = DA.dev_folds(df, k=K, seed=SPLIT_SEED)
orep = DA.oof_leakage_report(df, fold); assert orep['ok'], orep
groups_all = df['group'].values
groups_busi = np.arange(len(busi_clean))              # BUSI không có patient ID: mỗi ảnh một nhóm -> CI mức ảnh (lạc quan)
print('dev', len(dev_idx), 'test', len(test_idx), 'busi', len(busi_clean), '| fold malignant rate', DA.fold_malignant_rates(df, fold))
json.dump(dict(busbra=rep, busi=busi_rep, oof_leakage=orep, manifest_sha=mp, manifest_match_v3=MANIFEST_MATCH_V3), open(os.path.join(OUT_DIR, 'data_audit_v4.json'), 'w'), indent=1, default=str)
''')

md("## 4. Huấn luyện X6: mô hình OOF (mỗi fold x seed) và mô hình cuối (resume được)")
code('''
from sklearn.model_selection import StratifiedGroupKFold
SPEC = MX.EXPERIMENTS[BASE]
def inner_val(pool):
    sgk = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=SPLIT_SEED)
    tr, va = next(sgk.split(pool, y[pool], groups_all[pool]))
    return pool[tr], pool[va]
def train_one(tag, tr_idx, va_idx, seed):
    data = dict(X=X, M=M, y=y, train=tr_idx, val=va_idx)
    model, cfgd = MX.build_experiment(BASE, pretrained=PRETRAINED)
    meta = dict(experiment=BASE, num_classes=2, dim=128, base=32, img_size=IMG_SIZE, seed=seed, model_cfg=cfgd, run_mode=RUN_MODE, tag=tag,
                pretrained_loaded=bool(getattr(model, 'pretrained_loaded', False)))
    if PRETRAINED and not meta['pretrained_loaded']:
        print(f'CẢNH BÁO [{tag}]: không nạp được pretrained -> backbone ngẫu nhiên.')
    cfg_run = copy.copy(TRAIN_CFG); cfg_run.use_ema = bool(SPEC.get('ema', False))
    r = C2.train_with_oom_fallback(tag, model, C2.make_loss_cfg(SPEC), data, cfg=cfg_run, out_dir=CKPT_DIR, seed=seed, device=DEVICE,
                                   augment=SPEC.get('augment', True), meta=meta)
    del model; gc.collect()
    return r['best_path']
PATHS = {}
t0 = time.time()
for f in range(K):                                     # OOF: huấn luyện trên dev trừ fold f
    pool = np.where((fold >= 0) & (fold != f))[0]
    tr, va = inner_val(pool)
    assert not set(np.where(fold == f)[0]) & (set(tr) | set(va))
    for s in SEEDS:
        PATHS[('oof', f, s)] = train_one(f'{BASE}_oof{f}', tr, va, s)
for s in SEEDS:                                        # mô hình cuối: train/val theo manifest (giống V3)
    PATHS[('final', 0, s)] = train_one(f'{BASE}_final', np.where(split == 'train')[0], np.where(split == 'val')[0], s)
print('huấn luyện xong trong', round((time.time() - t0) / 60, 1), 'phút')
''')

md("## 5. Dự đoán OOF -> T-scaling và ngưỡng Youden CHỈ từ dữ liệu phát triển")
code('''
from sklearn.metrics import roc_curve
MAIN = SEEDS[0]
def predictor(kind, f, s):
    return INF.load_predictor(PATHS[(kind, f, s)], DEVICE)
logp_dev = np.zeros((len(df), 2))
for f in range(K):
    idx = np.where(fold == f)[0]
    pr = predictor('oof', f, MAIN)
    logp_dev[idx] = pr(X[idx])['logits']
TEMP = TRUST.fit_temperature(logp_dev[dev_idx], y[dev_idx])
p_dev = UN.softmax(logp_dev[dev_idx], TEMP)[:, 1]
fpr, tpr, th = roc_curve(y[dev_idx], p_dev)
THR = float(th[int(np.argmax(tpr - fpr))])
THR = float(np.clip(THR, 1e-3, 1 - 1e-3))
print(f'T={TEMP:.3f}  ngưỡng Youden (dev OOF)={THR:.3f}  [không dùng nhãn test]')
''')

md("## 6. Trích đặc trưng (có cache; ensemble = cùng fold khác seed, chỉ cho baseline B3)")
code('''
def extract_cached(name, imgs, ids, main, members):
    path = os.path.join(FEAT_DIR, f'{name}.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    d = PL.extract_dataframe(imgs, ids, main, base_seed=SPLIT_SEED, T=TEMP, thr=THR, seg_thr=SEG_THR, min_area_frac=MIN_AREA_FRAC, members=members, with_cf=True,
                             progress=lambda i: print(name, i, flush=True))
    d.to_csv(path, index=False); return d
parts = []
for f in range(K):
    idx = np.where(fold == f)[0]
    main = predictor('oof', f, MAIN); mem = [predictor('oof', f, s) for s in SEEDS[1:]]
    parts.append(extract_cached(f'dev_fold{f}', X[idx], idx, main, mem))
df_dev = pd.concat(parts).sort_values('id').reset_index(drop=True)
assert (df_dev['id'].values == dev_idx).all()
fin_main = predictor('final', 0, MAIN); fin_mem = [predictor('final', 0, s) for s in SEEDS[1:]]
df_test = extract_cached('busbra_test', X[test_idx], test_idx, fin_main, fin_mem)
df_busi = extract_cached('busi', Xb, np.arange(len(Xb)), fin_main, fin_mem)
y_dev, y_test, y_busi = y[dev_idx], y[test_idx], yb
tg = {n: PL.make_error_targets(yy, d['p_mal'].values, THR) for n, yy, d in (('dev', y_dev, df_dev), ('test', y_test, df_test), ('busi', y_busi, df_busi))}
for n, t in tg.items():
    print(n, 'n=', len(t['wrong']), 'sai=', int(t['wrong'].sum()), 'FN=', int(t['is_fn'].sum()), 'FP=', int(t['is_fp'].sum()))
''')

md("## 7. Giao thức benchmark (B0–B6, P1, ablation; Holm; bootstrap theo bệnh nhân)")
code('''
n_err = int(tg['dev']['wrong'].sum())
print('số lỗi trên dev OOF:', n_err)
RISK_FITTED = n_err >= 10
if not RISK_FITTED:
    assert SMOKE, f'Quá ít ca lỗi ({n_err}) trên dev: không thể huấn luyện bộ phát hiện — dừng (không chỉnh giao thức).'
    print('SMOKE: quá ít lỗi trên dữ liệu tổng hợp, bỏ qua fit risk (chỉ ở smoke).')
if RISK_FITTED:
    split_dev = split[dev_idx]; FR.assert_no_test(split_dev)
    RES = BM.run_protocol(df_dev, tg['dev']['wrong'], tg['dev']['is_fn'], groups_all[dev_idx], split_dev,
                          {'test': df_test, 'busi': df_busi}, {'test': tg['test']['wrong'], 'busi': tg['busi']['wrong']},
                          {'test': tg['test']['is_fn'], 'busi': tg['busi']['is_fn']}, {'test': groups_all[test_idx], 'busi': groups_busi},
                          seed=SPLIT_SEED, k=K, B=B, rates=RATES)
    display(RES['dev_table'].round(3)); display(RES['dev_pairs'].round(4))
''')

md("## 8. Khóa risk model, chính sách cảnh báo, cổng phạm vi, cấu hình triển khai")
code('''
def img_stats(imgs):
    return np.stack([FR.image_stats(a, PP.valid_content_mask(a)) for a in imgs])
if RISK_FITTED:
    model = RES['model']; policy = RES['policy']
    model.save(os.path.join(DEPLOY_DIR, 'risk'))
    gate = FR.ImageStatGate(q=0.995).fit(img_stats(X[dev_idx]))
    content = dict(t_review=policy['t_review'], t_high=policy['t_high'], gate=gate.to_dict(), source=policy['source'], n_dev=policy['n_dev'])
    pol_path = os.path.join(DEPLOY_DIR, 'alert_policy.locked.json')
    if os.path.exists(pol_path): os.remove(pol_path)    # chỉ cho phép khi chạy lại TOÀN BỘ từ đầu
    FR.lock_policy(pol_path, content)
    import shutil
    shutil.copy(PATHS[('final', 0, MAIN)], os.path.join(DEPLOY_DIR, 'model.pt'))
    dcfg = dict(img_size=IMG_SIZE, temperature=TEMP, threshold=THR, seg_threshold=SEG_THR, min_area_frac=MIN_AREA_FRAC, seed=CFG['counterfactual']['seed'],
                model_name=BASE, version=CFG['version'], checkpoint='model.pt', run_mode=RUN_MODE)
    json.dump(dcfg, open(os.path.join(DEPLOY_DIR, 'deploy_config.json'), 'w'), indent=1)
    print('đã khóa:', {k: v for k, v in content.items() if k != 'gate'})
''')

md("## 9. Ghi artifact kết quả cho ứng dụng web")
code('''
sys.path.insert(0, os.path.abspath('bsv4'))
V4 = dict(dev_table='v4_dev_nested_cv_table.csv', dev_pairs='v4_dev_paired_comparisons.csv', test_table='v4_busbra_test_table.csv', test_pairs='v4_busbra_test_paired_comparisons.csv',
          busi_table='v4_busi_external_table.csv', busi_pairs='v4_busi_external_paired_comparisons.csv', ablation='v4_ablation_dev.csv', robustness='v4_robustness_summary.csv',
          oracle='v4_oracle_experiment.csv', pred_dev='v4_predictions_dev_oof.csv', pred_test='v4_predictions_busbra_test.csv', pred_busi='v4_predictions_busi.csv',
          risk_coverage='v4_risk_coverage_curves.csv', classification='v4_classification_metrics.csv', status_table='v4_alert_status_counts.csv')
def W(key, d):
    d.to_csv(os.path.join(OUT_DIR, V4[key]), index=False)
def seg_dice(pr, Xs, Ms):
    ds = []
    for i in range(0, len(Xs), 32):
        o = pr(Xs[i:i + 32])
        for sp, m in zip(o['seg_prob'], Ms[i:i + 32]):
            ds.append(SG.dice(SG.postprocess_mask(sp, SEG_THR, MIN_AREA_FRAC)[0], m > 0))
    return float(np.mean(ds))
if RISK_FITTED:
    W('dev_table', RES['dev_table']); W('dev_pairs', RES['dev_pairs'])
    W('ablation', RES['dev_table'][RES['dev_table']['method'].str.startswith(('P1', 'B4', 'B5', 'B6'))])
    for key, nm in (('test', 'test'), ('busi', 'busi')):
        W(f'{key}_table', RES['held_out'][nm]['table']); W(f'{key}_pairs', RES['held_out'][nm]['pairs'])
    bundle = PL.Bundle(DEPLOY_DIR); assert bundle.risk_enabled
    pol = bundle.policy
    sets = dict(dev=(df_dev, RES['dev_scores'], tg['dev'], y_dev, X[dev_idx]), test=(df_test, RES['held_out']['test']['scores'], tg['test'], y_test, X[test_idx]),
                busi=(df_busi, RES['held_out']['busi']['scores'], tg['busi'], y_busi, Xb))
    rc_rows, cls_rows, st_rows = [], [], []
    for nm, (d, sc, t, yy, imgs) in sets.items():
        risk = sc['P1']
        ins = (d['mask_empty'].values == 0) & np.array([bundle.gate.in_scope(s) for s in img_stats(imgs)])
        cnt, st = BM.status_counts(risk, pol, ins)
        for k_, v in cnt.items(): st_rows.append(dict(dataset=nm, status=k_, n=int(v)))
        W(f'pred_{nm}', pd.DataFrame(dict(id=d['id'], y=yy, p_mal=d['p_mal'], pred=t['pred'], wrong=t['wrong'], is_fn=t['is_fn'], risk=risk, status=st)))
        for meth, s in sc.items():
            cov, sr, _ = SP.risk_coverage_curve(s, t['wrong']); sel = np.unique(np.linspace(0, len(cov) - 1, min(200, len(cov))).astype(int))
            rc_rows += [dict(dataset=nm, method=meth, coverage=float(cov[i]), selective_risk=float(sr[i])) for i in sel]
        cm = EV.classification_metrics(yy, d['p_mal'].values, THR)
        cm.update(dataset=nm, ece=EV.ece_binary(d['p_mal'].values, yy), brier=EV.brier_binary(d['p_mal'].values, yy), nll=EV.nll_binary(d['p_mal'].values, yy))
        if nm == 'test': cm['dice_pred_mask'] = seg_dice(fin_main, X[test_idx], M[test_idx])
        if nm == 'busi': cm['dice_pred_mask'] = seg_dice(fin_main, Xb, Mb)
        cls_rows.append(cm)
    W('risk_coverage', pd.DataFrame(rc_rows)); W('classification', pd.DataFrame(cls_rows)); W('status_table', pd.DataFrame(st_rows))
    json.dump({k: RES['held_out'][k]['risk_calibration'] for k in RES['held_out']}, open(os.path.join(OUT_DIR, 'risk_calibration.json'), 'w'), indent=1)
    display(pd.DataFrame(cls_rows).round(3)); display(pd.DataFrame(st_rows))
''')

md("## 10. Robustness Lab (label-preserving vs counterfactual tách riêng)")
code('''
if RISK_FITTED:
    rng = np.random.default_rng(SPLIT_SEED); sel = rng.choice(len(test_idx), size=min(ROB_N, len(test_idx)), replace=False)
    imgs = [X[test_idx][i] for i in sel]
    def risk_fn(img):
        row, _ = PL.extract_features(img, fin_main, T=TEMP, thr=THR, seg_thr=SEG_THR, min_area_frac=MIN_AREA_FRAC, seed=0, with_cf=True)
        d = PL.add_derived(pd.DataFrame([row]))
        for c in bundle.risk.feature_names:
            if c not in d: d[c] = np.nan
        return float(bundle.risk.predict_risk(d)[0])
    rob = RB.run_robustness_lab(imgs, fin_main, None, (1, 3, 5), seed=SPLIT_SEED, T=TEMP, thr=THR, seg_thr=SEG_THR, min_area_frac=MIN_AREA_FRAC, risk_fn=risk_fn)
    rob.to_csv(os.path.join(OUT_DIR, 'v4_robustness_raw.csv'), index=False)
    rs = RB.summarize_robustness(rob); W('robustness', rs); display(rs.round(3))
''')

md("## 11. Thí nghiệm ORACLE (dùng GT mask — CHỈ phân tích, tách khỏi suy luận thật)")
code('''
if RISK_FITTED:
    rows = []
    for i in sel:
        gi = test_idx[i]; img = X[gi]; gt = (M[gi] > 0).astype(np.uint8)
        if gt.sum() == 0: continue
        o = fin_main(img[None]); valid = PP.valid_content_mask(img)
        pm = SG.postprocess_mask(o['seg_prob'][0], SEG_THR, MIN_AREA_FRAC)[0]
        for src, mk in (('predicted', pm), ('oracle_gt', gt)):
            if mk is None or mk.sum() == 0: continue
            r = CF.verify_image(img, mk, fin_main, T=TEMP, thr=THR, cfg=CF.CFConfig(seed=0), valid=valid, evidence=o['evidence'][0], rng=np.random.default_rng([0, 17]))
            rows.append(dict(id=int(gi), y=int(y[gi]), mask_source=src, dice_pred_gt=SG.dice(pm, gt), status=r['status'], **{k: v for k, v in r['features'].items() if isinstance(v, (int, float, np.floating))}))
    orc = pd.DataFrame(rows); W('oracle', orc)
    display(orc.groupby('mask_source')[[c for c in ('lesion_sens', 'bg_sens', 'cf_specificity', 'lesion_bg_contrast', 'ev_faithfulness') if c in orc]].mean().round(3))
''')

md("## 12. RUN_MANIFEST và gói tải xuống cho web app")
code('''
def sha(p): return FR.sha256_file(p)
files = {f: sha(os.path.join(OUT_DIR, f)) for f in sorted(os.listdir(OUT_DIR)) if f.endswith(('.csv', '.json'))}
man = dict(version=CFG['version'], run_mode=RUN_MODE, scientific_validity='NONE (synthetic smoke test)' if SMOKE else 'real run — xem giới hạn trong báo cáo',
           created=time.strftime('%Y-%m-%d %H:%M:%S'), torch=torch.__version__, device=str(DEVICE), base_model=BASE, seeds=SEEDS, oof_folds=K, n_bootstrap=B,
           img_size=IMG_SIZE, temperature=TEMP, threshold=THR, alert_rates=RATES, primary_rate=PRIMARY_RATE, risk_fitted=bool(RISK_FITTED),
           manifest_match_v3=MANIFEST_MATCH_V3, config_sha256=hashlib.sha256(json.dumps(CFG, sort_keys=True).encode()).hexdigest(),
           test_labels_used_for_training_or_thresholds=False,
           notes=['BUS-BRA test và BUSI đã được khảo sát ở V3: không phải tập chưa từng quan sát',
                  'risk học từ dự đoán OOF của mô hình fold, áp dụng cho mô hình cuối (cross-fit, khai báo)',
                  'ensemble chỉ dùng cho baseline B3; risk triển khai dùng TTA vì web chỉ có một checkpoint',
                  'BUSI không có patient ID: CI mức ảnh lạc quan'], artifacts=files)
if RISK_FITTED:
    man['deploy'] = {f: sha(os.path.join(DEPLOY_DIR, f)) for f in ('model.pt', 'deploy_config.json', 'alert_policy.locked.json')}
json.dump(man, open(os.path.join(OUT_DIR, 'RUN_MANIFEST.json'), 'w'), indent=1, ensure_ascii=False)
import shutil
zp = shutil.make_archive(os.path.join(WORK_DIR, 'BreastVision_V4_run_artifacts'), 'zip', WORK_DIR, 'outputs')
if RISK_FITTED: shutil.make_archive(os.path.join(WORK_DIR, 'BreastVision_V4_deploy'), 'zip', WORK_DIR, 'deploy')
print('Chép outputs/*.csv|json vào artifacts/v4/ và deploy/ vào thư mục deploy/ của app. Gói:', zp)
''')

nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
out = os.path.join(ROOT, 'notebooks', 'BreastVision_XAI_V4_BUS_VERIFY_Kaggle.ipynb')
json.dump(nb, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote', out, len(cells), 'cells')
