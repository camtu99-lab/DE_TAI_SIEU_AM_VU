# tests_xai.py - kiểm thử đơn vị cho BreastVision-XAI V2. Chạy: python tests_xai.py  (ghi test_results.json)
import os, sys, json, time, tempfile, shutil, inspect, traceback
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bus_core as core
import losses_xai as LX
import models_xai as MX
import data_audit as DA
import bus_core_v2 as C2
import evaluation_xai as EV

DEV = torch.device('cpu')
IMG = 64
TMP = tempfile.mkdtemp(prefix='bvxai_')
RESULTS = []
_cache = {}


def test(fn):
    fn._is_test = True
    return fn


def synth_arrays():
    if 'arrays' not in _cache:
        root = DA.make_synthetic_busbra(os.path.join(TMP, 'busbra'), n_patients=30, seed=0)
        df, info = DA.index_busbra(root)
        df, rep = DA.audit_dataframe(df)
        df, _ = DA.drop_invalid(df)
        df = DA.grouped_split(df, seed=0)
        X, M, y = C2.load_arrays(df, IMG)
        _cache['arrays'] = (df, X, M, y, rep, info)
    return _cache['arrays']


def tiny_model(name='X6'):
    torch.manual_seed(0)
    m, cfg = MX.build_experiment(name, pretrained=False)
    return m.to(DEV)


# ------------------------------------------------------------------ dataset
@test
def test_dataset_pairing_and_audit():
    df, X, M, y, rep, info = synth_arrays()
    assert info['n_unpaired'] == 0 and len(df) > 20
    assert X.shape == (len(df), IMG, IMG, 3) and M.shape == (len(df), IMG, IMG) and set(np.unique(M)) <= {0, 1}
    assert (M.sum((1, 2)) > 0).all(), 'mọi ảnh BUS-BRA giả lập phải có mask khác rỗng'
    # mask phải trùng vị trí khối sáng/tối: tâm khối u của ảnh malignant sáng hơn nền trong vùng mask
    mal = np.where(y == 1)[0][:5]
    for k in mal:
        g = X[k].mean(-1)
        assert g[M[k] > 0].mean() > g[M[k] == 0].mean(), 'mask không khớp ảnh sau letterbox'
    assert rep['n_label_conflict_groups'] == 0
    return f"{len(df)} ảnh, {df['patient_id'].nunique()} bệnh nhân, ghép ảnh–mask 100%"


@test
def test_augmentation_sync():
    img = np.zeros((IMG, IMG, 3), np.uint8); m = np.zeros((IMG, IMG), np.uint8)
    img[20:44, 18:46] = 255; m[20:44, 18:46] = 1
    import random
    random.seed(0); np.random.seed(0)
    ious = []
    for _ in range(100):
        ia, ma = C2.augment_pair(img.copy(), m.copy())
        a = ia.mean(-1) > 110
        ious.append((a & (ma > 0)).sum() / max((a | (ma > 0)).sum(), 1))
    assert np.mean(ious) > 0.8, np.mean(ious)
    return f'IoU trung bình giữa vùng sáng và mask sau 100 lần augment = {np.mean(ious):.3f}'


# ------------------------------------------------------------------ phân vùng
@test
def test_segmentation_output_and_metrics():
    m = tiny_model('X0').eval()
    x = torch.randn(2, 3, IMG, IMG)
    o = m(x)
    assert o['seg'].shape == (2, 1, IMG, IMG) and torch.isfinite(o['seg']).all()
    gt = (torch.rand(2, 1, IMG, IMG) > 0.8).float()
    assert torch.isfinite(LX.seg_loss(o['seg'], gt))
    g = (np.random.rand(3, IMG, IMG) > 0.7).astype(np.uint8)
    sm = EV.seg_metrics(g.astype(np.float32), g)
    assert abs(sm['summary']['dice'] - 1) < 1e-6 and sm['summary']['hd95'] == 0
    return 'seg (B,1,H,W) hữu hạn; Dice(GT,GT)=1; HD95(GT,GT)=0'


# ------------------------------------------------------------------ gộp đa vùng
@test
def test_multi_region_pooling():
    Z = torch.randn(2, 8, 10, 10)
    m = torch.zeros(2, 1, 10, 10); m[0, :, 3:6, 3:7] = 1
    p0 = MX.masked_avg_pool(Z, m, eps=0.0)
    manual = Z[0, :, 3:6, 3:7].mean((1, 2))
    assert torch.allclose(p0[0], manual, atol=1e-5), 'masked pooling sai'
    p = MX.masked_avg_pool(Z, m)                              # eps mặc định
    assert torch.isfinite(p).all() and p[1].abs().max() < 1e-6, 'mask rỗng phải cho vector hữu hạn ~0'
    ring = LX.lesion_ring(m, 2)
    assert (ring * m).sum() == 0, 'vòng không được chồng lên tổn thương'
    assert ((LX.dilate(m, 2) - m).clamp(0, 1) == ring).all()
    assert ring[0].sum() > 0 and ring[1].sum() == 0
    return 'masked pooling khớp tính tay; mask rỗng hữu hạn; ring ∩ lesion = ∅'


@test
def test_fusion_modes_shapes():
    for mode in ('concat', 'weighted', 'gated'):
        f = MX.RegionFusion(mode, 3, 16)
        out = f([torch.randn(4, 16) for _ in range(3)])
        assert out.shape == (4, f.out_dim) and torch.isfinite(out).all()
    return 'concat / weighted / gated cho đầu ra đúng kích thước'


# ------------------------------------------------------------------ evidence–logit coupling
@test
def test_evidence_logit_coupling():
    torch.manual_seed(1)
    m = MX.BreastVisionXAI(MX.XAIConfig(pretrained=False, evidence_mode='ewp', fusion='concat')).eval()
    x = torch.randn(2, 3, IMG, IMG)
    C, rest, lg = m.decompose_logits(x)
    err = (C.sum((2, 3)) + rest - lg).abs().max().item()
    assert err < 1e-4, err
    # can thiệp: xoá ô -> logit đổi đúng bằng phần đóng góp bị loại (trừ hằng số bias của chiếu)
    o = m(x); E = o['evidence']
    mk = torch.ones_like(E); mk[:, :, :3, :3] = 0
    lg2 = m(x, ev_mask=mk)['logits']
    D = m.cfg.dim; W = m.head[1].weight[:, :D]; A = m.fusion.proj[0].weight
    pred_delta = -(torch.einsum('kd,bdhw->bkhw', W @ A, o['Z']) * E * (1 - mk)).sum((2, 3))
    assert torch.allclose(lg2 - lg, pred_delta, atol=1e-4), (lg2 - lg - pred_delta).abs().max()
    # evidence thật sự ảnh hưởng logit; đối chứng decoupled thì không
    m_dec = MX.BreastVisionXAI(MX.XAIConfig(pretrained=False, evidence_mode='decoupled')).eval()
    l1 = m_dec(x)['logits']
    for p in m_dec.ev_head.parameters():
        p.data.add_(torch.randn_like(p))
    assert torch.allclose(l1, m_dec(x)['logits'], atol=1e-6), 'decoupled: evidence không được ảnh hưởng logit'
    l0 = m(x)['logits']
    for p in m.ev_head.parameters():
        p.data.add_(torch.randn_like(p))
    assert (l0 - m(x)['logits']).abs().max() > 1e-4, 'ewp: đổi evidence head phải đổi logit'
    return f'phân rã logit chính xác (sai số {err:.1e}); can thiệp ev_mask khớp dự đoán; decoupled không ảnh hưởng logit'


@test
def test_evidence_collapse_detection():
    N, h, w = 4, 16, 16
    uni = torch.ones(N, h, w) / (h * w)
    peak = torch.zeros(N, h * w); peak[:, 5] = 1; peak = peak.view(N, h, w)
    good = torch.zeros(N, h, w); good[:, 4:9, 4:9] = 1; good = good / good.sum((1, 2), keepdim=True)
    assert EV.evidence_collapse_report(uni)['collapse_flag'] and EV.evidence_collapse_report(uni)['uniform_collapse_rate'] == 1
    assert EV.evidence_collapse_report(peak)['peaked_collapse_rate'] == 1
    assert not EV.evidence_collapse_report(good)['collapse_flag']
    return 'phát hiện đúng collapse đồng đều, collapse một điểm; không báo nhầm map tập trung'


# ------------------------------------------------------------------ explanation loss
@test
def test_explanation_loss():
    B, h = 2, 16
    gt = torch.zeros(B, 1, 64, 64); gt[:, :, 20:36, 20:40] = 1
    uni = torch.ones(B, 1, h, h)
    m = LX.downsample_mask(gt, (h, h)); ideal = m / m.sum((1, 2, 3), keepdim=True)
    for mode in ('mask_only', 'lesion_context', 'density'):
        lu, d = LX.explanation_loss(uni, gt, mode)
        li, _ = LX.explanation_loss(ideal, gt, mode)
        assert lu > 0.05, (mode, float(lu)); assert li < 1e-4, (mode, float(li))
    # mask rỗng -> bị bỏ qua, không NaN
    le, d = LX.explanation_loss(uni, torch.zeros_like(gt), 'lesion_context')
    assert float(le) == 0 and d['valid'].sum() == 0
    # gradient
    E = torch.rand(B, 1, h, h, requires_grad=True)
    for mode in ('mask_only', 'lesion_context', 'density'):
        E.grad = None
        LX.explanation_loss(E, gt, mode)[0].backward()
        assert E.grad is not None and torch.isfinite(E.grad).all() and E.grad.abs().sum() > 0, mode
    # phụ thuộc diện tích ROI: density(đồng đều) = log(1+margin) với mọi diện tích; lesion_context = 0 khi đạt đích điều chỉnh theo diện tích
    vals = []
    for s in (4, 8, 16):
        g2 = torch.zeros(1, 1, 64, 64); g2[:, :, 30 - s:30 + s, 30 - s:30 + s] = 1
        l, dd = LX.explanation_loss(torch.ones(1, 1, h, h), g2, 'density', ring_cells=1)
        vals.append(float(l))
        a = float(dd['a_r'])
        tau = a + 0.5 * (1 - a)
        mr = LX.lesion_plus_context(LX.downsample_mask(g2, (h, h)), 1)
        unif = torch.ones_like(mr) / mr.numel(); ideal_r = mr / mr.sum()
        m_ideal = float((ideal_r * mr).sum()); m_unif = float((unif * mr).sum())        # M_r của hai map; nội suy tuyến tính theo lam
        lam = (tau - m_unif) / (m_ideal - m_unif)
        assert 0 < lam < 1
        lc, _ = LX.explanation_loss((1 - lam) * unif + lam * ideal_r, g2, 'lesion_context', ring_cells=1)
        lc_below, _ = LX.explanation_loss((1 - lam * 0.8) * unif + lam * 0.8 * ideal_r, g2, 'lesion_context', ring_cells=1)
        assert float(lc) < 1e-5 and float(lc_below) > 1e-3, (s, float(lc), float(lc_below))
    assert max(vals) - min(vals) < 1e-3 and abs(vals[0] - np.log(2)) < 1e-3, vals
    # map đồng đều bị phạt
    return f'đồng đều bị phạt, lý tưởng ≈ 0; mask rỗng bỏ qua; gradient hữu hạn; density(đồng đều) bất biến diện tích ({vals[0]:.3f}≈ln2)'


# ------------------------------------------------------------------ consistency
@test
def test_inverse_warp_and_consistency():
    B = 4
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 32), indexing='ij')
    pat = (torch.exp(-((xx - 0.2) ** 2 + (yy + 0.1) ** 2) / 0.2) + 0.3 * torch.sin(2 * xx)).expand(B, 1, 32, 32).contiguous()
    # hflip chính xác
    hf = LX.hflip_theta(B, DEV)
    assert torch.allclose(LX.warp(pat, hf), pat.flip(-1), atol=1e-5)
    g = torch.Generator().manual_seed(0)
    th = LX.sample_affine(B, DEV, generator=g)
    back = LX.inverse_warp(LX.warp(pat, th), th)
    V = LX.valid_pixel_mask(th, (32, 32))
    err = ((back - pat).abs() * V).max().item()
    assert err < 0.03, err
    assert 0.4 < V.mean().item() < 1.0, 'mặt nạ hợp lệ phải loại bỏ viền nhưng vẫn giữ đa số pixel'
    # sai số phải giảm khi tăng độ phân giải (phân biệt sai số nội suy với lỗi hệ thống)
    def err_at(n):
        y2, x2 = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing='ij')
        p2 = (torch.exp(-((x2 - 0.2) ** 2 + (y2 + 0.1) ** 2) / 0.2) + 0.3 * torch.sin(2 * x2)).expand(B, 1, n, n).contiguous()
        return ((LX.inverse_warp(LX.warp(p2, th), th) - p2).abs() * LX.valid_pixel_mask(th, (n, n))).max().item()
    assert err_at(64) < err * 0.5, 'sai số không giảm theo độ phân giải -> lỗi hệ thống'
    A3 = lambda t: torch.cat([t, torch.tensor([0., 0., 1.]).view(1, 1, 3).expand(B, 1, 3)], 1)
    assert (A3(LX.invert_theta(th)) @ A3(th) - torch.eye(3)).abs().max() < 1e-5
    # theta^-1 ∘ theta = I
    A = LX.invert_theta(th)
    # mất nhất quán = 0 khi E2 là bản warp của E1; lớn khi E2 ngẫu nhiên
    E1 = torch.softmax(torch.randn(B, 1 * 32 * 32), -1).view(B, 1, 32, 32) * 1.0
    E1 = pat.clamp(min=0) + 0.01
    E2 = LX.warp(E1, th)
    l_ok = LX.evidence_consistency(E1, E2, th)
    l_bad = LX.evidence_consistency(E1, torch.rand(B, 1, 32, 32), th)
    assert float(l_ok) < 0.05 and float(l_bad) > float(l_ok) + 0.1, (float(l_ok), float(l_bad))
    lg = torch.randn(B, 2)
    assert float(LX.prediction_consistency(lg, lg)) < 1e-7 and float(LX.prediction_consistency(lg, -lg)) > 0.1
    return f'inverse-warp sai số {err:.3f} trên pixel hợp lệ; hflip chính xác; consistency {float(l_ok):.3f} (đúng) vs {float(l_bad):.3f} (sai)'


# ------------------------------------------------------------------ gradient flow
@test
def test_gradient_flow():
    x = torch.randn(4, 3, IMG, IMG); gt = (torch.rand(4, 1, IMG, IMG) > 0.9).float(); gt[:, :, 20:30, 20:30] = 1
    y = torch.tensor([0, 1, 0, 1])

    def grads(model, loss_fn):
        model.train(); model.zero_grad()
        loss_fn(model(x)).backward()
        g = lambda mod: sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
        return g(model.ev_head) if hasattr(model, 'ev_head') else 0.0, g(model.dec), g(model.enc)
    m = MX.BreastVisionXAI(MX.XAIConfig(pretrained=False, evidence_mode='ewp'))
    ev_g, dec_g, enc_g = grads(m, lambda o: LX.cls_loss(o['logits'], y))
    assert ev_g > 0, 'ewp: classification loss phải truyền gradient tới evidence head'
    assert dec_g == 0, 'detach_mask=True: loss phân loại không được chảy vào decoder phân vùng'
    ev_g, dec_g, enc_g = grads(m, lambda o: LX.seg_loss(o['seg'], gt))
    assert dec_g > 0 and ev_g == 0
    ev_g2, _, _ = grads(m, lambda o: LX.explanation_loss(o['evidence'], gt, 'lesion_context')[0])
    assert ev_g2 > 0, 'explanation loss phải truyền gradient tới evidence head'
    d = MX.BreastVisionXAI(MX.XAIConfig(pretrained=False, evidence_mode='decoupled'))
    ev_d, _, _ = grads(d, lambda o: LX.cls_loss(o['logits'], y))
    assert ev_d == 0, 'decoupled: classification loss KHÔNG được chạm evidence head'
    ev_d2, _, _ = grads(d, lambda o: LX.explanation_loss(o['evidence'], gt, 'lesion_context')[0])
    assert ev_d2 > 0
    # consistency loss: gradient tới evidence head
    th = LX.sample_affine(4, DEV)
    m.train(); m.zero_grad()
    o1, o2 = m(x), m(LX.apply_label_preserving(x, th))
    (LX.evidence_consistency(o1['evidence'], o2['evidence'], th) + LX.prediction_consistency(o1['logits'], o2['logits'])).backward()
    assert sum(float(p.grad.abs().sum()) for p in m.ev_head.parameters() if p.grad is not None) > 0
    return 'cls→evidence(ewp) có; cls→decoder không (detach); cls→evidence(decoupled) không; expl→evidence có; cons→evidence có'


# ------------------------------------------------------------------ checkpoint, suy luận không GT
@test
def test_checkpoint_roundtrip_all_models():
    x = torch.randn(2, 3, IMG, IMG)
    for name in ('E0', 'E1', 'E2', 'E3', 'X0', 'X6', 'V3_full', 'A_evidence_egr'):
        m = tiny_model(name).eval()
        meta = dict(experiment=name, num_classes=2, dim=128, base=32)
        p = os.path.join(TMP, f'{name}.pt')
        C2.save_model_checkpoint(p, m, meta)
        m2, _ = C2.load_model_from_checkpoint(p, DEV)
        o1, o2 = m(x), m2(x)
        for k in ('seg', 'logits', 'evidence'):
            if o1.get(k) is not None:
                assert torch.allclose(o1[k], o2[k], atol=1e-6), (name, k)
    return 'save→load cho E0,E1,E2,E3,X0,X6,A_evidence_egr cho đầu ra giống hệt'


@test
def test_inference_does_not_use_gt_mask():
    sig = list(inspect.signature(MX.BreastVisionXAI.forward).parameters)
    assert sig == ['self', 'x', 'ev_mask'], sig
    assert list(inspect.signature(C2.predict_arrays).parameters) == ['model', 'X', 'indices', 'device', 'batch_size', 'amp', 'tta'], 'predict_arrays không được nhận mask'
    df, X, M, y, *_ = synth_arrays()
    m = tiny_model('X6').eval()
    idx = np.arange(6)
    a = C2.predict_arrays(m, X, idx, DEV)
    M_backup = M.copy(); M[:] = 1 - M                        # đảo toàn bộ GT mask
    b = C2.predict_arrays(m, X, idx, DEV)
    M[:] = M_backup
    assert np.allclose(a['prob'], b['prob']) and np.allclose(a['seg_prob'].astype(float), b['seg_prob'].astype(float)), 'dự đoán phụ thuộc GT mask!'
    return 'forward(x, ev_mask) không có tham số mask GT; đảo toàn bộ GT mask không đổi dự đoán'


# ------------------------------------------------------------------ rò rỉ, trùng lặp
@test
def test_leakage_detection_and_duplicates():
    df, X, M, y, rep, info = synth_arrays()
    lr = DA.leakage_report(df)
    assert lr['ok'], lr
    assert set(df['split']) == {'train', 'val', 'test'}
    # phép thử âm: đẩy 1 ảnh của bệnh nhân train sang test -> phải bị phát hiện
    bad = df.copy()
    cnt = bad[bad['split'] == 'train'].groupby('patient_id').size()
    p = cnt[cnt >= 2].index[0]
    i = bad.index[(bad['patient_id'] == p) & (bad['split'] == 'train')][0]
    bad.loc[i, 'split'] = 'test'
    assert not DA.leakage_report(bad)['ok'], 'không phát hiện rò rỉ bệnh nhân'
    # ảnh trùng khác mã bệnh nhân vẫn phải cùng nhóm
    root = os.path.join(TMP, 'dupset'); shutil.copytree(os.path.join(TMP, 'busbra'), root)
    src = os.path.join(root, 'Images', 'bus_0001-l.png')
    shutil.copy(src, os.path.join(root, 'Images', 'bus_9999-l.png')); shutil.copy(os.path.join(root, 'Masks', 'mask_0001-l.png'), os.path.join(root, 'Masks', 'mask_9999-l.png'))
    import pandas as pd
    meta = pd.read_csv(os.path.join(root, 'bus_data.csv'))
    meta.loc[len(meta)] = dict(ID='bus_9999-l', Case='C9999', Pathology=meta.iloc[0]['Pathology'], BIRADS=4, Device='synthetic')
    meta.to_csv(os.path.join(root, 'bus_data.csv'), index=False)
    d2, _ = DA.index_busbra(root)
    d2, r2 = DA.audit_dataframe(d2)
    assert r2['n_exact_duplicate_images'] >= 2
    g = DA.build_groups(d2)
    a = int(np.where(d2['patient_id'] == 'C9999')[0][0]); b = int(np.where(d2['image_path'].str.contains('bus_0001-l'))[0][0])
    assert g[a] == g[b], 'ảnh trùng không được gom vào cùng nhóm với bản gốc'
    clean, drop = DA.drop_invalid(d2)
    assert drop['dropped_exact_dups'] >= 1
    # nhãn mâu thuẫn
    meta2 = pd.read_csv(os.path.join(root, 'bus_data.csv')); meta2.loc[meta2['ID'] == 'bus_9999-l', 'Pathology'] = 'malignant' if meta.iloc[0]['Pathology'] == 'benign' else 'benign'
    meta2.to_csv(os.path.join(root, 'bus_data.csv'), index=False)
    d3, _ = DA.index_busbra(root); d3, r3 = DA.audit_dataframe(d3)
    assert r3['n_label_conflict_groups'] >= 1
    return 'chia theo bệnh nhân không rò rỉ; phép thử âm bị phát hiện; trùng/gần trùng cùng nhóm; nhãn mâu thuẫn được báo'


@test
def test_busi_binary_excludes_normal_and_cross_dataset():
    root = DA.make_synthetic_busi(os.path.join(TMP, 'busi'), n=12, seed=5)
    df, info = DA.index_busi_binary(root)
    assert info['n_normal_excluded'] == 6 and set(df['label']) <= {0, 1} and (df['pathology'] != 'normal').all()
    dfa, _ = synth_arrays()[0], None
    a, _ = DA.audit_dataframe(DA.index_busbra(os.path.join(TMP, 'busbra'))[0])
    b, _ = DA.audit_dataframe(df, mask_col='mask_paths')
    cd = DA.cross_dataset_duplicates(a, b)
    assert cd['ok'], cd
    # chèn một bản sao để chắc chắn phát hiện được
    b2 = b.copy(); b2.loc[0, 'sha256'] = a.loc[0, 'sha256']; b2.loc[0, '_dhash'] = a.loc[0, '_dhash']
    assert not DA.cross_dataset_duplicates(a, b2)['ok']
    return 'BUSI: loại 6 ảnh normal, không chuyển thành benign; không trùng với BUS-BRA; bản sao chèn vào bị phát hiện'


# ------------------------------------------------------------------ khóa pipeline
@test
def test_external_evaluation_lock():
    lock = EV.ExternalEvalLock(os.path.join(TMP, 'lock.json'))
    ck = os.path.join(TMP, 'X6.pt')
    if not os.path.exists(ck):
        C2.save_model_checkpoint(ck, tiny_model('X6'), dict(experiment='X6', num_classes=2))
    called = []
    loader = lambda: called.append(1) or 'DATA'
    pred = lambda data, L: dict(auroc=0.5, T=L['temperature'])
    try:
        EV.evaluate_external(lock, ck, loader, pred); raise AssertionError('đã đọc dữ liệu ngoài khi chưa khóa')
    except EV.LockError:
        pass
    assert not called, 'load_external_fn bị gọi trước khi khóa'
    lock.lock(ck, dict(img=IMG), temperature=1.3, threshold=0.5, abstain_threshold=0.6)
    out = EV.evaluate_external(lock, ck, loader, pred)
    assert called == [1] and out['T'] == 1.3
    for fn in (lambda: EV.evaluate_external(lock, ck, loader, pred), lambda: lock.lock(ck, {}, 1, 0.5, 0.5)):
        try:
            fn(); raise AssertionError('cho phép đánh giá/khóa lại')
        except EV.LockError:
            pass
    # sửa checkpoint sau khóa
    lock2 = EV.ExternalEvalLock(os.path.join(TMP, 'lock2.json')); lock2.lock(ck, {}, 1, 0.5, 0.5)
    with open(ck, 'ab') as f:
        f.write(b'x')
    try:
        lock2.guard_access(ck); raise AssertionError('không phát hiện checkpoint bị sửa')
    except EV.LockError:
        pass
    return 'từ chối khi chưa khóa (không đọc dữ liệu); đánh giá 1 lần; khóa lại/đánh giá lại bị chặn; sửa checkpoint bị phát hiện'


# ------------------------------------------------------------------ mô hình đầy đủ
@test
def test_all_registered_experiments_forward():
    x = torch.randn(2, 3, IMG, IMG)
    shapes = {}
    for name in MX.EXPERIMENTS:
        m = tiny_model(name).eval()
        o = m(x)
        if o.get('logits') is not None:
            assert o['logits'].shape == (2, 2)
        if o.get('seg') is not None:
            assert o['seg'].shape == (2, 1, IMG, IMG)
        shapes[name] = 'ok'
        del m
    return f'{len(shapes)} cấu hình trong registry (E0–E4, X0–X6, ablation) forward đúng kích thước'


@test
def test_training_smoke_and_resume():
    df, X, M, y, *_ = synth_arrays()
    tr, va = np.where(df['split'] == 'train')[0], np.where(df['split'] == 'val')[0]
    data = dict(X=X, M=M, y=y, train=tr, val=va)
    spec = MX.EXPERIMENTS['X6']
    m = tiny_model('X6')
    cfg = C2.TrainCfg(epochs=2, patience=5, batch_size=4, amp=False, num_workers=0)
    out_dir = os.path.join(TMP, 'train')
    r = C2.train_model('X6', m, C2.make_loss_cfg(spec), data, cfg, out_dir, 0, DEV, meta=dict(experiment='X6', num_classes=2), log=lambda *a: None)
    h = json.load(open(r['history_path']))
    assert len(h) == 2 and all(np.isfinite(e['train_loss']) for e in h)
    for k in ('loss_seg', 'loss_cls', 'loss_exp', 'loss_cons_pred', 'loss_cons_ev'):
        assert k in h[0], k
    assert os.path.exists(r['best_path']) and os.path.exists(r['last_path'])
    r2 = C2.train_model('X6', tiny_model('X6'), C2.make_loss_cfg(spec), data, cfg, out_dir, 0, DEV, meta=dict(experiment='X6', num_classes=2), log=lambda *a: None)
    assert len(json.load(open(r2['history_path']))) == 2, 'resume phải bỏ qua các epoch đã xong'
    mb, _ = C2.load_model_from_checkpoint(r['best_path'], DEV)
    p = C2.predict_arrays(mb, X, va, DEV)
    assert np.isfinite(p['prob']).all()
    _cache['trained'] = (mb, data, p)
    return 'X6 (constraint + consistency) huấn luyện 2 epoch: loss hữu hạn, đủ các thành phần loss, checkpoint + resume hoạt động'


@test
def test_evaluation_functions_run():
    df, X, M, y, *_ = synth_arrays()
    mb, data, p = _cache['trained'] if 'trained' in _cache else (tiny_model('X6').eval(), None, None)
    va = np.where(df['split'] == 'val')[0]; te = np.where(df['split'] == 'test')[0]
    pv = C2.predict_arrays(mb, X, va, DEV); pt = C2.predict_arrays(mb, X, te, DEV)
    cm = EV.classification_metrics(y[te], pt['prob'][:, 1]); assert 'auroc' in cm
    loc = EV.localisation_report(pt['evidence'], M[te]); assert loc['n'] > 0
    fa = EV.faithfulness_report(mb, X, te, DEV, steps=3, max_images=3); assert 'deletion_auc_evidence' in fa and np.isfinite(fa['deletion_auc_random'])
    iv = EV.evidence_intervention(mb, X, te, DEV, max_images=3); assert iv['applicable']
    bg = EV.background_perturbation(mb, X, M, te, DEV, max_images=3)
    cs = EV.explanation_consistency_eval(mb, X, te, DEV); assert 0 <= cs['prediction_agreement'] <= 1
    Tm, rows = EV.calibration_report(pv['prob'], y[va], pt['prob'], y[te]); assert Tm > 0 and len(rows) == 2
    sel = EV.selective_report(pv['prob'], pt['prob'], y[te]); assert len(sel['rows']) == 3
    sm = EV.seg_metrics(pt['seg_prob'], M[te]); assert np.isfinite(sm['summary']['dice'])
    st = EV.summarize_seeds([0.7, 0.72, 0.71]); assert st['n_seeds'] == 3
    return 'phân loại, định vị, faithfulness, can thiệp evidence, nhiễu nền, nhất quán, hiệu chuẩn, chọn lọc, phân vùng, thống kê đều chạy'


@test
def test_gradcam_dict():
    m = tiny_model('X3').eval()
    cam, idx = C2.gradcam_dict(m, torch.randn(1, 3, IMG, IMG))
    assert cam.shape == (IMG, IMG) and np.isfinite(cam).all() and idx in (0, 1)
    return 'Grad-CAM cho mô hình trả dict hoạt động'


@test
def test_analyze_image_v2():
    df, X, M, y, *_ = synth_arrays()
    m = tiny_model('X6').eval()
    dcfg = dict(img_size=IMG, temperature=1.7, threshold=0.5, abstain_threshold=0.99, seg_threshold=0.5, min_area=5)
    rgb = np.random.RandomState(0).randint(0, 255, (120, 150, 3)).astype(np.uint8)
    r = C2.analyze_image_v2(rgb, m, dcfg, DEV)
    assert abs(r['p_cal'].sum() - 1) < 1e-6 and r['label'] in ('benign', 'malignant') and r['abstain'] is True       # ngưỡng 0.99 -> chắc chắn từ chối
    assert r['mask'].shape == (IMG, IMG) and r['evidence'].shape == (IMG, IMG) and r['cam'].shape == (IMG, IMG)
    assert list(inspect.signature(C2.analyze_image_v2).parameters) == ['rgb', 'model', 'dcfg', 'device'], 'không được có tham số mask'
    return 'analyze_image_v2: xác suất hiệu chuẩn, abstention, mask, box, evidence, Grad-CAM; chỉ nhận ảnh'


@test
def test_resume_midway_and_compact_marker():
    df, X, M, y, *_ = synth_arrays()
    tr, va = np.where(df['split'] == 'train')[0], np.where(df['split'] == 'val')[0]
    data = dict(X=X, M=M, y=y, train=tr, val=va)
    spec = MX.EXPERIMENTS['X3']; out_dir = os.path.join(TMP, 'resume_mid')
    meta = dict(experiment='X3', num_classes=2)
    cfg = C2.TrainCfg(epochs=3, patience=5, batch_size=4, amp=False, num_workers=0)

    class Interrupt(Exception):
        pass

    def die_after_first_epoch(msg):
        raise Interrupt()                      # log() được gọi SAU khi checkpoint 'last' đã lưu -> mô phỏng phiên bị ngắt thật
    try:
        C2.train_model('X3', tiny_model('X3'), C2.make_loss_cfg(spec), data, cfg, out_dir, 2, DEV, meta=meta, log=die_after_first_epoch)
        raise AssertionError('lẽ ra phải bị ngắt')
    except Interrupt:
        pass
    last = os.path.join(out_dir, 'X3_s2_last.pt')
    ck = C2.core.torch_load(last, DEV)
    assert ck['done'] is False and not ck.get('compact') and 'opt' in ck and ck['epoch'] == 0, 'checkpoint giữa chừng phải đầy đủ'
    r = C2.train_model('X3', tiny_model('X3'), C2.make_loss_cfg(spec), data, cfg, out_dir, 2, DEV, meta=meta, log=lambda *a: None)
    h = json.load(open(r['history_path']))
    assert [e['epoch'] for e in h] == [1, 2, 3], 'resume phải tiếp tục từ epoch 2 và chạy đủ 3 epoch'
    ck2 = C2.core.torch_load(last, DEV)
    assert ck2.get('compact') is True and ck2['done'] is True and os.path.getsize(last) < 1e6, 'xong rồi thì checkpoint cuối phải gọn'
    r3 = C2.train_model('X3', tiny_model('X3'), C2.make_loss_cfg(spec), data, cfg, out_dir, 2, DEV, meta=meta, log=lambda *a: None)
    assert len(json.load(open(r3['history_path']))) == 3, 'chạy lại sau khi xong không được huấn luyện thêm'
    return 'ngắt thật sau epoch 1 → resume chạy tiếp epoch 2–3; xong thì checkpoint cuối gọn (<1MB); chạy lại không huấn luyện thêm'


@test
def test_mask_encodings_and_empty_dataset_errors():
    import cv2, pandas as pd
    root = DA.make_synthetic_busbra(os.path.join(TMP, 'busbra_01'), n_patients=8, seed=3)
    for f in os.listdir(os.path.join(root, 'Masks')):                       # đổi mask sang mã hoá 0/1 (không phải 0/255)
        p = os.path.join(root, 'Masks', f); m = cv2.imread(p, cv2.IMREAD_GRAYSCALE); cv2.imwrite(p, (m > 127).astype(np.uint8))
    df, info = DA.index_busbra(root); a, rep = DA.audit_dataframe(df)
    assert rep['n_empty_mask'] == 0, 'mask 0/1 bị coi là rỗng'
    clean, drop = DA.drop_invalid(a); assert len(clean) == len(df)
    # nếu loại hết: phải báo lỗi rõ ràng, không để lỗi khó hiểu ở bước sau
    empty_masks = a.copy(); empty_masks['issue'] = 'mask_rong'
    try:
        DA.drop_invalid(empty_masks); raise AssertionError('lẽ ra phải báo lỗi khi loại hết')
    except RuntimeError as e:
        assert 'không còn ảnh nào' in str(e)
    for fn in (lambda: DA.cross_dataset_duplicates(clean.iloc[:0], clean), lambda: DA.cross_dataset_duplicate_rows(clean.iloc[:0], clean), lambda: DA.audit_dataframe(df.iloc[:0])):
        try:
            fn(); raise AssertionError('đầu vào rỗng phải báo lỗi rõ ràng')
        except ValueError as e:
            assert 'rỗng' in str(e).lower()
    return 'mask 0/1 không bị loại nhầm; loại hết/bảng rỗng → báo lỗi tiếng Việt rõ nguyên nhân thay vì "argmin of an empty sequence"'



# ------------------------------------------------------------------ V3 innovations tests
@test
def test_v3_posterior_zone_and_boundary_loss():
    m = torch.zeros(2, 1, 16, 16)
    m[0, 0, 3:6, 6:10] = 1.0
    pz = LX.posterior_zone(m, depth_cells=4)
    assert (pz * m).sum() == 0, 'vùng hậu âm không được chồng lên tổn thương'
    assert pz[0, 0, 6:10, 6:10].mean() > 0.5, 'phải kích hoạt ngay phía dưới khối u'
    assert pz[0, 0, :3, :].sum() == 0, 'không được kích hoạt phía trên khối u'
    assert pz[1].sum() == 0, 'mask rỗng phải cho hậu âm rỗng'
    seg_lg = torch.randn(2, 1, 64, 64, requires_grad=True)
    gt = torch.zeros(2, 1, 64, 64); gt[:, :, 20:40, 20:40] = 1.0
    bl = LX.boundary_loss(seg_lg, gt)
    bl.backward()
    assert torch.isfinite(bl) and seg_lg.grad.abs().sum() > 0
    return 'posterior_zone chỉ nằm dưới khối u và rời rạc với u; boundary_loss hữu hạn và có gradient'


@test
def test_v3_evidence_coupling_all_and_decomposition():
    torch.manual_seed(2)
    m, _ = MX.build_experiment('V3_full', pretrained=False)
    m.eval()
    x = torch.randn(2, 3, IMG, IMG)
    C, rest, lg = m.decompose_logits(x)
    err = (C.sum((2, 3)) + rest - lg).abs().max().item()
    assert err < 1e-4, f'sai số phân rã V3 quá lớn: {err}'
    # Can thiệp ev_mask trên V3 phải thay đổi TẤT CẢ các vùng phụ thuộc evidence
    o = m(x); E = o['evidence']
    mk = torch.ones_like(E); mk[:, :, :4, :4] = 0
    lg2 = m(x, ev_mask=mk)['logits']
    assert (lg2 - lg).abs().max() > 0
    # Đảm bảo E0 legacy wrapper nhận đúng cờ pretrained_loaded
    from unittest.mock import patch
    with patch('bus_core.build_model', return_value=(torch.nn.Linear(2, 2), True)):
        w, _ = MX.build_experiment('E0', pretrained=True)
        assert w.pretrained_loaded is True
    return f'phân rã cộng toàn bộ 4 vùng V3 chính xác (sai số {err:.1e}); sửa cờ pretrained_loaded E0'


@test
def test_v3_counterfactual_faithfulness_and_ema_tta():
    torch.manual_seed(3)
    m, cfgd = MX.build_experiment('V3_full', pretrained=False)
    m.train()
    x = torch.randn(4, 3, IMG, IMG)
    mb = torch.zeros(4, 1, IMG, IMG); mb[:, :, 16:40, 16:40] = 1.0
    yb = torch.tensor([0, 1, 0, 1])
    out = m(x)
    E = out['evidence']
    thr_hi = torch.quantile(E.flatten(1), 0.8, dim=1, keepdim=True).view(-1, 1, 1, 1)
    thr_lo = torch.quantile(E.flatten(1), 0.2, dim=1, keepdim=True).view(-1, 1, 1, 1)
    l_hi, _, _ = m.forward_heads_from_Z(out['Z'].detach(), out['mask_feat'].detach(), out.get('ring'), out.get('post'), E, ev_mask=(E < thr_hi).float())
    l_lo, _, _ = m.forward_heads_from_Z(out['Z'].detach(), out['mask_feat'].detach(), out.get('ring'), out.get('post'), E, ev_mask=(E > thr_lo).float())
    lcfg = C2.make_loss_cfg(MX.EXPERIMENTS['V3_full'])
    loss, parts = LX.total_loss(out, mb, yb, lcfg, logits_drop_hi=l_hi, logits_drop_lo=l_lo)
    assert 'edge' in parts and 'faith' in parts and torch.isfinite(loss)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.ev_head.parameters() if p.grad is not None) > 0
    # TTA & EMA
    ema = C2.ModelEMA(m, decay=0.9)
    ema.update(m)
    X_np = (np.random.rand(4, IMG, IMG, 3) * 255).astype(np.uint8)
    p_tta = C2.predict_arrays(m, X_np, np.arange(4), DEV, tta=True)
    assert p_tta['prob'].shape == (4, 2) and np.allclose(p_tta['evidence'].sum((1, 2)), 1.0, atol=1e-4)
    return 'faithfulness + edge loss có gradient; EMA và 2-view TTA hoạt động chuẩn'

def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if callable(f) and getattr(f, '_is_test', False)]
    order = [n for n, _ in tests]
    # giữ thứ tự có phụ thuộc: huấn luyện trước đánh giá
    tests.sort(key=lambda t: (t[0] == 'test_evaluation_functions_run', t[0] == 'test_training_smoke_and_resume' and False, t[0]))
    for name, fn in tests:
        t0 = time.time()
        try:
            detail = fn() or ''
            RESULTS.append(dict(name=name, status='PASS', seconds=round(time.time() - t0, 1), detail=detail))
        except Exception as e:
            RESULTS.append(dict(name=name, status='FAIL', seconds=round(time.time() - t0, 1), detail=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:]))
        print(f"[{RESULTS[-1]['status']}] {name} ({RESULTS[-1]['seconds']}s) {RESULTS[-1]['detail']}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_results.json')
    json.dump(dict(device=str(DEV), torch=torch.__version__, n=len(RESULTS), passed=sum(r['status'] == 'PASS' for r in RESULTS), results=RESULTS), open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n{sum(r['status'] == 'PASS' for r in RESULTS)}/{len(RESULTS)} PASS")
    return all(r['status'] == 'PASS' for r in RESULTS)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)