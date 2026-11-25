# bus_core_v2.py - dữ liệu, augmentation siêu âm (speckle + affine), EMA, TTA, engine huấn luyện/resume, Grad-CAM
import os, json, time, random, gc, copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import bus_core as core
import losses_xai as LX
from data_audit import read_rgb, read_union_mask, read_mask


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================ Nạp và tiền xử lý ============================
def load_arrays(df, img_size, mask_col='mask_path', desc=None):
    N = len(df)
    X = np.zeros((N, img_size, img_size, 3), np.uint8)
    M = np.zeros((N, img_size, img_size), np.uint8)
    for i, r in enumerate(df.itertuples(index=False)):
        img = read_rgb(r.image_path)
        assert img is not None, r.image_path
        mp = getattr(r, mask_col)
        paths = mp if isinstance(mp, (list, tuple)) else [mp]
        X[i] = core.preprocess_image(img, img_size)
        M[i] = core.preprocess_mask(read_union_mask(paths, img.shape[:2]), img_size)
    return X, M, df['label'].values.astype(np.int64)


def augment_pair(img, m, speckle=True):
    # augmentation đồng bộ ảnh–mask + mô phỏng nhiễu nhân speckle đặc trưng siêu âm (tăng chịu đựng domain shift)
    h, w = m.shape
    m0 = m
    if random.random() < 0.5:
        img, m = img[:, ::-1], m[:, ::-1]
    img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
    if random.random() < 0.8:
        A = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-15, 15), random.uniform(0.85, 1.15))
        A[:, 2] += (random.uniform(-0.08, 0.08) * w, random.uniform(-0.08, 0.08) * h)
        it = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mt = cv2.warpAffine(m, A, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if not (m0.sum() > 0 and mt.sum() == 0):
            img, m = it, mt
    out = img.astype(np.float32)
    if random.random() < 0.8:
        out = out * random.uniform(0.8, 1.2) + random.uniform(-20, 20)
    if random.random() < 0.3:
        out = 255.0 * (np.clip(out, 0, 255) / 255.0) ** random.uniform(0.8, 1.25)
    if speckle and random.random() < 0.25:
        # nhiễu nhân speckle + nhiễu cộng nhẹ
        out = out * (1.0 + np.random.normal(0, random.uniform(0.02, 0.06), out.shape[:2] + (1,)))
    elif random.random() < 0.3:
        out = out + np.random.normal(0, random.uniform(3, 10), out.shape)
    return np.clip(out, 0, 255).astype(np.uint8), m


class BUSDataset(Dataset):
    def __init__(self, X, M, y, indices, augment=False):
        self.X, self.M, self.y, self.indices, self.augment = X, M, y, np.asarray(indices), augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        k = int(self.indices[i])
        img, m = self.X[k], self.M[k]
        if self.augment:
            img, m = augment_pair(img, m)
        return (core.normalize_to_tensor(img), torch.from_numpy(np.ascontiguousarray(m[None].astype(np.float32))),
                torch.tensor(int(self.y[k]), dtype=torch.long), k)


def _winit(w):
    s = (torch.initial_seed() + w) % (2 ** 32)
    random.seed(s); np.random.seed(s)


def make_loader(X, M, y, indices, batch_size, augment, shuffle, num_workers=0, seed=42):
    g = torch.Generator(); g.manual_seed(seed)
    ds = BUSDataset(X, M, y, indices, augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=(shuffle and len(ds) > batch_size), num_workers=num_workers,
                      worker_init_fn=_winit, generator=g, pin_memory=torch.cuda.is_available())


# ============================ Dự đoán (chuẩn & 2-view TTA) ============================
@torch.no_grad()
def predict_arrays(model, X, indices, device, batch_size=32, amp=False, tta=False):
    model.eval()
    probs, segp, evid = [], [], []
    for i in range(0, len(indices), batch_size):
        idx = indices[i:i + batch_size]
        x = torch.stack([core.normalize_to_tensor(X[k]) for k in idx]).to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            o = model(x)
            if tta:
                of = model(x.flip(-1))
        if o.get('logits') is not None:
            p = F.softmax(o['logits'].float(), 1)
            if tta:
                p = 0.5 * (p + F.softmax(of['logits'].float(), 1))
            probs.append(p.cpu().numpy())
        if o.get('seg') is not None:
            sp = torch.sigmoid(o['seg'].float())
            if tta:
                sp = 0.5 * (sp + torch.sigmoid(of['seg'].float()).flip(-1))
            segp.append(sp.squeeze(1).cpu().numpy().astype(np.float16))
        if o.get('evidence') is not None:
            ev = o['evidence'].float()
            if tta:
                ev = 0.5 * (ev + of['evidence'].float().flip(-1))
                ev = ev / (ev.sum((2, 3), keepdim=True) + 1e-8)
            evid.append(ev.cpu().numpy()[:, 0])
    cat = lambda a: np.concatenate(a) if a else None
    return dict(idx=np.asarray(indices), prob=cat(probs), seg_prob=cat(segp), evidence=cat(evid))


# ============================ EMA helper (V3) ============================
class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow


# ============================ Engine huấn luyện ============================
class TrainCfg:
    def __init__(self, epochs=40, patience=10, batch_size=16, lr_enc=1e-4, lr_new=1e-3, weight_decay=1e-4, grad_clip=1.0,
                 amp=True, num_workers=2, lr_patience=4, nan_limit=5, class_weights=True, use_ema=False, ema_decay=0.995):
        self.__dict__.update(locals()); self.__dict__.pop('self')


def _param_groups(model, cfg):
    enc, new = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if (n.startswith('enc.') or n.startswith('model.features') or '.features.' in n) else new).append(p)
    gs = [{'params': new, 'lr': cfg.lr_new}] if new else []
    if enc:
        gs.append({'params': enc, 'lr': cfg.lr_enc})
    return gs


def _auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')


def val_metrics(pred, M, y):
    out = {}
    idx = pred['idx']
    if pred['prob'] is not None:
        out['auroc'] = _auroc(y[idx], pred['prob'][:, 1]) if pred['prob'].shape[1] == 2 else float('nan')
    if pred['seg_prob'] is not None:
        pb, gb = pred['seg_prob'].astype(np.float32) >= 0.5, M[idx] > 0
        inter = (pb & gb).sum((1, 2)); den = pb.sum((1, 2)) + gb.sum((1, 2))
        has = gb.sum((1, 2)) > 0
        out['dice'] = float((2 * inter / (den + 1e-7))[has].mean()) if has.any() else float('nan')
    return out


def selection_score(m):
    v = [m[k] for k in ('auroc', 'dice') if k in m and np.isfinite(m[k])]
    return float(np.mean(v)) if v else -1.0


def make_loss_cfg(spec):
    return LX.LossCfg(**spec.get('loss', {})) if spec['kind'] == 'xai' else LX.LossCfg()


def train_model(name, model, loss_cfg, data, cfg, out_dir, seed, device, augment=True, resume=True, meta=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    tag = f'{name}_s{seed}'
    best_p, last_p, hist_p = (os.path.join(out_dir, f'{tag}_{k}') for k in ('best.pt', 'last.pt', 'history.json'))
    seed_everything(seed)
    X, M, y = data['X'], data['M'], data['y']
    use_cons = loss_cfg.lam_cons_pred > 0 or loss_cfg.lam_cons_ev > 0
    use_faith = getattr(loss_cfg, 'lam_faith', 0.0) > 0
    tl = make_loader(X, M, y, data['train'], cfg.batch_size, augment, True, cfg.num_workers, seed)
    model = model.to(device)
    ema = ModelEMA(model, cfg.ema_decay) if getattr(cfg, 'use_ema', False) else None
    cw = None
    if cfg.class_weights:
        cnt = np.bincount(y[data['train']], minlength=int(y.max()) + 1).astype(float)
        cw = torch.tensor(len(data['train']) / (len(cnt) * np.maximum(cnt, 1)), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=cfg.lr_patience)
    use_amp = cfg.amp and device.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ep0, best, best_ep, bad, hist, done = 0, -1e9, -1, 0, [], False
    if resume and os.path.exists(last_p):
        ck = core.torch_load(last_p, device)
        if not ck.get('compact'):
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sch.load_state_dict(ck['sch']); scaler.load_state_dict(ck['scaler'])
            if ema is not None and 'ema' in ck:
                ema.shadow = ck['ema']
        ep0, best, best_ep, bad, hist, done = ck['epoch'] + 1, ck['best'], ck['best_ep'], ck['bad'], ck['hist'], ck['done']
        log(f'[{tag}] resume từ epoch {ep0}' + (' (đã hoàn tất)' if done else ''))
    for ep in range(ep0, cfg.epochs):
        if done:
            break
        model.train()
        tot, n, skipped, parts_sum = 0.0, 0, 0, {}
        t0 = time.time()
        for xb, mb, yb, _ in tl:
            xb, mb, yb = xb.to(device, non_blocking=True), mb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out_aug = theta = logits_hi = logits_lo = None
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(xb)
                if use_cons:
                    theta = LX.sample_affine(xb.shape[0], device)
                    out_aug = model(LX.apply_label_preserving(xb, theta))
                if use_faith and out.get('evidence') is not None and hasattr(model, 'forward_heads_from_Z'):
                    E = out['evidence']
                    q = getattr(loss_cfg, 'faith_q', 0.2)
                    thr_hi = torch.quantile(E.flatten(1), 1.0 - q, dim=1, keepdim=True).view(-1, 1, 1, 1)
                    thr_lo = torch.quantile(E.flatten(1), q, dim=1, keepdim=True).view(-1, 1, 1, 1)
                    keep_hi = (E < thr_hi).float()
                    keep_lo = (E > thr_lo).float()
                    logits_hi, _, _ = model.forward_heads_from_Z(out['Z'].detach(), out['mask_feat'].detach(), out.get('ring'), out.get('post'), E, ev_mask=keep_hi)
                    logits_lo, _, _ = model.forward_heads_from_Z(out['Z'].detach(), out['mask_feat'].detach(), out.get('ring'), out.get('post'), E, ev_mask=keep_lo)
            loss, parts = LX.total_loss(out, mb, yb, loss_cfg, cw, out_aug, theta, logits_hi, logits_lo)
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > cfg.nan_limit:
                    raise FloatingPointError('Quá nhiều batch NaN/Inf')
                continue
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt); scaler.update()
            if ema is not None:
                ema.update(model)
            tot += float(loss.detach()) * xb.size(0); n += xb.size(0)
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v * xb.size(0)
        if n == 0:
            raise RuntimeError('Epoch không có batch hợp lệ')
        if ema is not None:
            cur_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict())
            vp = predict_arrays(model, X, data['val'], device, amp=use_amp)
            eval_state = {k: v.detach().clone() for k, v in ema.state_dict().items()}
            model.load_state_dict(cur_state)
        else:
            vp = predict_arrays(model, X, data['val'], device, amp=use_amp)
            eval_state = model.state_dict()
        vm = val_metrics(vp, M, y)
        score = selection_score(vm)
        sch.step(score)
        imp = score > best + 1e-4
        if imp:
            best, best_ep, bad = score, ep, 0
            torch.save(dict(model=eval_state, meta=dict(meta or {}, name=name, seed=seed, best_score=best, best_epoch=ep + 1)), best_p)
        else:
            bad += 1
        done = bad >= cfg.patience or ep == cfg.epochs - 1
        rec = dict(epoch=ep + 1, train_loss=tot / n, skipped=skipped, score=float(score), time_s=time.time() - t0,
                   **{f'loss_{k}': v / n for k, v in parts_sum.items()}, **{f'val_{k}': float(v) for k, v in vm.items()})
        hist.append(rec)
        if done:
            torch.save(dict(compact=True, epoch=ep, best=best, best_ep=best_ep, bad=bad, hist=hist, done=True), last_p)
        else:
            ck_last = dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), scaler=scaler.state_dict(), epoch=ep, best=best,
                           best_ep=best_ep, bad=bad, hist=hist, done=False)
            if ema is not None:
                ck_last['ema'] = ema.state_dict()
            torch.save(ck_last, last_p)
        json.dump(hist, open(hist_p, 'w'))
        log(f'[{tag}] ep {ep + 1:3d}/{cfg.epochs} loss {rec["train_loss"]:.4f} ' + ' '.join(f'{k}={v:.3f}' for k, v in vm.items()) + f' score {score:.4f}' + (' *best*' if imp else ''))
        if bad >= cfg.patience:
            log(f'[{tag}] early stopping'); break
    del opt, tl
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return dict(name=name, seed=seed, best_path=best_p, last_path=last_p, history_path=hist_p, best_score=best, best_epoch=best_ep + 1)


def train_with_oom_fallback(*args, cfg=None, **kw):
    bs = cfg.batch_size
    while True:
        oom = False
        try:
            cfg.batch_size = bs
            return train_model(*args, cfg=cfg, **kw)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and bs > 2:
                oom = True
            else:
                raise
        if oom:
            gc.collect(); torch.cuda.empty_cache(); bs //= 2
            print('!! hết bộ nhớ GPU -> batch', bs)


# ============================ Checkpoint ============================
def save_model_checkpoint(path, model, meta):
    torch.save(dict(model=model.state_dict(), meta=meta), path)


def load_model_from_checkpoint(path, device, pretrained=False):
    from models_xai import build_experiment
    ck = core.torch_load(path, device)
    meta = ck['meta']
    model, _ = build_experiment(meta['experiment'], num_classes=meta.get('num_classes', 2), pretrained=pretrained,
                                dim=meta.get('dim', 128), base=meta.get('base', 32))
    model.load_state_dict(ck['model'])
    return model.to(device).eval(), meta


# ============================ Grad-CAM cho mô hình trả dict ============================
def gradcam_dict(model, x, class_idx=None, layer=None):
    layer = layer if layer is not None else (model.neck if hasattr(model, 'neck') else model.model.features[-1])
    store = {}

    def hook(m, i, o):
        store['a'] = o
        o.register_hook(lambda g: store.__setitem__('g', g))
    h = layer.register_forward_hook(hook)
    try:
        model.eval()
        with torch.enable_grad():
            xin = x.detach().clone().requires_grad_(True)
            lg = model(xin)['logits']
            idx = int(lg.argmax(1)) if class_idx is None else int(class_idx)
            model.zero_grad(set_to_none=True)
            lg[0, idx].backward()
        w = store['g'].detach().mean((2, 3), keepdim=True)
        cam = F.relu((w * store['a'].detach()).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), idx
    finally:
        h.remove()


# ============================ Phân tích một ảnh (dùng cho ứng dụng) ============================
def analyze_image_v2(rgb, model, dcfg, device):
    import bus_trust as T
    size = int(dcfg['img_size'])
    img = core.preprocess_image(rgb, size)
    x = core.normalize_to_tensor(img).unsqueeze(0).to(device)
    model.eval()
    use_tta = bool(dcfg.get('tta', False))
    with torch.no_grad():
        o = model(x)
        if use_tta:
            of = model(x.flip(-1))
            lg = 0.5 * (F.softmax(o['logits'].float(), 1) + F.softmax(of['logits'].float(), 1))
            raw = lg[0].cpu().numpy()
            sp_t = 0.5 * (torch.sigmoid(o['seg'].float()) + torch.sigmoid(of['seg'].float()).flip(-1))
        else:
            raw = F.softmax(o['logits'].float(), 1)[0].cpu().numpy()
            sp_t = torch.sigmoid(o['seg'].float())
    cal = T.softmax(np.log(np.clip(raw, 1e-12, 1)), float(dcfg['temperature']))
    p_mal = float(cal[1])
    conf = float(cal.max())
    res = dict(image=img, p_raw=raw, p_cal=cal, p_malignant=p_mal, label='malignant' if p_mal >= float(dcfg['threshold']) else 'benign',
               confidence=conf, abstain=bool(conf < float(dcfg['abstain_threshold'])), evidence=None, cam=None)
    sp = sp_t[0, 0].cpu().numpy()
    res['seg_prob'] = sp
    res['mask'] = (sp >= float(dcfg['seg_threshold'])).astype(np.uint8)
    res['boxes'] = core.mask_to_boxes(res['mask'], int(dcfg['min_area']))
    if o.get('evidence') is not None:
        E = o['evidence'].float()
        if use_tta:
            E = 0.5 * (E + of['evidence'].float().flip(-1))
        res['evidence'] = F.interpolate(E / (E.sum() + 1e-12), size=(size, size), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    res['cam'], _ = gradcam_dict(model, x, class_idx=int(np.argmax(cal)))
    return res
