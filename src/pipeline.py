"""Pipeline BUS-VERIFY dùng CHUNG cho notebook Kaggle và ứng dụng Streamlit (đảm bảo parity). Không import torch: mô hình được truyền như callable."""
import os, json, time
import numpy as np
import pandas as pd

from . import preprocessing as PP
from . import segmentation as SG
from . import uncertainty as UN
from . import robustness as RB
from . import counterfactual_verifier as CF
from . import failure_risk as FR
from . import selective_prediction as SP


def extract_features(img, predict_fn, T=1.0, thr=0.5, seg_thr=0.5, min_area_frac=0.0015, cf_cfg=None, seed=0, with_shortcut=False, with_cf=True, members=None):
    """Đặc trưng cho MỘT ảnh đã tiền xử lý (uint8 HxWx3). KHÔNG nhận ground-truth. Trả (dict hàng, dict phụ cho giao diện)."""
    cf_cfg = cf_cfg or CF.CFConfig(seed=seed)
    valid = PP.valid_content_mask(img)
    stab = RB.stability_features(img, predict_fn, T=T, thr=thr, seg_thr=seg_thr, valid=valid)
    base = stab['base']
    P = UN.softmax(base['logits'][None], T)[0]
    z0 = float(UN.margin(base['logits'], T)); zthr = float(np.log(thr / (1 - thr)))
    # p_raw: softmax CHƯA hiệu chuẩn (dùng cho baseline B1); thr_margin: khoảng cách tới NGƯỠNG QUYẾT ĐỊNH (khác |logit| khi thr != 0.5)
    row = dict(p_mal=float(P[1]), p_raw=float(UN.softmax(base['logits'][None], 1.0)[0, 1]), pred=int(P[1] >= thr), z=z0, thr_margin=abs(z0 - zthr))
    if members:                                  # ensemble THẬT: các mô hình cùng fold khác seed (không bao gồm mô hình chính)
        ml = np.stack([np.asarray(f(img[None])['logits'][0], np.float64) for f in members] + [np.asarray(base['logits'], np.float64)], 0)
        row.update(UN.ensemble_features(ml, T, thr))
    row.update(stab['features'])
    mask, info = (None, dict(empty=True, area_frac=0.0, n_components=0))
    if base['seg_prob'] is not None:
        mask, info = SG.postprocess_mask(base['seg_prob'], seg_thr, min_area_frac)
    row.update(mask_area_frac=info['area_frac'], mask_components=info['n_components'], mask_empty=int(info['empty']))
    extra = dict(mask=mask, valid=valid, seg_prob=base['seg_prob'], evidence=base['evidence'], mask_info=info, view_probs=stab['probs'])
    if with_cf and mask is not None:
        res = CF.verify_image(img, mask, predict_fn, T=T, thr=thr, cfg=cf_cfg, valid=valid, evidence=base['evidence'],
                              rng=np.random.default_rng([seed, 17]), return_records=True)
        row.update(res['features']); row['cf_status'] = res['status']
        extra['cf_records'] = res.get('records', [])
    else:
        row['cf_status'] = 'skipped' if mask is not None else 'no_segmentation'
    if with_shortcut:
        row.update(CF.shortcut_probe(img, valid, predict_fn, T, thr))
    row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
    return row, extra


def extract_dataframe(images, ids, predict_fn, base_seed=0, progress=None, **kw):  # kw: T, thr, members, with_cf, ...
    """images: iterable uint8 HxWx3 (đã letterbox+resize). Seed mỗi ảnh = [base_seed, idx] -> tái lập."""
    rows = []
    for i, (img, id_) in enumerate(zip(images, ids)):
        r, _ = extract_features(img, predict_fn, seed=int(base_seed) * 1000003 + i, **kw)
        r['id'] = id_
        rows.append(r)
        if progress and (i + 1) % 50 == 0:
            progress(i + 1)
    return add_derived(pd.DataFrame(rows))


def add_derived(df):
    return FR.add_derived_features(df)


def make_error_targets(y, p_mal, thr):
    pred = (np.asarray(p_mal) >= thr).astype(int)
    y = np.asarray(y).astype(int)
    return dict(pred=pred, wrong=(pred != y).astype(int), is_fn=((y == 1) & (pred == 0)).astype(int), is_fp=((y == 0) & (pred == 1)).astype(int))


class Bundle:
    """Gói triển khai đã khóa: cấu hình, mô hình risk, chính sách cảnh báo, cổng phạm vi. Thiếu artifact -> ném RiskModelMissing (KHÔNG giả lập)."""
    def __init__(self, directory):
        self.dir = directory
        cfgp = os.path.join(directory, 'deploy_config.json')
        if not os.path.exists(cfgp):
            raise FR.RiskModelMissing(f'Không có deploy_config.json trong {directory}')
        self.cfg = json.load(open(cfgp, encoding='utf-8'))
        self.risk = None; self.policy = None; self.gate = None; self.risk_error = None
        try:
            self.risk = FR.FailureRiskModel.load(os.path.join(directory, 'risk'))
            self.policy = FR.load_policy(os.path.join(directory, 'alert_policy.locked.json'))
            if self.policy.get('gate'):
                self.gate = FR.ImageStatGate.from_dict(self.policy['gate'])
        except (FR.RiskModelMissing, FR.PolicyTamperedError) as e:
            self.risk_error = str(e)                 # ghi lý do; giao diện hiển thị rõ, KHÔNG tạo điểm giả

    @property
    def risk_enabled(self):
        return self.risk is not None and self.policy is not None


def analyze_array(rgb, predict_fn, bundle, with_cf=True):
    """Phân tích một ảnh người dùng: validate -> tiền xử lý -> đặc trưng -> (nếu có artifact) risk + trạng thái. Lỗi ảnh -> InvalidImageError."""
    t0 = time.time()
    PP.validate_image(rgb)
    cfg = bundle.cfg
    img = PP.preprocess_image(rgb, int(cfg['img_size']))
    row, extra = extract_features(img, predict_fn, T=cfg['temperature'], thr=cfg['threshold'], seg_thr=cfg['seg_threshold'],
                                  min_area_frac=cfg['min_area_frac'], seed=int(cfg.get('seed', 0)), with_cf=with_cf or bundle.risk_enabled)   # risk cần đặc trưng CF: không bao giờ tính risk với CF bị bỏ trống
    df = add_derived(pd.DataFrame([row]))
    out = dict(image=img, row=df.iloc[0].to_dict(), extra=extra, p_malignant=float(row['p_mal']),
               label='malignant' if row['pred'] == 1 else 'benign', inference_seconds=time.time() - t0, risk=None, status=None, contributions=None)
    if bundle.risk_enabled:
        for c in bundle.risk.feature_names:
            if c not in df:
                df[c] = np.nan
        r = float(bundle.risk.predict_risk(df)[0])
        oos = SG.mask_plausibility(extra['mask_info']) if extra['mask'] is not None else ['no_segmentation']
        stats = FR.image_stats(img, extra['valid'])
        if bundle.gate is not None and not bundle.gate.in_scope(stats):
            oos.append('image_statistics_out_of_range')
        out.update(risk=r, oos_reasons=oos, status=SP.decide_status(r, bundle.policy['t_review'], bundle.policy['t_high'], in_scope=not oos),
                   contributions=bundle.risk.contributions(df.iloc[0]))
    return out
