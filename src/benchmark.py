"""Benchmark phát hiện lỗi BUS-VERIFY: B0..B6, P1, ablation; đánh giá (a) NESTED CV trên dữ liệu phát triển (xác nhận chính),
(b) fit-trên-dev -> áp dụng cho tập giữ lại (test BUS-BRA cũ, BUSI). Thuần numpy/sklearn.

Quy ước: điểm CAO = nguy cơ mô hình chính dự đoán SAI cao. Nhãn lỗi `wrong` do mô hình chính tạo ra. KHÔNG dùng nhãn test để huấn luyện/chọn C/ngưỡng."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import failure_risk as FR
from . import selective_prediction as SP
from . import statistical_tests as ST
from . import evaluation as EV

BASELINE_NOTES = {
    'B0': 'Random referral (cảnh báo ngẫu nhiên, seed cố định)',
    'B1': 'Maximum softmax confidence (CHƯA hiệu chuẩn, so với 0.5)',
    'B1b': 'Threshold margin: |logit - logit(ngưỡng quyết định)| (bổ sung: ngưỡng Youden != 0.5 nên B1 không phải khoảng cách tới ranh giới thật)',
    'B2': 'Calibrated confidence (T-scaling đơn điệu => CÙNG thứ hạng với B1; chỉ khác xác suất, không khác khả năng phát hiện lỗi)',
    'B3': 'Ensemble uncertainty (JS/BALD giữa các thành viên; nếu không có ensemble thật thì dùng bất đồng giữa các góc nhìn TTA và được GHI RÕ)',
    'B4': 'Learned confidence-only error detector (LR trên nhóm confidence)',
    'B5': 'Counterfactual features only (LR)',
    'B6': 'Uncertainty + stability, KHÔNG có counterfactual (LR)',
    'P1': 'BUS-VERIFY đầy đủ (LR trên confidence + bất định + ổn định + counterfactual)',
}


def present(df, names):
    """Chỉ giữ cột có mặt và có ít nhất một giá trị hữu hạn."""
    return [n for n in names if n in df and df[n].notna().any()]


ENS_COLS = ('ens_js', 'ens_dz_sd', 'ens_flip')


def feature_sets(df, drop_rank_duplicates=True, deploy_compatible=True):
    """FEATURE_SETS + ABLATIONS đã lọc theo cột thực có. Trả (dict tên -> list đặc trưng, dict ghi chú các cột bị bỏ)."""
    notes = {}
    out = {}
    for name, cols in {**FR.FEATURE_SETS, **{k: [c for c in FR.FEATURE_SETS['P1_full'] if c not in set(v)] for k, v in FR.ABLATIONS.items()}}.items():
        cols = present(df, cols)
        if deploy_compatible:                    # ứng dụng chỉ có MỘT checkpoint: đặc trưng ensemble chỉ dùng cho baseline B3, KHÔNG vào mô hình risk triển khai
            cols = [c for c in cols if c not in ENS_COLS]
        if drop_rank_duplicates:
            cols, dropped = FR.dedupe_rank_equivalent(df, cols)
            if dropped:
                notes[name] = dropped
        out[name] = cols
    return out, notes


def unlearned_scores(df, seed=0, ens_prefer=True):
    """B0..B3 (không học). Trả dict tên -> điểm nguy cơ (cao = nguy cơ sai cao) và nguồn của B3."""
    n = len(df)
    out = {'B0': np.random.default_rng(seed).random(n)}
    p_raw = df['p_raw'].to_numpy(float) if 'p_raw' in df else df['p_mal'].to_numpy(float)
    out['B1'] = 1.0 - np.maximum(p_raw, 1 - p_raw)
    out['B1b'] = -df['thr_margin'].to_numpy(float)
    pc = df['p_mal'].to_numpy(float)
    out['B2'] = 1.0 - np.maximum(pc, 1 - pc)
    src = 'ensemble' if (ens_prefer and 'ens_js' in df and df['ens_js'].notna().any()) else 'tta_views'
    out['B3'] = df['ens_js'].to_numpy(float) if src == 'ensemble' else df['view_js'].to_numpy(float)
    return out, src


def nested_oof_scores(df_dev, sets, wrong, groups, k=5, seed=0, C_grid=(0.01, 0.03, 0.1, 0.3, 1.0)):
    """Nested CV theo cụm: vòng NGOÀI k-fold cho điểm held-out; vòng TRONG chỉ chọn C trên phần train của vòng ngoài. Trả (scores, chosen_C)."""
    y = np.asarray(wrong).astype(int); g = np.asarray(groups)
    outer = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g))
    scores, chosen = {}, {}
    for name, cols in sets.items():
        if not cols:
            continue
        X = df_dev[cols].to_numpy(float)
        s = np.full(len(y), np.nan); cs = []
        for tr, te in outer:
            C, _ = FR.select_C(X[tr], y[tr], g[tr], grid=C_grid, k=3, seed=seed)
            m = FR.make_pipeline(C).fit(X[tr], y[tr])
            s[te] = m.predict_proba(X[te])[:, 1]; cs.append(C)
        scores[name] = s; chosen[name] = cs
    return scores, chosen


def fit_and_apply(df_dev, wrong_dev, groups_dev, split_dev, df_eval, sets, seed=0, k=5):
    """Huấn luyện trên TOÀN BỘ dev (assert_no_test) rồi chấm điểm tập giữ lại. Trả (scores, models)."""
    FR.assert_no_test(split_dev)
    models, scores = {}, {}
    for name, cols in sets.items():
        if not cols:
            continue
        m = FR.FailureRiskModel(cols, seed=seed, k=k).fit(df_dev, wrong_dev, groups_dev, split_dev)
        models[name] = m
        scores[name] = m.predict_risk(df_eval)
    return scores, models


PRIMARY_PAIRS = (('P1', 'B1'), ('P1', 'B3'), ('P1', 'B4'), ('P1', 'B6'))


def paired_comparisons(scores, wrong, is_fn, groups, pairs=PRIMARY_PAIRS, rate=0.2, B=2000, seed=0, alpha=0.05):
    """So sánh ghép cặp theo CỤM bệnh nhân (cùng mẫu bootstrap) cho hai điểm cuối chính: Error AUPRC và FN Alert Recall tại `rate`.
    Hiệu chỉnh Holm trên TOÀN BỘ các so sánh (bảo thủ). Báo cả effect size (Δ/SE) và CI."""
    wrong = np.asarray(wrong).astype(int); is_fn = np.asarray(is_fn).astype(bool); g = np.asarray(groups)

    def auprc(s):
        return lambda i: EV.safe_auprc(wrong[i], s[i])

    def fnr(s):
        def f(i):
            nf = is_fn[i].sum()
            if not nf:
                return float('nan')
            a = SP.alert_set(s[i], rate, rng=np.random.default_rng(seed))
            return float((a & is_fn[i]).sum() / nf)
        return f
    rows = []
    for a, b in pairs:
        if a not in scores or b not in scores:
            continue
        for metric, mk in (('error_auprc', auprc), (f'fn_alert_recall@{int(rate*100)}', fnr)):
            r = ST.paired_cluster_bootstrap(mk(np.asarray(scores[a], float)), mk(np.asarray(scores[b], float)), g, B=B, seed=seed, alpha=alpha)
            rows.append(dict(A=a, B=b, metric=metric, delta=r['delta'], ci_lo=r['lo'], ci_hi=r['hi'], p=r['p'], effect_z=r['z'], n_valid=r['n_valid']))
    df = pd.DataFrame(rows)
    if len(df):
        adj, rej = ST.holm_bonferroni(df['p'].fillna(1.0).to_numpy(), alpha)
        df['p_holm'] = adj; df['significant_holm'] = rej
        df['verdict'] = np.where(~df['significant_holm'], 'không đủ bằng chứng sau hiệu chỉnh đa so sánh',
                                 np.where(df['delta'] > 0, 'A tốt hơn B (có ý nghĩa sau Holm)', 'B tốt hơn A (có ý nghĩa sau Holm)'))
    return df


def risk_calibration_report(risk, wrong, n_bins=10):
    """Hiệu chuẩn của risk score (xác suất mô hình chính SAI) so với tần suất sai quan sát. KHÁC với hiệu chuẩn P(malignant)."""
    r = np.clip(np.asarray(risk, float), 0, 1); y = np.asarray(wrong, float)
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1)); edges[-1] += 1e-9
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum():
            rows.append(dict(mean_risk=float(r[m].mean()), observed_error=float(y[m].mean()), n=int(m.sum())))
            ece += m.mean() * abs(r[m].mean() - y[m].mean())
    return dict(ece=float(ece), brier=float(((r - y) ** 2).mean()), mean_risk=float(r.mean()), error_rate=float(y.mean()), bins=rows)


def make_alert_policy(risk_dev_oof, review_rate=0.20, high_rate=0.10):
    """Ngưỡng cảnh báo chỉ từ risk OOF của DỮ LIỆU PHÁT TRIỂN (phân vị), t_review <= t_high."""
    t_review = FR.threshold_for_alert_rate(risk_dev_oof, review_rate)
    t_high = FR.threshold_for_alert_rate(risk_dev_oof, high_rate)
    return dict(t_review=float(min(t_review, t_high)), t_high=float(t_high), review_rate_dev=review_rate, high_rate_dev=high_rate,
                n_dev=int(len(risk_dev_oof)), source='out-of-fold risk trên dữ liệu phát triển (không dùng test)')


def status_counts(risk, policy, in_scope=None):
    risk = np.asarray(risk, float)
    ins = np.ones(len(risk), bool) if in_scope is None else np.asarray(in_scope, bool)
    st = [SP.decide_status(r, policy['t_review'], policy['t_high'], in_scope=bool(i)) for r, i in zip(risk, ins)]
    return pd.Series(st).value_counts().to_dict(), st


def run_protocol(df_dev, wrong_dev, fn_dev, groups_dev, split_dev, df_eval_map, wrong_eval_map, fn_eval_map, groups_eval_map,
                 seed=0, k=5, B=1000, rates=(0.1, 0.2, 0.3), include_ablations=True):
    """Toàn bộ giao thức: (1) nested CV trên dev -> bảng xác nhận chính; (2) fit trên dev -> từng tập giữ lại trong df_eval_map.
    Trả dict kết quả (bảng + mô hình P1 + chính sách đã khóa)."""
    sets, notes = feature_sets(df_dev)
    if not include_ablations:
        sets = {k_: v for k_, v in sets.items() if k_ in FR.FEATURE_SETS}
    un_dev, src = unlearned_scores(df_dev, seed)
    nested, chosen = nested_oof_scores(df_dev, sets, wrong_dev, groups_dev, k=k, seed=seed)
    dev_scores = {**un_dev, **{('P1' if n == 'P1_full' else n.split('_')[0] if n in FR.FEATURE_SETS else n): s for n, s in nested.items()}}
    out = dict(sets=sets, dropped_rank_duplicates=notes, b3_source=src, chosen_C=chosen, dev_scores=dev_scores)
    out['dev_table'] = EV.error_detection_table(dev_scores, wrong_dev, fn_dev, groups_dev, rates=rates, B=B, seed=seed)
    out['dev_pairs'] = paired_comparisons(dev_scores, wrong_dev, fn_dev, groups_dev, B=B, seed=seed)
    _, models = fit_and_apply(df_dev, wrong_dev, groups_dev, split_dev, df_dev.iloc[:1], {n: c for n, c in sets.items() if n in FR.FEATURE_SETS}, seed=seed, k=k)
    p1 = models['P1_full']
    out['model'] = p1
    out['policy'] = make_alert_policy(p1.calib(p1.oof_raw))
    out['held_out'] = {}
    for name, dfe in df_eval_map.items():
        sc, _ = fit_and_apply(df_dev, wrong_dev, groups_dev, split_dev, dfe, {n: c for n, c in sets.items() if n in FR.FEATURE_SETS}, seed=seed, k=k)
        un, _ = unlearned_scores(dfe, seed)
        allsc = {**un, **{('P1' if n == 'P1_full' else n.split('_')[0]): s for n, s in sc.items()}}
        out['held_out'][name] = dict(table=EV.error_detection_table(allsc, wrong_eval_map[name], fn_eval_map[name], groups_eval_map[name], rates=rates, B=B, seed=seed),
                                     pairs=paired_comparisons(allsc, wrong_eval_map[name], fn_eval_map[name], groups_eval_map[name], B=B, seed=seed),
                                     scores=allsc, risk_calibration=risk_calibration_report(allsc['P1'], wrong_eval_map[name]))
    return out
