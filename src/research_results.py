"""Đọc kết quả nghiên cứu THẬT từ results/ (CSV/JSON/MD). Không có số liệu nhập tay; xác thực schema; thiếu cột -> lỗi rõ ràng."""
from __future__ import annotations

import io
import json
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src import config
from src.security import UserFacingError

REQUIRED_COLS = {
    'internal_results_per_seed.csv': ['model', 'seed', 'auroc', 'auprc', 'sensitivity_malignant', 'specificity', 'balanced_accuracy', 'f1', 'dice', 'iou', 'hd95'],
    'internal_results_mean_sd.csv': ['model', 'n_seeds', 'auroc_mean', 'auroc_sd', 'auprc_mean', 'auprc_sd', 'sensitivity_malignant_mean', 'sensitivity_malignant_sd',
                                     'specificity_mean', 'specificity_sd', 'balanced_accuracy_mean', 'balanced_accuracy_sd', 'f1_mean', 'f1_sd', 'dice_mean', 'dice_sd',
                                     'iou_mean', 'iou_sd', 'hd95_mean', 'hd95_sd'],
    'paired_comparisons.csv': ['A', 'B', 'metric', 'diff', 'ci_low', 'ci_high'],
    'calibration.csv': ['model', 'seed', 'hiệu_chuẩn', 'T', 'NLL', 'Brier', 'ECE'],
    'selective_prediction.csv': ['model', 'aurc', 'coverage_target', 'coverage_actual', 'selective_risk', 'abstain_threshold', 'sensitivity_malignant_accepted', 'malignant_abstained'],
    'xai_evaluation.csv': ['model', 'group', 'metric', 'value'],
}
JSON_FILES = ['external_busi_results.json']
TEXT_FILES = ['research_report.md']
EXTERNAL_REQUIRED = ['n', 'auroc', 'auprc', 'sensitivity_malignant', 'specificity', 'segmentation', 'calibration_locked_T']

XAI_GROUPS = {
    'Localisation (vị trí giải thích so với mask)': ['localisation_evidence', 'localisation_gradcam'],
    'Faithfulness (can thiệp/nhiễu loạn ảnh hưởng dự đoán)': ['faithfulness', 'evidence_intervention', 'background_perturbation'],
    'Consistency (nhất quán khi biến đổi ảnh)': ['consistency_hflip', 'consistency_affine'],
}
DIAGNOSTIC_GROUP = 'collapse'


class ResultsError(UserFacingError):
    def __init__(self, message_vi: str, technical: str = ''):
        super().__init__('results', message_vi, technical)


def _dir(results_dir) -> Path:
    return Path(results_dir or config.RESULTS_DIR)


@lru_cache(maxsize=64)
def _load_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def load_table(name: str, results_dir=None) -> pd.DataFrame:
    p = _dir(results_dir) / name
    if name not in REQUIRED_COLS:
        raise ResultsError(f'Bảng {name} không nằm trong danh sách artifact được hỗ trợ.')
    if not p.exists():
        raise ResultsError(f'Thiếu tệp kết quả {name}.', 'FileNotFoundError')
    try:
        df = _load_csv_cached(str(p), p.stat().st_mtime).copy()
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ResultsError(f'Không đọc được {name}.', f'{type(e).__name__}: {e}') from e
    miss = [c for c in REQUIRED_COLS[name] if c not in df.columns]
    if miss:
        raise ResultsError(f'{name} thiếu cột bắt buộc: {", ".join(miss)}.', f'missing={miss}')
    if name == 'xai_evaluation.csv':
        df['value'] = pd.to_numeric(df['value'], errors='coerce')    # cột value có thể được đọc dạng chuỗi
    return df


def load_external(results_dir=None) -> dict:
    p = _dir(results_dir) / 'external_busi_results.json'
    if not p.exists():
        raise ResultsError('Thiếu tệp external_busi_results.json.')
    try:
        d = json.loads(p.read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        raise ResultsError('external_busi_results.json hỏng (JSON không hợp lệ).', f'{type(e).__name__}: {e}') from e
    if not isinstance(d, dict) or not d:
        raise ResultsError('external_busi_results.json có cấu trúc không hợp lệ.')
    for m, v in d.items():
        miss = [k for k in EXTERNAL_REQUIRED if not isinstance(v, dict) or k not in v]
        if miss:
            raise ResultsError(f'external_busi_results.json: mô hình {m} thiếu khóa {", ".join(miss)}.')
    return d


def load_report_md(results_dir=None) -> str:
    p = _dir(results_dir) / 'research_report.md'
    if not p.exists():
        raise ResultsError('Thiếu research_report.md.')
    return p.read_text(encoding='utf-8')


def check_files(results_dir=None) -> list[str]:
    """Trả về danh sách tệp thiếu/lỗi (rỗng = đầy đủ)."""
    bad = []
    for n in REQUIRED_COLS:
        try:
            load_table(n, results_dir)
        except ResultsError:
            bad.append(n)
    try:
        load_external(results_dir)
    except ResultsError:
        bad.append('external_busi_results.json')
    if not (_dir(results_dir) / 'research_report.md').exists():
        bad.append('research_report.md')
    return bad


def verdict_from_ci(diff: float, lo: float, hi: float, a: str, b: str) -> str:
    """Kết luận tính TỪ CI (không dùng ngôn ngữ so sánh tuyệt đối)."""
    if lo <= 0.0 <= hi:
        return 'Chưa đủ bằng chứng về khác biệt (CI chứa 0).'
    if lo > 0:
        return f'CI > 0: {a} cao hơn {b} theo paired bootstrap trên tập test nội bộ.'
    return f'CI < 0: {b} cao hơn {a} theo paired bootstrap trên tập test nội bộ.'


def paired_table(a: str | None = None, b: str | None = None, results_dir=None) -> pd.DataFrame:
    df = load_table('paired_comparisons.csv', results_dir)
    if a:
        df = df[df['A'] == a]
    if b:
        df = df[df['B'] == b]
    df = df.copy()
    df['kết luận (tính từ CI)'] = [verdict_from_ci(r.diff, r.ci_low, r.ci_high, r.A, r.B) for r in df.itertuples()]
    return df


def kpis(model: str = 'V3_full', results_dir=None) -> list[dict]:
    """KPI lấy từ artifact; mỗi KPI ghi model + dataset + nguồn file. Không chọn/gộp seed tùy ý (mean ± sd qua toàn bộ seed trong file)."""
    ms = load_table('internal_results_mean_sd.csv', results_dir)
    row = ms[ms['model'] == model]
    if row.empty:
        raise ResultsError(f'Không có mô hình {model} trong internal_results_mean_sd.csv.')
    r = row.iloc[0]
    n = int(r['n_seeds'])
    src = 'internal_results_mean_sd.csv'
    ds = 'BUS-BRA nội bộ (test)'
    out = []
    for lab, key in (('AUROC', 'auroc'), ('Sensitivity malignant', 'sensitivity_malignant'), ('Dice', 'dice')):
        out.append(dict(label=lab, value=f"{float(r[key + '_mean']):.3f} ± {float(r[key + '_sd']):.3f}", model=model, dataset=ds, source=src, note=f'mean ± sd qua {n} seed'))
    ext = load_external(results_dir)
    if model in ext:
        e = ext[model]
        ci = e.get('auroc_ci95')
        val = f"{float(e['auroc']):.3f}" + (f" [{ci[0]:.3f}; {ci[1]:.3f}]" if ci else '')
        out.append(dict(label='AUROC ngoài (BUSI)', value=val, model=model, dataset=f"BUSI ngoài, n={int(e['n'])} (pipeline đã khóa)", source='external_busi_results.json', note='khoảng tin cậy 95% nếu có trong file'))
    return out


def model_comparison(models=('X0', 'X6', 'V3_full'), results_dir=None) -> pd.DataFrame:
    ms = load_table('internal_results_mean_sd.csv', results_dir)
    return ms[ms['model'].isin(models)].reset_index(drop=True)


def xai_pivot(group_keys: list[str], results_dir=None) -> pd.DataFrame:
    df = load_table('xai_evaluation.csv', results_dir)
    d = df[df['group'].isin(group_keys)]
    if d.empty:
        return pd.DataFrame()
    return d.pivot_table(index=['group', 'metric'], columns='model', values='value', aggfunc='mean').reset_index()


def df_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8-sig')


def tables_zip_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            z.writestr(name if name.endswith('.csv') else name + '.csv', df_csv_bytes(df))
    return buf.getvalue()


PRED_REQUIRED = ['case_id', 'y_true', 'p_malignant', 'pred', 'split', 'model', 'seed']
PRED_FORBIDDEN_SUBSTR = ('patient', 'path', 'filename', 'file_name', 'name')


def load_predictions(results_dir=None) -> tuple[pd.DataFrame | None, str]:
    """Bảng dự đoán theo ca (artifact tùy chọn). Không có -> (None, thông báo). Từ chối bảng chứa cột có thể định danh."""
    p = _dir(results_dir) / 'predictions_per_image.csv'
    if not p.exists():
        return None, 'Bộ artifact hiện tại chưa chứa bảng dự đoán theo từng ảnh để tái dựng FN/FP an toàn.'
    try:
        df = pd.read_csv(p)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None, 'predictions_per_image.csv không đọc được; tính năng bị tắt.'
    miss = [c for c in PRED_REQUIRED if c not in df.columns]
    if miss:
        return None, 'predictions_per_image.csv thiếu cột: ' + ', '.join(miss) + '; tính năng bị tắt.'
    bad = [c for c in df.columns if any(s in c.lower() for s in PRED_FORBIDDEN_SUBSTR) and c != 'case_id']
    if bad:
        return None, 'predictions_per_image.csv chứa cột có thể định danh (' + ', '.join(bad) + '); từ chối hiển thị để bảo vệ quyền riêng tư.'
    return df, ''


def outcome(y_true: int, pred: int) -> str:
    """malignant=1 là lớp dương: TP/TN/FP/FN."""
    return {(1, 1): 'TP', (0, 0): 'TN', (0, 1): 'FP', (1, 0): 'FN'}[(int(y_true), int(pred))]
