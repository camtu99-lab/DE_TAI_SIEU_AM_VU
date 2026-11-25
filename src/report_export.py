"""Xuất báo cáo nghiên cứu: JSON cho ca hiện tại (trong RAM) + HTML/Markdown từ artifact thật.
KHÔNG chứa: tên tệp gốc, bytes ảnh, EXIF, tên bệnh nhân, đường dẫn cục bộ."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

from src import config
from src.security import new_case_id

DISCLAIMER = ('Công cụ minh họa nghiên cứu, không phải thiết bị chẩn đoán y khoa. Đây là dự đoán của mô hình/kết quả nghiên cứu; '
              'cần chuyên gia đánh giá. Không dùng để quyết định lâm sàng.')


def build_case_report(rt, res, perturbation: list | None = None, perturbation_params: dict | None = None) -> dict:
    from src.preprocessing import preprocessing_config
    cfg = rt.cfg
    d = dict(
        report_type='research_case', app=config.APP_NAME, app_version=config.APP_VERSION, timestamp_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        model=dict(experiment=cfg['experiment'], seed=int(cfg['seed']), checkpoint_sha256=rt.sha256, run_mode=cfg['run_mode'], img_size=int(cfg['img_size'])),
        preprocessing=preprocessing_config(cfg),
        locked_parameters=dict(temperature=float(cfg['temperature']), classification_threshold=float(cfg['threshold']), abstain_threshold=float(cfg['abstain_threshold']),
                               note='đã khóa từ validation; không điều chỉnh theo ảnh người dùng'),
        input_size_px=dict(height=int(res.original_shape[0]), width=int(res.original_shape[1])),
        label_mapping={'benign': config.LABEL_BENIGN, 'malignant': config.LABEL_MALIGNANT},
        raw_logits=dict(benign=float(res.raw_logits[0]), malignant=float(res.raw_logits[1])),
        raw_probabilities=dict(benign=float(res.raw_probs[0]), malignant=float(res.raw_probs[1])),
        calibrated_probabilities=dict(benign=float(res.p_cal[0]), malignant=float(res.p_cal[1])),
        model_prediction=res.label, confidence_calibrated=float(res.confidence), abstain=bool(res.abstain),
        segmentation=dict(source='predicted mask (không dùng ground truth)', mask_area_fraction=float(res.mask_fraction), boxes_xyxy=[list(map(int, b)) for b in res.boxes]),
        warnings=list(res.warnings), disclaimer=DISCLAIMER)
    if perturbation:
        d['perturbation_lab'] = dict(params=perturbation_params or {}, note='độ nhạy của mô hình, không phải bằng chứng nhân quả',
                                     results=[dict(region=o.region, n_perturbed=o.n_perturbed, delta_logit_malignant=o.delta_logit_malignant,
                                                   orig_p_malignant_cal=o.orig_p_malignant_cal, pert_p_malignant_cal=o.pert_p_malignant_cal,
                                                   label_changed=o.label_changed, abstain_changed=o.abstain_changed) for o in perturbation])
    return d


def case_report_bytes(report: dict) -> tuple[str, bytes]:
    name = f'BreastVision_XAI_research_case_{new_case_id()}.json'
    return name, json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')


def research_html(tables: dict, report_md: str | None = None, title: str = 'BreastVision-XAI V3 — kết quả nghiên cứu (từ artifact)') -> bytes:
    parts = [f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>{html.escape(title)}</title>',
             '<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;color:#212529}table{border-collapse:collapse;font-size:13px}'
             'td,th{border:1px solid #ced4da;padding:4px 8px}th{background:#f1f3f5}pre{white-space:pre-wrap;background:#f8f9fa;padding:12px}</style></head><body>',
             f'<h1>{html.escape(title)}</h1><p><b>{html.escape(config.BADGE)}</b> — {html.escape(DISCLAIMER)}</p>']
    for name, df in tables.items():
        parts.append(f'<h2>{html.escape(name)}</h2>' + df.to_html(index=False, escape=True, float_format=lambda v: f'{v:.4f}'))
    if report_md:
        parts.append('<h2>research_report.md (nguyên văn từ notebook)</h2><pre>' + html.escape(report_md) + '</pre>')
    parts.append('</body></html>')
    return ''.join(parts).encode('utf-8')
