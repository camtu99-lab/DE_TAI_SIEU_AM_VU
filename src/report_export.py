"""Xuất báo cáo nghiên cứu (HTML tự chứa + JSON) cho MỘT phiên phân tích. Không ghi ảnh ra đĩa; ảnh được nhúng base64 trong chính báo cáo người dùng tải về."""
import base64, io, json, html, time
import numpy as np
from PIL import Image
from .app_support import DISCLAIMER, RISK_LIMITS, VERSION


def _png_b64(arr):
    buf = io.BytesIO(); Image.fromarray(np.asarray(arr, np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def build_report(record):
    """record: dict(images={tên: ndarray}, model={...}, results={...}, risk=None|{...}, cf=None|{...}, robustness=None|[...], limitations=[...], warnings=[...])."""
    e = html.escape
    imgs = ''.join(f'<figure><img src="data:image/png;base64,{_png_b64(a)}" width="256"><figcaption>{e(n)}</figcaption></figure>' for n, a in record.get('images', {}).items())

    def table(d):
        return '<table>' + ''.join(f'<tr><th>{e(str(k))}</th><td>{e(str(v))}</td></tr>' for k, v in d.items()) + '</table>'
    risk = record.get('risk')
    risk_html = table(risk) + f'<p class="note">{e(RISK_LIMITS)}</p>' if risk else '<p class="warn">Risk score KHÔNG được tính: mô hình risk/chính sách cảnh báo chưa có hoặc không hợp lệ. Không có số giả lập.</p>'
    cf = record.get('cf')
    cf_html = table(cf) if cf else '<p>Chưa chạy kiểm chứng phản thực cho ảnh này.</p>'
    lim = ''.join(f'<li>{e(x)}</li>' for x in record.get('limitations', []))
    warn = ''.join(f'<li>{e(x)}</li>' for x in record.get('warnings', []))
    return f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>{e(VERSION)} — báo cáo phiên</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:920px;margin:24px auto;color:#14213d}}h1{{color:#0b6e6e}}figure{{display:inline-block;margin:6px;text-align:center}}
table{{border-collapse:collapse;margin:8px 0}}th,td{{border:1px solid #cbd5e1;padding:4px 10px;text-align:left}}th{{background:#eef6f6}}.warn{{background:#fff4e5;border-left:4px solid #e67e22;padding:8px}}.note{{background:#eef2ff;padding:8px}}</style></head><body>
<h1>{e(VERSION)}</h1><p>Tạo lúc {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="warn"><b>Lưu ý:</b> {e(DISCLAIMER)}</div>
<h2>Hình ảnh</h2>{imgs}
<h2>Mô hình và phiên bản</h2>{table(record.get('model', {}))}
<h2>Kết quả phân tích</h2>{table(record.get('results', {}))}
<h2>Kiểm chứng phản thực</h2>{cf_html}
<h2>Failure-Risk</h2>{risk_html}
<h2>Cảnh báo</h2><ul>{warn or '<li>Không có cảnh báo bổ sung.</li>'}</ul>
<h2>Giới hạn</h2><ul>{lim}</ul></body></html>'''


def build_json(record):
    r = {k: v for k, v in record.items() if k != 'images'}
    return json.dumps(r, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
