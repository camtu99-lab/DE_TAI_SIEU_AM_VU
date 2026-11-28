import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import ui, app_support as AS, report_export as RE, evidence_maps as EM

st = ui.setup('Research Report', 'Xuất báo cáo phiên (HTML + JSON) và đọc tài liệu nghiên cứu')
ui.disclaimer(st)
t1, t2 = st.tabs(['Xuất báo cáo phiên', 'Tài liệu nghiên cứu'])
with t1:
    a = ui.need_analysis(st)
    bundle, pred, msgs = ui.runtime(st)
    if a is not None and bundle is not None:
        ex = a['extra']; cfg = bundle.cfg
        images = {'Ảnh gốc (tiền xử lý)': a['image']}
        if ex['mask'] is not None:
            images['Segmentation overlay'] = AS.overlay_mask(a['image'], ex['mask'])
        if ex['evidence'] is not None:
            images['Evidence map'] = AS.heat_overlay(a['image'], EM.upsample_evidence(ex['evidence'], a['image'].shape[:2]))
        if a.get('gradcam') is not None:
            images['Grad-CAM'] = AS.heat_overlay(a['image'], a['gradcam'])
        risk = None
        if a.get('risk') is not None:
            risk = dict(risk_score=round(a['risk'], 4), status=a['status'], t_review=bundle.policy['t_review'], t_high=bundle.policy['t_high'])
        cf = st.session_state.get('cf_result')
        rec = dict(images=images, model=dict(name=cfg.get('model_name'), version=cfg.get('version'), temperature=cfg['temperature'], threshold=cfg['threshold'], checkpoint=cfg.get('checkpoint')),
                   results=dict(label=a['label'], p_malignant=round(a['p_malignant'], 4), inference_seconds=round(a['inference_seconds'], 3), file=a.get('name')),
                   risk=risk, cf=(cf['features'] if cf else None),
                   warnings=([] if risk else ['Risk score không khả dụng (thiếu artifact hợp lệ).']) + ([f"Lý do ngoài phạm vi: {a['oos_reasons']}"] if a.get('oos_reasons') else []),
                   limitations=[AS.RISK_LIMITS, 'Tập test BUS-BRA đã được khảo sát ở V3; BUSI không có mã bệnh nhân.', 'Không phải thiết bị chẩn đoán y khoa.'])
        st.download_button('⬇️ Tải báo cáo HTML', RE.build_report(rec).encode('utf-8'), 'bus_verify_report.html', 'text/html')
        st.download_button('⬇️ Tải JSON', RE.build_json(rec).encode('utf-8'), 'bus_verify_report.json', 'application/json')
with t2:
    rd = os.path.join(AS.ROOT, 'reports')
    docs = sorted(f for f in os.listdir(rd) if f.endswith('.md')) if os.path.isdir(rd) else []
    if not docs:
        st.info('Chưa có tài liệu trong reports/.')
    else:
        d = st.selectbox('Tài liệu', docs)
        st.markdown(open(os.path.join(rd, d), encoding='utf-8').read())
