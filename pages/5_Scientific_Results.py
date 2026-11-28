import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd
from src import ui, app_support as AS

st = ui.setup('Scientific Results', 'Đọc trực tiếp CSV/JSON thực nghiệm — không số liệu nhập tay')
ui.disclaimer(st)
t4, t3 = st.tabs(['V4 — BUS-VERIFY', 'V3 — nền tảng (kết quả thật đã có)'])
with t4:
    man = AS.load_json(os.path.join(AS.V4_DIR, AS.V4_FILES['manifest']))
    if not man:
        ui.not_run(st, 'chưa có kết quả thực nghiệm V4', 'Bảng dưới sẽ tự hiện sau khi notebook Kaggle ghi artifacts/v4/*.csv.')
    else:
        st.json(man, expanded=False)
    import plotly.express as px
    for key, title in [('dev_table', 'Nested-CV trên dữ liệu phát triển (xác nhận chính)'), ('dev_pairs', 'So sánh ghép cặp (bootstrap theo bệnh nhân, Holm)'),
                       ('test_table', 'BUS-BRA test giữ lại (ĐÃ khảo sát ở V3 — nội bộ)'), ('test_pairs', 'So sánh ghép cặp — test giữ lại'),
                       ('busi_table', 'BUSI ngoài (CI mức ảnh: lạc quan)'), ('busi_pairs', 'So sánh ghép cặp — BUSI'), ('ablation', 'Ablation (dev)'),
                       ('oracle', 'Thí nghiệm ORACLE (dùng GT mask — báo cáo riêng, không phải suy luận thật)'), ('classification', 'Phân loại / hiệu chuẩn / phân vùng')]:
        t = AS.load_table(os.path.join(AS.V4_DIR, AS.V4_FILES[key]))
        st.subheader(title)
        if t is None:
            ui.not_run(st, os.path.basename(AS.V4_FILES[key]), '')
        else:
            st.dataframe(t, use_container_width=True, hide_index=True)
    rc = AS.load_table(os.path.join(AS.V4_DIR, AS.V4_FILES['risk_coverage']))
    if rc is not None:
        st.subheader('Risk–coverage curve')
        st.plotly_chart(px.line(rc, x='coverage', y='selective_risk', color='method', facet_col='dataset' if 'dataset' in rc else None, template='plotly_dark'), use_container_width=True)
with t3:
    st.caption('Nguồn: artifacts/v3_reference/ (xuất từ notebook V3).')
    for f, ttl in [('internal_results_mean_sd.csv', 'Nội bộ BUS-BRA (mean±sd theo seed)'), ('calibration.csv', 'Hiệu chuẩn'), ('selective_prediction.csv', 'Selective prediction (V3: độ tin cậy = max-prob)'),
                   ('paired_comparisons.csv', 'So sánh ghép cặp V3 (CI mức ảnh, chưa hiệu chỉnh đa so sánh)'), ('xai_evaluation.csv', 'Đánh giá XAI')]:
        t = AS.load_table(os.path.join(AS.V3_REF, f)); st.subheader(ttl)
        if t is not None:
            st.dataframe(t, use_container_width=True, hide_index=True)
    for f in ('external_busi_results.json', 'data_audit.json', 'pipeline_lock_primary.json'):
        j = AS.load_json(os.path.join(AS.V3_REF, f))
        if j:
            with st.expander(f):
                st.json(j)
    for img in ('curves_V3_full_s42.png', 'reliability_primary.png', 'examples_primary.png'):
        p = os.path.join(AS.V3_REF, img)
        if os.path.exists(p):
            st.image(p, caption=img, use_container_width=True)
