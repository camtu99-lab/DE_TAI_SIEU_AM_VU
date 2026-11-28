import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src import ui, app_support as AS, pipeline as PL, failure_risk as FR

st = ui.setup('Failure-Risk Detection', 'Nguy cơ MÔ HÌNH CHÍNH dự đoán sai — khác với xác suất ác tính')
ui.disclaimer(st)
bundle, pred, msgs = ui.runtime(st)
ui.show_runtime_messages(st, msgs)
if bundle is None or not bundle.risk_enabled:
    ui.not_run(st, 'chức năng risk score đang TẮT',
               'Cần risk_model (huấn luyện bằng out-of-fold) và alert_policy.locked.json hợp lệ trong gói triển khai. Ứng dụng không giả lập điểm bằng số ngẫu nhiên hay công thức tùy ý.')
    st.markdown(AS.RISK_LIMITS); st.stop()
a = ui.need_analysis(st)
if a is None or pred is None:
    st.stop()
if a.get('risk') is None:
    ui.not_run(st, 'ảnh này chưa có risk', 'Hãy phân tích lại ảnh ở trang Ultrasound Analysis (khi risk bật, đặc trưng phản thực được tính tự động).'); st.stop()
pol = bundle.policy
c = st.columns(4)
c[0].metric('Risk score (xác suất mô hình sai, đã hiệu chuẩn)', f"{a['risk']:.3f}")
c[1].markdown('**Trạng thái cảnh báo**<br>' + ui.status_html(a['status']), unsafe_allow_html=True)
c[2].metric('Ngưỡng "Cần xem xét lại" (khóa)', f"{pol['t_review']:.3f}")
c[3].metric('Ngưỡng "Nguy cơ cao" (khóa)', f"{pol['t_high']:.3f}")
if a.get('oos_reasons'):
    ui.box(st, 'off', 'Lý do ngoài phạm vi/không kiểm chứng được: ' + ', '.join(a['oos_reasons']))
st.subheader('Đặc trưng ảnh hưởng đến risk score')
con = a['contributions']
import plotly.graph_objects as go
con = con.head(12)[::-1]
st.plotly_chart(go.Figure(go.Bar(x=con.values, y=con.index, orientation='h', marker_color=['#fb7185' if v > 0 else '#2dd4bf' for v in con.values]))
                .update_layout(template='plotly_dark', height=420, title='Đóng góp vào logit risk (đỏ: tăng nguy cơ sai; xanh: giảm). Tổng + intercept = logit thô', xaxis_title='coef × giá trị chuẩn hóa'), use_container_width=True)
st.caption(f"Mô hình risk: {bundle.risk.card.get('features') and len(bundle.risk.card['features'])} đặc trưng, C={bundle.risk.card.get('C')}, huấn luyện trên {bundle.risk.card.get('n')} ca "
           f"({bundle.risk.card.get('n_errors')} lỗi) out-of-fold, hiệu chuẩn: {bundle.risk.card.get('calibrator')}. Ngưỡng lấy từ dữ liệu phát triển: {pol.get('source', '')}.")
st.subheader('Giới hạn của cảnh báo')
st.markdown(AS.RISK_LIMITS)
