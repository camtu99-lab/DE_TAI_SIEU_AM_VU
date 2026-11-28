import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src import ui, app_support as AS

st = ui.setup('Failure Case Explorer', 'Lọc FN / FP / TP / TN và lỗi độ tin cậy cao từ bảng dự đoán thật')
ui.disclaimer(st)
choices = {'Dev (out-of-fold)': 'pred_dev', 'BUS-BRA test giữ lại': 'pred_test', 'BUSI ngoài': 'pred_busi'}
ds = st.selectbox('Tập dữ liệu', list(choices))
path = os.path.join(AS.V4_DIR, AS.V4_FILES[choices[ds]])
df = AS.load_table(path)
if df is None:
    ui.not_run(st, f'chưa có {os.path.basename(path)}', 'V3 không lưu dự đoán theo từng ảnh nên không thể dựng lại các ca lỗi; cần chạy notebook V4.'); st.stop()
try:
    AS.validate_prediction_table(df)
except ValueError as e:
    st.error(str(e)); st.stop()
kind = st.radio('Loại ca', ['all', 'FN', 'FP', 'TP', 'TN', 'high_conf_error'], horizontal=True,
                format_func={'all': 'Tất cả', 'FN': 'FN (ác tính bị bỏ sót)', 'FP': 'FP', 'TP': 'TP', 'TN': 'TN', 'high_conf_error': 'Lỗi độ tin cậy cao'}.get)
thr = st.slider('Ngưỡng độ tin cậy cho "lỗi tin cậy cao"', 0.5, 0.99, 0.9) if kind == 'high_conf_error' else 0.9
sub = AS.case_filter(df, kind, thr)
if 'status' in sub:
    sts = st.multiselect('Trạng thái cảnh báo', sorted(sub['status'].dropna().unique()), default=sorted(sub['status'].dropna().unique()))
    sub = sub[sub['status'].isin(sts)]
c = st.columns(4)
c[0].metric('Số ca', len(sub)); c[1].metric('FN trong lọc', int(((sub.y == 1) & (sub.pred == 0)).sum()))
fn = df[(df.y == 1) & (df.pred == 0)]
if len(fn) and 'status' in df:
    c[2].metric('FN được cảnh báo (toàn tập)', f"{int((fn['status'] != 'Không phát cảnh báo').sum())}/{len(fn)}")
c[3].metric('Tỉ lệ sai trong lọc', f"{sub['wrong'].mean():.2%}" if len(sub) else '—')
st.dataframe(sub.sort_values('risk', ascending=False) if 'risk' in sub else sub, use_container_width=True, hide_index=True)
if len(sub):
    cid = st.selectbox('Xem chi tiết ca', sub['id'].tolist())
    st.json({k: (None if pd.isna(v) else v) for k, v in sub[sub['id'] == cid].iloc[0].to_dict().items()})
    st.caption('Ảnh bệnh nhân không được đóng gói; chỉ hiển thị số liệu của ca.')
