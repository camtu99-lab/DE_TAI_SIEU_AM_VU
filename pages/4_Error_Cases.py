"""Phân tích trường hợp sai: chỉ hiển thị khi có artifact dự đoán theo ca THẬT. Không tự tái dựng FN/FP."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import ui, research_results as RR  # noqa: E402

ui.setup_page('Error Cases', '🔍')
rt = ui.get_runtime()
ui.sidebar(rt)
ui.header()
st.subheader('Phân tích trường hợp sai (FN/FP)')
df, msg = RR.load_predictions()
if df is None:
    ui.card('warn', 'Chưa có artifact theo ca', msg)
    st.write('Khi có `results/predictions_per_image.csv` với các cột `case_id` (ẩn danh), `y_true`, `p_malignant`, `pred`, `split`, `model`, `seed`, trang này sẽ cho phép lọc FN/FP/TP/TN. '
             'Ứng dụng **không** suy đoán FN/FP từ bảng metric tổng hợp, từ tên thư mục hay từ hình ảnh.')
    st.stop()
df = df.copy()
df['outcome'] = [RR.outcome(a, b) for a, b in zip(df['y_true'], df['pred'])]
c1, c2, c3 = st.columns(3)
mdl = c1.selectbox('Mô hình', sorted(df['model'].unique()))
split = c2.selectbox('Split', sorted(df['split'].unique()))
oc = c3.multiselect('Loại', ['FN', 'FP', 'TP', 'TN'], default=['FN', 'FP'])
d = df[(df['model'] == mdl) & (df['split'] == split) & (df['outcome'].isin(oc))]
conf_sort = st.checkbox('Sắp xếp theo độ tự tin giảm dần', value=True)
if conf_sort:
    d = d.assign(_c=(d['p_malignant'] - 0.5).abs()).sort_values('_c', ascending=False).drop(columns='_c')
st.dataframe(d, hide_index=True, **ui.STRETCH)
st.caption('Chỉ dùng mã ca ẩn danh. Ảnh không được hiển thị trừ khi có tệp ảnh được phép phân phối (không có trong gói mặc định).')
