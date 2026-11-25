"""Thí nghiệm can thiệp vùng ảnh (equal-area mặc định) dựa trên MASK DỰ ĐOÁN. Minh họa độ nhạy, không phải nhân quả."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, ui, perturbation as PT, preprocessing as PP, visualization as V, research_results as RR  # noqa: E402
from src.security import UserFacingError  # noqa: E402

ui.setup_page('Perturbation Lab', '🧪')
rt = ui.get_runtime()
ui.sidebar(rt)
ui.header()
ui.require_ready(rt)

st.subheader('Thí nghiệm can thiệp vùng ảnh')
st.warning(PT.WARNING_TEXT)
res = st.session_state.get('current_result')
if res is None:
    st.info('Chưa có ca nào được phân tích. Hãy vào trang Analyze, phân tích một ảnh rồi quay lại.')
    st.stop()

c1, c2, c3 = st.columns(3)
kind = c1.selectbox('Kiểu perturbation', list(PT.KINDS), format_func=lambda k: PT.KINDS[k])
fraction = c2.slider('Tỉ lệ can thiệp (so với vùng nhỏ nhất khi equal-area)', 0.05, 1.0, 0.5, 0.05)
seed = int(c3.number_input('Random seed', 0, 10_000, 0, 1))
equal_area = st.checkbox('Equal-area mode (mọi vùng bị tác động cùng số pixel)', value=True)
opacity = st.slider('Độ đậm hiển thị vùng', 0.1, 0.9, 0.45, 0.05)
if not equal_area:
    st.warning('Equal-area đang TẮT: số pixel bị tác động khác nhau giữa các vùng nên kết quả không so sánh công bằng được.')

try:
    valid = PP.valid_region(res.original_shape, int(rt.cfg['img_size']))
    regions, skipped = PT.build_regions(res.mask, res.post_prob, valid)
except UserFacingError as e:
    ui.show_error(e)
    st.stop()

st.image(V.region_overlay(res.image, regions, opacity), caption='Các vùng (từ mask DỰ ĐOÁN): đỏ=tổn thương, vàng=quanh tổn thương, xanh=hậu âm, xám=nền xa', width=420)
st.write('Kích thước vùng (pixel): ' + ', '.join(f'{PT.REGION_LABELS[k]}={int(v.sum())}' for k, v in regions.items()))
for k, why in skipped.items():
    st.caption(f'Bỏ qua vùng "{PT.REGION_LABELS[k]}": {why}.')

if st.button('Chạy Perturbation Lab', type='primary'):
    try:
        with st.spinner('Đang chạy các lượt suy luận can thiệp...'):
            outs = PT.run_lab(rt, res, regions, kind, fraction, seed, equal_area)
        st.session_state['perturb_state'] = dict(results=outs, params=dict(kind=kind, fraction=fraction, seed=seed, equal_area=equal_area))
    except UserFacingError as e:
        ui.show_error(e)

ps = st.session_state.get('perturb_state')
if ps:
    outs = ps['results']
    rows = [{'Vùng': PT.REGION_LABELS[o.region], 'Số pixel vùng': o.region_size, 'Pixel bị tác động': o.n_perturbed,
             'Logit malignant gốc': round(o.orig_logit_malignant, 4), 'Logit malignant sau': round(o.pert_logit_malignant, 4), 'Δ logit': round(o.delta_logit_malignant, 4),
             'P(malignant) hiệu chuẩn gốc': round(o.orig_p_malignant_cal, 4), 'P(malignant) hiệu chuẩn sau': round(o.pert_p_malignant_cal, 4),
             'Đổi nhãn?': 'có' if o.label_changed else 'không', 'Đổi trạng thái từ chối?': 'có' if o.abstain_changed else 'không'} for o in outs]
    st.dataframe(pd.DataFrame(rows), hide_index=True, **ui.STRETCH)
    cols = st.columns(len(outs))
    for col, o in zip(cols, outs):
        col.image(V.region_overlay(o.image, {o.region: o.selected}, 0.35, {o.region: o.selected}), caption=f'{PT.REGION_LABELS[o.region]} · {o.n_perturbed}px', **ui.STRETCH)
    st.caption('Δ logit là chênh lệch logit thô của lớp malignant (ảnh gốc không lật). Không có “risk score” nào được tạo từ Δ logit.')
st.warning(PT.WARNING_TEXT)

try:
    x = RR.load_table('xai_evaluation.csv')
    d = x[(x['model'] == 'V3_full') & (x['group'] == 'background_perturbation') & x['metric'].isin(['mean_abs_dprob_lesion_shuffle', 'mean_abs_dprob_background_shuffle', 'n'])]
    v = dict(zip(d['metric'], d['value']))
    if 'mean_abs_dprob_lesion_shuffle' in v and 'mean_abs_dprob_background_shuffle' in v:
        st.info(f'Tham chiếu từ thí nghiệm của notebook (xai_evaluation.csv, V3_full, n={int(v.get("n", 0))}): |ΔP| trung bình khi shuffle tổn thương = {v["mean_abs_dprob_lesion_shuffle"]:.3f}; '
                f'khi shuffle nền = {v["mean_abs_dprob_background_shuffle"]:.3f}. Hai giá trị này được báo cáo nguyên trạng, không nhằm kết luận mô hình nhạy riêng với tổn thương.')
except UserFacingError:
    pass
