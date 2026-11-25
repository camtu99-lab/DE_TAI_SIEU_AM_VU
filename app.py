"""BreastVision-XAI V3 — Trang chủ: health check, tổng quan, Presentation Mode."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from src import config, ui, research_results as RR  # noqa: E402
from src.security import UserFacingError  # noqa: E402

ui.setup_page('Trang chủ')
rt = ui.get_runtime()
ui.sidebar(rt)
ui.header()

PIPELINE = ['Ảnh siêu âm', 'Tiền xử lý', 'Encoder EfficientNet-B3 dùng chung', 'Phân vùng (mask dự đoán)', 'Vùng lesion / quanh tổn thương / hậu âm (từ mask dự đoán)',
            'Gộp đặc trưng theo evidence', 'Logit', 'Hiệu chuẩn nhiệt độ', 'Ngưỡng quyết định đã khóa', 'Từ chối (tùy chọn)']


def flow():
    html = '<div class="bv-flow">' + '<span class="bv-arrow">→</span>'.join(f'<span class="bv-box">{p}</span>' for p in PIPELINE) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def kpi_cards():
    try:
        ks = RR.kpis('V3_full')
    except UserFacingError as e:
        ui.show_error(e)
        return
    for col, k in zip(st.columns(len(ks)), ks):
        col.markdown(f'<div class="bv-kpi"><div class="l">{k["label"]}</div><div class="v">{k["value"]}</div>'
                     f'<div class="s">{k["model"]} · {k["dataset"]}<br>{k["note"]}<br>nguồn: {k["source"]}</div></div>', unsafe_allow_html=True)
    st.caption('KPI lấy trực tiếp từ artifact; không chọn số đẹp. Checkpoint đang chạy trong demo là một seed (xem sidebar); KPI nội bộ là mean ± sd qua các seed trong file.')


def x6_vs_v3():
    try:
        comp = RR.model_comparison(('X6', 'V3_full'))
        paired = RR.paired_table('V3_full', 'X6')
    except UserFacingError as e:
        ui.show_error(e)
        return
    cols = ['model', 'n_seeds', 'auroc_mean', 'auroc_sd', 'sensitivity_malignant_mean', 'specificity_mean', 'dice_mean', 'dice_sd']
    st.dataframe(comp[cols], hide_index=True, **ui.STRETCH)
    st.dataframe(paired[['metric', 'diff', 'ci_low', 'ci_high', 'kết luận (tính từ CI)']], hide_index=True, **ui.STRETCH)
    st.caption('X6 chỉ là kết quả thực nghiệm lưu trữ (không có checkpoint X6 trong bản triển khai này).')


if st.session_state.get('presentation', False):
    st.subheader('Presentation Mode')
    kpi_cards()
    t1, t2, t3, t4 = st.tabs(['Pipeline', 'Ví dụ', 'X6 so với V3_full', 'Giới hạn'])
    with t1:
        flow()
        st.caption('Segmentation, evidence map và Grad-CAM là ba đầu ra/biểu diễn khác nhau.')
    with t2:
        res = st.session_state.get('current_result')
        if res is None:
            st.info('Chưa có ca nào được phân tích. Vào trang Analyze để tải ảnh của bạn (hoặc ca mẫu nếu có).')
        else:
            from src import visualization as V
            panels = V.analysis_panels(res, 0.45)
            for col, (t, im) in zip(st.columns(min(4, len(panels))), panels[:4]):
                col.image(im, caption=t, **ui.STRETCH)
            st.write(f'**Dự đoán mô hình: {config.CLASS_VI[res.label]}** · P(malignant) hiệu chuẩn {res.p_malignant:.3f} · '
                     f'{"mô hình từ chối dự đoán tin cậy" if res.abstain else "không bị từ chối"}')
            ui.disclaimer()
    with t3:
        x6_vs_v3()
    with t4:
        st.markdown('- Chỉ phục vụ nghiên cứu; chưa kiểm định lâm sàng.\n- Domain shift tồn tại (BUSI chỉ là dữ liệu ngoài theo protocol hiện có).\n'
                    '- Calibration/abstention không bảo đảm đúng; Grad-CAM/evidence không phải bản đồ bệnh lý.\n- CI chứa 0 nghĩa là chưa đủ bằng chứng về khác biệt.\n'
                    '- Chi tiết: trang *About Limits*.')
    st.stop()

st.subheader('Tổng quan')
st.write('Web demo này chạy **suy luận thật** từ checkpoint V3_full đã đóng băng, tái sử dụng mã tiền xử lý/suy luận gốc của notebook. '
         'Kết quả là **dự đoán của mô hình** để minh họa nghiên cứu; **cần chuyên gia đánh giá**.')
flow()
st.subheader('Kiểm tra artifact khi khởi động')
ui.show_health(rt)
if not rt.ok:
    ui.card('err', 'Chức năng phân tích bị khóa', rt.error.message_vi if rt.error else 'Kiểm tra toàn vẹn thất bại.')
    if rt.error:
        with st.expander('Chi tiết kỹ thuật'):
            st.code(f'component: {rt.error.component}\n{rt.error.technical or "(không có)"}')
else:
    ui.card('info', 'Sẵn sàng', 'Mở trang <b>Analyze</b> ở thanh bên để tải ảnh. Ảnh chỉ xử lý trong RAM, không lưu.')
    with st.expander('Tham số đã khóa (đọc từ deploy_config.json)'):
        c = rt.cfg
        st.json({k: c[k] for k in ('experiment', 'seed', 'img_size', 'temperature', 'threshold', 'abstain_threshold', 'seg_threshold', 'min_area', 'tta', 'run_mode', 'pretrained_loaded')})
    st.caption('Public deployment: không tải dữ liệu có thông tin nhận dạng cá nhân.')
