"""Trang phân tích ảnh: upload/ca mẫu -> dự đoán + phân vùng + Grad-CAM + evidence + phân rã. Ảnh chỉ xử lý trong RAM."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, ui, inference as INF, preprocessing as PP, visualization as V  # noqa: E402
from src import report_export as RE, demo_cases as DC  # noqa: E402
from src.security import UserFacingError, validate_upload  # noqa: E402

ui.setup_page('Analyze')
rt = ui.get_runtime()
ui.sidebar(rt)
ui.header()
ui.require_ready(rt)

st.subheader('Phân tích ảnh siêu âm')
st.caption('Ảnh được giải mã và xử lý hoàn toàn trong bộ nhớ; không lưu, không gửi ra ngoài. Không tải dữ liệu có thông tin nhận dạng cá nhân.')

up = st.file_uploader('Tải ảnh siêu âm (png, jpg, jpeg, bmp)', type=[e.lstrip('.') for e in config.ALLOWED_EXT])
cases, cases_msg = DC.list_demo_cases()
case = None
if cases:
    choice = st.selectbox('Hoặc chọn ca mẫu ẩn danh (mẫu dữ liệu nghiên cứu)', ['(không chọn)'] + [c.id for c in cases])
    case = next((c for c in cases if c.id == choice), None)
    if case:
        st.caption(f'Nguồn: {case.source} · Giấy phép: {case.license} · {case.description}')
else:
    st.caption('Ca mẫu ẩn danh: chưa khả dụng. ' + cases_msg)
want_cam = st.checkbox('Tạo Grad-CAM (tốn thêm thời gian trên CPU)', value=False)

data, token, err = None, None, None
try:
    if up is not None:
        data = up.getvalue()
        validate_upload(up.name, data)
    elif case is not None:
        data = case.path.read_bytes()
    if data is not None:
        token = hashlib.blake2b(data, digest_size=8).hexdigest()
except UserFacingError as e:
    data, err = None, e
if err:
    ui.show_error(err)

if st.button('Phân tích ảnh', type='primary', disabled=data is None):
    try:
        with st.spinner('Đang suy luận trên CPU...'):
            rgb = PP.decode_image(data)
            res = INF.infer_image(rt, rgb, with_gradcam=want_cam)
        h, w = rgb.shape[:2]
        scale = min(1.0, 768 / max(h, w))
        disp = rgb if scale >= 1.0 else cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        st.session_state.update(current_result=res, current_display=disp, current_stats=PP.image_stats(rgb), current_token=token,
                                current_case=case.id if case else None, perturb_state=None)
    except UserFacingError as e:
        ui.show_error(e)

res = st.session_state.get('current_result')
if res is None:
    st.info('Chọn ảnh rồi bấm "Phân tích ảnh".')
    st.stop()
if token is not None and token != st.session_state.get('current_token'):
    st.warning('Ảnh đã thay đổi so với kết quả đang hiển thị. Bấm "Phân tích ảnh" để cập nhật.')

stats = st.session_state.get('current_stats', {})
if min(stats.get('height', 999), stats.get('width', 999)) < 128:
    st.warning('Ảnh nhỏ: mô hình phải phóng to; kết quả kém tin cậy.')
if stats.get('chroma', 0) > 20:
    st.warning('Ảnh có nhiều màu (Doppler/chú thích/ảnh chụp màn hình?). Ảnh B-mode thường là thang xám; ảnh này có thể ngoài miền dữ liệu huấn luyện.')
for w_ in res.warnings:
    st.warning(w_)

# ----- Kết quả chính
st.subheader('Kết quả của mô hình')
if res.abstain:
    ui.card('warn', 'Mô hình từ chối đưa ra dự đoán tin cậy',
            f'Theo ngưỡng đã khóa trên tập validation (độ tin cậy hiệu chuẩn {res.confidence:.3f} &lt; {rt.cfg["abstain_threshold"]:.3f}). '
            f'Nhãn theo ngưỡng chỉ mang tính tham khảo nghiên cứu: {config.CLASS_VI[res.label]}.')
else:
    ui.card('info', f'Dự đoán mô hình: {config.CLASS_VI[res.label]}', 'Đây là kết quả nghiên cứu của mô hình, không phải kết luận chẩn đoán.')
c1, c2, c3, c4 = st.columns(4)
c1.metric('P(malignant) thô', f'{res.raw_probs[1]:.3f}')
c2.metric('P(malignant) đã hiệu chuẩn', f'{res.p_malignant:.3f}', help='Temperature scaling chỉ hiệu chỉnh độ tin cậy của xác suất; không làm thay đổi bản chất mô hình.')
c3.metric('Ngưỡng phân loại (đã khóa)', f'{rt.cfg["threshold"]:.4f}')
c4.metric('Độ tin cậy / ngưỡng từ chối', f'{res.confidence:.3f} / {rt.cfg["abstain_threshold"]:.3f}')
st.dataframe(pd.DataFrame({'': ['benign (0)', 'malignant (1)'], 'logit thô': res.raw_logits, 'xác suất thô': res.raw_probs, 'xác suất hiệu chuẩn': res.p_cal}),
             hide_index=True, **ui.STRETCH)
st.caption(f'Temperature T = {rt.cfg["temperature"]:.6f} (đã khóa, không fit lại). Logit thô tính trên ảnh gốc không lật; xác suất thô là trung bình TTA lật ngang ({"bật" if rt.cfg["tta"] else "tắt"}). '
           f'Nhãn = malignant nếu P(malignant) hiệu chuẩn ≥ ngưỡng khóa (không phải 0.5).')

alpha = st.slider('Độ đậm lớp phủ', 0.1, 0.9, 0.45, 0.05)
tabs = st.tabs(['Tổng quan', 'Phân vùng', 'Evidence & Grad-CAM', 'Phân rã quyết định', 'Báo cáo', 'Chi tiết kỹ thuật'])

with tabs[0]:
    panels = V.analysis_panels(res, alpha, original_rgb=st.session_state.get('current_display'))
    for i in range(0, len(panels), 3):
        for col, (t, im) in zip(st.columns(3), panels[i:i + 3]):
            col.image(im, caption=t, **ui.STRETCH)

with tabs[1]:
    a, b = st.columns(2)
    a.image(V.seg_panel(res.image, res.mask, res.boxes, alpha), caption='Mask dự đoán + contour + bounding box', **ui.STRETCH)
    b.metric('Tỉ lệ diện tích mask', f'{res.mask_fraction * 100:.2f}%')
    b.metric('Số vùng (sau min_area)', len(res.boxes))
    if res.boxes:
        bx = pd.DataFrame(res.boxes, columns=['x1', 'y1', 'x2', 'y2'])
        b.dataframe(bx, hide_index=True, **ui.STRETCH)
    st.caption('Không hiển thị Dice cho ảnh upload vì không có mask ground truth. Mask dùng cho các vùng lesion/quanh/hậu âm là mask DỰ ĐOÁN.')
    if res.post_prob is not None or res.ring_prob is not None:
        zs = []
        if res.ring_prob is not None:
            zs.append(('Vòng quanh tổn thương (từ mask dự đoán)', V.heat_panel(res.image, res.ring_prob, alpha)))
        if res.post_prob is not None:
            zs.append(('Vùng hậu âm BI-RADS (từ mask dự đoán)', V.heat_panel(res.image, res.post_prob, alpha)))
        for col, (t, im) in zip(st.columns(len(zs)), zs):
            col.image(im, caption=t, **ui.STRETCH)
    case_id = st.session_state.get('current_case')
    case_obj = next((c for c in cases if c.id == case_id), None) if case_id else None
    if case_obj is not None and case_obj.gt_mask_path is not None:
        if st.checkbox('Chế độ nghiên cứu — hiển thị ground truth (chỉ ca mẫu có GT hợp lệ)'):
            gt = PP.decode_image(case_obj.gt_mask_path.read_bytes())
            st.metric('Dice (predicted mask vs GT)', f'{DC.dice_vs_gt(res.mask, gt, int(rt.cfg["img_size"])):.3f}')
            if case_obj.gt_label:
                st.write(f'Nhãn ground truth của ca mẫu: **{case_obj.gt_label}**')

with tabs[2]:
    st.markdown('##### Evidence map — bản đồ bằng chứng nội tại của mô hình')
    if res.evidence is not None:
        st.image(V.heat_panel(res.image, res.evidence, alpha), width=380)
        st.caption('Evidence map tham gia vào đường tính logit của mô hình. Mức tập trung không đồng nghĩa với bằng chứng y khoa và localisation không đồng nghĩa với faithfulness.')
        if res.collapse is not None:
            (st.warning if res.collapse['collapse_flag'] else st.success)(
                ('Evidence map có dấu hiệu collapse (quá đồng đều/dồn vào 1–2 ô).' if res.collapse['collapse_flag']
                 else f"Evidence map không collapse: entropy chuẩn hóa {res.collapse['mean_norm_entropy']:.2f}, số ô hiệu dụng ≈ {res.collapse['mean_ess_fraction'] * 100:.0f}% lưới."))
        if res.flip_abs_diff is not None:
            st.metric('Chênh lệch P(malignant) thô khi lật ngang', f'{res.flip_abs_diff * 100:.1f} điểm %', help='Chênh lệch lớn: dự đoán kém ổn định với ảnh này.')
    else:
        st.info('Kiến trúc đang chạy không trả evidence map.')
    st.markdown('##### Grad-CAM — vùng kích hoạt ảnh hưởng đến dự đoán')
    if res.cam is None and res.cam_error is None:
        if st.button('Tạo Grad-CAM cho ca hiện tại'):
            INF.compute_gradcam(rt, res)
            st.rerun()
    if res.cam is not None:
        st.image(V.heat_panel(res.image, res.cam, alpha), width=380)
        st.caption('Grad-CAM không phải “vùng ung thư”. Đây là minh họa XAI về vùng kích hoạt ảnh hưởng đến dự đoán; không chứng minh mô hình suy luận đúng.')
    elif res.cam_error:
        st.warning('Không tạo được Grad-CAM cho ca này. Kết quả dự đoán chính vẫn hợp lệ.')

with tabs[3]:
    if res.contrib is None:
        st.info('Tab phân rã bị ẩn/không khả dụng: ' + (res.decomposition_note or 'kiến trúc không hỗ trợ.'))
    else:
        fig, margin, total = V.contrib_figure(res.contrib)
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(pd.DataFrame({'thành phần': list(res.contrib['names']) + ['Hằng số (bias)'], 'đóng góp (logit malignant − benign)': np.round(margin, 5)}),
                     hide_index=True, **ui.STRETCH)
        st.caption('Chỉ hiển thị vì tổng đóng góp khớp logit đầu ra của mô hình trong dung sai (kiểm tra tự động). Đây là phân rã của chính mô hình trên ảnh gốc không lật, '
                   'không phải quan hệ nhân quả lâm sàng; kết quả cuối cùng còn qua TTA và temperature scaling.')

with tabs[4]:
    ps = st.session_state.get('perturb_state')
    report = RE.build_case_report(rt, res, ps['results'] if ps else None, ps['params'] if ps else None)
    fname, payload = RE.case_report_bytes(report)
    st.download_button('⬇️ Tải báo cáo nghiên cứu ca hiện tại (JSON)', payload, file_name=fname, mime='application/json')
    st.caption('Báo cáo tạo trong RAM, không chứa tên tệp gốc, bytes ảnh, EXIF hay đường dẫn cục bộ. Không phải báo cáo y khoa.')
    with st.expander('Xem nội dung báo cáo'):
        st.json(report)

with tabs[5]:
    st.write(f'Thời gian suy luận: {res.timing_s:.2f}s · thiết bị: {rt.device_name} · kích thước ảnh gốc: {res.original_shape[1]}×{res.original_shape[0]}')
    if res.cam_error:
        st.code('Grad-CAM: ' + res.cam_error)
    if res.decomposition_note:
        st.code('Phân rã: ' + res.decomposition_note)
    if res.contrib:
        st.code(f'Sai số tổng đóng góp so với logit: {res.contrib["max_abs_err"]:.2e}')
ui.disclaimer()
