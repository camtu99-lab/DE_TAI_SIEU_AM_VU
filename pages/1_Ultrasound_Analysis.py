import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from src import ui, app_support as AS, pipeline as PL, preprocessing as PP, evidence_maps as EM

st = ui.setup('Ultrasound Analysis', 'Phân loại, phân vùng, bản đồ bằng chứng và xác suất đã hiệu chuẩn')
ui.disclaimer(st)
bundle, pred, msgs = ui.runtime(st)
ui.show_runtime_messages(st, msgs)

src_choice = st.radio('Nguồn ảnh', ['Tải ảnh lên', 'Ảnh mẫu ẩn danh'], horizontal=True)
rgb, name = None, None
if src_choice == 'Tải ảnh lên':
    up = st.file_uploader('Ảnh siêu âm vú (PNG/JPG, ≤15 MB)', type=['png', 'jpg', 'jpeg'])
    if up is not None:
        rgb, name = np.array(Image.open(up).convert('RGB')), up.name
else:
    samples = AS.list_samples()
    if not samples:
        st.info('Chưa có ảnh mẫu trong assets/samples/ (gói không phân phối ảnh bệnh nhân). Đặt ảnh ẩn danh hợp lệ vào thư mục đó để dùng.')
    else:
        pick = st.selectbox('Chọn ảnh mẫu', samples, format_func=os.path.basename)
        rgb, name = np.array(Image.open(pick).convert('RGB')), os.path.basename(pick)

if rgb is not None:
    try:
        PP.validate_image(rgb)
    except PP.InvalidImageError as e:
        st.error(f'Ảnh không hợp lệ: {e}'); st.stop()
    if pred is None or bundle is None:
        ui.not_run(st, 'không thể suy luận', 'Thiếu checkpoint/PyTorch/gói triển khai (xem thông báo ở trên). Không hiển thị kết quả giả.')
        st.image(rgb, caption='Ảnh gốc', width=360); st.stop()
    with st.spinner('Đang phân tích (CPU)…'):
        out = PL.analyze_array(rgb, pred, bundle, with_cf=False)
        out['name'] = name
        cam = None
        try:
            cam = pred.gradcam(out['image'])
        except Exception as e:                    # Grad-CAM tùy chọn: báo lý do, không giấu
            out['gradcam_error'] = f'{type(e).__name__}: {e}'
        out['gradcam'] = cam
    st.session_state['analysis'] = out
    ex = out['extra']
    cols = st.columns(4)
    cols[0].image(out['image'], caption='Ảnh gốc (đã tiền xử lý)', use_container_width=True)
    if ex['mask'] is not None:
        cols[1].image(AS.overlay_mask(out['image'], ex['mask']), caption='Segmentation overlay (dự đoán)', use_container_width=True)
    else:
        cols[1].warning('Không có đầu ra phân vùng.')
    if ex['evidence'] is not None:
        cols[2].image(AS.heat_overlay(out['image'], EM.upsample_evidence(ex['evidence'], out['image'].shape[:2])), caption='Evidence map', use_container_width=True)
    else:
        cols[2].warning('Mô hình không có evidence map.')
    if cam is not None:
        cols[3].image(AS.heat_overlay(out['image'], cam), caption='Grad-CAM', use_container_width=True)
    else:
        cols[3].info(f"Grad-CAM không khả dụng. {out.get('gradcam_error', '')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Dự đoán', 'Ác tính' if out['label'] == 'malignant' else 'Lành tính')
    m2.metric('P(ác tính) đã hiệu chuẩn', f"{out['p_malignant']:.3f}")
    m3.metric('Thời gian suy luận', f"{out['inference_seconds']:.2f} s")
    m4.metric('Ngưỡng quyết định (khóa)', f"{bundle.cfg['threshold']:.3f}")
    st.caption(f"Mô hình: {bundle.cfg.get('model_name', '?')} · phiên bản {bundle.cfg.get('version', '?')} · T={bundle.cfg['temperature']:.3f} · ảnh {name} (không lưu, chỉ trong phiên).")
    if ex['mask_info']['empty']:
        st.warning('Mask dự đoán rỗng — không thể kiểm chứng bằng chứng; xem trang Failure-Risk.')
    st.info('Xác suất này là P(ác tính) của mô hình, KHÁC với risk score (nguy cơ mô hình dự đoán sai) ở trang Failure-Risk.')
