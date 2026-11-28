# app.py - BreastVision-XAI V3: ứng dụng Streamlit phân tích ảnh siêu âm vú (công cụ hỗ trợ nghiên cứu)
import os, json
import numpy as np
import cv2
import torch
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import bus_core as core
import bus_core_v2 as C2

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
st.set_page_config(page_title='BreastVision-XAI V3', layout='wide')
page = st.sidebar.radio('Trang', ['Phân tích ảnh', 'Kết quả nghiên cứu', 'Hướng dẫn và giới hạn'])
st.sidebar.warning('Công cụ hỗ trợ NGHIÊN CỨU. Không phải thiết bị chẩn đoán y tế. Không nhập ảnh có thông tin định danh bệnh nhân.')


def load_cfg():
    p = os.path.join(HERE, 'deploy_config.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


@st.cache_resource
def get_model(ckpt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m, meta = C2.load_model_from_checkpoint(ckpt, device)
    return m, device


if page == 'Phân tích ảnh':
    st.title('BreastVision-XAI V3: phân tích ảnh siêu âm vú')
    cfg = load_cfg()
    if cfg is None:
        st.error('Không tìm thấy deploy_config.json. Hãy chạy notebook BreastVision_XAI_V3.ipynb (mục 27) và đặt các file xuất ra cạnh app.py.'); st.stop()
    if cfg.get('run_mode') != 'full_experiment':
        st.error('MÔ HÌNH NÀY TỪ CHẾ ĐỘ SMOKE TEST (dữ liệu tổng hợp). Kết quả KHÔNG có ý nghĩa khoa học hay y khoa.')
    if not cfg.get('pretrained_loaded', False):
        st.warning('Backbone không dùng trọng số pretrained ImageNet trong lần huấn luyện này.')
    ck = os.path.join(HERE, cfg['checkpoint'])
    if not os.path.exists(ck):
        st.error('Không tìm thấy checkpoint: ' + ck); st.stop()
    model, device = get_model(ck)
    st.sidebar.caption(f"Mô hình: {cfg['experiment']} | img_size {cfg['img_size']} | T={cfg['temperature']:.2f} | thiết bị: {device}")
    up = st.file_uploader('Tải ảnh siêu âm (png/jpg/bmp). Ảnh chỉ được xử lý trong bộ nhớ, không lưu.', type=['png', 'jpg', 'jpeg', 'bmp'])
    if up is None:
        st.info('Hãy tải lên một ảnh siêu âm.')
    else:
        bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error('Không đọc được ảnh.'); st.stop()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with st.spinner('Đang phân tích...'):
            r = C2.analyze_image_v2(rgb, model, cfg, device)
        if r['abstain']:
            st.warning('HỆ THỐNG TỪ CHỐI DỰ ĐOÁN: độ tin cậy sau hiệu chuẩn thấp hơn ngưỡng đã chọn trên tập validation. Cần chuyên gia xem xét.')
        else:
            st.info('Không bị từ chối. Điều này KHÔNG có nghĩa dự đoán là đúng.')
        c1, c2 = st.columns(2)
        with c1:
            st.subheader('Dự đoán của mô hình')
            st.write(f"Nhãn theo ngưỡng đã khóa: **{r['label']}** (kết quả nghiên cứu, không phải chẩn đoán)")
            st.table({'Lớp': ['benign', 'malignant'], 'Xác suất đã hiệu chuẩn (%)': [round(float(p) * 100, 2) for p in r['p_cal']],
                      'Xác suất thô (%)': [round(float(p) * 100, 2) for p in r['p_raw']]})
        with c2:
            st.subheader('Vùng nghi ngờ do mô hình phân vùng')
            st.write(f"Tỉ lệ diện tích mask: {r['mask'].mean() * 100:.2f}% | số vùng (bounding box): {len(r['boxes'])}")
            st.write(r['boxes'])
        panels = [('Ảnh đã tiền xử lý', r['image']), ('Mask dự đoán', core.overlay_mask(r['image'], r['mask'])),
                  (f"Bounding box ({len(r['boxes'])})", core.draw_boxes(r['image'], r['boxes'])), ('Grad-CAM', core.overlay_cam(r['image'], r['cam']))]
        if r['evidence'] is not None:
            ev = r['evidence'] / (r['evidence'].max() + 1e-12)
            panels.append(('Evidence map (nối trực tiếp vào logit)', core.overlay_cam(r['image'], ev)))
        fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.2))
        for a, (t, im) in zip(np.atleast_1d(ax), panels):
            a.imshow(im); a.set_title(t, fontsize=9); a.axis('off')
        st.pyplot(fig)
        st.caption('Grad-CAM và evidence map là giải thích về cách mô hình tính điểm, chưa phải bằng chứng y khoa. Độ trung thực (faithfulness) của chúng chỉ được xác nhận bằng các thử nghiệm trong báo cáo nghiên cứu.')
        st.caption('Giới hạn: huấn luyện trên một bộ dữ liệu công khai, chưa kiểm định lâm sàng; ngưỡng từ chối chỉ là công cụ hỗ trợ.')

elif page == 'Kết quả nghiên cứu':
    st.title('Kết quả thực nghiệm đã lưu')
    cfg = load_cfg()
    if cfg is not None and cfg.get('run_mode') != 'full_experiment':
        st.error('Các tệp kết quả hiện có đến từ SMOKE TEST (dữ liệu tổng hợp): chỉ kiểm tra mã, không có giá trị kết luận.')
    st.caption('Chỉ hiển thị các tệp do notebook tạo ra; không có số liệu nhập tay.')
    shown = False
    if os.path.isdir(RES):
        for fn in sorted(os.listdir(RES)):
            p = os.path.join(RES, fn)
            if fn.endswith('.csv'):
                st.subheader(fn); st.dataframe(pd.read_csv(p)); shown = True
            elif fn.endswith('.png'):
                st.image(p, caption=fn); shown = True
    if not shown:
        st.warning('Chưa có tệp kết quả trong thư mục results/.')

else:
    st.title('Hướng dẫn và giới hạn')
    st.markdown("""
**Cài đặt:** `pip install -r requirements.txt` rồi `streamlit run app.py`.

**Kiến trúc:** encoder EfficientNet-B3 chung → decoder phân vùng → gộp đặc trưng 3 vùng (toàn ảnh, tổn thương, quanh tổn thương) bằng *mask dự đoán* →
evidence map nối trực tiếp vào logit → hiệu chuẩn nhiệt độ và từ chối dự đoán chọn trên validation.

**Giới hạn:**
- Không dùng mask ground-truth khi suy luận; mask hiển thị là do mô hình dự đoán.
- Ứng dụng không lưu ảnh và không ghi thông tin định danh.
- Mô hình chưa được kiểm định lâm sàng; mọi kết quả cần chuyên gia đánh giá.
- Từ chối dự đoán không đảm bảo các dự đoán còn lại là đúng.
""")