# app.py - BreastVision-XAI V3: web demo Streamlit phân tích ảnh siêu âm vú (công cụ hỗ trợ NGHIÊN CỨU)
# Chạy: streamlit run app.py
# Cần cạnh file này: các module .py của notebook, deploy_config.json, checkpoint .pt (xem mục 27 của notebook).
import os, io, sys, json, glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

st.set_page_config(page_title='BreastVision-XAI V3', page_icon='🩻', layout='wide')

import bus_core as core          # noqa: E402
import bus_core_v2 as C2         # noqa: E402
import bus_trust as T            # noqa: E402
import evaluation_xai as EV      # noqa: E402

# tương thích nhiều phiên bản Streamlit: bản mới dùng width='stretch', bản cũ dùng use_container_width=True
_v = tuple(int(x) for x in st.__version__.split('.')[:2] if x.isdigit())
STRETCH = {'width': 'stretch'} if _v >= (1, 50) else {'use_container_width': True}

CLASSES = ['benign', 'malignant']
CLASS_VI = {'benign': 'benign (lành tính)', 'malignant': 'malignant (ác tính)'}
IMG_TYPES = ['png', 'jpg', 'jpeg', 'bmp']

st.markdown("""
<style>
.bv-card{border-radius:12px;padding:16px 20px;margin:4px 0 14px 0;border:1px solid rgba(128,128,128,.25)}
.bv-card h3{margin:0 0 4px 0;font-size:1.35rem}
.bv-card p{margin:0;opacity:.85}
.bv-ok{background:rgba(11,114,133,.10);border-left:6px solid #0b7285}
.bv-abs{background:rgba(240,140,0,.12);border-left:6px solid #f08c00}
.bv-smoke{background:rgba(224,49,49,.10);border-left:6px solid #e03131}
</style>
""", unsafe_allow_html=True)


# ============================ Cấu hình / nạp mô hình ============================
def find_model_dir():
    cands = [os.environ.get('BV_MODEL_DIR', ''), HERE, os.path.join(HERE, 'deploy_xai')]
    for d in cands:
        if d and os.path.exists(os.path.join(d, 'deploy_config.json')):
            return d
    return None


def load_cfg(model_dir):
    with open(os.path.join(model_dir, 'deploy_config.json'), encoding='utf-8') as f:
        return json.load(f)


@st.cache_resource(show_spinner='Đang nạp mô hình...')
def get_model(ckpt, mtime):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, meta = C2.load_model_from_checkpoint(ckpt, device)
    return model, device


# ============================ Phân tích một ảnh ============================
def _extras(model, rgb, cfg, device):
    """Thông tin giải thích bổ sung từ một lượt forward (không lật): phân rã logit theo vùng, vùng hậu âm, collapse, ổn định khi lật."""
    size = int(cfg['img_size'])
    img = core.preprocess_image(rgb, size)
    x = core.normalize_to_tensor(img).unsqueeze(0).to(device)
    out = {}
    with torch.no_grad():
        o = model(x)
        of = model(x.flip(-1))
        out['p_mal_orig'] = float(F.softmax(o['logits'].float(), 1)[0, 1])
        out['p_mal_flip'] = float(F.softmax(of['logits'].float(), 1)[0, 1])
        for key in ('ring', 'post'):
            if o.get(key) is not None:
                out[key] = F.interpolate(o[key].float(), size=(size, size), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        if o.get('evidence') is not None:
            out['collapse'] = EV.evidence_collapse_report(o['evidence'].float().cpu())
        mc = getattr(model, 'cfg', None)
        if mc is not None and getattr(model, 'fusion', None) is not None and model.fusion.mode == 'concat':
            names = ['Toàn ảnh (evidence)']
            if mc.use_lesion:
                names.append('Tổn thương')
            if mc.use_peri:
                names.append('Quanh tổn thương')
            if mc.use_posterior:
                names.append('Hậu âm (BI-RADS)')
            D = mc.dim
            W, b = model.head[1].weight, model.head[1].bias
            vals = []
            for i, r in enumerate(o['regions']):
                v = model.fusion.proj[i](r)
                vals.append((v @ W[:, i * D:(i + 1) * D].T)[0].float().cpu().numpy())
            out['contrib'] = dict(names=names, vals=np.array(vals), bias=b.float().cpu().numpy(), logits=o['logits'][0].float().cpu().numpy())
    return out


@st.cache_data(show_spinner=False, max_entries=6)
def run_analysis(_model, _device, img_bytes, cfg_json, ckpt):
    cfg = json.loads(cfg_json)
    bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = C2.analyze_image_v2(rgb, _model, cfg, _device)
    r['extras'] = _extras(_model, rgb, cfg, _device)
    h, w = rgb.shape[:2]
    chroma = float((rgb.max(2).astype(np.int16) - rgb.min(2).astype(np.int16)).mean())
    r['input_info'] = dict(height=int(h), width=int(w), chroma=chroma)
    if r['evidence'] is not None:
        r['evidence_in_mask'] = float((r['evidence'] * (r['mask'] > 0)).sum() / (r['evidence'].sum() + 1e-12)) if r['mask'].any() else float('nan')
    r['cam_in_mask'] = float(T.cam_alignment(r['cam'], r['mask'])[0][0]) if r['mask'].any() else float('nan')
    return r


# ============================ Hiển thị ============================
def norm01(a):
    a = np.asarray(a, np.float32)
    return a / (a.max() + 1e-12)


def panel_set(r, alpha, show_boxes):
    img = r['image']
    mask_ov = core.overlay_mask(img, r['mask'], alpha=alpha)
    if show_boxes and r['boxes']:
        mask_ov = core.draw_boxes(mask_ov, r['boxes'])
    panels = [('Ảnh đã tiền xử lý', img), (f"Mask dự đoán ({len(r['boxes'])} vùng)", mask_ov), ('Grad-CAM', core.overlay_cam(img, r['cam'], alpha=alpha))]
    if r['evidence'] is not None:
        panels.append(('Evidence map (nối vào logit)', core.overlay_cam(img, norm01(r['evidence']), alpha=alpha)))
    return panels


def contrib_figure(c):
    names = c['names'] + ['Hằng số (bias)']
    margin = list(c['vals'][:, 1] - c['vals'][:, 0]) + [float(c['bias'][1] - c['bias'][0])]
    total = float(sum(margin))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.2))
    cols = ['#e03131' if v > 0 else '#1c7ed6' for v in margin]
    ax.barh(names[::-1], margin[::-1], color=cols[::-1])
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('Đóng góp vào (logit malignant − logit benign), trước hiệu chuẩn')
    ax.set_title(f'Tổng = {total:+.3f}  →  nghiêng về {"malignant" if total > 0 else "benign"}', fontsize=10)
    plt.tight_layout()
    return fig, margin, total


def composite_png(r, cfg, alpha):
    panels = panel_set(r, alpha, True)
    ex = r['extras']
    if ex.get('post') is not None:
        panels.append(('Vùng hậu âm BI-RADS', core.overlay_cam(r['image'], norm01(ex['post']), alpha=alpha)))
    fig, ax = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.3))
    for a, (t, im) in zip(np.atleast_1d(ax), panels):
        a.imshow(im); a.set_title(t, fontsize=9); a.axis('off')
    pm = r['p_cal'][1] * 100
    fig.suptitle(f"BreastVision-XAI ({cfg['experiment']}) | nhãn theo ngưỡng: {r['label']} | P(malignant) hiệu chuẩn = {pm:.1f}% | "
                 f"{'TỪ CHỐI DỰ ĐOÁN' if r['abstain'] else 'không bị từ chối'} | công cụ hỗ trợ nghiên cứu, không phải chẩn đoán", fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=140); plt.close(fig)
    return buf.getvalue()


def report_json(r, cfg):
    ex = r['extras']
    d = dict(model=cfg['experiment'], run_mode=cfg.get('run_mode'), img_size=cfg['img_size'], seed=cfg.get('seed'), temperature=cfg['temperature'],
             classification_threshold=cfg['threshold'], abstain_threshold=cfg['abstain_threshold'], label=r['label'], abstain=r['abstain'],
             p_calibrated=dict(zip(CLASSES, [float(p) for p in r['p_cal']])), p_raw=dict(zip(CLASSES, [float(p) for p in r['p_raw']])),
             confidence=float(r['confidence']), mask_area_fraction=float(r['mask'].mean()), n_regions=len(r['boxes']), boxes_xyxy=[list(b) for b in r['boxes']],
             stability_flip_abs_diff=abs(ex['p_mal_orig'] - ex['p_mal_flip']), note='Công cụ hỗ trợ nghiên cứu; không phải chẩn đoán y khoa.')
    return json.dumps(d, ensure_ascii=False, indent=2)


# ============================ Thanh bên ============================
page = st.sidebar.radio('Trang', ['Phân tích ảnh', 'Kết quả nghiên cứu', 'Hướng dẫn và giới hạn'])
st.sidebar.warning('Công cụ hỗ trợ NGHIÊN CỨU, không phải thiết bị chẩn đoán y tế. Không tải ảnh có thông tin định danh bệnh nhân.')
model_dir = find_model_dir()
cfg = load_cfg(model_dir) if model_dir else None


# ============================ Trang 1: Phân tích ảnh ============================
if page == 'Phân tích ảnh':
    st.title('🩻 BreastVision-XAI V3')
    st.caption('Phân loại benign/malignant, phân vùng tổn thương và giải thích bằng evidence map trên ảnh siêu âm vú (B-mode).')

    if cfg is None:
        st.error('Không tìm thấy `deploy_config.json`.')
        st.markdown('Chạy notebook `BreastVision_XAI_V3.ipynb` đến **mục 27**, rồi đặt thư mục `deploy_xai/` (gồm checkpoint `.pt` và `deploy_config.json`) '
                    'cạnh `app.py`, hoặc đặt biến môi trường `BV_MODEL_DIR` trỏ tới thư mục đó.')
        st.stop()
    if cfg.get('run_mode') != 'full_experiment':
        st.markdown('<div class="bv-card bv-smoke"><h3>⚠️ MÔ HÌNH SMOKE TEST</h3><p>Checkpoint này huấn luyện trên dữ liệu tổng hợp/rất ít epoch. '
                    'Kết quả KHÔNG có ý nghĩa khoa học hay y khoa; chỉ dùng để kiểm tra giao diện.</p></div>', unsafe_allow_html=True)
    if not cfg.get('pretrained_loaded', False):
        st.warning('Backbone không dùng trọng số pretrained ImageNet trong lần huấn luyện này.')
    if str(cfg.get('experiment', '')).startswith('E'):
        st.error('Ứng dụng này cần mô hình họ X/V3 (có evidence map). Checkpoint hiện tại là baseline E*.'); st.stop()

    ck = os.path.join(model_dir, cfg['checkpoint'])
    if not os.path.exists(ck):
        st.error('Không tìm thấy checkpoint: ' + ck); st.stop()
    model, device = get_model(ck, os.path.getmtime(ck))
    st.sidebar.divider()
    st.sidebar.caption(f"Mô hình: **{cfg['experiment']}** · img_size {cfg['img_size']} · T={cfg['temperature']:.2f} · ngưỡng={cfg['threshold']:.3f} · "
                       f"ngưỡng từ chối={cfg['abstain_threshold']:.3f} · thiết bị: {device}")
    st.sidebar.subheader('Hiển thị')
    alpha = st.sidebar.slider('Độ đậm lớp phủ', 0.1, 0.9, 0.45, 0.05)
    show_boxes = st.sidebar.checkbox('Vẽ bounding box', True)

    up = st.file_uploader('Tải ảnh siêu âm (png/jpg/bmp). Ảnh chỉ được xử lý trong bộ nhớ, không lưu.', type=IMG_TYPES)
    samples = sorted(p for ext in IMG_TYPES for p in glob.glob(os.path.join(HERE, 'samples', f'*.{ext}')))
    sample_choice = None
    if samples:
        sample_choice = st.selectbox('Hoặc chọn ảnh mẫu', ['(không chọn)'] + [os.path.basename(p) for p in samples])

    img_bytes = None
    if up is not None:
        img_bytes = up.getvalue()
    elif sample_choice and sample_choice != '(không chọn)':
        with open(os.path.join(HERE, 'samples', sample_choice), 'rb') as fh:
            img_bytes = fh.read()
    if img_bytes is None:
        st.info('Hãy tải lên một ảnh siêu âm' + (' hoặc chọn ảnh mẫu.' if samples else '.'))
        st.stop()

    with st.spinner('Đang phân tích...'):
        r = run_analysis(model, device, img_bytes, json.dumps(cfg, sort_keys=True), ck)
    if r is None:
        st.error('Không đọc được ảnh. Hãy thử định dạng png/jpg/bmp khác.'); st.stop()

    ex, info = r['extras'], r['input_info']
    # ---- cảnh báo chất lượng đầu vào
    if min(info['height'], info['width']) < 100:
        st.warning(f"Ảnh rất nhỏ ({info['width']}×{info['height']} px); mô hình sẽ phải phóng to, kết quả kém tin cậy.")
    if info['chroma'] > 20:
        st.warning('Ảnh có nhiều màu. Ảnh siêu âm B-mode thường là thang xám; ảnh có Doppler, chú thích màu hoặc ảnh chụp màn hình có thể nằm ngoài miền dữ liệu huấn luyện.')

    # ---- kết luận
    p_mal = float(r['p_cal'][1])
    if r['abstain']:
        st.markdown(f"<div class='bv-card bv-abs'><h3>⏸ Hệ thống TỪ CHỐI dự đoán</h3><p>Độ tin cậy sau hiệu chuẩn ({r['confidence']*100:.1f}%) thấp hơn ngưỡng "
                    f"{cfg['abstain_threshold']*100:.1f}% (chọn trên tập validation). Cần chuyên gia xem xét. Nhãn nghiêng về: {CLASS_VI[r['label']]}.</p></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bv-card bv-ok'><h3>Nghiêng về: {CLASS_VI[r['label']]}</h3><p>Kết quả nghiên cứu theo ngưỡng đã khóa; không bị từ chối. "
                    "Điều này KHÔNG có nghĩa dự đoán là đúng.</p></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('P(malignant) đã hiệu chuẩn', f'{p_mal*100:.1f}%')
    c2.metric('Ngưỡng phân loại', f"{cfg['threshold']*100:.1f}%")
    c3.metric('Độ tin cậy', f"{r['confidence']*100:.1f}%", delta=f"{(r['confidence']-cfg['abstain_threshold'])*100:+.1f} so với ngưỡng từ chối", delta_color='off')
    c4.metric('Diện tích mask', f"{r['mask'].mean()*100:.1f}%")
    pc1, pc2 = st.columns(2)
    pc1.progress(float(r['p_cal'][0]), text=f"benign: {r['p_cal'][0]*100:.1f}% (thô {r['p_raw'][0]*100:.1f}%)")
    pc2.progress(float(r['p_cal'][1]), text=f"malignant: {r['p_cal'][1]*100:.1f}% (thô {r['p_raw'][1]*100:.1f}%)")
    if not r['boxes']:
        st.warning('Mô hình phân vùng không tìm thấy vùng nghi ngờ nào (mask rỗng hoặc dưới diện tích tối thiểu). Dự đoán khi đó chủ yếu dựa trên đặc trưng toàn ảnh.')

    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Tổng quan', 'Vùng giải phẫu', 'Đóng góp theo vùng', 'Độ tin cậy giải thích', 'Tải xuống'])

    with tab1:
        panels = panel_set(r, alpha, show_boxes)
        for col, (t, im) in zip(st.columns(len(panels)), panels):
            col.image(im, caption=t, **STRETCH)
        st.caption('Grad-CAM và evidence map là giải thích về cách mô hình tính điểm, chưa phải bằng chứng y khoa. '
                   'Độ trung thực (faithfulness) chỉ được xác nhận bằng các thử nghiệm trong báo cáo nghiên cứu.')
        if r['boxes']:
            bx = pd.DataFrame(r['boxes'], columns=['x1', 'y1', 'x2', 'y2'])
            bx['rộng'], bx['cao'] = bx.x2 - bx.x1, bx.y2 - bx.y1
            st.dataframe(bx, hide_index=True, **STRETCH)
            st.caption('Tọa độ pixel trên ảnh đã tiền xử lý (letterbox vuông rồi resize về img_size).')

    with tab2:
        st.markdown('Các vùng dùng để gộp đặc trưng đều **suy ra từ mask do mô hình dự đoán** (không dùng ground-truth lúc suy luận).')
        zones = [('Tổn thương (mask dự đoán)', core.overlay_mask(r['image'], r['mask'], alpha=alpha))]
        if ex.get('ring') is not None:
            zones.append(('Vòng quanh tổn thương', core.overlay_cam(r['image'], norm01(ex['ring']), alpha=alpha)))
        if ex.get('post') is not None:
            zones.append(('Vùng hậu âm BI-RADS (phía dưới tổn thương)', core.overlay_cam(r['image'], norm01(ex['post']), alpha=alpha)))
        for col, (t, im) in zip(st.columns(len(zones)), zones):
            col.image(im, caption=t, **STRETCH)
        if ex.get('post') is not None:
            st.caption('Vùng hậu âm được tính theo trục dọc của chùm tia siêu âm phía dưới tổn thương (bóng lưng / tăng âm sau), phục vụ đặc trưng BI-RADS.')

    with tab3:
        c = ex.get('contrib')
        if c is None:
            st.info('Mô hình này không hỗ trợ phân rã logit theo vùng (cần fusion dạng concat).')
        else:
            fig, margin, total = contrib_figure(c)
            st.pyplot(fig); plt.close(fig)
            st.dataframe(pd.DataFrame(dict(vùng=c['names'] + ['Hằng số (bias)'], đóng_góp_logit_margin=np.round(margin, 4))), hide_index=True, **STRETCH)
            lg = c['logits']
            st.caption(f'Tổng các đóng góp = logit malignant − logit benign = {lg[1]-lg[0]:+.4f} (khớp chính xác, tính trên ảnh gốc không lật). '
                       'Giá trị dương đẩy về malignant, âm đẩy về benign. Đây là phân rã của chính mô hình, không phải quan hệ nhân quả lâm sàng. '
                       'Kết quả cuối cùng dùng thêm TTA lật ngang và hiệu chuẩn nhiệt độ nên có thể lệch nhẹ.')

    with tab4:
        a, b, c_ = st.columns(3)
        if r['evidence'] is not None and not np.isnan(r.get('evidence_in_mask', np.nan)):
            a.metric('Evidence nằm trong mask dự đoán', f"{r['evidence_in_mask']*100:.1f}%",
                     help=f"Diện tích mask chiếm {r['mask'].mean()*100:.1f}% ảnh; nếu evidence rải đều thì giá trị xấp xỉ con số này.")
        else:
            a.metric('Evidence nằm trong mask dự đoán', 'n/a')
        b.metric('Grad-CAM nằm trong mask dự đoán', 'n/a' if np.isnan(r['cam_in_mask']) else f"{r['cam_in_mask']*100:.1f}%")
        d = abs(ex['p_mal_orig'] - ex['p_mal_flip'])
        c_.metric('Chênh lệch P(malignant) khi lật ngang', f'{d*100:.1f} điểm %', help='Dự đoán thô trên ảnh gốc và ảnh lật; chênh lệch lớn là dấu hiệu dự đoán không ổn định.')
        col = ex.get('collapse')
        if col is not None:
            if col['collapse_flag']:
                st.warning('Evidence map có dấu hiệu **collapse** (quá đồng đều hoặc dồn vào 1–2 ô): giải thích này kém thông tin.')
            else:
                st.success(f"Evidence map không collapse: entropy chuẩn hóa {col['mean_norm_entropy']:.2f}, số ô hiệu dụng ≈ {col['mean_ess_fraction']*100:.0f}% lưới đặc trưng.")
        if d > 0.15:
            st.warning('Dự đoán thay đổi đáng kể khi lật ảnh: nên diễn giải rất thận trọng.')
        st.caption('Các chỉ số này chỉ mô tả hành vi của mô hình trên ảnh này. Định vị (localisation) khác với độ trung thực (faithfulness) và nhất quán (consistency); '
                   'chỉ các thử nghiệm deletion/insertion/intervention trong báo cáo mới nói về ảnh hưởng nhân quả tới dự đoán.')

    with tab5:
        st.download_button('⬇️ Tải hình tổng hợp (PNG)', composite_png(r, cfg, alpha), file_name='breastvision_result.png', mime='image/png')
        st.download_button('⬇️ Tải báo cáo (JSON, không chứa ảnh)', report_json(r, cfg), file_name='breastvision_result.json', mime='application/json')
        st.caption('Báo cáo không chứa ảnh gốc hay tên file; ảnh không được lưu trên máy chủ.')

    st.caption('Giới hạn: huấn luyện trên dữ liệu công khai (BUS-BRA), kiểm định ngoài trên BUSI; chưa kiểm định lâm sàng. Từ chối dự đoán không đảm bảo các dự đoán còn lại là đúng.')


# ============================ Trang 2: Kết quả nghiên cứu ============================
elif page == 'Kết quả nghiên cứu':
    st.title('Kết quả thực nghiệm đã lưu')
    if cfg is not None and cfg.get('run_mode') != 'full_experiment':
        st.error('Các tệp kết quả hiện có đến từ SMOKE TEST (dữ liệu tổng hợp): chỉ kiểm tra mã, không có giá trị kết luận.')
    st.caption('Chỉ hiển thị các tệp do notebook tạo ra trong thư mục results/; không có số liệu nhập tay.')
    res_dir = os.path.join(model_dir or HERE, 'results')
    if not os.path.isdir(res_dir) or not os.listdir(res_dir):
        st.warning('Chưa có tệp kết quả trong thư mục results/.'); st.stop()
    files = sorted(os.listdir(res_dir))
    # không hiển thị manifest chia dữ liệu (chứa ID bệnh nhân/đường dẫn ảnh) trên demo công khai
    csvs, pngs, mds = [f for f in files if f.endswith('.csv') and 'manifest' not in f], [f for f in files if f.endswith('.png')], [f for f in files if f.endswith('.md')]
    t1, t2, t3 = st.tabs(['Bảng số liệu', 'Hình', 'Báo cáo'])
    with t1:
        main = [f for f in ('internal_results_mean_sd.csv', 'calibration.csv', 'selective_prediction.csv', 'paired_comparisons.csv', 'xai_evaluation.csv') if f in csvs]
        for f in main + [f for f in csvs if f not in main]:
            with st.expander(f, expanded=f == 'internal_results_mean_sd.csv'):
                st.dataframe(pd.read_csv(os.path.join(res_dir, f)), **STRETCH)
        ext = os.path.join(res_dir, 'external_busi_results.json')
        if os.path.exists(ext):
            with st.expander('external_busi_results.json (đánh giá ngoài BUSI, pipeline đã khóa)'):
                with open(ext, encoding='utf-8') as fh:
                    st.json(json.load(fh))
        if not csvs:
            st.info('Không có file CSV.')
    with t2:
        if not pngs:
            st.info('Không có hình.')
        for f in pngs:
            st.image(os.path.join(res_dir, f), caption=f, **STRETCH)
    with t3:
        for f in mds:
            with open(os.path.join(res_dir, f), encoding='utf-8') as fh:
                st.markdown(fh.read())
        if not mds:
            st.info('Không có báo cáo .md.')


# ============================ Trang 3: Hướng dẫn và giới hạn ============================
else:
    st.title('Hướng dẫn và giới hạn')
    st.markdown("""
**Cách dùng:** tải ảnh siêu âm vú (B-mode) ở trang *Phân tích ảnh*. Ứng dụng trả về nhãn theo ngưỡng đã khóa, xác suất đã hiệu chuẩn, trạng thái từ chối dự đoán,
mask/bounding box, Grad-CAM, evidence map và phân rã đóng góp theo vùng.

**Kiến trúc (V3):** encoder EfficientNet-B3 dùng chung → decoder phân vùng (SCSE) → gộp đặc trưng 4 vùng (toàn ảnh theo evidence, tổn thương, quanh tổn thương, hậu âm BI-RADS)
dựa trên *mask dự đoán* → evidence map nối trực tiếp vào logit (phân rã cộng chính xác) → hiệu chuẩn nhiệt độ → từ chối dự đoán theo ngưỡng chọn trên validation.

**Đọc kết quả thế nào**
- *Xác suất đã hiệu chuẩn* là xác suất sau temperature scaling; *độ tin cậy* là xác suất lớn nhất. Dưới ngưỡng từ chối thì ứng dụng không đưa ra dự đoán.
- *Evidence nằm trong mask* chỉ cho biết giải thích tập trung ở đâu, chưa chứng minh nó ảnh hưởng tới dự đoán.
- *Đóng góp theo vùng* cho biết vùng nào đẩy logit về malignant hay benign trong chính mô hình.

**Giới hạn**
- Không dùng mask ground-truth khi suy luận; mask hiển thị do mô hình dự đoán.
- Ứng dụng không lưu ảnh và không ghi thông tin định danh.
- Huấn luyện trên một bộ dữ liệu công khai, kiểm định ngoài trên BUSI; chưa kiểm định lâm sàng; ảnh từ máy siêu âm hoặc quốc gia khác có thể lệch miền.
- Từ chối dự đoán không đảm bảo các dự đoán còn lại là đúng. Mọi kết quả cần chuyên gia đánh giá.
- Đây là công cụ hỗ trợ nghiên cứu, không phải thiết bị chẩn đoán y tế.

**Chạy cục bộ:** `pip install -r requirements.txt` rồi `streamlit run app.py`. Có thể đặt `BV_MODEL_DIR` để trỏ tới thư mục chứa `deploy_config.json` và checkpoint.
Thư mục `samples/` (tùy chọn) chứa ảnh mẫu xuất hiện trong mục chọn ảnh; chỉ dùng ảnh có giấy phép cho phép và không có thông tin định danh.
""")
