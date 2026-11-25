import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import config, ui  # noqa: E402

ui.setup_page('Methodology', '🧭')
rt = ui.get_runtime()
ui.sidebar(rt)
ui.header()
st.subheader('Phương pháp')
FLOW = ['Ultrasound image', 'preprocessing', 'shared encoder', 'segmentation', 'predicted lesion / perilesional / posterior regions', 'evidence-aware feature fusion',
        'logits', 'temperature calibration', 'locked decision threshold', 'optional abstention']
st.markdown('<div class="bv-flow">' + '<span class="bv-arrow">→</span>'.join(f'<span class="bv-box">{p}</span>' for p in FLOW) + '</div>', unsafe_allow_html=True)
st.markdown("""
- **Shared EfficientNet-B3 encoder**, decoder phân vùng kiểu U-Net (SCSE ở V3), biểu diễn đa vùng: toàn ảnh, lesion, perilesional, posterior-zone — các vùng đều suy ra từ **mask dự đoán**.
- **Evidence map** tham gia trực tiếp vào đường tính logit của mô hình; logit phân rã được thành đóng góp theo vùng + hằng số.
- Các thành phần **chỉ thuộc huấn luyện** (boundary-aware loss, counterfactual faithfulness loss, label smoothing, EMA) **không chạy lúc suy luận**; web chỉ tái sử dụng forward path của checkpoint.
- **TTA** lật ngang trung bình xác suất; **temperature scaling** `softmax(log p / T)`; **ngưỡng phân loại** và **ngưỡng từ chối** được khóa từ validation.
- Nhãn: benign = 0, malignant = 1.
""")
st.info('Segmentation, evidence map và Grad-CAM là ba đầu ra/biểu diễn khác nhau. Vùng evidence trùng lesion không tự động chứng minh explanation faithful. Faithfulness được đánh giá riêng trong thí nghiệm.')
if rt.cfg:
    st.markdown('**Tham số đã khóa (đọc từ deploy_config.json, không hard-code):**')
    c = rt.cfg
    st.table({'tham số': ['temperature', 'ngưỡng phân loại', 'ngưỡng từ chối', 'seg_threshold', 'min_area', 'img_size', 'TTA'],
              'giá trị': [f"{c['temperature']:.6f}", f"{c['threshold']:.6f}", f"{c['abstain_threshold']:.6f}", str(c['seg_threshold']), str(c['min_area']), str(c['img_size']), str(c['tta'])]})
st.markdown('**BUS-VERIFY failure-risk detector** hiện **tắt**: chưa có artifact đã huấn luyện/kiểm định trong bộ thực nghiệm đã kiểm toán. Ứng dụng không tạo “risk score” từ confidence, entropy, Δlogit hay abstain score.')
ui.disclaimer()
