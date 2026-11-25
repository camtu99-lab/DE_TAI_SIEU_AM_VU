import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import ui  # noqa: E402

ui.setup_page('About & Limits', 'ℹ️')
rt = ui.get_runtime()
ui.sidebar(rt)
ui.header()
st.subheader('Giới hạn và phạm vi sử dụng')
ui.card('warn', 'Chỉ phục vụ nghiên cứu', 'Không phải thiết bị chẩn đoán y tế và chưa được kiểm định lâm sàng. Mọi kết quả là dự đoán của mô hình, cần chuyên gia đánh giá.')
st.markdown("""
- Chưa được kiểm định lâm sàng; không thay thế đánh giá của bác sĩ.
- **Domain shift tồn tại**: ảnh từ máy siêu âm/quy trình khác có thể làm hiệu năng giảm; BUSI chỉ được dùng như tập ngoài theo protocol hiện có.
- Lớp *normal* không được gộp thành *benign*.
- Confidence/calibration **không bảo đảm** dự đoán đúng; abstention không bảo đảm bắt hết lỗi.
- Grad-CAM và evidence map **không phải bản đồ bệnh lý**; localisation không đồng nghĩa faithfulness.
- Perturbation có thể tạo ảnh ngoài phân phối huấn luyện; thay đổi dự đoán không chứng minh nguyên nhân bệnh lý.
- Khi khoảng tin cậy của hiệu số chứa 0: **chưa đủ bằng chứng về khác biệt** (ví dụ V3_full so với X6 ở AUROC nếu file thật cho CI chứa 0).
- Kết quả có thể kém hơn ở một số chỉ số (ví dụ specificity ngoài BUSI, AURC) — xem Research Dashboard; failure case không bị che giấu.
- Dữ liệu: ứng dụng không lưu ảnh; không dùng cho dữ liệu nhận dạng cá nhân.
""")
st.caption('Quyền riêng tư: ảnh chỉ xử lý trong RAM của phiên; không database, không analytics bên thứ ba, không gọi API ngoài.')
