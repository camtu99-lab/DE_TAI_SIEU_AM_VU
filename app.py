"""BreastVision-XAI V4 — BUS-VERIFY. Trang 1: Research Overview. Chạy: streamlit run app.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from src import ui, app_support as AS

st = ui.setup('Research Overview', 'BUS-VERIFY: kiểm chứng bằng chứng phản thực và cảnh báo nguy cơ sai lệch dự đoán của AI trên ảnh siêu âm vú')
ui.disclaimer(st)

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader('Câu hỏi nghiên cứu')
    st.markdown('''
Không chỉ hỏi *AI dự đoán gì* và *chú ý ở đâu*, mà hỏi: **có thể phát hiện nguy cơ AI dự đoán sai hay không, nếu kiểm chứng bằng chứng
(can thiệp phản thực lên tổn thương / lân cận / nền), bất định và độ ổn định?**

- **H1**: Nhạy cảm phản thực theo vùng bổ sung thông tin phát hiện lỗi vượt trên confidence/entropy/ensemble (Error AUPRC, FN Alert Recall).
- **H2**: Ca mô hình đi tắt (nhạy với nền hơn tổn thương) có nguy cơ sai cao hơn.
- **H3**: Bộ phát hiện huấn luyện trên BUS-BRA giữ được giá trị khi sang BUSI (kiểm định ngoài) — *giả thuyết có thể bị bác bỏ*.
''')
    st.subheader('Quy trình')
    st.graphviz_chart('''digraph G{rankdir=LR;node[shape=box,style=rounded,color="#2dd4bf",fontcolor="#e6edf7"];edge[color="#64748b"];
img[label="Ảnh siêu âm"];m[label="Mô hình chính\\n(phân loại+phân vùng+evidence)"];reg[label="Vùng DỰ ĐOÁN\\nlesion/peri/background"];
cf[label="Counterfactual Verifier\\n+ đối chứng khớp diện tích"];unc[label="Bất định + ổn định\\n(TTA, ensemble)"];risk[label="Failure-Risk\\n(LR đã hiệu chuẩn, OOF)"];sel[label="Selective prediction\\n(ngưỡng khóa)"];
img->m->reg->cf->risk;m->unc->risk;risk->sel;}''')
with c2:
    st.subheader('Trạng thái artifact V4')
    s = AS.v4_status()
    rows = [dict(thành_phần=k, có=('✅' if v[0] else '⛔ NOT RUN')) for k, v in s.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if not AS.v4_has_results():
        ui.not_run(st, 'chưa có kết quả thực nghiệm V4', 'Chạy notebook Kaggle V4 rồi chép thư mục artifacts/v4 và gói deploy vào đây. Các trang kết quả sẽ tự đọc.')

st.subheader('Nền tảng V3 (kết quả thật đã công bố trong dự án)')
v3 = AS.load_table(os.path.join(AS.V3_REF, 'internal_results_mean_sd.csv'))
ext = AS.load_json(os.path.join(AS.V3_REF, 'external_busi_results.json'))
if v3 is not None:
    st.dataframe(v3, use_container_width=True, hide_index=True)
if ext:
    st.caption('Đánh giá ngoài BUSI (V3) — xem trang Scientific Results để biết chi tiết và hạn chế.')
st.subheader('Giới hạn quan trọng')
st.markdown('''
- Tập test BUS-BRA (272 ảnh, 73 ác tính) **đã được khảo sát ở V3** → không phải tập "chưa từng quan sát"; kết quả trên đó là đánh giá nội bộ giữ lại.
- Chỉ ~7 ca false-negative trong test → FN Alert Recall có khoảng tin cậy rất rộng; bằng chứng chính dựa trên nested-CV trên dữ liệu phát triển.
- BUSI không có mã bệnh nhân → CI mức ảnh là lạc quan; BUSI có ảnh trùng lặp.
''')
