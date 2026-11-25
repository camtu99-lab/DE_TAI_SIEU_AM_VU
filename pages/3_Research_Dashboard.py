"""Dashboard kết quả nghiên cứu: đọc trực tiếp CSV/JSON thật trong results/. Không nhập tay số liệu."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config, ui, research_results as RR, report_export as RE, visualization as V  # noqa: E402
from src.security import UserFacingError  # noqa: E402

ui.setup_page('Research Dashboard', '📊')
rt = ui.get_runtime()
ui.sidebar(rt)
ui.header()
st.subheader('Dashboard kết quả nghiên cứu')
st.caption('Mọi số liệu đọc trực tiếp từ tệp trong results/ (do notebook tạo ra). Checkpoint demo là một seed; các bảng nội bộ là mean ± sd qua các seed trong file.')

try:
    ms = RR.load_table('internal_results_mean_sd.csv')
    per = RR.load_table('internal_results_per_seed.csv')
    cal = RR.load_table('calibration.csv')
    sel = RR.load_table('selective_prediction.csv')
    ext = RR.load_external()
except UserFacingError as e:
    ui.show_error(e)
    st.stop()

tabs = st.tabs(['A. Nội bộ BUS-BRA', 'B. So sánh X0 / X6 / V3_full', 'C. Calibration', 'D. Selective prediction', 'E. Ngoài BUSI', 'F. XAI', 'Báo cáo & tải xuống'])
export = {}

with tabs[0]:
    st.markdown('**BUS-BRA test (nội bộ)** — mean ± sd qua seed. Ghi chú: n_seeds = 1 nghĩa là không có sd.')
    cols = ['model', 'n_seeds'] + [f'{m}_{s}' for m in ('auroc', 'auprc', 'sensitivity_malignant', 'specificity', 'balanced_accuracy', 'f1', 'dice', 'iou', 'hd95') for s in ('mean', 'sd')]
    st.dataframe(ms[cols].round(4), hide_index=True, **ui.STRETCH)
    st.markdown('**Từng seed**')
    st.dataframe(per.round(4), hide_index=True, **ui.STRETCH)
    export['internal_results_mean_sd'] = ms
    export['internal_results_per_seed'] = per

with tabs[1]:
    st.markdown('**X0, X6, V3_full** (X6 là kết quả thực nghiệm lưu trữ; không có checkpoint X6 để chạy live).')
    comp = RR.model_comparison(('X0', 'X6', 'V3_full'))
    st.dataframe(comp.round(4), hide_index=True, **ui.STRETCH)
    c1, c2 = st.columns(2)
    for col, (m, t) in zip((c1, c2), (('auroc', 'AUROC — BUS-BRA test, mean ± sd qua seed'), ('dice', 'Dice — BUS-BRA test, mean ± sd qua seed'))):
        fig = V.model_bar_figure(comp, m, t)
        col.pyplot(fig)
        plt.close(fig)
    st.markdown('**So sánh cặp (paired bootstrap) — kết luận tính từ CI**')
    pt = pd.concat([RR.paired_table('V3_full', 'X6'), RR.paired_table('V3_full', 'X0'), RR.paired_table('X6', 'X0')])
    st.dataframe(pt.round(4), hide_index=True, **ui.STRETCH)
    row = RR.paired_table('V3_full', 'X6')
    r_auc = row[row['metric'] == 'auroc']
    if not r_auc.empty:
        r = r_auc.iloc[0]
        st.info(f'AUROC V3_full − X6 = {r["diff"]:+.4f} (CI95 [{r["ci_low"]:+.4f}; {r["ci_high"]:+.4f}]). {r["kết luận (tính từ CI)"]}'
                + (' V3_full CHƯA chứng minh AUROC cao hơn X6.' if r['ci_low'] <= 0 <= r['ci_high'] else ''))
    export['paired_comparisons'] = pt

with tabs[2]:
    st.markdown('**ECE / NLL / Brier trước và sau temperature scaling** (mỗi mô hình một seed theo file).')
    st.dataframe(cal.round(4), hide_index=True, **ui.STRETCH)
    p = config.RESULTS_DIR / 'figures' / 'reliability_primary.png'
    if p.exists():
        st.image(str(p), caption='Reliability diagram (notebook)', **ui.STRETCH)
    export['calibration'] = cal

with tabs[3]:
    st.markdown('**AURC, coverage, selective risk, malignant bị từ chối** (AURC thấp hơn = tốt hơn).')
    st.dataframe(sel.round(4), hide_index=True, **ui.STRETCH)
    a = sel.groupby('model')['aurc'].first()
    st.caption('AURC trong file: ' + ', '.join(f'{m}={v:.4f}' for m, v in a.items()) + '. Được báo cáo nguyên trạng; không diễn giải thành ưu thế của mô hình nào.')
    export['selective_prediction'] = sel

with tabs[4]:
    st.markdown('**Đánh giá ngoài BUSI** (pipeline đã khóa; Normal không bị đổi thành benign).')
    rows = []
    for m, e in ext.items():
        ci = e.get('auroc_ci95')
        rows.append({'model': m, 'n': e['n'], 'AUROC': round(e['auroc'], 4), 'AUROC CI95': f'[{ci[0]:.3f}; {ci[1]:.3f}]' if ci else 'n/a', 'AUPRC': round(e['auprc'], 4),
                     'sensitivity': round(e['sensitivity_malignant'], 4), 'specificity': round(e['specificity'], 4), 'Dice': round(e['segmentation']['dice'], 4),
                     'IoU': round(e['segmentation']['iou'], 4), 'HD95': round(e['segmentation']['hd95'], 2), 'ECE (T khóa)': round(e['calibration_locked_T']['ECE'], 4)})
    ed = pd.DataFrame(rows)
    st.dataframe(ed, hide_index=True, **ui.STRETCH)
    st.caption('Domain shift: đây là dữ liệu từ nguồn khác; độ đặc hiệu ngoài thấp hơn nội bộ cần được thảo luận trung thực.')
    export['external_busi'] = ed

with tabs[5]:
    st.markdown('Ba nhóm được tách riêng, **không gộp thành một “XAI score”**.')
    for title, keys in RR.XAI_GROUPS.items():
        pv = RR.xai_pivot(keys)
        st.markdown(f'**{title}**')
        if pv.empty:
            st.caption('Không có dữ liệu trong xai_evaluation.csv.')
        else:
            st.dataframe(pv.round(4), hide_index=True, **ui.STRETCH)
            export['xai_' + keys[0]] = pv
    st.markdown('**Chẩn đoán collapse của evidence map** (không phải điểm giải thích)')
    cv = RR.xai_pivot([RR.DIAGNOSTIC_GROUP])
    if not cv.empty:
        st.dataframe(cv.round(4), hide_index=True, **ui.STRETCH)
    st.caption('Localisation ≠ faithfulness ≠ consistency. Mỗi nhóm trả lời một câu hỏi khác nhau.')

with tabs[6]:
    st.download_button('⬇️ Tải các bảng đang hiển thị (zip CSV)', RR.tables_zip_bytes(export), file_name='breastvision_research_tables.zip', mime='application/zip')
    md = RR.load_report_md() if (config.RESULTS_DIR / 'research_report.md').exists() else ''
    st.download_button('⬇️ Tải research_report.md (từ artifact)', md.encode('utf-8'), file_name='research_report.md', mime='text/markdown')
    st.download_button('⬇️ Tải báo cáo HTML (bảng + báo cáo từ artifact)', RE.research_html(export, md), file_name='breastvision_research_report.html', mime='text/html')
    with st.expander('research_report.md'):
        st.markdown(md)
ui.disclaimer()
