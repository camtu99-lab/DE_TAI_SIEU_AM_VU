import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src import ui, app_support as AS, robustness as RB, segmentation as SG

st = ui.setup('Robustness Lab', 'Biến đổi có kiểm soát: tách bảo toàn nhãn và can thiệp phản thực')
ui.disclaimer(st)
bundle, pred, msgs = ui.runtime(st)
st.markdown('**Phân loại biến đổi.** *Bảo toàn nhãn* (nhiễu, độ sáng, tương phản, mờ, hình học, che nền) kỳ vọng không đổi nhãn → dùng đo **độ ổn định**. '
            '*Phản thực* (che tổn thương) có thể làm mất thông tin bệnh lý → là phép **thử**, không phải đo ổn định.')
tab1, tab2 = st.tabs(['Ảnh đang phân tích', 'Kết quả tổng hợp (V4)'])
with tab1:
    a = ui.need_analysis(st)
    if a is not None and pred is not None:
        name = st.selectbox('Biến đổi', list(RB.CORRUPTIONS), format_func=lambda k: f"{k} — {'bảo toàn nhãn' if RB.CORRUPTIONS[k]['label_preserving'] else 'PHẢN THỰC'} ({RB.CORRUPTIONS[k]['desc']})")
        lv = st.slider('Mức', 1, len(RB.CORRUPTIONS[name]['levels']), 2)
        cfg = bundle.cfg
        try:
            df = RB.run_robustness_lab([a['image']], pred, [name], (lv,), seed=int(cfg.get('seed', 0)), T=cfg['temperature'], thr=cfg['threshold'], seg_thr=cfg['seg_threshold'],
                                       risk_fn=None)
        except Exception as e:
            st.error(f'Lỗi khi chạy biến đổi: {type(e).__name__}: {e}'); st.stop()
        r = df.iloc[0]
        if r['status'] != 'ok':
            ui.box(st, 'warn', r['status'])
        else:
            out, _ = RB.apply_corruption(a['image'], name, lv, np.random.default_rng(int(cfg.get('seed', 0))), pred_mask=a['extra']['mask'], valid=a['extra']['valid'])
            c1, c2 = st.columns(2); c1.image(a['image'], caption='Gốc'); c2.image(out, caption=f'{name} mức {lv}')
            m = st.columns(4)
            m[0].metric('|Δ P(ác tính)|', f"{r['abs_dp']:.3f}"); m[1].metric('Đảo nhãn', 'Có' if r['flip'] else 'Không')
            m[2].metric('Dice mask vs sạch', f"{r.get('seg_dice', float('nan')):.3f}"); m[3].metric('Evidence cosine', f"{r.get('evidence_cosine', float('nan')):.3f}")
            st.caption('Ổn định risk score: chạy bằng notebook (tốn kém trên CPU) — xem tab kết quả tổng hợp.')
with tab2:
    t = AS.load_table(os.path.join(AS.V4_DIR, AS.V4_FILES['robustness']))
    if t is None:
        ui.not_run(st, 'chưa có kết quả Robustness Lab V4', 'Chạy giai đoạn 7 của notebook Kaggle.')
    else:
        st.dataframe(t, use_container_width=True, hide_index=True)
        import plotly.express as px
        st.plotly_chart(px.line(t, x='level', y='abs_dp', color='corruption', line_dash='label_preserving', template='plotly_dark', title='|Δ P(ác tính)| theo mức biến đổi'), use_container_width=True)
