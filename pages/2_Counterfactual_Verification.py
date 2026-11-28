import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from src import ui, app_support as AS, segmentation as SG, preprocessing as PP, pipeline as PL, counterfactual_verifier as CF

st = ui.setup('Counterfactual Evidence Verification', 'Can thiệp có kiểm soát lên vùng do mô hình DỰ ĐOÁN (không dùng ground-truth)')
ui.disclaimer(st)
bundle, pred, msgs = ui.runtime(st)
a = ui.need_analysis(st)
if a is None or pred is None:
    if a is not None:
        ui.not_run(st, 'không có mô hình', 'Cần checkpoint để thực hiện can thiệp.')
    st.stop()
img, ex = a['image'], a['extra']
if ex['mask'] is None or ex['mask_info']['empty']:
    ui.not_run(st, 'mask dự đoán rỗng', 'Không có vùng tổn thương dự đoán nên không dựng được 3 vùng.'); st.stop()
cfg = bundle.cfg
reg = SG.build_regions(ex['mask'], ex['valid'], int(CF.CFConfig().ring_px))
if not reg.ok:
    ui.not_run(st, f'vùng không hợp lệ ({reg.reason})', ''); st.stop()
c1, c2, c3 = st.columns(3)
region_key = c1.selectbox('Vùng can thiệp', list(AS.REGION_KEYS), format_func=AS.REGION_KEYS.get)
op = c2.selectbox('Loại biến đổi', ['blur', 'replace', 'permute'], format_func={'blur': 'Local blurring', 'replace': 'Local intensity replacement', 'permute': 'Intensity permutation'}.get)
lo, hi = AS.RANGES[op]
default = {'blur': 6.0, 'replace': 1.0, 'permute': 1.0}[op]
strength = c3.slider('Mức (blur: sigma px; replace/permute: tỉ lệ pixel)', float(lo), float(hi), float(default))
img2, changed = AS.interactive_intervention(img, reg, region_key, op, strength, seed=int(cfg.get('seed', 0)))
r = AS.logit_prob_change(pred, img, img2, cfg['temperature'], cfg['threshold'])
q = CF.perturbation_qa(img, img2, changed)
ca, cb = st.columns(2)
ca.image(img, caption='Trước', use_container_width=True); cb.image(img2, caption=f'Sau ({changed.mean()*100:.1f}% diện tích ảnh bị đổi)', use_container_width=True)
m = st.columns(4)
m[0].metric('Δ logit', f"{r['dz']:+.3f}"); m[1].metric('Δ xác suất', f"{r['dp']:+.3f}", f"{r['p0']:.3f} → {r['p1']:.3f}")
m[2].metric('Đảo nhãn', 'Có' if r['flip'] else 'Không'); m[3].metric('Diện tích can thiệp (px)', int(changed.sum()))
st.caption('QA của can thiệp (kiểm tra artefact ngoài phân phối): ' + ', '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in q.items()))
st.markdown('**Cách đọc:** một thay đổi logit lớn khi che tổn thương nhưng nhỏ khi che nền *có thể* cho thấy mô hình dựa vào tổn thương; ngược lại gợi ý đi tắt. '
            'Phải so với đối chứng khớp diện tích bên dưới vì bản thân can thiệp có thể tạo artefact.')
st.divider()
if st.button('Chạy kiểm chứng đầy đủ (toàn bộ phép biến đổi + đối chứng)'):
    with st.spinner('Đang chạy kiểm chứng phản thực (CPU, có thể vài chục giây)…'):
        res = CF.verify_image(img, ex['mask'], pred, T=cfg['temperature'], thr=cfg['threshold'], valid=ex['valid'], evidence=ex['evidence'],
                              rng=np.random.default_rng(int(cfg.get('seed', 0))))
    st.session_state['cf_result'] = res
res = st.session_state.get('cf_result')
if res:
    st.write(f"Trạng thái: **{res['status']}**")
    f = pd.DataFrame([{k: v for k, v in res['features'].items()}]).T.rename(columns={0: 'giá trị'})
    st.dataframe(f, use_container_width=True)
    feat = res['features']
    import plotly.graph_objects as go
    names = ['lesion_sens', 'peri_sens', 'bg_sens', 'matched_les_sens', 'random_sens']
    st.plotly_chart(go.Figure(go.Bar(x=['Tổn thương', 'Lân cận', 'Nền', 'Đối chứng khớp diện tích', 'Đối chứng ngẫu nhiên'], y=[feat.get(n, float('nan')) for n in names],
                                     marker_color=['#2dd4bf', '#38bdf8', '#f59e0b', '#94a3b8', '#64748b'])).update_layout(template='plotly_dark', title='Độ nhạy = |Δ logit| trung bình qua các phép biến đổi', height=360), use_container_width=True)
