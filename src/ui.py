"""Thành phần giao diện Streamlit dùng chung (import streamlit TRONG hàm để module này không làm hỏng kiểm thử không-GUI)."""
import os, sys
import numpy as np
from . import app_support as AS

CSS = """
<style>
.block-container{padding-top:1.2rem;max-width:1320px}
h1,h2,h3{letter-spacing:.2px}
.bv-hero{background:linear-gradient(120deg,#0f2a43 0%,#0b6e6e 100%);border-radius:14px;padding:18px 24px;margin-bottom:14px}
.bv-hero h1{margin:0;font-size:1.7rem}.bv-hero p{margin:.2rem 0 0;opacity:.9}
.bv-card{background:#111a2e;border:1px solid #223049;border-radius:12px;padding:14px 16px;margin-bottom:10px}
.bv-warn{background:#3a2a10;border-left:4px solid #f59e0b;border-radius:6px;padding:10px 14px;margin:8px 0}
.bv-off{background:#2a1620;border-left:4px solid #f43f5e;border-radius:6px;padding:10px 14px;margin:8px 0}
.bv-ok{background:#0f2a24;border-left:4px solid #2dd4bf;border-radius:6px;padding:10px 14px;margin:8px 0}
.bv-status-high{color:#fb7185;font-weight:700}.bv-status-review{color:#fbbf24;font-weight:700}.bv-status-none{color:#2dd4bf;font-weight:700}.bv-status-oos{color:#a78bfa;font-weight:700}
</style>
"""
STATUS_CLASS = {'Nguy cơ dự đoán sai cao': 'high', 'Cần xem xét lại': 'review', 'Không phát cảnh báo': 'none', 'Đầu vào ngoài phạm vi đã kiểm định': 'oos'}


def setup(title, subtitle=''):
    import streamlit as st
    root = AS.ROOT
    if root not in sys.path:
        sys.path.insert(0, root)
    st.set_page_config(page_title=f'{title} · BUS-VERIFY', page_icon='🩻', layout='wide')
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="bv-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f'**{AS.VERSION}**')
        st.caption('Công cụ nghiên cứu — không phải thiết bị chẩn đoán y khoa.')
    return st


def box(st, kind, text):
    st.markdown(f'<div class="bv-{kind}">{text}</div>', unsafe_allow_html=True)


def disclaimer(st):
    box(st, 'warn', '⚠️ ' + AS.DISCLAIMER)


def not_run(st, what, how):
    box(st, 'off', f'<b>NOT RUN — {what}.</b> {how}')


def runtime(st):
    """Nạp bundle/predictor một lần cho mỗi tiến trình. Trả (bundle, predictor, messages)."""
    @st.cache_resource(show_spinner='Đang nạp mô hình (CPU)…')
    def _load():
        return AS.load_runtime()
    return _load()


def get_current(st):
    return st.session_state.get('analysis')


def need_analysis(st):
    a = get_current(st)
    if a is None:
        st.info('Chưa có ảnh được phân tích. Hãy sang trang **Ultrasound Analysis**, tải ảnh lên rồi quay lại.')
    return a


def status_html(status):
    return f'<span class="bv-status-{STATUS_CLASS.get(status, "none")}">{status}</span>'


def show_runtime_messages(st, msgs):
    for m in msgs:
        box(st, 'off', m)
