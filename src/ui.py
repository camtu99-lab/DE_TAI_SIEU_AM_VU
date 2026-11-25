"""Thành phần giao diện dùng chung (CSS, header, sidebar, health check). Không chứa logic suy luận."""
from __future__ import annotations

import streamlit as st

from src import config
from src.security import UserFacingError

_V = tuple(int(x) for x in st.__version__.split('.')[:2] if x.isdigit())
STRETCH = {'width': 'stretch'} if _V >= (1, 50) else {'use_container_width': True}

CSS = """
<style>
.bv-badge{display:inline-block;padding:3px 10px;border-radius:999px;background:#fff4e6;color:#a05a00;border:1px solid #ffd8a8;font-weight:600;font-size:.78rem;letter-spacing:.02em}
.bv-title{font-size:2rem;font-weight:700;margin:0;color:#212529}
.bv-sub{color:#495057;margin:2px 0 8px 0}
.bv-card{border-radius:10px;padding:14px 18px;margin:6px 0 12px 0;border:1px solid #dee2e6;background:#f8f9fa}
.bv-info{border-left:6px solid #0b7285;background:#e3fafc}
.bv-warn{border-left:6px solid #e67700;background:#fff4e6}
.bv-err{border-left:6px solid #c92a2a;background:#fff5f5}
.bv-card h3{margin:0 0 4px 0;font-size:1.25rem}.bv-card p{margin:0;color:#343a40}
.bv-kpi{border:1px solid #dee2e6;border-radius:10px;padding:14px 16px;background:#fff;height:100%}
.bv-kpi .v{font-size:1.7rem;font-weight:700;color:#0b7285}.bv-kpi .l{font-weight:600}.bv-kpi .s{font-size:.78rem;color:#6c757d}
.bv-flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0}
.bv-box{padding:8px 12px;border:1px solid #ced4da;border-radius:8px;background:#fff;font-size:.9rem}
.bv-arrow{color:#868e96;font-weight:700}
</style>
"""
PRESENTATION_CSS = "<style>html{font-size:118%} section[data-testid='stSidebar']{min-width:220px;max-width:240px}</style>"


def setup_page(title: str, icon: str = '🩻') -> None:
    st.set_page_config(page_title=f'{title} · {config.APP_NAME}', page_icon=icon, layout='wide')
    st.markdown(CSS, unsafe_allow_html=True)
    if st.session_state.get('presentation', False):
        st.markdown(PRESENTATION_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner='Đang kiểm tra artifact và nạp mô hình (CPU)...')
def get_runtime():
    from src.checkpoint import build_runtime
    return build_runtime()


@st.cache_data(show_spinner=False)
def cached_table(name: str):
    from src import research_results as RR
    return RR.load_table(name)


def header() -> None:
    st.markdown(f'<p class="bv-title">{config.APP_NAME}</p><p class="bv-sub">“{config.APP_SUBTITLE}”</p>'
                f'<span class="bv-badge">{config.BADGE}</span>', unsafe_allow_html=True)
    st.write('')


def card(kind: str, title: str, body: str = '') -> None:
    st.markdown(f'<div class="bv-card bv-{kind}"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)


def disclaimer() -> None:
    st.caption('Công cụ minh họa nghiên cứu, không phải thiết bị chẩn đoán y khoa. Đây là dự đoán của mô hình, cần chuyên gia đánh giá.')


def show_error(e: UserFacingError) -> None:
    card('err', 'Không thể hoàn tất thao tác', e.message_vi)
    with st.expander('Chi tiết kỹ thuật'):
        st.code(f'component: {e.component}\n{e.technical or "(không có)"}')


def sidebar(rt) -> None:
    sb = st.sidebar
    sb.markdown(f'**{config.APP_NAME}**')
    sb.markdown(f'<span class="bv-badge">{config.BADGE}</span>', unsafe_allow_html=True)
    sb.checkbox('Presentation Mode', key='presentation', help='Chữ lớn hơn và hiển thị KPI/bảng so sánh trên trang chủ.')
    sb.divider()
    status = {r.name: r for r in rt.rows}
    def mark(n):
        s = status[n].status if n in status else 'FAIL'
        return {'OK': '✅', 'WARN': '⚠️', 'DISABLED': '⏸️'}.get(s, '❌')
    cfg = rt.cfg or {}
    sb.markdown(f"**Model:** {cfg.get('experiment', 'n/a')}  \n**Seed:** {cfg.get('seed', 'n/a')}  \n**Input:** {cfg.get('img_size', 'n/a')}×{cfg.get('img_size', 'n/a')}  \n"
                f"**Thiết bị:** {rt.device_name.upper()}  \n**Checkpoint:** {mark('Checkpoint')}  \n**Hash SHA-256:** {mark('Checkpoint SHA256')}  \n"
                f"**Calibration:** {mark('Calibration parameters')}  \n**TTA (lật ngang):** {'bật' if cfg.get('tta') else 'tắt'}")
    sb.caption('🔒 Ảnh không được lưu mặc định bởi ứng dụng.')
    sb.caption(f'v{config.APP_VERSION}')


def require_ready(rt) -> None:
    """Khóa trang nếu checkpoint/config không toàn vẹn (fail-closed)."""
    if rt.ok:
        return
    card('err', 'Hệ thống chưa sẵn sàng — chức năng phân tích bị khóa',
         rt.error.message_vi if rt.error else 'Một hoặc nhiều kiểm tra toàn vẹn artifact thất bại.')
    show_health(rt)
    if rt.error:
        with st.expander('Chi tiết kỹ thuật'):
            st.code(f'component: {rt.error.component}\n{rt.error.technical or "(không có)"}')
    st.stop()


def show_health(rt) -> None:
    import pandas as pd
    df = pd.DataFrame([dict(Kiểm_tra=r.name, Trạng_thái=r.status, Chi_tiết=r.detail) for r in rt.rows])
    st.dataframe(df, hide_index=True, **STRETCH)
