# app.py - BreastVision-XAI V3: giao diện Streamlit phân tích ảnh siêu âm vú (công cụ hỗ trợ NGHIÊN CỨU)
#
# Hai chế độ:
#   * "model": có deploy_config.json + checkpoint (xuất từ mục 27 của notebook) -> dùng pipeline thật analyze_image_v2.
#   * "demo" : chưa có checkpoint -> chỉ minh họa giao diện bằng xử lý ảnh cổ điển. KHÔNG phải dự đoán của mô hình.
import os, json, hashlib
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(HERE, "results")
SAMPLE_NAME = "bus_sample.png"

st.set_page_config(page_title="BreastVision-XAI V3", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")

# =====================================================================================
# Giao diện (CSS)
# =====================================================================================
def _icon_uri(paths, color="white"):
    svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%s' stroke-width='2' "
           "stroke-linecap='round' stroke-linejoin='round'>%s</svg>") % (color, paths)
    return 'url("data:image/svg+xml;utf8,%s")' % svg.replace("#", "%23")


NAV_ICONS = [
    "<path d='M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/><path d='M9 22V12h6v10'/>",
    "<path d='M12 20V10M18 20V4M6 20v-4'/>",
    "<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/><path d='M9 12l2 2 4-4'/>",
    "<path d='M3 20h18M7 20v-6M12 20V8M17 20v-9'/>",
    "<path d='M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z'/><path d='M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z'/>",
    "<circle cx='12' cy='12' r='10'/><path d='M12 16v-4M12 8h.01'/>",
]
NAV_ITEMS = ["Trang chủ", "Phân tích tổn thương", "Kiểm chứng bằng chứng (BUS-VERIFY)", "Kết quả thực nghiệm",
             "Thông tin phương pháp", "Hướng dẫn sử dụng"]

_nav_css = "\n".join(
    f'[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type({i + 1})::before'
    f'{{background-image:{_icon_uri(p)};}}' for i, p in enumerate(NAV_ICONS))

CSS = """
<style>
:root{--navy:#0b2a6b;--navy2:#0a1f55;--blue:#1060e8;--bg:#f2f6fc;--line:#e1e9f5;--txt:#0f2455;--mut:#5b6b8c;}
html,body,[data-testid="stAppViewContainer"],.stApp{background:var(--bg);font-family:'Inter','Segoe UI',Roboto,Arial,sans-serif;}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer{display:none !important;}
[data-testid="stSidebarCollapseButton"],[data-testid="collapsedControl"],[data-testid="stSidebarHeader"]{display:none !important;}
.block-container{padding:128px 28px 28px 28px !important;max-width:100% !important;}

/* ---------- Header ---------- */
.bv-header{position:fixed;top:0;left:0;right:0;height:106px;z-index:1000;display:flex;align-items:center;
  justify-content:space-between;padding:0 32px;background:linear-gradient(90deg,#0a2262 0%,#103a8c 100%);color:#fff;}
.bv-brand{display:flex;align-items:center;gap:18px;}
.bv-title{font-size:38px;font-weight:800;letter-spacing:.3px;line-height:1.1;display:flex;align-items:center;gap:14px;}
.bv-v{font-size:18px;font-weight:700;background:#1a4fc4;border:1px solid #3d74e6;padding:3px 14px;border-radius:9px;}
.bv-sub{font-size:18px;opacity:.95;margin-top:4px;}
.bv-warn{background:#fde4e4;color:#c4162a;font-weight:700;font-size:17px;padding:14px 22px;border-radius:14px;
  display:flex;align-items:center;gap:10px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.18);}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"]{top:106px;height:calc(100vh - 106px);min-width:285px !important;max-width:285px !important;
  background:linear-gradient(180deg,var(--navy) 0%,var(--navy2) 100%);}
[data-testid="stSidebar"] > div:first-child{background:transparent;}
[data-testid="stSidebarUserContent"]{padding:18px 10px 10px 10px;}
[data-testid="stSidebar"] *{color:#fff;}
[data-testid="stSidebar"] div[role="radiogroup"]{gap:6px;}
[data-testid="stSidebar"] div[role="radiogroup"] > label{padding:15px 16px 15px 52px;border-radius:12px;position:relative;
  width:100%;cursor:pointer;transition:background .15s;}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{background:rgba(255,255,255,.08);}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{display:none;}
[data-testid="stSidebar"] div[role="radiogroup"] > label p{font-size:17px;line-height:1.25;margin:0;}
[data-testid="stSidebar"] div[role="radiogroup"] > label::before{content:"";position:absolute;left:16px;top:50%;
  width:24px;height:24px;margin-top:-12px;background-repeat:no-repeat;background-size:contain;}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){background:var(--blue);}
__NAV_CSS__
.bv-side-note{margin:18px 8px 0 8px;font-size:12.5px;line-height:1.45;color:#b9c8ee !important;}
.bv-side-note b{color:#fff !important;}

/* ---------- Thẻ (card) ---------- */
[data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border:1px solid var(--line) !important;border-radius:18px !important;
  box-shadow:0 2px 10px rgba(16,48,120,.06);}
.bv-h1{display:flex;align-items:center;gap:12px;font-size:30px;font-weight:800;color:var(--txt);margin:4px 0 2px 0;}
.bv-h1 svg{stroke:var(--navy);}
.bv-lead{color:var(--mut);font-size:16px;margin:0 0 6px 38px;}
.bv-ct{display:flex;align-items:center;gap:10px;font-size:21px;font-weight:800;color:var(--txt);margin:2px 0 10px 0;}
.bv-ct svg{stroke:var(--blue);flex:none;}
.bv-cap{background:#eef3fb;border-radius:10px;text-align:center;padding:9px 6px;font-size:15px;color:var(--txt);margin-top:8px;}
.bv-ph{height:220px;border:2px dashed #c9d7ee;border-radius:12px;display:flex;align-items:center;justify-content:center;
  color:var(--mut);font-size:14px;text-align:center;padding:10px;background:#f7faff;}
[data-testid="stImage"] img{border-radius:12px;}

/* ---------- Hàng kết quả / badge ---------- */
.bv-row{display:flex;align-items:center;justify-content:space-between;padding:12px 12px;font-size:17px;color:var(--txt);}
.bv-row .l{display:flex;align-items:center;gap:12px;}
.bv-row svg{stroke:var(--navy);flex:none;}
.bv-top{background:linear-gradient(90deg,#eaf0fb,#f6e4e6);border-radius:12px;padding:10px 12px;}
.bv-top.ok{background:linear-gradient(90deg,#eaf0fb,#e2f5ea);} .bv-top.warn{background:linear-gradient(90deg,#eaf0fb,#fff1d2);}
.bv-top.gray{background:#eef2f9;}
.bv-badge{font-size:14px;font-weight:700;border-radius:9px;padding:5px 12px;white-space:nowrap;}
.bv-badge.ok{background:#e3f6ea;color:#17874a;} .bv-badge.bad{background:#fbe0e0;color:#d12a2a;}
.bv-badge.warn{background:#fff0cf;color:#b36a00;} .bv-badge.info{background:#dfeafc;color:#1f5fd0;}
.bv-badge.gray{background:#e9edf5;color:#58688a;}
.bv-badge.big{font-size:23px;padding:8px 26px;border-radius:11px;}
.bv-sep{border-top:1px solid #edf1f8;}
.bv-banner{background:#fff3d6;border-radius:12px;padding:12px 14px;color:#6b4a00;font-size:15px;margin-top:10px;display:flex;gap:10px;align-items:center;}
.bv-banner b{color:#5a3d00;}

/* ---------- Nút & uploader ---------- */
.stButton > button,.stDownloadButton > button{background:var(--blue);color:#fff;border:0;border-radius:11px;font-weight:700;font-size:17px;
  padding:12px 20px;width:100%;}
.stButton > button:hover{background:#0b4fc4;color:#fff;border:0;}
[data-testid="stFileUploader"] section{background:#fff;border:1px solid var(--line);border-radius:12px;padding:8px 12px;}
[data-testid="stFileUploader"] section > div{padding:0;}
[data-testid="stFileUploaderDropzoneInstructions"] span{font-size:14px;}
</style>
"""
st.markdown(CSS.replace("__NAV_CSS__", "<!-- -->\n" + _nav_css), unsafe_allow_html=True)

# =====================================================================================
# Thành phần HTML dùng lại
# =====================================================================================
ICONS = {
    "monitor": "<rect x='2' y='3' width='20' height='14' rx='2'/><path d='M8 21h8M12 17v4'/>",
    "clip": "<path d='M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2'/><rect x='8' y='2' width='8' height='4' rx='1'/>",
    "shield": "<path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/>",
    "bars": "<path d='M12 20V10M18 20V4M6 20v-4'/>",
    "image": "<rect x='3' y='3' width='18' height='18' rx='2'/><circle cx='8.5' cy='8.5' r='1.5'/><path d='M21 15l-5-5L5 21'/>",
    "target": "<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='4'/>",
    "layers": "<path d='M12 2l10 5-10 5L2 7z'/><path d='M2 12l10 5 10-5M2 17l10 5 10-5'/>",
    "cpu": "<rect x='5' y='5' width='14' height='14' rx='2'/><path d='M9 1v4M15 1v4M9 19v4M15 19v4M1 9h4M1 15h4M19 9h4M19 15h4'/>",
    "dash": "<rect x='4' y='4' width='16' height='16' rx='3' stroke-dasharray='3 3'/>",
    "wave": "<path d='M3 18l5-8 4 5 3-4 6 7z'/>",
    "info": "<circle cx='12' cy='12' r='10'/><path d='M12 16v-4M12 8h.01'/>",
}


def svg(name, size=22, stroke="currentColor"):
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' stroke='{stroke}' stroke-width='2' "
            f"stroke-linecap='round' stroke-linejoin='round'>{ICONS[name]}</svg>")


def card_title(icon, text):
    st.markdown(f"<div class='bv-ct'>{svg(icon)}<span>{text}</span></div>", unsafe_allow_html=True)


def page_title(icon, title, lead):
    st.markdown(f"<div class='bv-h1'>{svg(icon, 30)}<span>{title}</span></div><div class='bv-lead'>{lead}</div>",
                unsafe_allow_html=True)


def row(icon, label, badge, cls, sep=True):
    st.markdown(f"<div class='bv-row {'bv-sep' if sep else ''}'><span class='l'>{svg(icon, 22)}{label}</span>"
                f"<span class='bv-badge {cls}'>{badge}</span></div>", unsafe_allow_html=True)


def caption(text):
    st.markdown(f"<div class='bv-cap'>{text}</div>", unsafe_allow_html=True)


def placeholder(text):
    st.markdown(f"<div class='bv-ph'>{text}</div>", unsafe_allow_html=True)


# Header cố định (giống ảnh mẫu)
st.markdown("""
<div class="bv-header">
  <div class="bv-brand">
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="#7fb2ff" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M44 8c8 0 14 6 14 14 0 10-9 14-14 22l-6 10"/><path d="M44 8c-9 0-16 6-17 15-1 8 3 14 8 18"/>
      <path d="M12 30c0-9 5-16 13-18M12 30c0 8 4 14 10 18l8 6"/><circle cx="46" cy="22" r="5" stroke="#fff"/>
      <path d="M50 26l8 10" stroke="#fff"/>
    </svg>
    <div>
      <div class="bv-title">BreastVision-XAI <span class="bv-v">V3</span></div>
      <div class="bv-sub">Hệ thống hỗ trợ phân tích ảnh siêu âm vú bằng học sâu có khả năng giải thích</div>
    </div>
  </div>
  <div class="bv-warn"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#c4162a" stroke-width="2.2" stroke-linecap="round">
    <circle cx="12" cy="12" r="10"/><path d="M12 7v6M12 17h.01"/></svg>GIAO DIỆN MINH HỌA • CHỈ DÙNG CHO NGHIÊN CỨU</div>
</div>
""", unsafe_allow_html=True)

# =====================================================================================
# Cấu hình & nạp mô hình
# =====================================================================================
def load_cfg():
    p = os.path.join(HERE, "deploy_config.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:
        return None


CFG = load_cfg()
CKPT = os.path.join(HERE, CFG["checkpoint"]) if CFG and CFG.get("checkpoint") else None
MODE = "model" if (CFG and CKPT and os.path.exists(CKPT)) else "demo"


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def get_model(ckpt):
    import torch
    import bus_core_v2 as C2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = C2.load_model_from_checkpoint(ckpt, device)
    return model, device


# =====================================================================================
# Xử lý ảnh
# =====================================================================================
def make_sample_image(h=384, w=384, seed=7):
    """Ảnh siêu âm TỔNG HỢP (khối giảm âm tối + nhiễu speckle) chỉ để thử giao diện, không phải ảnh bệnh nhân."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 85 + 40 * np.exp(-((yy - h * 0.15) / (h * 0.5)) ** 2)
    for y0 in (0.12, 0.3, 0.52, 0.8):                                   # các lớp mạc/mô sáng nằm ngang
        base += 55 * np.exp(-((yy - h * y0 - 14 * np.sin(xx / 55)) / 5.0) ** 2) * (0.6 + 0.4 * np.sin(xx / 70 + y0 * 9))
    cx, cy = w * 0.47, h * 0.5
    ang = np.arctan2(yy - cy, xx - cx)
    r = np.hypot((xx - cx) / 1.3, yy - cy)
    rad = 50 + 8 * np.sin(3 * ang + 1) + 6 * np.sin(5 * ang + 2) + 4 * np.sin(8 * ang)
    soft = 1 / (1 + np.exp((r - rad) / 3.0))
    base = base * (1 - soft) + 22 * soft
    post = np.exp(-((xx - cx) / 70) ** 2) * (yy > cy + 60) * np.exp(-(yy - cy - 60) / 120.0)   # tăng âm phía sau
    base += 28 * post
    sp = cv2.GaussianBlur(rng.rayleigh(1.0, (h, w)).astype(np.float32), (0, 0), 1.1)
    img = np.clip(base * (0.55 + 0.45 * sp), 0, 255).astype(np.uint8)
    return np.stack([img] * 3, -1)


def letterbox(rgb, size):
    h, w = rgb.shape[:2]
    s = max(h, w)
    scale = size / s
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), np.uint8)
    valid = np.zeros((size, size), bool)
    t, l = (size - nh) // 2, (size - nw) // 2
    canvas[t:t + nh, l:l + nw] = r
    valid[t:t + nh, l:l + nw] = True
    return canvas, valid


def to_rgb_uint8(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = (img.astype(np.float32) / max(float(img.max()), 1.0) * 255).clip(0, 255).astype(np.uint8)
    return img


def demo_pipeline(rgb, size=384):
    """CHẾ ĐỘ DEMO: tách vùng tối lớn nhất gần tâm bằng ngưỡng Otsu. Không dùng mô hình học sâu."""
    img, valid = letterbox(rgb, size)
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), size / 70)
    inner = cv2.erode(valid.astype(np.uint8), np.ones((size // 25 * 2 + 1,) * 2, np.uint8)).astype(bool)
    mask = np.zeros((size, size), np.uint8)
    if inner.sum() > 100:
        thr = cv2.threshold(g[inner].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        dark = ((g < thr * 0.8) & inner).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 40 * 2 + 1,) * 2)
        dark = cv2.morphologyEx(cv2.morphologyEx(dark, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, lab, st_, cen = cv2.connectedComponentsWithStats(dark)
        best, bs = 0, 0.0
        for i in range(1, n):
            a = st_[i, cv2.CC_STAT_AREA]
            if a < 0.004 * size * size:
                continue
            d = np.hypot(cen[i][0] - size / 2, cen[i][1] - size / 2) / size
            s = a * np.exp(-(d / 0.3) ** 2)
            if s > bs:
                best, bs = i, s
        if best:
            mask = (lab == best).astype(np.uint8)
            mask = (cv2.GaussianBlur(mask.astype(np.float32), (0, 0), size / 90) > 0.5).astype(np.uint8)
    heat = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), size / 12)
    heat = heat / (heat.max() + 1e-8) if mask.any() else np.zeros_like(heat)
    return dict(image=img, mask=mask, xai=heat, xai_kind="demo", boxes=[], mode="demo",
                label_text="Demo", cls="gray", sub="Chưa nạp mô hình - kết quả chỉ để minh họa giao diện",
                p_mal=None, abstain=False)


def model_pipeline(rgb):
    import bus_core as core
    import bus_core_v2 as C2
    model, device = get_model(CKPT)
    r = C2.analyze_image_v2(rgb, model, CFG, device)
    if r["evidence"] is not None:
        xai, kind = r["evidence"] / (r["evidence"].max() + 1e-12), "evidence"
    else:
        xai, kind = r["cam"], "gradcam"
    if r["abstain"]:
        label_text, cls = "Từ chối dự đoán", "warn"
    elif r["label"] == "malignant":
        label_text, cls = "Ác tính", "bad"
    else:
        label_text, cls = "Lành tính", "ok"
    sub = (f"P(ác tính) đã hiệu chuẩn = {r['p_malignant'] * 100:.1f}% | ngưỡng đã khóa = {float(CFG['threshold']):.2f} | "
           f"độ tin cậy = {r['confidence'] * 100:.1f}%")
    return dict(image=r["image"], mask=r["mask"], xai=xai, xai_kind=kind, boxes=r["boxes"], mode="model",
                label_text=label_text, cls=cls, sub=sub, p_mal=r["p_malignant"], abstain=r["abstain"])


def draw_contour(img, mask, color=(40, 230, 70)):
    out = np.ascontiguousarray(img.copy())
    if mask is not None and mask.any():
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, color, max(2, out.shape[0] // 140), cv2.LINE_AA)
    return out


def heat_overlay(img, m, alpha=0.62):
    heat = cv2.applyColorMap((np.clip(m, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return ((1 - alpha) * img + alpha * heat).astype(np.uint8)


def region_shares(xai, mask):
    """Phân bổ khối lượng bản đồ giải thích theo 3 vùng: tổn thương / lân cận / nền (theo mask dự đoán)."""
    if mask is None or not mask.any() or xai is None:
        return None
    m = mask.astype(np.uint8)
    k = max(5, int(0.25 * np.sqrt(m.sum())))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    ring = (cv2.dilate(m, ker) > 0) & (m == 0)
    bg = (m == 0) & ~ring
    x = np.clip(np.asarray(xai, np.float64), 0, None)
    tot = x.sum() + 1e-12
    out = {}
    for name, reg in (("Vùng tổn thương", m > 0), ("Vùng lân cận", ring), ("Vùng nền", bg)):
        out[name] = (float(x[reg].sum() / tot), float(reg.mean()))
    return out


# =====================================================================================
# Biểu đồ minh họa (đúng như ảnh mẫu: gắn nhãn "minh họa")
# =====================================================================================
BLUE, LBLUE, GRID = "#1a5fd6", "#7db1fa", "#d9e2f2"


def _fig(w=3.7, h=2.7):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    return fig, ax


def chart_roc():
    fig, ax = _fig()
    x = np.linspace(0, 1, 200)
    ax.plot(x, 1 - (1 - x) ** 5.5, color=BLUE, lw=2.2)
    ax.plot([0, 1], [0, 1], "--", color="#8a94a8", lw=1.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout(pad=0.4)
    return fig


def chart_cm():
    fig, ax = _fig(3.4, 2.7)
    im = ax.imshow(np.array([[0.95, 0.22], [0.2, 0.9]]), cmap="Blues", vmin=0, vmax=1.15)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.06)
    cb.set_ticks([]); cb.outline.set_visible(False)
    fig.tight_layout(pad=0.4)
    return fig


def chart_models():
    fig, ax = _fig()
    ax.barh([2, 1, 0], [0.97, 0.85, 0.74], color=[LBLUE, BLUE, LBLUE], height=0.62)
    ax.set_xlim(0, 1.02)
    ax.set_yticks([])
    fig.tight_layout(pad=0.4)
    return fig


def show_charts():
    with st.container(border=True):
        card_title("bars", "Biểu đồ minh họa kết quả")
        c1, c2, c3 = st.columns(3)
        for col, title, fn in ((c1, "Đường cong ROC (minh họa)", chart_roc), (c2, "Ma trận nhầm lẫn (minh họa)", chart_cm),
                               (c3, "So sánh mô hình (minh họa)", chart_models)):
            with col:
                st.markdown(f"<div style='text-align:center;color:#0f2455;font-size:16px;margin-bottom:4px'>{title}</div>",
                            unsafe_allow_html=True)
                fig = fn()
                st.pyplot(fig, clear_figure=True, width="stretch")
                plt.close(fig)
        st.caption("Các biểu đồ trên chỉ để minh họa bố cục, KHÔNG phải số liệu thực nghiệm. Số liệu thật nằm ở mục 'Kết quả thực nghiệm'.")


# =====================================================================================
# Sidebar điều hướng
# =====================================================================================
page = st.sidebar.radio("Điều hướng", NAV_ITEMS, index=1, label_visibility="collapsed", key="nav")
st.sidebar.markdown(
    "<div class='bv-side-note'><b>Công cụ hỗ trợ NGHIÊN CỨU.</b> Không phải thiết bị chẩn đoán y tế. "
    "Không tải ảnh có thông tin định danh bệnh nhân. Ảnh chỉ xử lý trong bộ nhớ, không lưu.<br><br>"
    + (f"Chế độ: <b>mô hình {CFG['experiment']}</b> (T={float(CFG['temperature']):.2f})" if MODE == "model"
       else "Chế độ: <b>DEMO</b> (chưa có checkpoint)") + "</div>", unsafe_allow_html=True)

# =====================================================================================
# TRANG: Phân tích tổn thương
# =====================================================================================
def page_analyze():
    if MODE == "model" and CFG.get("run_mode") != "full_experiment":
        st.error("MÔ HÌNH NÀY TỪ CHẾ ĐỘ SMOKE TEST (dữ liệu tổng hợp). Kết quả KHÔNG có ý nghĩa khoa học hay y khoa.")
    if MODE == "model" and not CFG.get("pretrained_loaded", False):
        st.warning("Backbone không dùng trọng số pretrained ImageNet trong lần huấn luyện này.")

    left, right = st.columns([1.95, 1], gap="medium")
    ss = st.session_state

    with left:
        page_title("monitor", "Phân tích ảnh siêu âm vú",
                   "Phân loại tổn thương, phân vùng tổn thương và giải thích quyết định của mô hình.")
        with st.container(border=True):
            c_up, c_btn = st.columns([3.2, 1], vertical_alignment="center")
            with c_up:
                up = st.file_uploader("Tải ảnh siêu âm vú (PNG, JPG)", type=["png", "jpg", "jpeg", "bmp"],
                                      label_visibility="collapsed")
                use_sample = st.checkbox("Dùng ảnh mẫu tổng hợp (bus_sample.png)", value=(up is None and "res" not in ss))
            with c_btn:
                go = st.button("▷  Phân tích", width="stretch")

        # xác định ảnh nguồn
        rgb, name = None, None
        if up is not None:
            data = up.getvalue()
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if bgr is None:
                st.error("Không đọc được ảnh.")
            else:
                if bgr.ndim == 3 and bgr.shape[2] >= 3:
                    bgr = bgr[:, :, :3][:, :, ::-1]
                rgb, name = to_rgb_uint8(bgr), up.name
        elif use_sample:
            rgb, name = make_sample_image(), SAMPLE_NAME
        key = None if rgb is None else hashlib.md5(rgb.tobytes()).hexdigest()
        if ss.get("res_key") != key:
            ss.pop("res", None)

        if go and rgb is not None:
            try:
                with st.spinner("Đang phân tích..."):
                    ss["res"] = model_pipeline(rgb) if MODE == "model" else demo_pipeline(rgb)
                    ss["res_key"], ss["res_name"] = key, name
            except Exception as e:                                           # noqa: BLE001
                st.error(f"Lỗi khi phân tích: {e}")
        elif go:
            st.warning("Hãy tải ảnh hoặc chọn ảnh mẫu trước.")

        res = ss.get("res")
        with st.container(border=True):
            card_title("image", "Kết quả phân tích")
            c1, c2, c3 = st.columns(3)
            if res is not None:
                with c1:
                    st.image(res["image"], width="stretch"); caption("Ảnh đầu vào")
                with c2:
                    st.image(draw_contour(res["image"], res["mask"]), width="stretch"); caption("Phân vùng tổn thương")
                with c3:
                    st.image(heat_overlay(res["image"], res["xai"]), width="stretch")
                    caption("Bản đồ giải thích (XAI)")
                if not res["mask"].any():
                    st.info("Mô hình không phát hiện vùng tổn thương nào đạt ngưỡng phân vùng.")
                if res["mode"] == "demo":
                    st.warning("CHẾ ĐỘ DEMO: vùng viền xanh được tách bằng ngưỡng Otsu và bản đồ nhiệt chỉ là làm mờ mask, "
                               "KHÔNG phải kết quả của mô hình BreastVision-XAI. Đặt deploy_config.json và checkpoint cạnh app.py để dùng mô hình thật.")
                else:
                    kind = "evidence map (nối trực tiếp vào logit)" if res["xai_kind"] == "evidence" else "Grad-CAM"
                    st.caption(f"Bản đồ XAI hiển thị: {kind}. Đây là giải thích về cách mô hình tính điểm, chưa phải bằng chứng y khoa.")
            else:
                with c1:
                    if rgb is not None:
                        st.image(letterbox(rgb, 384)[0], width="stretch")
                    else:
                        placeholder("Chưa có ảnh")
                    caption("Ảnh đầu vào")
                with c2:
                    placeholder("Nhấn <b>Phân tích</b> để phân vùng"); caption("Phân vùng tổn thương")
                with c3:
                    placeholder("Nhấn <b>Phân tích</b> để tạo bản đồ giải thích"); caption("Bản đồ giải thích (XAI)")

    with right:
        res = ss.get("res")
        with st.container(border=True):
            card_title("clip", "Kết quả mô hình")
            if res is None:
                st.markdown(f"<div class='bv-row bv-top gray'><span class='l'>{svg('cpu')}Kết quả phân loại</span>"
                            f"<span class='bv-badge big gray'>—</span></div>", unsafe_allow_html=True)
                row("target", "Phân vùng tổn thương", "Chưa thực hiện", "gray", sep=False)
                row("layers", "Giải thích mô hình (XAI)", "Chưa tạo", "gray", sep=False)
            else:
                top_cls = {"bad": "", "ok": "ok", "warn": "warn", "gray": "gray"}[res["cls"]]
                st.markdown(f"<div class='bv-row bv-top {top_cls}'><span class='l'>{svg('cpu')}Kết quả phân loại</span>"
                            f"<span class='bv-badge big {res['cls']}'>{res['label_text']}</span></div>", unsafe_allow_html=True)
                row("target", "Phân vùng tổn thương", "✔ Đã thực hiện" if res["mask"].any() else "Không phát hiện",
                    "ok" if res["mask"].any() else "warn", sep=False)
                row("layers", "Giải thích mô hình (XAI)", "✔ Đã tạo", "ok", sep=False)
                st.caption(res["sub"])
                if res["abstain"]:
                    st.warning("Độ tin cậy sau hiệu chuẩn thấp hơn ngưỡng đã chọn trên validation. Cần chuyên gia xem xét.")

        with st.container(border=True):
            card_title("shield", "Kiểm chứng bằng chứng (BUS-VERIFY)")
            shares = region_shares(res["xai"], res["mask"]) if res is not None else None
            for i, (nm, ic) in enumerate((("Vùng tổn thương", "target"), ("Vùng lân cận", "dash"), ("Vùng nền", "wave"))):
                if shares is None:
                    row(ic, nm, "Đang phân tích" if res is None else "Không có", "info" if res is None else "gray", sep=(i > 0))
                else:
                    sh, ar = shares[nm]
                    row(ic, nm, f"{sh * 100:.0f}% bản đồ", "info", sep=(i > 0))
            st.markdown("<div class='bv-banner'><span>ⓘ</span><span><b>BUS-VERIFY:</b> Đang nghiên cứu – chưa xác thực</span></div>",
                        unsafe_allow_html=True)

    show_charts()


# =====================================================================================
# CÁC TRANG KHÁC
# =====================================================================================
def page_home():
    page_title("monitor", "Trang chủ", "BreastVision-XAI V3 - phân tích ảnh siêu âm vú đa nhiệm có giải thích.")
    with st.container(border=True):
        card_title("info", "Giới thiệu")
        st.markdown("""
Ứng dụng nhận một ảnh siêu âm vú và trả về: **phân loại** benign/malignant (xác suất đã hiệu chuẩn nhiệt độ, có cơ chế từ chối dự đoán),
**phân vùng tổn thương** và **bản đồ giải thích** (evidence map / Grad-CAM).

**Kiến trúc:** encoder EfficientNet-B3 dùng chung → decoder phân vùng (SCSE) → gộp đặc trưng các vùng (toàn ảnh, tổn thương, quanh tổn thương,
hậu âm BI-RADS) bằng *mask dự đoán* → evidence map nối trực tiếp vào logit → hiệu chuẩn nhiệt độ và ngưỡng từ chối chọn trên validation.

Chọn **Phân tích tổn thương** ở thanh bên để bắt đầu.
""")
    if MODE == "demo":
        st.warning("Đang chạy chế độ DEMO vì chưa tìm thấy deploy_config.json và checkpoint. Xem mục 'Hướng dẫn sử dụng'.")


def page_verify():
    page_title("shield", "Kiểm chứng bằng chứng (BUS-VERIFY)", "Phân bổ bản đồ giải thích theo vùng giải phẫu do mô hình phân vùng.")
    res = st.session_state.get("res")
    with st.container(border=True):
        card_title("shield", "Phân bổ khối lượng bản đồ giải thích")
        shares = region_shares(res["xai"], res["mask"]) if res is not None else None
        if shares is None:
            st.info("Chưa có kết quả: hãy phân tích một ảnh (có phát hiện tổn thương) ở trang 'Phân tích tổn thương'.")
        else:
            df = pd.DataFrame([{"Vùng": k, "Tỉ lệ khối lượng bản đồ (%)": round(v[0] * 100, 1),
                                "Tỉ lệ diện tích (%)": round(v[1] * 100, 1),
                                "Độ tập trung (×)": round(v[0] / max(v[1], 1e-9), 2)} for k, v in shares.items()])
            st.dataframe(df, hide_index=True, width="stretch")
            if res["mode"] == "demo":
                st.warning("Dữ liệu trên lấy từ chế độ DEMO, không phản ánh mô hình.")
        st.markdown("""
**Cách tính:** chia ảnh thành *tổn thương* (mask dự đoán), *vùng lân cận* (mask giãn nở, bán kính ≈ 0,25·√diện tích mask) và *nền*;
tính phần khối lượng của bản đồ giải thích rơi vào mỗi vùng. "Độ tập trung" = tỉ lệ khối lượng / tỉ lệ diện tích.

**Lưu ý:** đây mới là thống kê mô tả, **chưa phải kiểm chứng nhân quả hay đã được xác thực**. Phần BUS-VERIFY đang trong giai đoạn nghiên cứu.
""")


def page_results():
    page_title("bars", "Kết quả thực nghiệm", "Chỉ hiển thị các tệp do notebook tạo ra, không có số liệu nhập tay.")
    if MODE == "demo":
        st.warning("Chưa có kết quả mô hình thật (chế độ DEMO).")
    shown = False
    if os.path.isdir(RES_DIR):
        for fn in sorted(os.listdir(RES_DIR)):
            p = os.path.join(RES_DIR, fn)
            if fn.endswith(".csv"):
                with st.container(border=True):
                    card_title("bars", fn)
                    st.dataframe(pd.read_csv(p), width="stretch")
                shown = True
            elif fn.endswith(".png"):
                with st.container(border=True):
                    st.image(p, caption=fn, width="stretch")
                shown = True
    if not shown:
        st.info("Thư mục results/ chưa có tệp .csv hoặc .png. Hãy chép thư mục results/ từ deploy_xai/ của notebook vào cạnh app.py.")


def page_method():
    page_title("info", "Thông tin phương pháp", "Tóm tắt pipeline suy luận.")
    with st.container(border=True):
        st.markdown("""
- **Tiền xử lý:** letterbox thành hình vuông, resize về `img_size`, chuẩn hóa ImageNet.
- **Mô hình:** BreastVisionXAI (EfficientNet-B3 + SCSE, vùng hậu âm BI-RADS, evidence-weighted pooling) - mô hình không nhận mask khi suy luận.
- **Hiệu chuẩn:** softmax với nhiệt độ T chọn trên validation; nhãn theo ngưỡng đã khóa; từ chối dự đoán khi độ tin cậy dưới ngưỡng.
- **Giải thích:** evidence map (nối trực tiếp vào logit) và Grad-CAM; bản đồ hiển thị ưu tiên evidence map.
- **Phân vùng:** mask dự đoán (ngưỡng `seg_threshold`), viền xanh trong giao diện là đường bao của mask.
""")


def page_help():
    page_title("info", "Hướng dẫn sử dụng", "Cách chạy ứng dụng và giới hạn.")
    with st.container(border=True):
        st.markdown("""
**Chạy:** `pip install -r requirements.txt` rồi `streamlit run app.py`.

**Dùng mô hình thật:** chạy mục 27 của notebook `BreastVision_XAI_V3.ipynb` để tạo `deploy_xai/`, rồi chép `deploy_config.json`,
file checkpoint (`.pt`) và thư mục `results/` vào cùng thư mục với `app.py`. Khi đó app tự chuyển từ chế độ DEMO sang chế độ mô hình.

**Giới hạn:**
- Công cụ nghiên cứu, không phải thiết bị chẩn đoán; chưa kiểm định lâm sàng.
- Huấn luyện trên bộ dữ liệu công khai; mask hiển thị do mô hình dự đoán.
- Từ chối dự đoán không đảm bảo các dự đoán còn lại là đúng.
- Không tải ảnh có thông tin định danh bệnh nhân.
""")


{"Trang chủ": page_home, "Phân tích tổn thương": page_analyze,
 "Kiểm chứng bằng chứng (BUS-VERIFY)": page_verify, "Kết quả thực nghiệm": page_results,
 "Thông tin phương pháp": page_method, "Hướng dẫn sử dụng": page_help}[page]()
