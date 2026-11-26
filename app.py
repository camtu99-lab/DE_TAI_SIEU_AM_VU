# app.py - BreastVision-XAI V3: giao diện Streamlit (công cụ hỗ trợ NGHIÊN CỨU, không phải thiết bị chẩn đoán)
# Chạy:  streamlit run app.py
#  - Có deploy_config.json + checkpoint (xuất từ mục 27 của notebook) và các module .py cạnh app.py -> dùng mô hình thật.
#  - Không có -> chế độ DEMO (phân vùng bằng xử lý ảnh cổ điển, KHÔNG phải mô hình học sâu) để xem giao diện.
import os, json
import numpy as np
import cv2
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import bus_core_v2 as C2
    HAS_BACKEND = True
except Exception:
    HAS_BACKEND = False

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
NAV = ["🏠  Trang chủ", "📊  Phân tích tổn thương", "🛡  Kiểm chứng bằng chứng (BUS-VERIFY)",
       "📈  Kết quả thực nghiệm", "📖  Thông tin phương pháp", "ⓘ  Hướng dẫn sử dụng"]

st.set_page_config(page_title="BreastVision-XAI V3", page_icon="🩺", layout="wide")

# ============================================================ CSS ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root{--navy:#0a2a6e;--navy2:#061c4d;--blue:#0b5bd3;--ink:#0c2461;--muted:#5a6b8c;--bg:#eef3fb;--card:#ffffff;
      --line:#e1e8f5;--okbg:#e4f6ea;--ok:#1b8a3f;--badbg:#fde4e4;--bad:#d62828;--warnbg:#fff3d6;--warn:#8a5a00;--chip:#e3efff}
html,body,.stApp,[class*="css"]{font-family:'Be Vietnam Pro','Segoe UI',system-ui,sans-serif}
.stApp{background:var(--bg)}
header[data-testid="stHeader"]{display:none}
.block-container{padding:104px 1.6rem 2rem 1.6rem;max-width:100%}
/* ---- thanh tiêu đề cố định ---- */
.bv-header{position:fixed;top:0;left:0;right:0;height:84px;z-index:999990;display:flex;align-items:center;gap:16px;
  padding:0 28px;background:linear-gradient(90deg,#0a2a6e 0%,#0c3a8f 60%,#0a2a6e 100%);color:#fff;box-shadow:0 2px 10px rgba(5,20,60,.35)}
.bv-header h1{margin:0;font-size:2rem;font-weight:800;letter-spacing:.2px;color:#fff;line-height:1.1;padding:0}
.bv-header .ver{background:#1d5fd8;border-radius:8px;padding:2px 12px;font-weight:700;font-size:1rem;margin-left:10px;vertical-align:middle}
.bv-header p{margin:2px 0 0 0;font-size:.95rem;color:#dbe7ff}
.bv-badge{margin-left:auto;background:#fde4e4;color:#c01818;border-radius:12px;padding:12px 22px;font-weight:700;font-size:.95rem;
  display:flex;gap:8px;align-items:center;box-shadow:0 2px 6px rgba(0,0,0,.2)}
/* ---- sidebar ---- */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b2f78 0%,#061c4d 100%);border-right:0}
section[data-testid="stSidebar"]>div:first-child{padding-top:96px}
section[data-testid="stSidebar"] *{color:#e8f0ff}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:6px}
section[data-testid="stSidebar"] div[role="radiogroup"] label{padding:12px 14px;border-radius:12px;cursor:pointer;width:100%;transition:background .15s}
section[data-testid="stSidebar"] div[role="radiogroup"] label>div:first-child{display:none}
section[data-testid="stSidebar"] div[role="radiogroup"] label p{font-size:1.02rem;font-weight:500;margin:0}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:rgba(255,255,255,.08)}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){background:var(--blue);box-shadow:0 2px 8px rgba(0,0,0,.25)}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p{font-weight:700;color:#fff}
.bv-sidefoot{margin-top:28px;padding:12px 14px;border-radius:12px;background:rgba(255,255,255,.07);font-size:.82rem;line-height:1.45;color:#cfe0ff}
/* ---- thẻ ---- */
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"] .bv-card-title),
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border:1px solid var(--line)!important;border-radius:16px;box-shadow:0 2px 10px rgba(20,50,110,.06)}
.bv-page-title{font-size:1.9rem;font-weight:800;color:var(--ink);margin:0 0 2px 0;display:flex;gap:10px;align-items:center}
.bv-page-sub{color:var(--muted);margin:0 0 14px 4px;font-size:1rem}
.bv-card-title{font-size:1.15rem;font-weight:700;color:var(--ink);margin:0 0 10px 0;display:flex;gap:8px;align-items:center}
.bv-up-title{font-weight:700;color:var(--ink);line-height:1.25}
.bv-up-sub{color:var(--muted);font-size:.85rem}
.bv-cap{background:#eef3fb;border-radius:10px;text-align:center;padding:8px 6px;color:var(--ink);font-size:.95rem;margin-top:6px}
.bv-row{display:flex;justify-content:space-between;align-items:center;padding:9px 6px;border-top:1px solid var(--line);color:var(--ink);font-size:1rem}
.bv-row:first-of-type{border-top:0}
.bv-main{display:flex;justify-content:space-between;align-items:center;background:#eef3fb;border-radius:12px;padding:0 0 0 14px;margin-bottom:8px;overflow:hidden;color:var(--ink);font-weight:600}
.bv-main .res{padding:16px 26px;font-weight:800;font-size:1.2rem;min-width:46%;text-align:center}
.res-mal{background:var(--badbg);color:var(--bad)} .res-ben{background:var(--okbg);color:var(--ok)} .res-abs{background:var(--warnbg);color:var(--warn)}
.pill{padding:3px 12px;border-radius:999px;font-size:.82rem;font-weight:600;white-space:nowrap}
.pill-ok{background:var(--okbg);color:var(--ok)} .pill-blue{background:var(--chip);color:#1b4fb0} .pill-warn{background:var(--warnbg);color:var(--warn)}
.bv-warn{background:var(--warnbg);color:#6b4500;border-radius:10px;padding:10px 14px;margin-top:10px;font-size:.92rem;border:1px solid #f1dc9d}
.bv-note{color:var(--muted);font-size:.82rem;margin-top:6px}
.bv-feat{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px;height:100%;box-shadow:0 2px 10px rgba(20,50,110,.05)}
.bv-feat h4{margin:0 0 6px 0;color:var(--ink);font-size:1.05rem} .bv-feat p{margin:0;color:var(--muted);font-size:.93rem;line-height:1.5}
.bv-hero{background:linear-gradient(120deg,#0a2a6e,#0b5bd3);color:#fff;border-radius:18px;padding:30px 34px;margin-bottom:16px}
.bv-hero h2{color:#fff;margin:0 0 8px 0;font-size:1.7rem} .bv-hero p{margin:0;color:#dbe7ff;font-size:1.02rem;line-height:1.55}
/* ---- nút & tải tệp ---- */
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:var(--blue);border:0;border-radius:10px;color:#fff;font-weight:700;padding:.55rem 1.1rem}
.stButton>button[kind="primary"]:hover,button[data-testid="stBaseButton-primary"]:hover{background:#0948ab;color:#fff}
[data-testid="stFileUploaderDropzone"]{padding:.45rem .8rem;border-radius:10px;background:#f6f9ff}
[data-testid="stFileUploaderDropzoneInstructions"] small{display:none}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="bv-header">
  <svg width="54" height="54" viewBox="0 0 64 64" fill="none"><path d="M32 6c-9 0-15 7-15 15 0 7 4 12 9 18l-9 17 10-9c8 9 22 3 22-12 0-5-2-9-5-12 1-5-1-9-6-14-1-2-4-3-6-3z"
  stroke="#bfe0ff" stroke-width="3.2" stroke-linejoin="round"/><circle cx="43" cy="43" r="4.5" fill="#fff"/></svg>
  <div><h1>BreastVision-XAI<span class="ver">V3</span></h1><p>Hệ thống hỗ trợ phân tích ảnh siêu âm vú bằng học sâu có khả năng giải thích</p></div>
  <div class="bv-badge">ⓘ GIAO DIỆN MINH HỌA • CHỈ DÙNG CHO NGHIÊN CỨU</div>
</div>""", unsafe_allow_html=True)

# ============================================================ Tiện ích ảnh ============================================================
def show_img(img, **kw):
    try:
        st.image(img, width="stretch", **kw)
    except TypeError:
        st.image(img, use_column_width=True, **kw)


def square_resize(rgb, size=384):
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = rgb
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_LINEAR)


def overlay_heat(img, cam, alpha=0.55):
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return ((1 - alpha) * img + alpha * heat).astype(np.uint8)


def draw_contour(img, mask, color=(40, 230, 40), th=3):
    out = np.ascontiguousarray(img.copy())
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, th)
    return out


def synthetic_sample(size=384, seed=7):
    """Ảnh TỔNG HỢP mô phỏng siêu âm (khối giảm âm bờ không đều) - không phải dữ liệu bệnh nhân."""
    rng = np.random.default_rng(seed)
    base = 95 + 25 * np.linspace(1, 0, size)[:, None] * np.ones((1, size))
    for y in rng.integers(20, size - 20, 7):                       # lớp mô sáng dạng dải
        base[max(0, y - 2):y + 2] += rng.uniform(25, 55)
    yy, xx = np.mgrid[:size, :size]
    cx, cy = size * 0.5, size * 0.52
    ang = np.arctan2(yy - cy, xx - cx)
    rad = 58 + 9 * np.sin(3 * ang + 1) + 6 * np.sin(5 * ang + 2) + 4 * np.sin(8 * ang)
    base[np.hypot(yy - cy, xx - cx) < rad] = 18
    img = base * rng.rayleigh(0.75, (size, size))                  # nhiễu nhân speckle
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.6)
    return np.stack([np.clip(img, 0, 255).astype(np.uint8)] * 3, -1)


# ============================================================ Backend ============================================================
@st.cache_resource(show_spinner=False)
def load_backend():
    """Trả về (model, cfg, device) nếu đủ file triển khai, ngược lại None."""
    if not HAS_BACKEND:
        return None
    try:
        p = os.path.join(HERE, "deploy_config.json")
        if not os.path.exists(p):
            return None
        cfg = json.load(open(p, encoding="utf-8"))
        ck = os.path.join(HERE, cfg["checkpoint"])
        if not os.path.exists(ck):
            return None
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = C2.load_model_from_checkpoint(ck, dev)
        return model, cfg, dev
    except Exception as e:                                          # noqa
        st.session_state["backend_error"] = str(e)[:300]
        return None


def demo_segment(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0, 0), img.shape[0] / 70)
    t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (g < t).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    best, score = 0, -1
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1 or a < 0.002 * H * W:
            continue
        s = a / (1 + np.hypot(cen[i][0] - W / 2, cen[i][1] - H / 2) / W)
        if s > score:
            best, score = i, s
    return (lab == best).astype(np.uint8) if best else np.zeros_like(m)


def demo_analyze(rgb):
    img = square_resize(rgb, 384)
    mask = demo_segment(img)
    p_mal, conf = 0.5, 0.5
    if mask.any():
        cnt = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
        a, per = cv2.contourArea(cnt), cv2.arcLength(cnt, True)
        sol = a / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
        circ = 4 * np.pi * a / (per ** 2 + 1e-6)
        p_mal = float(1 / (1 + np.exp(-(10 * (0.9 - sol) + 5 * (0.72 - circ)))))
        conf = max(p_mal, 1 - p_mal)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) if mask.any() else np.zeros(mask.shape, np.float32)
    xai = cv2.GaussianBlur(dist, (0, 0), 14) if mask.any() else dist
    xai = xai / (xai.max() + 1e-8)
    return dict(mode="demo", image=img, mask=mask, xai=xai, xai_name="Bản đồ minh họa (không phải XAI thật)", p_mal=p_mal, conf=conf,
                label="malignant" if p_mal >= 0.5 else "benign", abstain=False, exp="DEMO (xử lý ảnh cổ điển)", smoke=False)


def model_analyze(rgb, bk):
    model, cfg, dev = bk
    r = C2.analyze_image_v2(rgb, model, cfg, dev)
    use_ev = r.get("evidence") is not None
    xai = r["evidence"] if use_ev else r["cam"]
    xai = np.clip(xai, 0, None)
    xai = xai / (xai.max() + 1e-12)
    return dict(mode="model", image=r["image"], mask=r["mask"], xai=xai, xai_name="Bản đồ giải thích (XAI)" if use_ev else "Grad-CAM",
                p_mal=float(r["p_malignant"]), conf=float(r["confidence"]), label=r["label"], abstain=bool(r["abstain"]),
                exp=cfg["experiment"], smoke=cfg.get("run_mode") != "full_experiment")


def region_mass(xai, mask):
    """Tỉ lệ khối lượng bằng chứng nằm trong tổn thương / vòng lân cận / nền (đo ĐỊNH VỊ, chưa phải kiểm chứng nhân quả)."""
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return None
    e = np.clip(xai, 0, None).astype(np.float64)
    e /= e.sum() + 1e-12
    k = max(3, int(0.06 * m.shape[0]))
    ring = np.clip(cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(int) - m, 0, 1)
    bg = 1 - np.clip(m + ring, 0, 1)
    return [(float((e * z).sum()), float(z.mean())) for z in (m, ring, bg)]


# ============================================================ Biểu đồ minh họa ============================================================
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, length=0)


def fig_roc():
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    x = np.linspace(0, 1, 100)
    ax.plot(x, 1 - (1 - x) ** 5, color="#0b5bd3", lw=2.2)
    ax.plot([0, 1], [0, 1], "--", color="#9aa6bd", lw=1)
    ax.set_xticks([]); ax.set_yticks([]); _style(ax)
    ax.set_xlabel("FPR", fontsize=7); ax.set_ylabel("TPR", fontsize=7); fig.tight_layout()
    return fig


def fig_cm():
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.imshow(np.array([[0.9, 0.2], [0.15, 0.85]]), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Lành", "Ác"], fontsize=7); ax.set_yticklabels(["Lành", "Ác"], fontsize=7); _style(ax); fig.tight_layout()
    return fig


def fig_bars():
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.barh([2, 1, 0], [0.9, 0.78, 0.7], color=["#7fb2ff", "#0b5bd3", "#9cc4ff"], height=0.55)
    ax.set_yticks([]); ax.set_xticks([]); _style(ax); fig.tight_layout()
    return fig


# ============================================================ Trang ============================================================
def go(i):
    st.session_state["page"] = NAV[i]


def page_home():
    st.markdown("""<div class="bv-hero"><h2>Phân loại – Phân vùng – Giải thích trong một quy trình</h2>
    <p>BreastVision-XAI V3 là mô hình đa nhiệm cho ảnh siêu âm vú: dự đoán lành tính/ác tính, khoanh vùng tổn thương và tạo bản đồ bằng chứng
    nối trực tiếp vào logit để người nghiên cứu kiểm tra vì sao mô hình đưa ra dự đoán. Kết quả không phải kết luận lâm sàng.</p></div>""",
                unsafe_allow_html=True)
    items = [("🧠 Đa nhiệm", "Encoder EfficientNet-B3 chung cho phân loại và phân vùng, kèm SCSE attention ở decoder/neck."),
             ("🔊 Vùng hậu âm BI-RADS", "Vùng phía dưới khối u tính từ mask dự đoán, bổ sung cùng toàn ảnh, tổn thương và vùng quanh tổn thương."),
             ("🔍 Evidence map", "Bản đồ bằng chứng điều chế mọi vùng, phân rã cộng chính xác logit nên đọc được đóng góp từng vị trí."),
             ("⚖️ Faithfulness loss", "Huấn luyện để che vùng bằng chứng cao nhất làm giảm độ tự tin rõ hơn che vùng bằng chứng thấp.")]
    cols = st.columns(4, gap="medium")
    for c, (t, d) in zip(cols, items):
        c.markdown(f'<div class="bv-feat"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.button("▷  Bắt đầu phân tích ảnh", type="primary", on_click=go, args=(1,))


def verdict_html(r):
    if r["abstain"]:
        cls, txt = "res-abs", "Từ chối dự đoán"
    elif r["label"] == "malignant":
        cls, txt = "res-mal", "Ác tính"
    else:
        cls, txt = "res-ben", "Lành tính"
    tag = " (DEMO)" if r["mode"] == "demo" else ""
    return (f'<div class="bv-main"><span>▣&nbsp; Kết quả phân loại</span><span class="res {cls}">{txt}{tag}</span></div>'
            f'<div class="bv-row"><span>◎&nbsp; Phân vùng tổn thương</span><span class="pill pill-ok">✔ Đã thực hiện</span></div>'
            f'<div class="bv-row"><span>◈&nbsp; {r["xai_name"]}</span><span class="pill pill-ok">✔ Đã tạo</span></div>'
            f'<div class="bv-row"><span>P(ác tính)</span><span class="pill pill-blue">{r["p_mal"] * 100:.1f}%</span></div>'
            f'<div class="bv-row"><span>Mô hình</span><span class="pill pill-blue">{r["exp"]}</span></div>')


def verify_html(r):
    vm = region_mass(r["xai"], r["mask"])
    names = ["◎&nbsp; Vùng tổn thương", "⛶&nbsp; Vùng lân cận", "△&nbsp; Vùng nền"]
    if vm is None:
        rows = "".join(f'<div class="bv-row"><span>{n}</span><span class="pill pill-blue">Không có mask</span></div>' for n in names)
    else:
        rows = "".join(f'<div class="bv-row"><span>{n}</span><span class="pill pill-blue">{m * 100:.1f}% bằng chứng · {a * 100:.1f}% diện tích</span></div>'
                       for n, (m, a) in zip(names, vm))
    return rows + '<div class="bv-warn">ⓘ <b>BUS-VERIFY:</b> Đang nghiên cứu – chưa xác thực (đây là đo định vị, chưa chứng minh tính nhân quả)</div>'


def page_analyze():
    r = st.session_state.get("result")
    left, right = st.columns([2.15, 1], gap="medium")
    with left:
        st.markdown('<div class="bv-page-title">🖥 Phân tích ảnh siêu âm vú</div>'
                    '<div class="bv-page-sub">Phân loại tổn thương, phân vùng tổn thương và giải thích quyết định của mô hình.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2, c3 = st.columns([1.05, 1.6, 0.7], vertical_alignment="center")
            c1.markdown('<div class="bv-up-title">Tải ảnh siêu âm vú</div><div class="bv-up-sub">Chọn tệp ảnh (PNG, JPG)</div>', unsafe_allow_html=True)
            up = c2.file_uploader("Tải ảnh", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
            run = c3.button("▷  Phân tích", type="primary", width="stretch")
            use_sample = st.checkbox("Dùng ảnh mẫu tổng hợp (không phải dữ liệu bệnh nhân)", value=False)
        if run:
            rgb = None
            if up is not None:
                bgr = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                rgb = None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb is None:
                    st.error("Không đọc được ảnh.")
            elif use_sample:
                rgb = synthetic_sample()
            else:
                st.warning("Hãy tải một ảnh hoặc chọn ảnh mẫu tổng hợp.")
            if rgb is not None:
                bk = load_backend()
                with st.spinner("Đang phân tích..."):
                    r = model_analyze(rgb, bk) if bk else demo_analyze(rgb)
                st.session_state["result"] = r          # chỉ giữ trong bộ nhớ phiên, không ghi ra đĩa
        with st.container(border=True):
            st.markdown('<div class="bv-card-title">🖼 Kết quả phân tích</div>', unsafe_allow_html=True)
            if r is None:
                st.info("Chưa có kết quả. Tải ảnh (hoặc dùng ảnh mẫu) rồi bấm Phân tích.")
            else:
                cols = st.columns(3, gap="small")
                panels = [("Ảnh đầu vào", r["image"]), ("Phân vùng tổn thương", draw_contour(r["image"], r["mask"])),
                          ("Bản đồ giải thích (XAI)", overlay_heat(r["image"], r["xai"]))]
                for c, (t, im) in zip(cols, panels):
                    with c:
                        show_img(im)
                        st.markdown(f'<div class="bv-cap">{t}</div>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="bv-card-title">📋 Kết quả mô hình</div>', unsafe_allow_html=True)
            if r is None:
                st.caption("Đang chờ ảnh đầu vào.")
            else:
                st.markdown(verdict_html(r), unsafe_allow_html=True)
                if r["mode"] == "demo":
                    st.markdown('<div class="bv-warn">⚠ <b>CHẾ ĐỘ DEMO:</b> chưa nạp mô hình. Nhãn, mask và bản đồ dưới đây chỉ để xem giao diện, '
                                'tạo bằng xử lý ảnh cổ điển, KHÔNG phải kết quả của BreastVision-XAI.</div>', unsafe_allow_html=True)
                if r["smoke"]:
                    st.error("Mô hình từ SMOKE TEST (dữ liệu tổng hợp): không có giá trị khoa học.")
                if r["abstain"]:
                    st.markdown('<div class="bv-warn">Độ tin cậy sau hiệu chuẩn thấp hơn ngưỡng từ chối: cần chuyên gia xem xét.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="bv-card-title">🛡 Kiểm chứng bằng chứng (BUS-VERIFY)</div>', unsafe_allow_html=True)
            if r is None:
                st.caption("Đang chờ ảnh đầu vào.")
            else:
                st.markdown(verify_html(r), unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="bv-card-title">📊 Biểu đồ minh họa kết quả</div>', unsafe_allow_html=True)
        cc = st.columns(3, gap="large")
        for c, t, f in zip(cc, ["Đường cong ROC (minh họa)", "Ma trận nhầm lẫn (minh họa)", "So sánh mô hình (minh họa)"], [fig_roc, fig_cm, fig_bars]):
            with c:
                st.markdown(f'<div style="text-align:center;color:#0c2461;font-weight:500">{t}</div>', unsafe_allow_html=True)
                st.pyplot(f(), clear_figure=True)
        st.markdown('<div class="bv-note">Các biểu đồ trên chỉ là hình minh họa bố cục, không chứa số liệu thật. Số liệu thật xem ở mục “Kết quả thực nghiệm”.</div>',
                    unsafe_allow_html=True)


def page_verify():
    st.markdown('<div class="bv-page-title">🛡 Kiểm chứng bằng chứng (BUS-VERIFY)</div>'
                '<div class="bv-page-sub">Kiểm tra bản đồ giải thích có tập trung vào vùng tổn thương hay không.</div>', unsafe_allow_html=True)
    r = st.session_state.get("result")
    with st.container(border=True):
        if r is None:
            st.info("Hãy phân tích một ảnh ở mục “Phân tích tổn thương” trước.")
        else:
            st.markdown(verify_html(r), unsafe_allow_html=True)
            vm = region_mass(r["xai"], r["mask"])
            if vm:
                fig, ax = plt.subplots(figsize=(5, 2))
                ax.barh(["Nền", "Lân cận", "Tổn thương"], [vm[2][0] * 100, vm[1][0] * 100, vm[0][0] * 100], color=["#9cc4ff", "#5b9bff", "#0b5bd3"])
                ax.set_xlabel("% khối lượng bằng chứng", fontsize=8); _style(ax); fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
    st.markdown("""
**Cách đọc:** *khối lượng bằng chứng* là phần tổng bản đồ giải thích nằm trong từng vùng; nếu một vùng chiếm tỉ lệ bằng chứng lớn hơn nhiều so với tỉ lệ diện tích,
mô hình đang tập trung vào vùng đó. Mask lấy từ chính mô hình (không dùng mask chuẩn khi suy luận).
Đây mới là đo **định vị**. Tính **trung thực** (xoá/chèn vùng, can thiệp bằng chứng) chỉ được đánh giá trên tập dữ liệu trong notebook (mục 20).""")


def page_results():
    st.markdown('<div class="bv-page-title">📈 Kết quả thực nghiệm</div><div class="bv-page-sub">Chỉ hiển thị tệp do notebook xuất ra thư mục results/.</div>',
                unsafe_allow_html=True)
    import pandas as pd
    shown = False
    if os.path.isdir(RES):
        for fn in sorted(os.listdir(RES)):
            p = os.path.join(RES, fn)
            if fn.endswith(".csv"):
                with st.container(border=True):
                    st.markdown(f'<div class="bv-card-title">{fn}</div>', unsafe_allow_html=True)
                    st.dataframe(pd.read_csv(p), width="stretch")
                shown = True
            elif fn.endswith(".png"):
                with st.container(border=True):
                    show_img(p, caption=fn)
                shown = True
    if not shown:
        st.warning("Chưa có tệp trong results/. Chạy mục 27 của notebook và đặt thư mục results/ cạnh app.py.")


def page_method():
    st.markdown('<div class="bv-page-title">📖 Thông tin phương pháp</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
**Kiến trúc (BreastVision-XAI V3):** encoder EfficientNet-B3 chung → decoder phân vùng (SCSE) → gộp đặc trưng theo vùng
(toàn ảnh, tổn thương, quanh tổn thương, hậu âm BI-RADS) bằng *mask dự đoán* → evidence map nối trực tiếp vào logit → hiệu chuẩn nhiệt độ và ngưỡng từ chối chọn trên validation.

**Đóng góp V3:** vùng hậu âm + SCSE + boundary loss; evidence coupling toàn vùng (phân rã cộng chính xác logit); counterfactual faithfulness loss;
label smoothing, speckle augmentation, EMA, 2-view TTA.

**Dữ liệu:** huấn luyện/validation/test chia theo bệnh nhân trên BUS-BRA; đánh giá ngoài trên BUSI (khóa pipeline trước khi đọc BUSI).""")


def page_guide():
    st.markdown('<div class="bv-page-title">ⓘ Hướng dẫn sử dụng</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
1. Vào **Phân tích tổn thương**, tải ảnh PNG/JPG (hoặc tick *ảnh mẫu tổng hợp*) rồi bấm **Phân tích**.
2. Xem nhãn, mặt nạ phân vùng, bản đồ giải thích và mục BUS-VERIFY.
3. Ảnh chỉ xử lý trong bộ nhớ, không lưu; **không tải ảnh có thông tin định danh bệnh nhân**.

**Dùng mô hình thật:** chạy mục 27 của notebook, chép `deploy_config.json`, checkpoint `.pt`, các module `.py` và thư mục `results/` cạnh `app.py`,
sau đó `pip install -r requirements.txt` và `streamlit run app.py`. Nếu thiếu, ứng dụng tự chuyển sang chế độ DEMO.

**Giới hạn:** công cụ hỗ trợ nghiên cứu, chưa kiểm định lâm sàng; từ chối dự đoán không bảo đảm các dự đoán còn lại là đúng.""")


# ============================================================ Điều hướng ============================================================
if "page" not in st.session_state:
    st.session_state["page"] = NAV[1]
with st.sidebar:
    page = st.radio("Điều hướng", NAV, key="page", label_visibility="collapsed")
    bk_state = "Mô hình thật" if load_backend() else "DEMO (chưa nạp mô hình)"
    st.markdown(f'<div class="bv-sidefoot"><b>Chế độ:</b> {bk_state}<br>Công cụ hỗ trợ <b>NGHIÊN CỨU</b>. Không phải thiết bị chẩn đoán y tế. '
                'Không tải ảnh có thông tin định danh.</div>', unsafe_allow_html=True)

{NAV[0]: page_home, NAV[1]: page_analyze, NAV[2]: page_verify, NAV[3]: page_results, NAV[4]: page_method, NAV[5]: page_guide}[page]()
