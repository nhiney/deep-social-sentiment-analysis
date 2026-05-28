"""Streamlit demo — Late Fusion Vietnamese Emotion Classifier.

Run from the project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.explainer import TextExplainer
from app.inference import LateFusionPredictor

logger = logging.getLogger("app")
logging.basicConfig(level=logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
CHECKPOINT   = str(_PROJECT_ROOT / "models" / "best_model")
FIGURES      = _PROJECT_ROOT / "reports" / "figures"
REPORTS      = _PROJECT_ROOT / "reports"

EMOTION_META = {
    "joy":      {"vi": "Vui vẻ",   "emoji": "😊", "color": "#FFD600"},
    "sadness":  {"vi": "Buồn bã",  "emoji": "😢", "color": "#42A5F5"},
    "anger":    {"vi": "Tức giận", "emoji": "😠", "color": "#EF5350"},
    "fear":     {"vi": "Sợ hãi",   "emoji": "😨", "color": "#AB47BC"},
    "disgust":  {"vi": "Ghê tởm",  "emoji": "🤢", "color": "#66BB6A"},
    "surprise": {"vi": "Ngạc nhiên","emoji": "😲", "color": "#FF7043"},
    "neutral":  {"vi": "Trung tính","emoji": "😐", "color": "#78909C"},
}

EXAMPLE_TEXTS = {
    "😊 Vui — Sinh nhật bạn thân": "Hôm nay sinh nhật đứa bạn thân nhất, vui quá trời!! Chúc mừng sinh nhật nha 🎂🎉 mãi yêu bạn lắm luôn",
    "😢 Buồn — Nhớ quê hương":    "Xa nhà mấy tháng rồi mà nhớ quá, nhớ mẹ, nhớ em... Hôm nay trời mưa lại càng buồn hơn 😢",
    "😠 Tức giận — Kẹt xe":       "Kẹt xe 2 tiếng đồng hồ mà không nhúc nhích được, điên thật sự!!! Thành phố này giao thông tệ quá đi",
    "😨 Sợ hãi — Thi cử":         "Ngày mai thi mà chưa học được gì hết, lo sợ quá... Không biết có qua môn không nữa 😰",
    "🤢 Chê — Đồ ăn tệ":          "Quán này quảng cáo ngon mà ăn vào thất vọng ghê gớm, vừa đắt vừa dở, không bao giờ quay lại nữa",
    "😲 Ngạc nhiên — Tin tức":     "Ủa thật không?! Không thể tin được là chuyện đó lại xảy ra, shock nặng luôn á 😱 ai mà ngờ được",
    "😐 Trung tính — Thông báo":   "Phòng họp thay đổi sang phòng 3B, bắt đầu lúc 9 giờ sáng mai. Mọi người chú ý nhé.",
}

TEENCODE_EXAMPLES = [
    ("ko / k / hok", "không"),
    ("đc / đk", "được"),
    ("mk / mik", "mình"),
    ("vui lắm lun", "vui lắm luôn"),
    ("iu / iu thik", "yêu / yêu thích"),
    ("ck / vk", "chồng / vợ"),
    ("😊", "[SMILE]"),
    ("🔥", "[FIRE]"),
    ("💯", "[HUNDRED]"),
]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #F8F9FA; }
[data-testid="stSidebar"] { background: #1E2433; }
[data-testid="stSidebar"] * { color: #E0E4EF !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #A0AABF !important; }
h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1565C0 0%, #283593 50%, #1A237E 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(21,101,192,0.25);
}
.hero h1 { font-size: 2.1rem; font-weight: 800; margin: 0 0 .4rem; color: white; }
.hero .subtitle { font-size: 1.05rem; opacity: .88; margin: 0 0 1.2rem; }
.hero .badges span {
    display: inline-block; background: rgba(255,255,255,0.15);
    border-radius: 20px; padding: .2rem .75rem; margin: .15rem .2rem;
    font-size: .82rem; font-weight: 600; letter-spacing: .3px;
    border: 1px solid rgba(255,255,255,0.25);
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 130px;
    background: white; border-radius: 12px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.08);
    border-top: 4px solid;
    text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 800; line-height: 1.1; }
.metric-card .lbl { font-size: .78rem; color: #607D8B; margin-top: .25rem; font-weight: 500; text-transform: uppercase; letter-spacing: .5px; }

/* ── Section headers ── */
.section-header {
    font-size: 1.15rem; font-weight: 700; color: #1565C0;
    border-left: 4px solid #1565C0; padding-left: .7rem;
    margin: 1.6rem 0 .8rem;
}

/* ── Confidence bars ── */
.conf-bar-wrap { margin: .3rem 0; }
.conf-bar-label { display: flex; justify-content: space-between; font-size: .88rem; margin-bottom: .18rem; font-weight: 500; }
.conf-bar-outer { background: #ECEFF1; border-radius: 6px; height: 14px; overflow: hidden; }
.conf-bar-inner { height: 100%; border-radius: 6px; transition: width .4s ease; }

/* ── Prediction result card ── */
.pred-card {
    background: white; border-radius: 14px;
    padding: 1.5rem 1.8rem; margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,.1);
    border: 2px solid;
}
.pred-label { font-size: 2.4rem; font-weight: 900; line-height: 1; }
.pred-vi    { font-size: 1.2rem; color: #546E7A; margin: .2rem 0; }
.pred-conf  { font-size: .9rem; color: #90A4AE; }

/* ── LIME box ── */
.lime-box {
    background: #FAFAFA; border: 1px solid #E0E0E0;
    border-radius: 10px; padding: 1.2rem 1.4rem; margin-top: .8rem;
    font-size: .95rem; line-height: 1.8;
}

/* ── Method step cards ── */
.step-card {
    display: flex; align-items: flex-start; gap: 1rem;
    background: white; border-radius: 10px; padding: 1rem 1.2rem;
    margin: .6rem 0; box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.step-num {
    background: #1565C0; color: white;
    border-radius: 50%; width: 32px; height: 32px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: .95rem; flex-shrink: 0;
}
.step-body .step-title { font-weight: 700; font-size: .98rem; color: #263238; }
.step-body .step-desc  { font-size: .86rem; color: #607D8B; margin-top: .15rem; }

/* ── Info box ── */
.info-box {
    background: #E3F2FD; border-left: 4px solid #1565C0;
    border-radius: 8px; padding: .9rem 1.1rem; margin: .8rem 0;
    font-size: .9rem; color: #1A237E;
}

/* ── Ablation row highlight ── */
.winner-row { background: #E8F5E9 !important; font-weight: 700; }

/* ── Tab styling ── */
[data-testid="stTab"] button { font-weight: 600; font-size: .95rem; }
</style>
"""


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model() -> Optional[LateFusionPredictor]:
    ckpt = Path(CHECKPOINT)
    if not ckpt.exists():
        return None
    try:
        return LateFusionPredictor(
            checkpoint_dir=str(ckpt),
            class_names=CLASS_NAMES,
            device="auto",
            max_length=128,
            apply_normalizer=True,
        )
    except Exception as e:
        logger.error("Failed to load predictor: %s", e)
        return None


@st.cache_resource(show_spinner=False)
def load_explainer(_predictor: LateFusionPredictor) -> Optional[TextExplainer]:
    if _predictor is None:
        return None
    try:
        return TextExplainer(
            class_names=_predictor.class_names,
            predict_proba_fn=_predictor.predict_proba_for_lime,
            num_samples=200,
            num_features=12,
            bow=False,
        )
    except Exception as e:
        logger.error("Failed to load explainer: %s", e)
        return None


# ── Helper renderers ──────────────────────────────────────────────────────────
def _metric_cards(metrics: dict) -> str:
    cards = [
        ("F1-Macro",   f"{metrics['f1_macro']:.4f}",   "#1565C0"),
        ("Accuracy",   f"{metrics['accuracy']:.4f}",    "#2E7D32"),
        ("F1-Weighted",f"{metrics['f1_weighted']:.4f}", "#E65100"),
        ("Test size",  "9 616",                          "#6A1B9A"),
    ]
    html = '<div class="metric-row">'
    for lbl, val, color in cards:
        html += (
            f'<div class="metric-card" style="border-top-color:{color}">'
            f'<div class="val" style="color:{color}">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>'
        )
    html += "</div>"
    return html


def _confidence_bars(probs: np.ndarray) -> str:
    html = ""
    order = np.argsort(probs)[::-1]
    for idx in order:
        cls  = CLASS_NAMES[idx]
        meta = EMOTION_META[cls]
        pct  = probs[idx] * 100
        html += (
            f'<div class="conf-bar-wrap">'
            f'<div class="conf-bar-label">'
            f'<span>{meta["emoji"]} {meta["vi"]}</span>'
            f'<span style="font-weight:700">{pct:.1f}%</span></div>'
            f'<div class="conf-bar-outer"><div class="conf-bar-inner" '
            f'style="width:{pct:.1f}%;background:{meta["color"]}"></div></div>'
            f'</div>'
        )
    return html


def _load_metrics() -> dict:
    p = REPORTS / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _load_ablation() -> pd.DataFrame:
    p = REPORTS / "ablation_results_with_phobert.csv"
    if not p.exists():
        p = REPORTS / "ablation_results.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def _figure(name: str):
    p = FIGURES / name
    if p.exists():
        return str(p)
    return None


# ── Tab 1 : Tổng quan ─────────────────────────────────────────────────────────
def tab_overview(metrics: dict) -> None:
    st.markdown("""
<div class="hero">
  <h1>🧠 Deep Social Sentiment Analysis</h1>
  <div class="subtitle">
    Phân loại cảm xúc tiếng Việt 7 lớp trên mạng xã hội — Đồ án tốt nghiệp
  </div>
  <div class="badges">
    <span>XLM-RoBERTa</span>
    <span>FT-Transformer</span>
    <span>Late Fusion</span>
    <span>LIME Explainability</span>
    <span>7-class Ekman</span>
    <span>Teencode NLP</span>
  </div>
</div>
""", unsafe_allow_html=True)

    if metrics:
        st.markdown(_metric_cards(metrics), unsafe_allow_html=True)

    # Architecture
    st.markdown('<div class="section-header">Kiến trúc mô hình</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.code("""
┌─────────────────────────────────────────────┐
│             INPUT LAYER                     │
│  ┌──────────────┐   ┌─────────────────────┐ │
│  │   Văn bản    │   │  Hành vi người dùng │ │
│  │  (text post) │   │  likes/cmts/shares  │ │
│  └──────┬───────┘   └──────────┬──────────┘ │
│         │                      │            │
│  ┌──────▼───────┐   ┌──────────▼──────────┐ │
│  │  Teencode    │   │  TabularPreprocessor│ │
│  │ Normalizer   │   │  (num + cat encode) │ │
│  └──────┬───────┘   └──────────┬──────────┘ │
│         │                      │            │
│  ┌──────▼───────┐   ┌──────────▼──────────┐ │
│  │XLM-RoBERTa   │   │  FT-Transformer     │ │
│  │[CLS] → 768d  │   │    → 192d           │ │
│  └──────┬───────┘   └──────────┬──────────┘ │
│         │                      │            │
│         └──────────┬───────────┘            │
│                    │ Concat (960d)           │
│           ┌────────▼────────┐               │
│           │   Fusion MLP    │               │
│           │ Linear→ReLU→Drop│               │
│           └────────┬────────┘               │
│                    │                        │
│           ┌────────▼────────┐               │
│           │ Softmax (7 lớp) │               │
│           └─────────────────┘               │
└─────────────────────────────────────────────┘
""", language=None)
    with col2:
        st.markdown("""
**Text Branch** (XLM-RoBERTa)
- Pre-train đa ngôn ngữ 100+
- Teencode normalization trước
- `[CLS]` pooling → **768d**

**Tabular Branch** (FT-Transformer)
- 10 numerical + 4 categorical features
- Embedding → Transformer block → **192d**

**Fusion Head**
- Concat 768 + 192 = **960d**
- Linear(960→256) → ReLU → Dropout
- Linear(256→7) → Softmax

**Explainability**
- LIME: tô màu token đóng góp
- SHAP: feature importance tabular
""")
        st.markdown('<div class="info-box">💡 Teencode normalization đóng góp <strong>+3.1 F1-Macro</strong> so với raw text.</div>', unsafe_allow_html=True)

    # Ablation table
    st.markdown('<div class="section-header">Kết quả Ablation Study</div>', unsafe_allow_html=True)
    ablation_data = {
        "Thí nghiệm": ["Exp1: XLM-R only", "Exp2: + Teencode", "Exp3: Full Fusion", "Exp4: PhoBERT-v2 ⭐"],
        "Backbone":   ["XLM-RoBERTa", "XLM-RoBERTa", "XLM-RoBERTa", "PhoBERT-v2"],
        "Teencode":   ["❌", "✅", "✅", "✅"],
        "Tabular":    ["❌", "❌", "✅", "❌"],
        "F1-Macro":   [0.6235, 0.6548, 0.6454, 0.7186],
        "Accuracy":   [0.6424, 0.6647, 0.6587, 0.7212],
    }
    df_abl = pd.DataFrame(ablation_data)
    st.dataframe(
        df_abl.style.highlight_max(subset=["F1-Macro", "Accuracy"], color="#C8E6C9"),
        use_container_width=True, hide_index=True,
    )

    fig_abl = _figure("ablation_with_phobert.png") or _figure("ablation_results.png")
    if fig_abl:
        st.image(fig_abl, caption="Ablation Study — F1-Macro so sánh 4 thí nghiệm", use_container_width=True)

    # EDA figures row
    st.markdown('<div class="section-header">Phân tích dữ liệu (EDA)</div>', unsafe_allow_html=True)
    eda_figs = [
        ("label_distribution.png",     "Phân phối nhãn cảm xúc"),
        ("text_length_per_emotion.png", "Độ dài văn bản theo cảm xúc"),
        ("correlation_heatmap.png",     "Tương quan đặc trưng hành vi"),
    ]
    cols = st.columns(len(eda_figs))
    for col, (fname, caption) in zip(cols, eda_figs):
        p = _figure(fname)
        if p:
            col.image(p, caption=caption, use_container_width=True)

    eda_figs2 = [
        ("boxplots_interaction_per_emotion.png", "Boxplot tương tác theo cảm xúc"),
        ("mean_interaction_heatmap.png",          "Heatmap tương tác trung bình"),
        ("violin_interaction_per_emotion.png",    "Violin plot phân phối"),
    ]
    cols2 = st.columns(len(eda_figs2))
    for col, (fname, caption) in zip(cols2, eda_figs2):
        p = _figure(fname)
        if p:
            col.image(p, caption=caption, use_container_width=True)


# ── Tab 2 : Demo trực tiếp ────────────────────────────────────────────────────
def tab_demo(predictor: Optional[LateFusionPredictor], explainer: Optional[TextExplainer]) -> None:
    if predictor is None:
        st.warning("⚠️ Chưa tải được mô hình. Kiểm tra checkpoint tại `models/best_model/`.")
        st.info("Cài đặt mô hình: tải checkpoint từ Google Drive và đặt vào `models/best_model/`")
        return

    st.markdown('<div class="section-header">Nhập văn bản để phân tích cảm xúc</div>', unsafe_allow_html=True)

    # Example selector
    example_key = st.selectbox(
        "Chọn câu ví dụ (hoặc nhập văn bản bên dưới):",
        ["— nhập tự do —"] + list(EXAMPLE_TEXTS.keys()),
    )
    default_text = EXAMPLE_TEXTS.get(example_key, "") if example_key != "— nhập tự do —" else ""

    col_input, col_config = st.columns([3, 1])

    with col_input:
        text = st.text_area(
            "Văn bản tiếng Việt:",
            value=default_text,
            height=130,
            placeholder="Nhập bài đăng Facebook, comment, tweet tiếng Việt...",
        )

    with col_config:
        st.markdown("**Tín hiệu hành vi** *(tùy chọn)*")
        likes    = st.number_input("👍 Likes",    min_value=0, max_value=100000, value=0, step=10)
        comments = st.number_input("💬 Comments", min_value=0, max_value=50000,  value=0, step=5)
        shares   = st.number_input("🔁 Shares",   min_value=0, max_value=20000,  value=0, step=5)
        use_lime = st.checkbox("Giải thích LIME", value=True)

    predict_btn = st.button("🔍 Phân tích cảm xúc", type="primary", use_container_width=True)

    if not predict_btn:
        st.markdown('<div class="info-box">👆 Nhập văn bản và nhấn <strong>Phân tích cảm xúc</strong> để bắt đầu.</div>', unsafe_allow_html=True)
        return

    if not text.strip():
        st.error("Vui lòng nhập văn bản trước khi phân tích.")
        return

    tabular_overrides = {}
    if likes > 0:    tabular_overrides["likes"]    = float(likes)
    if comments > 0: tabular_overrides["comments"] = float(comments)
    if shares > 0:   tabular_overrides["shares"]   = float(shares)

    with st.spinner("Đang phân tích..."):
        result = predictor.predict(
            [text],
            tabular_overrides=tabular_overrides if tabular_overrides else None,
        )[0]

    pred_class = result["label"]
    probs      = np.array([result["probabilities"][c] for c in CLASS_NAMES])
    meta       = EMOTION_META[pred_class]
    conf       = probs.max() * 100

    # Result card
    st.markdown(f"""
<div class="pred-card" style="border-color:{meta['color']}">
  <div class="pred-label" style="color:{meta['color']}">{meta['emoji']} {meta['vi']}</div>
  <div class="pred-vi"><code>{pred_class}</code></div>
  <div class="pred-conf">Độ tin cậy: <strong>{conf:.1f}%</strong></div>
</div>
""", unsafe_allow_html=True)

    # Confidence bars
    st.markdown("**Xác suất từng cảm xúc:**")
    st.markdown(
        f'<div style="background:white;padding:1rem 1.2rem;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.07)">'
        f'{_confidence_bars(probs)}</div>',
        unsafe_allow_html=True,
    )

    # LIME explanation
    if use_lime and explainer is not None:
        st.markdown('<div class="section-header">Giải thích LIME — Token nào đóng vai trò quan trọng?</div>', unsafe_allow_html=True)
        with st.spinner("Đang tạo giải thích LIME (~10-15 giây)..."):
            try:
                explanation = explainer.explain(text)
                st.markdown(
                    f'<div class="lime-box">{explanation.highlight_html}</div>',
                    unsafe_allow_html=True,
                )
                # Token table
                tokens = sorted(explanation.tokens, key=lambda x: abs(x[1]), reverse=True)[:10]
                if tokens:
                    df_tok = pd.DataFrame(tokens, columns=["Token", "LIME weight"])
                    df_tok["Hướng"] = df_tok["LIME weight"].apply(
                        lambda w: f"✅ Ủng hộ {pred_class}" if w > 0 else f"❌ Phản đối {pred_class}"
                    )
                    df_tok["LIME weight"] = df_tok["LIME weight"].round(4)
                    st.dataframe(df_tok, use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"LIME không khả dụng: {e}")
    elif use_lime and explainer is None:
        st.info("LIME chưa tải được. Kiểm tra lại cài đặt.")


# ── Tab 3 : Kết quả nghiên cứu ────────────────────────────────────────────────
def tab_results(metrics: dict) -> None:
    st.markdown('<div class="section-header">Hiệu suất trên tập kiểm tra (9 616 mẫu)</div>', unsafe_allow_html=True)

    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F1-Macro",    f"{metrics['f1_macro']:.4f}",    delta="+0.0641 vs baseline")
        c2.metric("Accuracy",    f"{metrics['accuracy']:.4f}")
        c3.metric("F1-Weighted", f"{metrics['f1_weighted']:.4f}")
        c4.metric("Recall-Macro",f"{metrics['recall_macro']:.4f}")

        # Per-class table
        st.markdown('<div class="section-header">Kết quả theo từng cảm xúc</div>', unsafe_allow_html=True)
        pc = metrics.get("per_class", {})
        rows = []
        for cls in CLASS_NAMES:
            if cls in pc:
                m    = pc[cls]
                meta = EMOTION_META[cls]
                rows.append({
                    "Cảm xúc":  f"{meta['emoji']} {meta['vi']}",
                    "F1":       round(m["f1"], 4),
                    "Precision":round(m["precision"], 4),
                    "Recall":   round(m["recall"], 4),
                    "Support":  int(m["support"]),
                })
        df_pc = pd.DataFrame(rows)
        st.dataframe(
            df_pc.style.highlight_max(subset=["F1"], color="#C8E6C9")
                       .highlight_min(subset=["F1"], color="#FFCCBC"),
            use_container_width=True, hide_index=True,
        )

    # Visual results
    st.markdown('<div class="section-header">Biểu đồ phân tích</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        p = _figure("learning_curves.png")
        if p: st.image(p, caption="Learning curves (Train vs Val Loss & F1)", use_container_width=True)
        p = _figure("per_class_metrics.png")
        if p: st.image(p, caption="F1 / Precision / Recall theo từng lớp cảm xúc", use_container_width=True)
    with col2:
        p = _figure("confusion_matrix.png")
        if p: st.image(p, caption="Confusion Matrix — tập kiểm tra 9 616 mẫu", use_container_width=True)
        p = _figure("error_analysis.png")
        if p: st.image(p, caption="Phân tích lỗi — mẫu bị phân loại sai", use_container_width=True)

    # Ablation detail
    st.markdown('<div class="section-header">Chi tiết Ablation Study</div>', unsafe_allow_html=True)
    df_abl = _load_ablation()
    if not df_abl.empty:
        rename_map = {
            "experiment":       "Thí nghiệm",
            "text_model":       "Backbone",
            "use_normalizer":   "Teencode",
            "use_tabular":      "Tabular",
            "best_epoch":       "Best Epoch",
            "train_seconds":    "Train (s)",
            "f1_macro":         "F1-Macro",
            "precision_macro":  "Precision",
            "recall_macro":     "Recall",
            "accuracy":         "Accuracy",
            "f1_weighted":      "F1-Weighted",
        }
        pretty = df_abl.rename(columns=rename_map)
        disp_cols = [c for c in rename_map.values() if c in pretty.columns]
        for col in ["F1-Macro", "Precision", "Recall", "Accuracy", "F1-Weighted"]:
            if col in pretty.columns:
                pretty[col] = pretty[col].round(4)
        st.dataframe(
            pretty[disp_cols].style.highlight_max(subset=["F1-Macro", "Accuracy"], color="#C8E6C9"),
            use_container_width=True, hide_index=True,
        )

    p = _figure("ablation_with_phobert.png") or _figure("ablation_results.png")
    if p:
        st.image(p, caption="So sánh F1-Macro qua 4 thí nghiệm ablation", use_container_width=True)

    st.markdown("""
<div class="info-box">
<strong>Kết luận:</strong> PhoBERT-v2 (pre-train chuyên biệt tiếng Việt) đạt <strong>F1=0.7186</strong>,
cao hơn XLM-R Exp2 <strong>+6.37%</strong>. Teencode normalization đóng góp +3.1 F1-Macro
so với raw text. Full fusion (tabular) tăng accuracy nhưng không cải thiện F1-Macro trong thiết lập này.
</div>
""", unsafe_allow_html=True)


# ── Tab 4 : Phương pháp ───────────────────────────────────────────────────────
def tab_methodology() -> None:
    # Dataset
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    df_data = pd.DataFrame({
        "Nguồn":        ["crawled_emotions.xlsx", "UIT-VSMEC.csv", "pseudo_labeled_apify.csv", "Tổng train"],
        "Mẫu":          [2034, 6927, 655, "~6700"],
        "Mô tả":        [
            "Facebook posts tự crawl — 7 cảm xúc, annotated thủ công",
            "Facebook comments công khai (UIT-NLP benchmark)",
            "990 posts Apify → zero-shot NLI (mDeBERTa) với confidence ≥ 0.35",
            "Sau stratified split 70/15/15",
        ],
    })
    st.dataframe(df_data, use_container_width=True, hide_index=True)

    # Pipeline steps
    st.markdown('<div class="section-header">Pipeline 7 bước</div>', unsafe_allow_html=True)
    steps = [
        ("Thu thập dữ liệu",          "Web crawling Facebook + UIT-VSMEC + pseudo-labeling bằng mDeBERTa zero-shot NLI"),
        ("Tiền xử lý văn bản",        "Teencode normalization: 170+ từ lóng + 70 emoji token → chuẩn hóa tiếng Việt mạng"),
        ("Feature engineering tabular","10 numerical (text_length, n_words, likes,...) + 4 categorical features"),
        ("Training Late Fusion",       "XLM-R + FT-Transformer + MLP fusion, AdamW, AMP, early stopping, F1-Macro monitor"),
        ("Evaluation",                 "F1-Macro, Accuracy, Precision/Recall per-class, Confusion Matrix trên test set 9616 mẫu"),
        ("Explainability",             "LIME: tô màu token đóng góp | SHAP KernelExplainer: feature importance tabular branch"),
        ("Ablation Study",             "4 thí nghiệm: Exp1→Exp2 (+Teencode) →Exp3 (+Tabular) →Exp4 (PhoBERT-v2 backbone)"),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
<div class="step-card">
  <div class="step-num">{i}</div>
  <div class="step-body">
    <div class="step-title">{title}</div>
    <div class="step-desc">{desc}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Emotion taxonomy
    st.markdown('<div class="section-header">Phân loại cảm xúc — Mô hình Ekman 7 lớp</div>', unsafe_allow_html=True)
    cols = st.columns(7)
    for col, cls in zip(cols, CLASS_NAMES):
        meta = EMOTION_META[cls]
        col.markdown(
            f'<div style="text-align:center;background:white;border-radius:10px;'
            f'padding:.8rem .4rem;box-shadow:0 2px 8px rgba(0,0,0,.07);'
            f'border-top:4px solid {meta["color"]}">'
            f'<div style="font-size:2rem">{meta["emoji"]}</div>'
            f'<div style="font-weight:700;font-size:.85rem;color:#263238">{meta["vi"]}</div>'
            f'<div style="font-size:.75rem;color:#90A4AE;font-style:italic">{cls}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Teencode normalizer
    st.markdown('<div class="section-header">Teencode Normalizer — ví dụ</div>', unsafe_allow_html=True)
    df_tc = pd.DataFrame(TEENCODE_EXAMPLES, columns=["Input (mạng xã hội)", "Output (chuẩn hóa)"])
    st.dataframe(df_tc, use_container_width=True, hide_index=True)
    st.markdown("""
<div class="info-box">
📝 TeencodeNormalizer áp dụng <strong>dictionary 170+ cụm từ</strong> và chuyển đổi
<strong>70 emoji</strong> thành token đặc biệt (ví dụ 😊 → <code>[SMILE]</code>).
Điều này giúp XLM-RoBERTa xử lý được văn bản mạng xã hội tiếng Việt hiệu quả hơn.
</div>
""", unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="section-header">Tech Stack</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Deep Learning**")
        st.markdown("- PyTorch 2.x\n- HuggingFace Transformers\n- XLM-RoBERTa-base\n- PhoBERT-v2 (VinAI)")
    with c2:
        st.markdown("**ML / Analysis**")
        st.markdown("- scikit-learn\n- LIME\n- SHAP\n- NumPy / Pandas")
    with c3:
        st.markdown("**App / API**")
        st.markdown("- Streamlit (demo app)\n- FastAPI (REST API)\n- Google Colab T4\n- pytest (185 tests)")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧠 Deep Social\nSentiment Analysis")
        st.divider()
        st.markdown("**Đồ án tốt nghiệp**")
        st.markdown("Phân loại cảm xúc tiếng Việt 7 lớp trên mạng xã hội")
        st.divider()

        # Model status
        ckpt = Path(CHECKPOINT)
        if ckpt.exists():
            st.success("✅ Checkpoint sẵn sàng")
        else:
            st.error("❌ Chưa có checkpoint")
            st.caption(f"Đặt model tại:\n`{CHECKPOINT}`")

        st.divider()
        st.markdown("**Kết quả chính**")
        st.markdown("- F1-Macro: **0.6877**")
        st.markdown("- Accuracy: **0.7020**")
        st.markdown("- PhoBERT F1: **0.7186**")

        st.divider()
        st.markdown("**Links**")
        st.markdown("[📁 GitHub](https://github.com/nhiney/deep-social-sentiment-analysis)")
        st.caption("Model: XLM-RoBERTa-base + FT-Transformer")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Deep Social Sentiment Analysis",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    metrics   = _load_metrics()
    predictor = load_model()
    explainer = load_explainer(predictor) if predictor else None

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Tổng quan",
        "🔍 Demo trực tiếp",
        "📊 Kết quả nghiên cứu",
        "📐 Phương pháp",
    ])

    with tab1:
        tab_overview(metrics)
    with tab2:
        tab_demo(predictor, explainer)
    with tab3:
        tab_results(metrics)
    with tab4:
        tab_methodology()


if __name__ == "__main__":
    main()
