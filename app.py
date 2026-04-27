"""
Visual Product Search Engine — Streamlit Demo (visual-only build)

Production-valid pipeline that uses ONLY visual signals at inference time:

    Category filter  +  CLIP B/32 image embedding  +  48D color histogram
    fused with α=0.4 (40% CLIP visual, 60% color)
    R@1 = 0.683 · R@5 = 0.862 · R@10 = 0.913 on 1,027 DeepFashion queries

No text descriptions are used — those are query-side metadata that's rarely
available in real e-commerce visual-search deployments.

Four tabs:
    Browse — pick from the 1,027-query test set (visual retrieval)
    Color  — filter by hand-picked color palette
    Upload — drop in your own product photo
    Research — leaderboard + ablation + phase timeline
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lookmatch — visual product search",
    page_icon="🪡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Theme ─────────────────────────────────────────────────────────────────────
INDIGO = "#4F46E5"
INDIGO_LIGHT = "#EEF2FF"
EMERALD = "#10B981"
EMERALD_LIGHT = "#ECFDF5"
AMBER = "#F59E0B"
CORAL = "#F43F5E"
SLATE_900 = "#0F172A"
SLATE_700 = "#334155"
SLATE_500 = "#64748B"
SLATE_300 = "#CBD5E1"
SLATE_100 = "#F1F5F9"
SLATE_50 = "#F8FAFC"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox, .stTextArea {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}

.stApp {{ background: linear-gradient(180deg, {SLATE_50} 0%, #FFFFFF 280px); }}

#MainMenu, footer, [data-testid="stDeployButton"] {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.hero {{
    background: linear-gradient(135deg, {INDIGO} 0%, #7C3AED 60%, #DB2777 100%);
    color: white;
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 18px;
    box-shadow: 0 12px 32px rgba(79, 70, 229, 0.22);
    position: relative;
    overflow: hidden;
}}
.hero::after {{
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 95% 0%, rgba(255,255,255,0.18), transparent 45%);
    pointer-events: none;
}}
.hero h1 {{ font-size: 2.0em; font-weight: 800; margin: 0 0 4px 0; letter-spacing: -0.02em; }}
.hero p {{ font-size: 1.0em; margin: 0; color: rgba(255,255,255,0.9); max-width: 760px; }}
.hero-pill {{
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.28);
    color: rgba(255,255,255,0.95);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.7em;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.hero-stats {{ display: flex; gap: 28px; margin-top: 16px; flex-wrap: wrap; }}
.hero-stat {{
    background: rgba(255,255,255,0.16);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 8px 14px;
    border-radius: 10px;
    min-width: 90px;
}}
.hero-stat .v {{ font-size: 1.4em; font-weight: 800; color: white; font-variant-numeric: tabular-nums; line-height: 1.1; }}
.hero-stat .l {{ font-size: 0.72em; color: rgba(255,255,255,0.85); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: {SLATE_100}; padding: 4px; border-radius: 12px;
    border: 1px solid {SLATE_300};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px; padding: 8px 16px; font-weight: 600; color: {SLATE_700};
    background: transparent; transition: all 0.15s ease;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: white; color: {INDIGO}; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 24px; }}

section[data-testid="stSidebar"] {{ background: white; border-right: 1px solid {SLATE_100}; }}
.sidebar-section {{ margin-bottom: 18px; }}
.sidebar-section h4 {{
    font-size: 0.72em; font-weight: 700; color: {SLATE_500};
    text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 10px 0;
}}
.kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.kpi {{
    background: {SLATE_50}; border: 1px solid {SLATE_100};
    border-radius: 10px; padding: 10px 12px;
}}
.kpi .v {{ font-size: 1.25em; font-weight: 800; color: {SLATE_900}; font-variant-numeric: tabular-nums; line-height: 1.0; }}
.kpi .l {{ font-size: 0.7em; color: {SLATE_500}; margin-top: 2px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
.kpi.accent .v {{ color: {INDIGO}; }}
.kpi.success .v {{ color: {EMERALD}; }}

.pipeline-step {{
    display: flex; align-items: flex-start; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid {SLATE_100};
}}
.pipeline-step:last-child {{ border-bottom: none; }}
.pipeline-step .num {{
    background: {INDIGO_LIGHT}; color: {INDIGO};
    border-radius: 6px; width: 22px; height: 22px;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.78em; flex-shrink: 0;
}}
.pipeline-step .body {{ flex: 1; }}
.pipeline-step .title {{ font-weight: 600; font-size: 0.88em; color: {SLATE_900}; }}
.pipeline-step .sub {{ font-size: 0.76em; color: {SLATE_500}; margin-top: 2px; line-height: 1.4; }}

.callout {{
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border-left: 3px solid {AMBER};
    border-radius: 10px; padding: 12px 14px; margin: 8px 0 0 0;
}}
.callout .head {{ font-weight: 700; font-size: 0.82em; color: #92400E; margin-bottom: 4px; }}
.callout .body {{ font-size: 0.78em; color: #78350F; line-height: 1.45; }}

.cat-row {{ display: flex; align-items: center; justify-content: space-between; margin: 4px 0; font-size: 0.82em; }}
.cat-row .name {{ flex: 0 0 80px; color: {SLATE_700}; font-weight: 500; }}
.cat-row .bar-wrap {{ flex: 1; background: {SLATE_100}; border-radius: 999px; height: 6px; overflow: hidden; margin: 0 8px; }}
.cat-row .bar {{ height: 100%; border-radius: 999px; }}
.cat-row .v {{ flex: 0 0 36px; text-align: right; color: {SLATE_700}; font-variant-numeric: tabular-nums; font-weight: 600; font-size: 0.85em; }}

.result-card {{
    border: 1px solid {SLATE_100};
    border-radius: 14px;
    overflow: hidden;
    background: white;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    margin-bottom: 12px;
    position: relative;
}}
.result-card:hover {{ transform: translateY(-2px); box-shadow: 0 14px 28px rgba(15, 23, 42, 0.10); }}
.result-card.is-correct {{ border-color: {EMERALD}; box-shadow: 0 0 0 3px {EMERALD_LIGHT}; }}
.result-card .rank {{
    position: absolute; top: 8px; left: 8px;
    background: rgba(15, 23, 42, 0.78); color: white; font-weight: 700;
    font-size: 0.74em; padding: 3px 8px; border-radius: 6px; z-index: 2; letter-spacing: 0.04em;
}}
.result-card.is-correct .rank {{ background: {EMERALD}; }}
.result-card .match-badge {{
    position: absolute; top: 8px; right: 8px;
    background: {EMERALD}; color: white; font-weight: 700;
    font-size: 0.7em; padding: 3px 8px; border-radius: 6px; z-index: 2;
    text-transform: uppercase; letter-spacing: 0.06em;
}}
.result-card .score-pill {{
    position: absolute; bottom: 8px; right: 8px;
    background: rgba(255, 255, 255, 0.96); color: {SLATE_900};
    font-weight: 700; font-size: 0.76em; padding: 4px 9px; border-radius: 999px;
    z-index: 2; box-shadow: 0 1px 4px rgba(0,0,0,0.10); font-variant-numeric: tabular-nums;
}}
.result-meta {{ padding: 10px 12px; font-size: 0.82em; }}
.result-meta .top {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }}
.result-meta .color-chip {{ display: inline-flex; align-items: center; gap: 6px; color: {SLATE_700}; font-weight: 500; }}
.result-meta .swatch {{ width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(0,0,0,0.05); display: inline-block; }}
.result-meta .cat {{ color: {SLATE_500}; font-weight: 600; text-transform: uppercase; font-size: 0.7em; letter-spacing: 0.06em; }}

.score-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }}
.score-comp {{ background: {SLATE_50}; border-radius: 6px; padding: 4px 6px; font-size: 0.7em; }}
.score-comp .l {{ color: {SLATE_500}; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }}
.score-comp .v {{ color: {SLATE_900}; font-weight: 700; font-variant-numeric: tabular-nums; }}
.score-comp .micro-bar {{ height: 3px; background: {SLATE_100}; border-radius: 2px; margin-top: 3px; overflow: hidden; }}
.score-comp .micro-fill {{ height: 100%; border-radius: 2px; }}

.query-panel {{
    background: white; border: 1px solid {SLATE_100};
    border-radius: 16px; padding: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    position: sticky; top: 12px;
}}
.query-panel h3 {{
    font-size: 0.78em; font-weight: 700; color: {SLATE_500};
    text-transform: uppercase; letter-spacing: 0.1em; margin: 0 0 8px 0;
}}
.query-panel .meta {{ font-size: 0.85em; color: {SLATE_700}; line-height: 1.6; margin-top: 8px; }}
.query-panel .meta b {{ color: {SLATE_900}; }}
.tag {{
    display: inline-block; background: {INDIGO_LIGHT}; color: {INDIGO};
    padding: 2px 8px; border-radius: 999px; font-size: 0.75em;
    font-weight: 600; margin-right: 4px;
}}
.tag.success {{ background: {EMERALD_LIGHT}; color: {EMERALD}; }}
.tag.warning {{ background: #FEF3C7; color: #92400E; }}

.color-palette {{ display: flex; gap: 4px; margin-top: 10px; }}
.color-palette .swatch {{ flex: 1; height: 18px; border-radius: 4px; border: 1px solid rgba(0,0,0,0.05); }}

.hit-banner {{
    border-radius: 12px; padding: 10px 14px;
    font-weight: 600; font-size: 0.92em; margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
}}
.hit-banner.hit {{ background: {EMERALD_LIGHT}; color: #065F46; border: 1px solid #A7F3D0; }}
.hit-banner.miss {{ background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }}
.hit-banner.info {{ background: {INDIGO_LIGHT}; color: #3730A3; border: 1px solid #C7D2FE; }}
.hit-banner .icon {{
    width: 26px; height: 26px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 0.85em;
}}
.hit-banner.hit .icon {{ background: {EMERALD}; color: white; }}
.hit-banner.miss .icon {{ background: {AMBER}; color: white; }}
.hit-banner.info .icon {{ background: {INDIGO}; color: white; }}

.section-h {{ display: flex; align-items: baseline; justify-content: space-between; margin: 4px 0 12px 0; }}
.section-h h2 {{ font-size: 1.05em; font-weight: 700; color: {SLATE_900}; margin: 0; }}
.section-h .sub {{ font-size: 0.85em; color: {SLATE_500}; }}

div[data-testid="stButton"] > button {{
    background: {INDIGO}; color: white; border: none; border-radius: 10px;
    padding: 8px 18px; font-weight: 600; transition: all 0.15s ease;
}}
div[data-testid="stButton"] > button:hover {{
    background: #4338CA; transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
}}
div[data-testid="stButton"] > button:active {{ transform: translateY(0); }}

div[data-baseweb="select"] > div {{ border-radius: 10px !important; }}
.stTextInput input, .stTextArea textarea {{
    border-radius: 10px !important; border: 1px solid {SLATE_300} !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {INDIGO} !important; box-shadow: 0 0 0 3px {INDIGO_LIGHT} !important;
}}

[data-testid="stFileUploader"] section {{
    border-radius: 14px; border: 2px dashed {SLATE_300};
    background: {SLATE_50}; padding: 28px 16px;
    transition: all 0.15s ease;
}}
[data-testid="stFileUploader"] section:hover {{ border-color: {INDIGO}; background: {INDIGO_LIGHT}; }}

div[data-baseweb="color-picker"] {{ width: 100%; }}

.swatch-tile {{
    width: 100%; height: 60px; border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 8px;
}}

.timeline {{ position: relative; padding-left: 28px; }}
.timeline::before {{
    content: ""; position: absolute;
    left: 8px; top: 4px; bottom: 4px; width: 2px;
    background: linear-gradient(180deg, {INDIGO} 0%, {EMERALD} 100%);
    border-radius: 2px;
}}
.timeline-item {{
    position: relative; margin-bottom: 16px; padding: 10px 14px;
    background: white; border: 1px solid {SLATE_100}; border-radius: 10px;
}}
.timeline-item::before {{
    content: ""; position: absolute; left: -24px; top: 14px;
    width: 12px; height: 12px; background: white;
    border: 2px solid {INDIGO}; border-radius: 50%;
}}
.timeline-item.champion::before {{ border-color: {EMERALD}; background: {EMERALD}; }}
.timeline-item .phase {{
    font-size: 0.7em; color: {SLATE_500}; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
}}
.timeline-item .title {{ font-weight: 700; color: {SLATE_900}; font-size: 0.95em; margin: 2px 0 4px 0; }}
.timeline-item .body {{ font-size: 0.85em; color: {SLATE_700}; line-height: 1.5; }}
.timeline-item .badge {{
    display: inline-block; background: {INDIGO_LIGHT}; color: {INDIGO};
    padding: 2px 8px; border-radius: 999px; font-size: 0.74em;
    font-weight: 700; margin-right: 6px; font-variant-numeric: tabular-nums;
}}
.timeline-item.champion .badge {{ background: {EMERALD_LIGHT}; color: {EMERALD}; }}

.footer-text {{
    text-align: center; color: {SLATE_500}; font-size: 0.78em;
    padding: 30px 0 10px 0; border-top: 1px solid {SLATE_100}; margin-top: 32px;
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading search engine…")
def load_engine():
    from src.search_engine import ProductSearchEngine

    engine = ProductSearchEngine()
    engine.load_gallery()
    return engine


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    proc = PROJECT_ROOT / "data" / "processed"
    gallery_df = pd.read_csv(proc / "gallery.csv")
    query_df = pd.read_csv(proc / "query.csv")
    return gallery_df, query_df


@st.cache_data(show_spinner="Loading visual embeddings…")
def load_query_embeddings():
    """Visual-only: CLIP image encoder + 48D color histogram."""
    cache = PROJECT_ROOT / "data" / "processed" / "emb_cache"

    def normed(name):
        emb = np.load(cache / f"{name}.npy").astype(np.float32)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)

    return {
        "visual": normed("clip_b32_query"),  # CLIP B/32 image encoder
        "color": normed("color48_query"),    # 48D RGB histogram
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_image(item_id: str) -> Image.Image | None:
    img_path = PROJECT_ROOT / "data" / "raw" / "images" / f"{item_id}.jpg"
    if img_path.exists():
        return Image.open(img_path).convert("RGB")
    return None


def dominant_palette(img: Image.Image, n_colors: int = 5) -> list[str]:
    small = img.convert("RGB").resize((48, 48), Image.LANCZOS)
    arr = np.array(small).reshape(-1, 3)
    bins = (arr // 64).astype(np.int32)
    keys = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
    unique, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)
    out = []
    for k in unique[order][:n_colors]:
        mask = keys == k
        if not mask.any():
            continue
        rgb = arr[mask].mean(axis=0).astype(int)
        out.append("#{:02X}{:02X}{:02X}".format(*rgb))
    while len(out) < n_colors and out:
        out.append(out[-1])
    return out


def color_name_to_hex(name: str) -> str:
    table = {
        "black": "#1A1A1A", "white": "#F5F5F5", "grey": "#9CA3AF", "gray": "#9CA3AF",
        "red": "#DC2626", "blue": "#2563EB", "navy": "#1E3A8A", "green": "#16A34A",
        "yellow": "#EAB308", "orange": "#F97316", "pink": "#EC4899", "purple": "#9333EA",
        "brown": "#92400E", "beige": "#D4B996", "tan": "#C8A165", "khaki": "#A19164",
        "olive": "#6B7C32", "burgundy": "#7C1E2D", "cream": "#F5E6D3", "ivory": "#F8F1E4",
        "silver": "#C0C0C0", "gold": "#D4A017", "maroon": "#800000",
    }
    if not isinstance(name, str):
        return "#9CA3AF"
    key = name.strip().lower().split()[0] if name.strip() else ""
    return table.get(key, "#9CA3AF")


def hex_to_color_histogram(hexes: list[str], bins_per_channel: int = 8) -> np.ndarray:
    """Synthesize a 48D RGB histogram that emphasizes the picked hex colors.

    Mimics what extract_color_palette would produce for an image dominated by
    these hues. Lets the user 'paint' a query without an image.
    """
    feat = np.zeros((bins_per_channel, 3), dtype=np.float32)
    for hx in hexes:
        h = hx.lstrip("#")
        if len(h) != 6:
            continue
        try:
            r = int(h[0:2], 16) / 255.0
            g = int(h[2:4], 16) / 255.0
            b = int(h[4:6], 16) / 255.0
        except ValueError:
            continue
        for ch_idx, val in enumerate((r, g, b)):
            bin_idx = min(int(val * bins_per_channel), bins_per_channel - 1)
            feat[bin_idx, ch_idx] += 1.0
    flat = feat.T.reshape(-1)  # (3 * bins,) — channels first to match feature_engineering output order
    s = flat.sum()
    if s > 0:
        flat = flat / s
    return flat.astype(np.float32)


def describe_label(item_id: str) -> str:
    parts = item_id.split("_")
    pid = next((p for p in parts if p.startswith("id")), item_id[:14])
    return pid[:14]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-size:1.4em; font-weight:800; color:{SLATE_900}; '
        'letter-spacing:-0.02em; margin: -8px 0 4px 0;">🪡 Lookmatch</div>'
        f'<div style="font-size:0.78em; color:{SLATE_500}; margin-bottom:18px;">'
        'visual product search · v1.2</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section"><h4>Performance</h4>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi success"><div class="v">0.683</div><div class="l">Recall@1</div></div>
            <div class="kpi success"><div class="v">0.862</div><div class="l">Recall@5</div></div>
            <div class="kpi accent"><div class="v">0.913</div><div class="l">Recall@10</div></div>
            <div class="kpi"><div class="v">300</div><div class="l">Gallery</div></div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section"><h4>Pipeline</h4>', unsafe_allow_html=True)
    steps = [
        ("1", "Category filter", "hard constraint — same category only (+10.3pp R@1 on this system)"),
        ("2", "CLIP B/32 image embed", "512D semantic visual descriptor (40% weight)"),
        ("3", "48D color histogram", "RGB color distribution (60% weight)"),
    ]
    for num, title, sub in steps:
        st.markdown(
            f"""
            <div class="pipeline-step">
                <span class="num">{num}</span>
                <div class="body">
                    <div class="title">{title}</div>
                    <div class="sub">{sub}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sidebar-section">
            <div class="callout">
                <div class="head">⚡ Pure-visual signal</div>
                <div class="body">No text descriptions enter the pipeline at any stage —
                the system runs on a raw photo with no query-side metadata. CLIP's image
                encoder carries the most signal (removing it costs −17pp R@1); color
                histograms add another +11pp on top.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section"><h4>Per-category R@1</h4>', unsafe_allow_html=True)
    # Per-category numbers — measured live via scripts/verify_ui_numbers.py
    # against the CLIP B/32 + cat + color α=0.4 production system on the
    # full 1,027-query test set.
    cat_perf = [
        ("suiting", 1.000), ("sweaters", 0.905), ("shirts", 0.868),
        ("jackets", 0.734), ("tees", 0.684), ("sweatshirts", 0.669),
        ("denim", 0.649), ("pants", 0.632), ("shorts", 0.475),
    ]
    for name, r1 in cat_perf:
        pct = int(r1 * 100)
        color = EMERALD if r1 >= 0.75 else AMBER if r1 >= 0.60 else CORAL
        st.markdown(
            f"""
            <div class="cat-row">
                <span class="name">{name}</span>
                <span class="bar-wrap"><span class="bar" style="width:{pct}%;background:{color};"></span></span>
                <span class="v">{r1:.3f}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="margin-top:18px; padding-top:14px; border-top: 1px solid {SLATE_100};
                    font-size:0.74em; color:{SLATE_500}; line-height:1.55;">
            <b>Dataset</b> DeepFashion In-Shop · 300 products · 1,027 queries · 9 categories<br>
            <b>Backbone</b> CLIP ViT-B/32 (image encoder) · 48D RGB histogram · α = 0.4<br>
            <b>Built by</b> Mark Rodrigues × Anthony Rodrigues
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <span class="hero-pill">visual-only · production-valid</span>
        <h1>Find any fashion product from a photo. No description needed.</h1>
        <p>CLIP image embeddings fused with a 48D color histogram, filtered by category.
        Works on a raw photo with no query-side metadata — the honest production number.</p>
        <div class="hero-stats">
            <div class="hero-stat"><div class="v">68.3%</div><div class="l">recall @ 1</div></div>
            <div class="hero-stat"><div class="v">86.2%</div><div class="l">recall @ 5</div></div>
            <div class="hero-stat"><div class="v">91.3%</div><div class="l">recall @ 10</div></div>
            <div class="hero-stat"><div class="v">9</div><div class="l">categories</div></div>
            <div class="hero-stat"><div class="v">5</div><div class="l">research phases</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_browse, tab_color, tab_upload, tab_research = st.tabs(
    ["🔍 Browse", "🎨 Color filter", "📷 Upload", "📊 Research"]
)

engine = load_engine()
gallery_df, query_df = load_data()
query_embs = load_query_embeddings()


# ── Result card renderer ──────────────────────────────────────────────────────
def render_results_grid(response, correct_pid: str | None, n_cols: int = 4,
                        score_labels: tuple[str, str] = ("Visual", "Color")):
    """Render a grid of result cards.

    The two per-component score labels match what the engine put into the
    SearchResult — visual_score lives in `text_score` for backwards compat
    when running visual-only retrieval (see search_engine._search_visual).
    """
    if not response.results:
        st.info("No results.")
        return
    rows = [response.results[i : i + n_cols] for i in range(0, len(response.results), n_cols)]
    label_a, label_b = score_labels

    for row in rows:
        cols = st.columns(n_cols)
        for col, r in zip(cols, row):
            with col:
                is_correct = correct_pid is not None and r.product_id == correct_pid
                card_cls = "result-card is-correct" if is_correct else "result-card"
                img = get_image(r.item_id)
                st.markdown(f'<div class="{card_cls}">', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="rank">#{r.rank}</span>'
                    + (f'<span class="match-badge">match</span>' if is_correct else "")
                    + f'<span class="score-pill">{r.combined_score:.3f}</span>',
                    unsafe_allow_html=True,
                )
                if img:
                    st.image(img, use_column_width=True)

                color_hex = color_name_to_hex(r.color)
                a_w = max(0.0, min(1.0, r.visual_score))
                b_w = max(0.0, min(1.0, r.color_score))
                st.markdown(
                    f"""
                    <div class="result-meta">
                        <div class="top">
                            <span class="color-chip">
                                <span class="swatch" style="background:{color_hex};"></span>
                                {(r.color or '—').title()}
                            </span>
                            <span class="cat">{r.category}</span>
                        </div>
                        <div class="score-grid">
                            <div class="score-comp">
                                <div class="l">{label_a}</div>
                                <div class="v">{r.visual_score:.3f}</div>
                                <div class="micro-bar"><div class="micro-fill" style="width:{int(a_w*100)}%;background:{INDIGO};"></div></div>
                            </div>
                            <div class="score-comp">
                                <div class="l">{label_b}</div>
                                <div class="v">{r.color_score:.3f}</div>
                                <div class="micro-bar"><div class="micro-fill" style="width:{int(b_w*100)}%;background:{AMBER};"></div></div>
                            </div>
                        </div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ── TAB 1: Browse ─────────────────────────────────────────────────────────────
with tab_browse:
    st.markdown(
        '<div class="section-h"><h2>Browse the held-out test set</h2>'
        '<span class="sub">1,027 queries · visual retrieval (CLIP image + color, no text)</span></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([2, 4, 2])
    with c1:
        cats = sorted(query_df["category2"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", ["All"] + cats, key="b_cat")
    filtered_q = query_df if cat_filter == "All" else query_df[query_df["category2"] == cat_filter]
    sample_options = filtered_q.head(60)[["item_id", "product_id", "color", "category2"]].copy()
    with c2:
        labels = [
            f"{describe_label(r['item_id'])} — {(r['color'] or 'unknown').title()} {r['category2']}"
            for _, r in sample_options.iterrows()
        ]
        sel_idx = st.selectbox(
            "Query product",
            range(len(labels)),
            format_func=lambda i: labels[i] if labels else "(none)",
            key="b_query",
        )
    with c3:
        top_k = st.slider("Show top-K", 4, 12, 8, key="b_k")

    if not labels:
        st.warning("No queries in this category.")
    else:
        sel_row = sample_options.iloc[sel_idx]
        q_row = query_df[query_df["item_id"] == sel_row["item_id"]].iloc[0]
        q_idx = q_row.name

        response = engine.search_by_precomputed(
            q_visual=query_embs["visual"][q_idx],
            q_color=query_embs["color"][q_idx],
            category=q_row["category2"],
            k=top_k,
        )
        correct_pid = q_row["product_id"]
        hit_rank = next((r.rank for r in response.results if r.product_id == correct_pid), None)

        col_q, col_r = st.columns([1, 3])

        with col_q:
            st.markdown('<div class="query-panel">', unsafe_allow_html=True)
            st.markdown('<h3>Query</h3>', unsafe_allow_html=True)
            q_img = get_image(q_row["item_id"])
            if q_img:
                st.image(q_img, use_column_width=True)
                palette = dominant_palette(q_img, n_colors=5)
                st.markdown(
                    '<div class="color-palette">'
                    + "".join(f'<div class="swatch" style="background:{c};"></div>' for c in palette)
                    + "</div>",
                    unsafe_allow_html=True,
                )

            color_hex = color_name_to_hex(q_row.get("color", ""))
            cat = q_row["category2"]
            st.markdown(
                f"""
                <div class="meta">
                    <div style="margin-top:8px;">
                        <span class="tag" style="background:{color_hex}33; color:{color_hex};">
                        ● {(q_row.get('color') or 'unknown').title()}</span>
                        <span class="tag">{cat}</span>
                    </div>
                    <div style="margin-top:10px;">
                        <b>{response.n_gallery_candidates}</b> candidates
                        · <b>{response.latency_ms:.1f}ms</b> latency
                        · <b>visual-only</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            if hit_rank:
                st.markdown(
                    f"""
                    <div class="hit-banner hit">
                        <span class="icon">✓</span>
                        Correct product retrieved at rank <b style="margin:0 4px;">#{hit_rank}</b>
                        from {response.n_gallery_candidates} {q_row['category2']} candidates
                        — using image pixels only.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="hit-banner miss">
                        <span class="icon">!</span>
                        Correct product not in top-{top_k}. Visual ambiguity is more common in
                        categories with high intra-class diversity — {q_row['category2']} sits at
                        R@1 = {dict(cat_perf).get(q_row['category2'], 0):.3f} on the production system.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            render_results_grid(response, correct_pid=correct_pid, n_cols=4,
                                score_labels=("Visual", "Color"))


# ── TAB 2: Color filter ───────────────────────────────────────────────────────
with tab_color:
    st.markdown(
        '<div class="section-h"><h2>Filter by color palette</h2>'
        '<span class="sub">Pick up to three hues + a category — pure-color retrieval, no image required</span></div>',
        unsafe_allow_html=True,
    )

    cc1, cc2, cc3, cc4 = st.columns([1.2, 1.2, 1.2, 2])
    if "color_a" not in st.session_state:
        st.session_state["color_a"] = "#1A1A1A"
        st.session_state["color_b"] = "#2563EB"
        st.session_state["color_c"] = "#F5F5F5"
    with cc1:
        c_a = st.color_picker("Hue 1", st.session_state["color_a"], key="c_pick_a")
    with cc2:
        c_b = st.color_picker("Hue 2", st.session_state["color_b"], key="c_pick_b")
    with cc3:
        c_c = st.color_picker("Hue 3", st.session_state["color_c"], key="c_pick_c")
    with cc4:
        cf_cat = st.selectbox(
            "Category",
            sorted(query_df["category2"].dropna().unique().tolist()),
            key="cf_cat",
        )
        cf_k = st.slider("Top-K", 4, 12, 8, key="cf_k")

    PRESETS = [
        ("Monochrome", ["#1A1A1A", "#777777", "#F5F5F5"]),
        ("Denim", ["#1E3A8A", "#3B82F6", "#1A1A1A"]),
        ("Earth tones", ["#92400E", "#D4B996", "#6B7C32"]),
        ("Pastels", ["#FBCFE8", "#BAE6FD", "#FDE68A"]),
        ("Black & white", ["#0F172A", "#FFFFFF", "#9CA3AF"]),
        ("Burgundy & cream", ["#7C1E2D", "#F5E6D3", "#1A1A1A"]),
    ]
    pcols = st.columns(len(PRESETS))
    for (name, swatches), pc in zip(PRESETS, pcols):
        with pc:
            tile_html = "".join(
                f'<div style="height:18px; background:{s}; border-bottom:1px solid rgba(0,0,0,0.05);"></div>'
                for s in swatches
            )
            st.markdown(
                f'<div style="border-radius:10px; overflow:hidden; border:1px solid {SLATE_100};">{tile_html}</div>',
                unsafe_allow_html=True,
            )
            if st.button(name, key=f"preset_{name}", use_container_width=True):
                st.session_state["color_a"], st.session_state["color_b"], st.session_state["color_c"] = swatches
                st.rerun()

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    run_color = st.button("Find products with this palette", key="run_color", use_container_width=False)

    chosen = [c_a, c_b, c_c]
    if run_color:
        q_color = hex_to_color_histogram(chosen, bins_per_channel=8)
        if q_color.sum() == 0:
            st.warning("Pick at least one color.")
        else:
            response = engine.search_by_color(q_color=q_color, category=cf_cat, k=cf_k)

            qcol, scol = st.columns([1, 3])
            with qcol:
                st.markdown('<div class="query-panel">', unsafe_allow_html=True)
                st.markdown('<h3>Your palette</h3>', unsafe_allow_html=True)
                tiles = "".join(
                    f'<div class="swatch-tile" style="background:{c};"></div>'
                    for c in chosen
                )
                st.markdown(tiles, unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="meta" style="margin-top:6px;">
                        <span class="tag">{cf_cat}</span>
                        <span class="tag warning">color-only retrieval</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with scol:
                st.markdown(
                    f"""
                    <div class="hit-banner info">
                        <span class="icon">i</span>
                        Pure color retrieval — no semantic visual signal.
                        Returned <b>{len(response.results)}</b> {cf_cat} matches in
                        <b>{response.latency_ms:.1f} ms</b>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                render_results_grid(response, correct_pid=None, n_cols=4,
                                    score_labels=("Visual", "Color"))
    else:
        st.markdown(
            f'<div style="margin-top:14px; padding:60px 16px; text-align:center; '
            f'color:{SLATE_500}; background:{SLATE_50}; border-radius:14px;">'
            f'<div style="font-size:2.4em;">🎨</div>'
            f'<div style="font-weight:600; color:{SLATE_700}; margin-top:6px;">'
            'Pick three colors above (or use a preset), then click <b>Find products</b>.'
            '</div></div>',
            unsafe_allow_html=True,
        )


# ── TAB 3: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        '<div class="section-h"><h2>Upload your own product photo</h2>'
        '<span class="sub">Drop in any image — pure visual retrieval (CLIP image + color, no text)</span></div>',
        unsafe_allow_html=True,
    )

    uc1, uc2 = st.columns([1, 1])
    with uc1:
        uploaded = st.file_uploader(
            "Drag a fashion photo here (JPG/PNG, ideally a single clothing item on a clean background)",
            type=["png", "jpg", "jpeg"],
            key="upload_img",
        )
    with uc2:
        upload_cat = st.selectbox(
            "Category",
            sorted(query_df["category2"].dropna().unique().tolist()),
            key="u_cat",
            help="Required for the +10pp R@1 lift from category filtering.",
        )
        upload_k = st.slider("Top-K", 4, 12, 8, key="u_k")
        run_upload = st.button("Find similar", key="run_upload", use_container_width=True)

    if uploaded is not None:
        try:
            user_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        except Exception as e:
            st.error(f"Could not read image: {e}")
            user_img = None

        if user_img is not None:
            qcol, scol = st.columns([1, 3])
            with qcol:
                st.markdown('<div class="query-panel">', unsafe_allow_html=True)
                st.markdown('<h3>Your photo</h3>', unsafe_allow_html=True)
                st.image(user_img, use_column_width=True)
                palette = dominant_palette(user_img, n_colors=6)
                st.markdown(
                    '<div class="color-palette">'
                    + "".join(f'<div class="swatch" style="background:{c};"></div>' for c in palette)
                    + "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <div class="meta" style="margin-top:10px;">
                        <span class="tag">{upload_cat}</span>
                        <span class="tag success">visual-only</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with scol:
                if run_upload:
                    with st.spinner("Embedding image with CLIP B/32 + extracting color histogram…"):
                        response = engine.search_by_image(
                            img=user_img,
                            category=upload_cat,
                            k=upload_k,
                        )
                    st.markdown(
                        f"""
                        <div class="hit-banner hit">
                            <span class="icon">→</span>
                            Found <b>{len(response.results)}</b> visually similar products in
                            <b>{response.latency_ms:.1f} ms</b>
                            ({response.n_gallery_candidates} {upload_cat} candidates).
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    render_results_grid(response, correct_pid=None, n_cols=4,
                                        score_labels=("Visual", "Color"))
                else:
                    st.markdown(
                        f'<div style="padding:60px 16px; text-align:center; color:{SLATE_500};">'
                        '<div style="font-size:2em; margin-bottom:8px;">↗</div>'
                        f'<div style="font-weight:600; color:{SLATE_700};">Click <b>Find similar</b> '
                        'to embed and search.</div>'
                        f'<div style="font-size:0.86em; margin-top:6px;">First image takes ~30 ms '
                        '(CLIP forward pass); search itself is sub-millisecond.</div></div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.markdown(
            f'<div style="padding:50px 16px; text-align:center; color:{SLATE_500};">'
            '<div style="font-size:2.4em; margin-bottom:6px;">📷</div>'
            f'<div style="font-weight:600; color:{SLATE_700};">No photo yet.</div>'
            '<div style="font-size:0.86em; margin-top:6px;">'
            'Drop a single clothing item on a clean background for the best results.</div></div>',
            unsafe_allow_html=True,
        )


# ── TAB 4: Research ───────────────────────────────────────────────────────────
with tab_research:
    st.markdown(
        '<div class="section-h"><h2>Five phases of research, condensed</h2>'
        '<span class="sub">Anthony × Mark · 32 configurations · the leaderboard plus the surprises</span></div>',
        unsafe_allow_html=True,
    )

    timeline = [
        ("P1", "ResNet50 baseline", "Jackets are 2.8× harder than shirts. Generic ImageNet features collapse visually diverse categories.", 0.307, False),
        ("P2", "Foundation models", "CLIP ViT-L/14 dominates DINOv2 by 2× (0.553 vs 0.243). Vision-language pretraining > self-supervised for products.", 0.642, False),
        ("P3", "Visual champion", "CLIP B/32 + cat filter + 48D color α=0.4 — visual-only system reaches R@1 = 0.683.", 0.683, True),
        ("P4", "96D color hurts", "Doubling color resolution catastrophically drops R@1 by -23pp. Coarse 8-bin histograms beat fine 16-bin.", 0.695, False),
        ("P5", "Optuna-tuned visual", "CLIP L/14 + color + spatial + cat filter, fusion weights tuned across 300 trials. R@1 climbs to 0.729.", 0.729, True),
        ("P6", "Production champion", "Visual-only stack ships. R@1 = 0.683 with the B/32 backbone and 48D color, sub-millisecond search.", 0.683, True),
    ]
    st.markdown('<div class="timeline">', unsafe_allow_html=True)
    for phase, title, body, r1, champ in timeline:
        cls = "timeline-item champion" if champ else "timeline-item"
        st.markdown(
            f"""
            <div class="{cls}">
                <div class="phase">{phase}</div>
                <div class="title">{title}</div>
                <div class="body">{body}</div>
                <div style="margin-top:6px;"><span class="badge">R@1 = {r1:.3f}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-h" style="margin-top:24px;"><h2>Visual-only leaderboard</h2>'
        '<span class="sub">Every row works on raw pixels — no query-side text at inference</span></div>',
        unsafe_allow_html=True,
    )

    leaderboard = [
        ("CLIP L/14 + color + spatial + cat (Optuna)", 0.729, 0.882, 0.974, "P5"),
        ("Per-category alpha oracle", 0.695, 0.866, 0.911, "P4"),
        ("CLIP B/32 + cat + color α=0.4 ★ shipping", 0.683, 0.862, 0.913, "P3"),
        ("CLIP L/14 + color α=0.5", 0.642, 0.808, 0.857, "P3"),
        ("CLIP B/32 + color α=0.5", 0.576, 0.789, 0.858, "P2"),
        ("CLIP L/14 bare", 0.553, 0.748, 0.805, "P2"),
        ("CLIP B/32 bare", 0.480, 0.722, 0.807, "P2"),
        ("ResNet50 + color rerank α=0.5", 0.405, 0.647, 0.757, "P1"),
        ("EfficientNet-B0 + color (aug)", 0.383, 0.612, 0.694, "P1"),
        ("Color-only 48D histogram", 0.338, 0.524, 0.613, "P1"),
        ("ResNet50 baseline", 0.307, 0.493, 0.590, "P1"),
        ("DINOv2 ViT-B/14 bare", 0.243, 0.450, 0.560, "P2"),
    ]
    df = pd.DataFrame(leaderboard, columns=["System", "R@1", "R@5", "R@10", "Phase"])

    def color_phase(val):
        m = {"P1": "#E2E8F0", "P2": "#DBEAFE", "P3": "#FEF3C7",
             "P4": "#EDE9FE", "P5": "#DCFCE7", "P6": "#A7F3D0"}
        return f"background-color: {m.get(str(val), 'white')}; font-weight:600; text-align:center"

    styled = (
        df.style
        .format({"R@1": "{:.3f}", "R@5": "{:.3f}", "R@10": "{:.3f}"})
        .applymap(color_phase, subset=["Phase"])
        .bar(subset=["R@1"], color="#A5B4FC", vmin=0.20, vmax=0.80)
        .bar(subset=["R@5"], color="#86EFAC", vmin=0.40, vmax=0.95)
    )
    st.dataframe(styled, use_container_width=True, height=470, hide_index=True)

    st.markdown(
        '<div class="section-h" style="margin-top:24px;"><h2>Visual-only ablation</h2>'
        '<span class="sub">What each component contributes to the R@1 = 0.683 champion</span></div>',
        unsafe_allow_html=True,
    )

    # Numbers measured live via scripts/verify_ui_numbers.py on the full
    # 1,027-query test set. Each row removes one component from the production
    # champion (cat filter + CLIP B/32 image + 48D color hist, α = 0.4).
    ablation = [
        ("Full: cat filter + CLIP image + color hist", 0.6826, 0.0000, INDIGO),
        ("Remove color histogram (CLIP + cat)", 0.5686, -0.1140, AMBER),
        ("Remove CLIP image (color + cat)", 0.5131, -0.1695, CORAL),
        ("Remove category filter (CLIP + color)", 0.5794, -0.1032, CORAL),
        ("Color-only baseline (no CLIP, no cat)", 0.3505, -0.3321, CORAL),
    ]
    abl_html = "".join(
        f"""
        <div style="display:grid; grid-template-columns: 2.4fr 0.7fr 0.7fr 1fr; gap:8px; align-items:center;
                    padding:10px 12px; background:white; border:1px solid {SLATE_100}; border-radius:10px;
                    margin-bottom:6px;">
            <div style="font-size:0.92em; color:{SLATE_900}; font-weight:600;">{name}</div>
            <div style="font-size:0.9em; font-variant-numeric: tabular-nums; color:{SLATE_700};">R@1 = <b>{r1:.4f}</b></div>
            <div style="font-size:0.9em; font-variant-numeric: tabular-nums; color:{color}; font-weight:700;">
                {('+' if delta >= 0 else '')}{delta:+.4f}</div>
            <div style="background:{SLATE_100}; border-radius:999px; height:10px; overflow:hidden; position:relative;">
                <div style="position:absolute; left:50%; top:0; bottom:0; width:1px; background:{SLATE_300};"></div>
                <div style="position:absolute; left:{50 + (delta * 120) if delta < 0 else 50}%;
                            width:{abs(delta * 120)}%; top:0; bottom:0; background:{color};
                            border-radius: 999px 0 0 999px;"></div>
            </div>
        </div>
        """
        for (name, r1, delta, color) in ablation
    )
    st.markdown(abl_html, unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="callout" style="margin-top:14px;">
            <div class="head">Visual-only takeaways</div>
            <div class="body">
                <b>1.</b> CLIP's image encoder carries the most visual signal — removing it costs
                <b>−17pp</b> R@1 (drops from 0.683 to 0.513).<br>
                <b>2.</b> Color histograms are the second-strongest visual feature — removing them
                costs <b>−11pp</b> even with CLIP intact.<br>
                <b>3.</b> Category filter is pure upside on top of those: <b>+10pp</b> R@1
                (0.579 → 0.683) with zero new features.<br>
                <b>4.</b> Color-only retrieval without category filter falls all the way to
                R@1 = 0.351 — your floor when you can't trust CLIP at all.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer-text">
        <b>Lookmatch</b> · Visual-only Product Search · DeepFashion In-Shop benchmark<br>
        Built by <b>Mark Rodrigues</b> × <b>Anthony Rodrigues</b> · 2026
    </div>
    """,
    unsafe_allow_html=True,
)
