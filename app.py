"""
Visual Product Search Engine — Streamlit Demo

Polished e-commerce-style interface for the Phase 6 production pipeline:
    Category filter + 48D color histogram + CLIP B/32 text embeddings
    R@1 = 0.941 on DeepFashion In-Shop (300 gallery, 1,027 queries)

Four tabs:
    Browse — pick from the 1,027-query test set
    Text   — describe a product in words
    Upload — drop in your own product photo
    Research — leaderboard, ablation, phase timeline
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

.stApp {{
    background: linear-gradient(180deg, {SLATE_50} 0%, #FFFFFF 280px);
}}

/* Hide Streamlit chrome we don't want */
#MainMenu, footer, [data-testid="stDeployButton"] {{
    visibility: hidden;
}}
header[data-testid="stHeader"] {{
    background: transparent;
}}

/* Hero header */
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
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 95% 0%, rgba(255,255,255,0.18), transparent 45%);
    pointer-events: none;
}}
.hero h1 {{
    font-size: 2.0em;
    font-weight: 800;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}}
.hero p {{
    font-size: 1.0em;
    margin: 0;
    color: rgba(255, 255, 255, 0.90);
    max-width: 760px;
}}
.hero-stats {{
    display: flex;
    gap: 28px;
    margin-top: 16px;
    flex-wrap: wrap;
}}
.hero-stat {{
    background: rgba(255,255,255,0.16);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.22);
    padding: 8px 14px;
    border-radius: 10px;
    min-width: 90px;
}}
.hero-stat .v {{
    font-size: 1.4em;
    font-weight: 800;
    color: white;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
}}
.hero-stat .l {{
    font-size: 0.72em;
    color: rgba(255,255,255,0.85);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: {SLATE_100};
    padding: 4px;
    border-radius: 12px;
    border: 1px solid {SLATE_300};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    color: {SLATE_700};
    background: transparent;
    transition: all 0.15s ease;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: white;
    color: {INDIGO};
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 24px;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: white;
    border-right: 1px solid {SLATE_100};
}}
.sidebar-section {{
    margin-bottom: 18px;
}}
.sidebar-section h4 {{
    font-size: 0.72em;
    font-weight: 700;
    color: {SLATE_500};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 10px 0;
}}
.kpi-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}}
.kpi {{
    background: {SLATE_50};
    border: 1px solid {SLATE_100};
    border-radius: 10px;
    padding: 10px 12px;
}}
.kpi .v {{
    font-size: 1.25em;
    font-weight: 800;
    color: {SLATE_900};
    font-variant-numeric: tabular-nums;
    line-height: 1.0;
}}
.kpi .l {{
    font-size: 0.7em;
    color: {SLATE_500};
    margin-top: 2px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.kpi.accent .v {{ color: {INDIGO}; }}
.kpi.success .v {{ color: {EMERALD}; }}

.pipeline-step {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid {SLATE_100};
}}
.pipeline-step:last-child {{ border-bottom: none; }}
.pipeline-step .num {{
    background: {INDIGO_LIGHT};
    color: {INDIGO};
    border-radius: 6px;
    width: 22px; height: 22px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.78em;
    flex-shrink: 0;
}}
.pipeline-step .body {{ flex: 1; }}
.pipeline-step .title {{ font-weight: 600; font-size: 0.88em; color: {SLATE_900}; }}
.pipeline-step .sub {{ font-size: 0.76em; color: {SLATE_500}; margin-top: 2px; line-height: 1.4; }}

.callout {{
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border-left: 3px solid {AMBER};
    border-radius: 10px;
    padding: 12px 14px;
    margin: 8px 0 0 0;
}}
.callout .head {{
    font-weight: 700;
    font-size: 0.82em;
    color: #92400E;
    margin-bottom: 4px;
}}
.callout .body {{
    font-size: 0.78em;
    color: #78350F;
    line-height: 1.45;
}}

.cat-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 4px 0;
    font-size: 0.82em;
}}
.cat-row .name {{
    flex: 0 0 80px;
    color: {SLATE_700};
    font-weight: 500;
}}
.cat-row .bar-wrap {{
    flex: 1;
    background: {SLATE_100};
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    margin: 0 8px;
}}
.cat-row .bar {{
    height: 100%;
    border-radius: 999px;
}}
.cat-row .v {{
    flex: 0 0 36px;
    text-align: right;
    color: {SLATE_700};
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 0.85em;
}}

/* Result cards */
.result-card {{
    border: 1px solid {SLATE_100};
    border-radius: 14px;
    overflow: hidden;
    background: white;
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    margin-bottom: 12px;
    position: relative;
}}
.result-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 14px 28px rgba(15, 23, 42, 0.10);
}}
.result-card.is-correct {{
    border-color: {EMERALD};
    box-shadow: 0 0 0 3px {EMERALD_LIGHT};
}}
.result-card .rank {{
    position: absolute;
    top: 8px; left: 8px;
    background: rgba(15, 23, 42, 0.78);
    color: white;
    font-weight: 700;
    font-size: 0.74em;
    padding: 3px 8px;
    border-radius: 6px;
    z-index: 2;
    letter-spacing: 0.04em;
}}
.result-card.is-correct .rank {{
    background: {EMERALD};
}}
.result-card .match-badge {{
    position: absolute;
    top: 8px; right: 8px;
    background: {EMERALD};
    color: white;
    font-weight: 700;
    font-size: 0.7em;
    padding: 3px 8px;
    border-radius: 6px;
    z-index: 2;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.result-card .score-pill {{
    position: absolute;
    bottom: 8px; right: 8px;
    background: rgba(255, 255, 255, 0.96);
    color: {SLATE_900};
    font-weight: 700;
    font-size: 0.76em;
    padding: 4px 9px;
    border-radius: 999px;
    z-index: 2;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
    font-variant-numeric: tabular-nums;
}}
.result-meta {{
    padding: 10px 12px;
    font-size: 0.82em;
}}
.result-meta .top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}}
.result-meta .color-chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: {SLATE_700};
    font-weight: 500;
}}
.result-meta .swatch {{
    width: 10px; height: 10px; border-radius: 50%;
    border: 1px solid rgba(0,0,0,0.05);
    display: inline-block;
}}
.result-meta .cat {{
    color: {SLATE_500};
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.7em;
    letter-spacing: 0.06em;
}}

.score-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-top: 6px;
}}
.score-comp {{
    background: {SLATE_50};
    border-radius: 6px;
    padding: 4px 6px;
    font-size: 0.7em;
}}
.score-comp .l {{
    color: {SLATE_500};
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
}}
.score-comp .v {{
    color: {SLATE_900};
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}
.score-comp .micro-bar {{
    height: 3px;
    background: {SLATE_100};
    border-radius: 2px;
    margin-top: 3px;
    overflow: hidden;
}}
.score-comp .micro-fill {{ height: 100%; border-radius: 2px; }}

/* Query card */
.query-panel {{
    background: white;
    border: 1px solid {SLATE_100};
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    position: sticky;
    top: 12px;
}}
.query-panel h3 {{
    font-size: 0.78em;
    font-weight: 700;
    color: {SLATE_500};
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 8px 0;
}}
.query-panel .meta {{
    font-size: 0.85em;
    color: {SLATE_700};
    line-height: 1.6;
    margin-top: 8px;
}}
.query-panel .meta b {{ color: {SLATE_900}; }}
.tag {{
    display: inline-block;
    background: {INDIGO_LIGHT};
    color: {INDIGO};
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.75em;
    font-weight: 600;
    margin-right: 4px;
}}
.tag.success {{ background: {EMERALD_LIGHT}; color: {EMERALD}; }}
.tag.warning {{ background: #FEF3C7; color: #92400E; }}

.color-palette {{
    display: flex;
    gap: 4px;
    margin-top: 10px;
}}
.color-palette .swatch {{
    flex: 1;
    height: 18px;
    border-radius: 4px;
    border: 1px solid rgba(0,0,0,0.05);
}}

/* Hit/miss banner */
.hit-banner {{
    border-radius: 12px;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 0.92em;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.hit-banner.hit {{
    background: {EMERALD_LIGHT};
    color: #065F46;
    border: 1px solid #A7F3D0;
}}
.hit-banner.miss {{
    background: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
}}
.hit-banner .icon {{
    width: 26px; height: 26px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.85em;
}}
.hit-banner.hit .icon {{ background: {EMERALD}; color: white; }}
.hit-banner.miss .icon {{ background: {AMBER}; color: white; }}

/* Suggestion pills */
.suggest-pill {{
    display: inline-block;
    background: white;
    border: 1px solid {SLATE_300};
    color: {SLATE_700};
    padding: 6px 12px;
    border-radius: 999px;
    font-size: 0.82em;
    margin: 3px 4px 3px 0;
    cursor: pointer;
    transition: all 0.15s ease;
}}
.suggest-pill:hover {{ border-color: {INDIGO}; color: {INDIGO}; }}

/* Section heading */
.section-h {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin: 4px 0 12px 0;
}}
.section-h h2 {{
    font-size: 1.05em;
    font-weight: 700;
    color: {SLATE_900};
    margin: 0;
}}
.section-h .sub {{
    font-size: 0.85em;
    color: {SLATE_500};
}}

/* Buttons */
div[data-testid="stButton"] > button {{
    background: {INDIGO};
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 600;
    transition: all 0.15s ease;
}}
div[data-testid="stButton"] > button:hover {{
    background: #4338CA;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
}}
div[data-testid="stButton"] > button:active {{ transform: translateY(0); }}

/* Selectbox / inputs */
div[data-baseweb="select"] > div {{ border-radius: 10px !important; }}
.stTextInput input, .stTextArea textarea {{
    border-radius: 10px !important;
    border: 1px solid {SLATE_300} !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {INDIGO} !important;
    box-shadow: 0 0 0 3px {INDIGO_LIGHT} !important;
}}

/* Upload zone */
[data-testid="stFileUploader"] section {{
    border-radius: 14px;
    border: 2px dashed {SLATE_300};
    background: {SLATE_50};
    padding: 28px 16px;
    transition: all 0.15s ease;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: {INDIGO};
    background: {INDIGO_LIGHT};
}}

.timeline {{
    position: relative;
    padding-left: 28px;
}}
.timeline::before {{
    content: "";
    position: absolute;
    left: 8px; top: 4px; bottom: 4px;
    width: 2px;
    background: linear-gradient(180deg, {INDIGO} 0%, {EMERALD} 100%);
    border-radius: 2px;
}}
.timeline-item {{
    position: relative;
    margin-bottom: 16px;
    padding: 10px 14px;
    background: white;
    border: 1px solid {SLATE_100};
    border-radius: 10px;
}}
.timeline-item::before {{
    content: "";
    position: absolute;
    left: -24px; top: 14px;
    width: 12px; height: 12px;
    background: white;
    border: 2px solid {INDIGO};
    border-radius: 50%;
}}
.timeline-item.champion::before {{ border-color: {EMERALD}; background: {EMERALD}; }}
.timeline-item .phase {{
    font-size: 0.7em;
    color: {SLATE_500};
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
.timeline-item .title {{
    font-weight: 700;
    color: {SLATE_900};
    font-size: 0.95em;
    margin: 2px 0 4px 0;
}}
.timeline-item .body {{
    font-size: 0.85em;
    color: {SLATE_700};
    line-height: 1.5;
}}
.timeline-item .badge {{
    display: inline-block;
    background: {INDIGO_LIGHT};
    color: {INDIGO};
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.74em;
    font-weight: 700;
    margin-right: 6px;
    font-variant-numeric: tabular-nums;
}}
.timeline-item.champion .badge {{ background: {EMERALD_LIGHT}; color: {EMERALD}; }}

footer-mark {{ display: none; }}
.footer-text {{
    text-align: center;
    color: {SLATE_500};
    font-size: 0.78em;
    padding: 30px 0 10px 0;
    border-top: 1px solid {SLATE_100};
    margin-top: 32px;
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading search engine...")
def load_engine():
    from src.search_engine import ProductSearchEngine

    engine = ProductSearchEngine()
    engine.load_gallery()
    return engine


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    proc = PROJECT_ROOT / "data" / "processed"
    gallery_df = pd.read_csv(proc / "gallery.csv")
    query_df = pd.read_csv(proc / "query.csv")
    return gallery_df, query_df


@st.cache_data(show_spinner="Loading embeddings...")
def load_query_embeddings():
    cache = PROJECT_ROOT / "data" / "processed" / "emb_cache"

    def normed(name):
        emb = np.load(cache / f"{name}.npy").astype(np.float32)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)

    return {"text": normed("clip_b32_text_query"), "color": normed("color48_query")}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_image(item_id: str) -> Image.Image | None:
    img_path = PROJECT_ROOT / "data" / "raw" / "images" / f"{item_id}.jpg"
    if img_path.exists():
        return Image.open(img_path).convert("RGB")
    return None


def dominant_palette(img: Image.Image, n_colors: int = 5) -> list[str]:
    """Return n_colors hex strings approximating the image's dominant colors."""
    small = img.convert("RGB").resize((48, 48), Image.LANCZOS)
    arr = np.array(small).reshape(-1, 3)
    # Quantize to a coarse 4x4x4 cube, count, take the busiest cells.
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
    """Best-effort mapping of DeepFashion color labels to a representative hex."""
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


def describe_label(item_id: str) -> str:
    """A shorter human label for a DeepFashion item id."""
    parts = item_id.split("_")
    pid = next((p for p in parts if p.startswith("id")), item_id[:14])
    return pid[:14]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:1.4em; font-weight:800; color:{}; letter-spacing:-0.02em; margin: -8px 0 4px 0;">'
        '🪡 Lookmatch</div>'
        '<div style="font-size:0.78em; color:{}; margin-bottom:18px;">visual product search · v1.0</div>'.format(
            SLATE_900, SLATE_500
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section"><h4>Performance</h4>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi success"><div class="v">0.941</div><div class="l">Recall@1</div></div>
            <div class="kpi success"><div class="v">1.000</div><div class="l">Recall@5</div></div>
            <div class="kpi accent"><div class="v">0.10ms</div><div class="l">Latency</div></div>
            <div class="kpi"><div class="v">300</div><div class="l">Gallery</div></div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section"><h4>Pipeline</h4>', unsafe_allow_html=True)
    steps = [
        ("1", "Category filter", "hard constraint — same category only"),
        ("2", "48D color histogram", "RGB color distribution"),
        ("3", "CLIP B/32 text", "semantic match, w=0.80"),
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
                <div class="head">⚡ Counterintuitive finding</div>
                <div class="body">Removing CLIP <em>visual</em> features <b>improves</b> R@1 by +1.35pp.
                Text descriptions carry the discriminative signal — visual embeddings add noise from
                lighting and pose.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section"><h4>Per-category R@1</h4>', unsafe_allow_html=True)
    cat_perf = [
        ("suiting", 1.000), ("jackets", 0.987), ("sweaters", 0.987),
        ("shirts", 0.975), ("denim", 0.948), ("sweatshirts", 0.937),
        ("pants", 0.931), ("tees", 0.926), ("shorts", 0.905),
    ]
    for name, r1 in cat_perf:
        pct = int(r1 * 100)
        color = EMERALD if r1 >= 0.95 else AMBER if r1 >= 0.92 else CORAL
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
            <b>Backbone</b> CLIP ViT-B/32 (text encoder only)<br>
            <b>Built by</b> Mark Rodrigues × Anthony Rodrigues
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>Find any fashion product from a photo or a sentence.</h1>
        <p>Trained on 1,027 queries across the DeepFashion In-Shop benchmark.
        Answers in 0.10 ms. Beats GPT-4V at 30,000× lower cost.</p>
        <div class="hero-stats">
            <div class="hero-stat"><div class="v">94.1%</div><div class="l">recall @ 1</div></div>
            <div class="hero-stat"><div class="v">100%</div><div class="l">recall @ 5</div></div>
            <div class="hero-stat"><div class="v">0.10 ms</div><div class="l">latency</div></div>
            <div class="hero-stat"><div class="v">9</div><div class="l">categories</div></div>
            <div class="hero-stat"><div class="v">5</div><div class="l">research phases</div></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab_browse, tab_text, tab_upload, tab_research = st.tabs(
    ["🔍 Browse", "✍️ Describe", "📷 Upload", "📊 Research"]
)

engine = load_engine()
gallery_df, query_df = load_data()
query_embs = load_query_embeddings()


# ── Result card renderer (shared) ─────────────────────────────────────────────
def render_results_grid(response, correct_pid: str | None, n_cols: int = 4):
    if not response.results:
        st.info("No results.")
        return

    rows = [response.results[i : i + n_cols] for i in range(0, len(response.results), n_cols)]
    max_score = max(r.combined_score for r in response.results) or 1.0

    for row in rows:
        cols = st.columns(n_cols)
        for col, r in zip(cols, row):
            with col:
                is_correct = correct_pid is not None and r.product_id == correct_pid
                card_cls = "result-card is-correct" if is_correct else "result-card"

                img = get_image(r.item_id)

                # Render the card scaffold (image goes via st.image, the rest is HTML)
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
                text_w = max(0.0, min(1.0, r.text_score))
                color_w = max(0.0, min(1.0, r.color_score))
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
                                <div class="l">Text</div>
                                <div class="v">{r.text_score:.3f}</div>
                                <div class="micro-bar"><div class="micro-fill" style="width:{int(text_w*100)}%;background:{INDIGO};"></div></div>
                            </div>
                            <div class="score-comp">
                                <div class="l">Color</div>
                                <div class="v">{r.color_score:.3f}</div>
                                <div class="micro-bar"><div class="micro-fill" style="width:{int(color_w*100)}%;background:{AMBER};"></div></div>
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
        '<span class="sub">1,027 queries · pick one to see retrieval in action</span></div>',
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
            q_text=query_embs["text"][q_idx],
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
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            desc = str(q_row.get("description", "") or "")
            if desc:
                with st.expander("Product description"):
                    st.write(desc[:500] + ("…" if len(desc) > 500 else ""))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            if hit_rank:
                st.markdown(
                    f"""
                    <div class="hit-banner hit">
                        <span class="icon">✓</span>
                        Correct product retrieved at rank <b style="margin:0 4px;">#{hit_rank}</b>
                        from {response.n_gallery_candidates} {q_row['category2']} candidates.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="hit-banner miss">
                        <span class="icon">!</span>
                        Correct product not in top-{top_k}. This is one of the 5.9% of queries the system gets wrong.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            render_results_grid(response, correct_pid=correct_pid, n_cols=4)


# ── TAB 2: Describe ───────────────────────────────────────────────────────────
with tab_text:
    st.markdown(
        '<div class="section-h"><h2>Describe what you\'re looking for</h2>'
        '<span class="sub">Pure-text retrieval via CLIP B/32 — no image required</span></div>',
        unsafe_allow_html=True,
    )

    EXAMPLES = [
        ("Black skinny jeans, zip fly, slim fit, slight stretch", "denim"),
        ("Classic white oxford shirt, button-down collar, slim fit", "shirts"),
        ("Navy crewneck sweatshirt, fleece-lined, kangaroo pocket", "sweatshirts"),
        ("Light wash denim shorts, frayed hem, summer", "shorts"),
        ("Camel wool blazer, two-button, peak lapel", "jackets"),
        ("Chunky cable-knit cream sweater, oversized fit", "sweaters"),
    ]
    if "text_query" not in st.session_state:
        st.session_state["text_query"] = EXAMPLES[0][0]
        st.session_state["text_cat"] = EXAMPLES[0][1]

    pill_cols = st.columns(len(EXAMPLES))
    for i, ((prompt, c), col) in enumerate(zip(EXAMPLES, pill_cols)):
        with col:
            if st.button(c.title(), key=f"sugg_{i}", use_container_width=True):
                st.session_state["text_query"] = prompt
                st.session_state["text_cat"] = c

    c1, c2, c3 = st.columns([5, 2, 1])
    with c1:
        text_query = st.text_area(
            "Describe a product",
            value=st.session_state.get("text_query", EXAMPLES[0][0]),
            height=80,
            key="text_query_input",
            label_visibility="collapsed",
            placeholder="e.g. cropped denim jacket, raw hem, indigo wash",
        )
    with c2:
        cats = ["Auto"] + sorted(query_df["category2"].dropna().unique().tolist())
        default_cat = st.session_state.get("text_cat", "Auto")
        try:
            default_idx = cats.index(default_cat)
        except ValueError:
            default_idx = 0
        text_cat = st.selectbox("Category", cats, index=default_idx, key="t_cat")
        text_k = st.slider("Top-K", 4, 12, 8, key="t_k")
    with c3:
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        run_text = st.button("Search", key="run_text", use_container_width=True)

    if run_text and text_query.strip():
        category = None if text_cat == "Auto" else text_cat
        with st.spinner("Searching the catalog…"):
            response = engine.search_by_text(description=text_query, category=category, k=text_k)
        st.markdown(
            f"""
            <div style="margin: 12px 0 14px 0; color:{SLATE_700}; font-size:0.92em;">
                <b>{len(response.results)} results</b> · category
                <span class="tag">{response.query_category}</span>
                · {response.n_gallery_candidates} candidates
                · <b>{response.latency_ms:.1f} ms</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_results_grid(response, correct_pid=None, n_cols=4)
    elif text_query.strip():
        st.markdown(
            f'<div style="color:{SLATE_500}; font-size:0.88em; margin-top:14px;">'
            'Click <b>Search</b>, or pick a category pill above to load an example.</div>',
            unsafe_allow_html=True,
        )


# ── TAB 3: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(
        '<div class="section-h"><h2>Upload your own product photo</h2>'
        '<span class="sub">Drop in any image — we\'ll extract a color palette and find similar products</span></div>',
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
            help="Tell the engine what kind of item this is — required for the +6.9pp R@1 lift.",
        )
        upload_desc = st.text_input(
            "Optional description",
            value="",
            key="u_desc",
            placeholder="e.g. black slim-fit denim jeans",
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
                        {f'<span class="tag success">"{upload_desc[:40]}"</span>' if upload_desc else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

            with scol:
                if run_upload:
                    with st.spinner("Extracting features and searching…"):
                        if upload_desc.strip():
                            response = engine.search_by_image_and_text(
                                img=user_img,
                                description=upload_desc,
                                category=upload_cat,
                                k=upload_k,
                            )
                            mode_label = "image + text"
                        else:
                            # Use image colour signal only (text branch is zeroed inside the engine)
                            from src.feature_engineering import extract_color_palette

                            q_color = extract_color_palette(user_img, bins_per_channel=8)
                            q_color = q_color / max(np.linalg.norm(q_color), 1e-8)
                            # Search by precomputed with zero text vector — fall through to colour-only
                            zero_text = np.zeros(512, dtype=np.float32)
                            response = engine.search_by_precomputed(
                                q_text=zero_text,
                                q_color=q_color,
                                category=upload_cat,
                                k=upload_k,
                            )
                            mode_label = "color only"
                    st.markdown(
                        f"""
                        <div class="hit-banner hit">
                            <span class="icon">→</span>
                            Found <b>{len(response.results)}</b> similar products in
                            <b>{response.latency_ms:.1f} ms</b> using <b>{mode_label}</b> retrieval
                            ({response.n_gallery_candidates} {upload_cat} candidates).
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    render_results_grid(response, correct_pid=None, n_cols=4)
                else:
                    st.markdown(
                        f'<div style="padding:60px 16px; text-align:center; color:{SLATE_500};">'
                        '<div style="font-size:2em; margin-bottom:8px;">↗</div>'
                        f'<div style="font-weight:600; color:{SLATE_700};">Click <b>Find similar</b> to run retrieval.</div>'
                        f'<div style="font-size:0.86em; margin-top:6px;">Add an optional description to use the text branch — '
                        'descriptions outperform image-only by ~9pp R@1.</div></div>',
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

    # Phase timeline
    timeline = [
        ("P1", "ResNet50 baseline", "Jackets are 2.8× harder than shirts. Generic ImageNet features collapse visually diverse categories.", 0.307, False),
        ("P2", "Foundation models", "CLIP ViT-L/14 dominates DINOv2 by 2× (0.553 vs 0.243). Vision-language pretraining > self-supervised for products.", 0.642, False),
        ("P3", "Category filter", "Restricting search to the same category adds +6.9pp R@1 with zero new features. 0/1027 queries hurt.", 0.683, False),
        ("P4", "96D color hurts", "Doubling color resolution catastrophically drops R@1 by -23pp. Coarse 8-bin histograms beat fine 16-bin.", 0.695, False),
        ("P5", "Text rerank", "Two-stage visual→text rerank reaches R@1=0.907. Removing CLIP visual entirely improves it to 0.920.", 0.920, False),
        ("P6", "Production champion", "Cat + 48D color + CLIP text. R@1=0.941, R@5=1.000, 0.10ms/query. Beats GPT-4V at 30,000× lower cost.", 0.941, True),
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

    # Leaderboard
    st.markdown(
        '<div class="section-h" style="margin-top:24px;"><h2>Full leaderboard — 15 selected configurations</h2>'
        '<span class="sub">Sorted by R@1; production-valid systems only</span></div>',
        unsafe_allow_html=True,
    )

    leaderboard = [
        ("Production champion (cat + color + text)", 0.941, 1.000, 1.000, "P6", "★★★"),
        ("Three-stage cat + CLIP + color + text", 0.907, 0.944, 0.944, "P5", ""),
        ("Visual top-K + text rerank (no color)", 0.920, 0.990, 0.990, "P5", "★★"),
        ("Per-category alpha oracle", 0.695, 0.866, 0.911, "P4", ""),
        ("CLIP B/32 + cat + color α=0.4", 0.683, 0.862, 0.913, "P3", "★"),
        ("CLIP L/14 + color + spatial + cat (Optuna)", 0.729, 0.882, 0.974, "P5", ""),
        ("CLIP L/14 + color α=0.5", 0.642, 0.808, 0.857, "P3", ""),
        ("Text-only CLIP L/14 prompts", 0.602, 0.957, 1.000, "P3", ""),
        ("CLIP B/32 + color α=0.5", 0.576, 0.789, 0.858, "P2", ""),
        ("CLIP L/14 bare", 0.553, 0.748, 0.805, "P2", ""),
        ("CLIP B/32 bare", 0.480, 0.722, 0.807, "P2", ""),
        ("ResNet50 + color rerank α=0.5", 0.405, 0.647, 0.757, "P1", ""),
        ("EfficientNet-B0 + color (aug)", 0.383, 0.612, 0.694, "P1", ""),
        ("ResNet50 baseline", 0.307, 0.493, 0.590, "P1", ""),
        ("DINOv2 ViT-B/14 bare", 0.243, 0.450, 0.560, "P2", ""),
    ]
    df = pd.DataFrame(leaderboard, columns=["System", "R@1", "R@5", "R@10", "Phase", "Note"])

    def color_phase(val):
        m = {"P1": "#E2E8F0", "P2": "#DBEAFE", "P3": "#FEF3C7",
             "P4": "#EDE9FE", "P5": "#DCFCE7", "P6": "#A7F3D0"}
        return f"background-color: {m.get(str(val), 'white')}; font-weight:600; text-align:center"

    styled = (
        df.style
        .format({"R@1": "{:.3f}", "R@5": "{:.3f}", "R@10": "{:.3f}"})
        .applymap(color_phase, subset=["Phase"])
        .bar(subset=["R@1"], color="#A5B4FC", vmin=0.20, vmax=1.00)
        .bar(subset=["R@5"], color="#86EFAC", vmin=0.40, vmax=1.00)
    )
    st.dataframe(styled, use_container_width=True, height=520, hide_index=True)

    # Ablation
    st.markdown(
        '<div class="section-h" style="margin-top:24px;"><h2>Ablation — what each component contributes</h2>'
        '<span class="sub">Phase 5 ablation on the three-stage pipeline (R@1 = 0.907 baseline)</span></div>',
        unsafe_allow_html=True,
    )

    ablation = [
        ("Full: cat + color + CLIP visual + text", 0.9065, 0.0000, INDIGO),
        ("Remove text rerank", 0.6699, -0.2366, CORAL),
        ("Remove category filter", 0.8345, -0.0721, CORAL),
        ("Remove color histogram", 0.8491, -0.0574, AMBER),
        ("Remove CLIP visual ★", 0.9200, +0.0135, EMERALD),
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
                <div style="position:absolute; left:{50 + (delta * 200) if delta < 0 else 50}%;
                            width:{abs(delta * 200)}%; top:0; bottom:0; background:{color};
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
            <div class="head">Two findings worth a LinkedIn post</div>
            <div class="body">
                <b>1.</b> Text reranking is the single biggest component — removing it drops R@1 by <b>−23.7pp</b>.<br>
                <b>2.</b> Removing CLIP <em>visual</em> embeddings <em>improves</em> R@1 by +1.35pp. When descriptions
                are paragraph-length and precise, text carries all the discriminative signal — visual embeddings
                add noise from lighting, pose, and background.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="footer-text">
        <b>Lookmatch</b> · Visual Product Search Engine · DeepFashion In-Shop benchmark<br>
        Built by <b>Mark Rodrigues</b> × <b>Anthony Rodrigues</b> · 2026
    </div>
    """,
    unsafe_allow_html=True,
)
