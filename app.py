"""
Visual Product Search Engine — Streamlit Demo

Demonstrates the best pipeline from 5 phases of research:
  Category filter + Color histogram + CLIP text embeddings → R@1=0.94

Key finding: REMOVING the CLIP visual backbone IMPROVES accuracy.
Text descriptions alone outperform image embeddings for product retrieval.
"""

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
    page_title="Fashion Product Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f0f2f6; border-radius: 8px; padding: 12px 16px;
    margin: 4px 0; text-align: center;
}
.metric-value { font-size: 1.8em; font-weight: 700; color: #1f77b4; }
.metric-label { font-size: 0.8em; color: #666; }
.result-card {
    border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px;
    margin: 4px; text-align: center; background: white;
}
.rank-badge {
    display: inline-block; background: #1f77b4; color: white;
    border-radius: 4px; padding: 2px 8px; font-weight: 700; font-size: 0.9em;
}
.score-bar-fill { background: #1f77b4; height: 6px; border-radius: 3px; }
.score-bar-bg { background: #e0e0e0; height: 6px; border-radius: 3px; margin: 4px 0; }
.finding-box {
    background: #fff3cd; border-left: 4px solid #ffc107;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 8px 0;
}
.pipeline-step {
    background: #e8f4fd; border-left: 4px solid #1f77b4;
    padding: 8px 12px; border-radius: 0 6px 6px 0; margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
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

    return {
        "text": normed("clip_b32_text_query"),
        "color": normed("color48_query"),
    }


def get_image(item_id: str) -> Image.Image | None:
    img_path = PROJECT_ROOT / "data" / "raw" / "images" / f"{item_id}.jpg"
    if img_path.exists():
        return Image.open(img_path).convert("RGB")
    return None


def color_bar(value: float, max_value: float = 1.0, color: str = "#1f77b4") -> str:
    pct = min(100, int(value / max(max_value, 1e-8) * 100))
    return f"""
    <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Model Performance")

    perf_data = [
        ("R@1", "0.941", "#27ae60"),
        ("R@5", "1.000", "#27ae60"),
        ("R@10", "1.000", "#27ae60"),
        ("Latency", "0.10ms/query", "#1f77b4"),
    ]
    cols = st.columns(2)
    for i, (label, value, color) in enumerate(perf_data):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Pipeline")
    steps = [
        ("1", "Category Filter", "Hard constraint — only same-category products"),
        ("2", "Color Histogram (48D)", "RGB color distribution match"),
        ("3", "CLIP Text Embedding", "Semantic description match (w=0.8)"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <b>Step {num}: {title}</b><br>
            <span style="font-size:0.8em; color:#555">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Key Finding")
    st.markdown("""
    <div class="finding-box">
        <b>Removing CLIP visual improves R@1 by +1.35pp</b><br>
        <span style="font-size:0.85em">Text descriptions capture color, cut, and style better than images.
        The visual model sees lighting variation and pose noise.
        The text says exactly what the product is.</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Per-Category R@1")
    cat_perf = {
        "suiting": 1.000, "jackets": 0.987, "sweaters": 0.987,
        "shirts": 0.975, "denim": 0.948, "sweatshirts": 0.937,
        "pants": 0.931, "tees": 0.926, "shorts": 0.905,
    }
    for cat, r1 in cat_perf.items():
        bar_pct = int(r1 * 100)
        color = "#27ae60" if r1 >= 0.95 else "#f39c12" if r1 >= 0.92 else "#e74c3c"
        st.markdown(f"""
        <div style="margin:3px 0; font-size:0.85em;">
            <b>{cat}</b> {r1:.3f}
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{bar_pct}%;background:{color};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **Dataset:** DeepFashion In-Shop
    **Gallery:** 300 products | **Queries:** 1,027
    **Model:** CLIP ViT-B/32 + 48D color hist
    """)


# ── Main content ──────────────────────────────────────────────────────────────
st.title("👗 Fashion Product Search Engine")
st.markdown(
    "Visual product retrieval with **R@1=0.941** on DeepFashion In-Shop. "
    "Counterintuitive finding: text descriptions alone beat visual embeddings."
)

tab1, tab2, tab3 = st.tabs(["🔍 Browse Query Set", "✍️ Text Search", "📈 Experiments"])

engine = load_engine()
gallery_df, query_df = load_data()
query_embs = load_query_embeddings()


# ── TAB 1: Browse Query Set ───────────────────────────────────────────────────
with tab1:
    st.markdown("Pick any query product and see what the search engine finds.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 3, 2])
    with col_ctrl1:
        categories = sorted(query_df["category2"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", ["All"] + categories, key="browse_cat")

    filtered_q = query_df if cat_filter == "All" else query_df[query_df["category2"] == cat_filter]

    with col_ctrl2:
        sample_options = filtered_q.head(50)[["item_id", "product_id", "color", "category2"]].copy()
        display_labels = [
            f"{row['item_id'].split('_')[-3][:20]} | {row['color']} {row['category2']}"
            for _, row in sample_options.iterrows()
        ]
        selected_idx = st.selectbox(
            "Select query product",
            range(len(display_labels)),
            format_func=lambda i: display_labels[i],
            key="browse_query",
        )

    with col_ctrl3:
        top_k = st.slider("Results to show", 3, 15, 8, key="browse_k")

    st.markdown("---")
    selected_row = sample_options.iloc[selected_idx]
    query_row = query_df[query_df["item_id"] == selected_row["item_id"]].iloc[0]
    query_original_idx = query_row.name

    # Get pre-computed query embeddings
    q_text_emb = query_embs["text"][query_original_idx]
    q_color_emb = query_embs["color"][query_original_idx]

    response = engine.search_by_precomputed(
        q_text=q_text_emb,
        q_color=q_color_emb,
        category=query_row["category2"],
        k=top_k,
    )

    # Layout: query on left, results on right
    col_q, col_r = st.columns([1, 3])

    with col_q:
        st.markdown("**Query Product**")
        q_img = get_image(query_row["item_id"])
        if q_img:
            st.image(q_img, use_column_width=True)
        st.markdown(f"""
        **Category:** {query_row['category2']}
        **Color:** {query_row.get('color', 'N/A')}
        **Latency:** {response.latency_ms:.1f}ms
        **Candidates:** {response.n_gallery_candidates}
        """)
        desc = str(query_row.get("description", ""))
        if desc:
            with st.expander("Description"):
                st.write(desc[:400] + ("..." if len(desc) > 400 else ""))

    with col_r:
        st.markdown(f"**Top-{top_k} Results** | Category: `{response.query_category}`")

        # Check if correct answer is in results
        correct_pid = query_row["product_id"]
        hit_rank = next(
            (r.rank for r in response.results if r.product_id == correct_pid), None
        )
        if hit_rank:
            st.success(f"Correct product found at rank #{hit_rank}")
        else:
            st.warning("Correct product not in top results")

        n_cols = 4
        result_rows = [response.results[i:i+n_cols] for i in range(0, len(response.results), n_cols)]

        for row_results in result_rows:
            row_cols = st.columns(n_cols)
            for col, result in zip(row_cols, row_results):
                with col:
                    is_correct = result.product_id == correct_pid
                    border_style = "border: 2px solid #27ae60;" if is_correct else ""
                    img = get_image(result.item_id)
                    if img:
                        st.image(img, use_column_width=True)
                    else:
                        st.markdown("🖼️ No image")

                    label = f"✅ #{result.rank}" if is_correct else f"#{result.rank}"
                    st.markdown(f"""
                    <div style="text-align:center; font-size:0.8em; {border_style} padding:4px; border-radius:4px;">
                        <b>{label}</b> {result.color}<br>
                        <span style="color:#555">Score: {result.combined_score:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Scores", expanded=False):
                        st.markdown(f"Text: `{result.text_score:.4f}`")
                        st.markdown(color_bar(result.text_score, 1.0, "#1f77b4"), unsafe_allow_html=True)
                        st.markdown(f"Color: `{result.color_score:.4f}`")
                        st.markdown(color_bar(result.color_score, 1.0, "#e67e22"), unsafe_allow_html=True)


# ── TAB 2: Text Search ────────────────────────────────────────────────────────
with tab2:
    st.markdown("Search for products using a text description (no image needed).")
    st.info(
        "This uses CLIP B/32 text embeddings — the same model that beats visual embeddings. "
        "Describe what you're looking for and the system finds the most similar products."
    )

    col_inp1, col_inp2, col_inp3 = st.columns([4, 2, 1])
    with col_inp1:
        search_text = st.text_area(
            "Product description",
            value="Black skinny jeans, zip fly, 79% cotton, slight stretch, slim fit",
            height=80,
            key="text_search_desc",
        )
    with col_inp2:
        search_cat = st.selectbox(
            "Category (optional)",
            ["Auto-detect"] + sorted(query_df["category2"].dropna().unique().tolist()),
            key="text_search_cat",
        )
        text_k = st.slider("Top K", 3, 15, 8, key="text_search_k")
    with col_inp3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_search = st.button("Search", type="primary", key="run_text_search")

    if run_search and search_text.strip():
        category = None if search_cat == "Auto-detect" else search_cat
        with st.spinner("Searching..."):
            response = engine.search_by_text(
                description=search_text, category=category, k=text_k
            )

        st.markdown(f"""
        **{len(response.results)} results** | Category: `{response.query_category}` |
        Candidates: {response.n_gallery_candidates} | Latency: **{response.latency_ms:.1f}ms**
        """)
        st.markdown("---")

        n_cols = 4
        result_rows = [response.results[i:i+n_cols] for i in range(0, len(response.results), n_cols)]
        for row_results in result_rows:
            row_cols = st.columns(n_cols)
            for col, result in zip(row_cols, row_results):
                with col:
                    img = get_image(result.item_id)
                    if img:
                        st.image(img, use_column_width=True)
                    st.markdown(f"""
                    <div style="text-align:center; font-size:0.82em; padding:4px;">
                        <b>#{result.rank}</b> {result.color} {result.category}<br>
                        <span style="color:#1f77b4">Score: {result.combined_score:.3f}</span>
                    </div>""", unsafe_allow_html=True)
                    with st.expander("Description", expanded=False):
                        st.write(result.description[:200] + "...")

    elif not run_search:
        st.markdown("""
        **Try these searches:**
        - `"Black skinny jeans, zip fly, 79% cotton, slight stretch"` + category: denim
        - `"Classic white polo shirt, short sleeve, slim fit"` + category: shirts
        - `"Navy blue crewneck sweatshirt, fleece lined, kangaroo pocket"` + category: sweatshirts
        - `"Light wash denim shorts, frayed hem, casual summer"` + category: shorts
        """)


# ── TAB 3: Experiments ────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 5-Phase Research Journey")
    st.markdown("Full experiment log — all approaches compared.")

    leaderboard_data = [
        ("ResNet50 baseline (Anthony P1)", 0.307, 0.559, 0.659, "P1"),
        ("EfficientNet-B0 (Mark P1)", 0.369, 0.607, 0.708, "P1"),
        ("ResNet50 + color rerank α=0.5 (M P1)", 0.405, 0.647, 0.757, "P1"),
        ("CLIP B/32 bare (Mark P2)", 0.480, 0.722, 0.807, "P2"),
        ("DINOv2 ViT-B/14 bare (Anthony P2)", 0.292, 0.589, 0.704, "P2"),
        ("CLIP B/32 + color α=0.5 (Mark P2)", 0.576, 0.789, 0.858, "P2"),
        ("CLIP L/14 bare (Anthony P3)", 0.553, 0.748, 0.805, "P3"),
        ("Text-only CLIP L/14 prompt (Anthony P3)", 0.602, 0.957, 1.000, "P3"),
        ("CLIP L/14 + color α=0.5 (Anthony P3)", 0.642, 0.808, 0.857, "P3"),
        ("CLIP B/32 + cat + color α=0.4 (M P3)", 0.683, 0.862, 0.913, "P3 ★"),
        ("Per-cat alpha oracle (Mark P4)", 0.695, 0.866, 0.911, "P4"),
        ("Two-stage visual→text K=10 (M P5)", 0.861, 0.887, 0.887, "P5"),
        ("Three-stage cat+CLIP+color+text (M P5)", 0.907, 0.944, 0.944, "P5"),
        ("Cat + color + text (ablation best, M P5)", 0.920, 0.990, 0.990, "P5 ★★"),
        ("Production pipeline (Phase 6)", 0.941, 1.000, 1.000, "P6 ★★★"),
    ]

    df_lb = pd.DataFrame(leaderboard_data, columns=["Model", "R@1", "R@5", "R@10", "Phase"])
    df_lb = df_lb.sort_values("R@1", ascending=False).reset_index(drop=True)
    df_lb.index = df_lb.index + 1

    def style_phase(val):
        colors = {"P1": "#e8e8e8", "P2": "#dbeafe", "P3": "#fef9c3",
                  "P4": "#ede9fe", "P5": "#dcfce7", "P6": "#bbf7d0"}
        for prefix, color in colors.items():
            if str(val).startswith(prefix):
                return f"background-color: {color}"
        return ""

    styled = (
        df_lb.style
        .format({"R@1": "{:.3f}", "R@5": "{:.3f}", "R@10": "{:.3f}"})
        .applymap(style_phase, subset=["Phase"])
        .bar(subset=["R@1"], color="#93c5fd", vmin=0, vmax=1.0)
    )
    st.dataframe(styled, use_container_width=True, height=500)

    st.markdown("---")
    st.markdown("### Ablation Study (Phase 5)")

    ablation_data = [
        ("Full: cat + color + CLIP visual + text", 0.9065, 0.0, "baseline"),
        ("Remove text rerank", 0.6699, -0.2366, "critical"),
        ("Remove color histogram", 0.8491, -0.0574, "moderate"),
        ("Remove category filter", 0.8345, -0.0721, "high"),
        ("Remove CLIP visual ★", 0.9200, +0.0135, "BETTER"),
    ]

    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        df_abl = pd.DataFrame(ablation_data, columns=["System", "R@1", "Delta", "Impact"])
        st.dataframe(df_abl, use_container_width=True, hide_index=True)

    with col_a2:
        st.markdown("""
        <div class="finding-box">
        <b>Counterintuitive finding #1</b>: Removing CLIP <em>visual</em> embeddings improves R@1 from 0.906 → 0.920.<br><br>
        <b>Why</b>: When product descriptions are paragraph-length and precise, CLIP text embeddings
        carry all the discriminative signal. Adding visual embeddings introduces noise from
        lighting variation, pose differences, and background clutter.<br><br>
        <b>Text reranking is the biggest component</b>: removing it drops R@1 by -23.7pp.
        This is the single most impactful finding from 5 phases of research.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Phase 4 Counterintuitive: 96D Color Hurts")
    st.markdown("""
    Adding more color bins (96D vs 48D) **catastrophically hurts** R@1 by -23.2pp.
    Reason: 16 bins/channel creates too-fine histograms that are sensitive to lighting variation.
    Coarser 8-bin histograms are more robust — less precision, more recall.
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8em'>"
    "Visual Product Search Engine | DeepFashion In-Shop | Mark Rodrigues × Anthony Rodrigues | 2026"
    "</div>",
    unsafe_allow_html=True,
)
