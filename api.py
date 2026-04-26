"""FastAPI service for the Visual Product Search Engine.

Production deployment alternative to the Streamlit demo. Loads the FAISS
index once at startup and serves search requests over HTTP — designed to be
the same engine the e-commerce site would call from its product page.

Anthony's Phase 7 added the Streamlit UI (good for human exploration). This
FastAPI surface is the machine-callable equivalent: image upload in,
ranked JSON out, suitable for Docker + load balancer.

Endpoints:
    GET  /health        — liveness/readiness probe
    GET  /info          — model + index metadata (dim, gallery size, config)
    GET  /categories    — list of categories present in the index
    POST /search        — multipart image upload, returns top-K matches
    GET  /              — redirect to /docs (FastAPI auto-docs)

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

The first request triggers CLIP weight download (~1.5GB) — preload by
hitting /health after startup, or warm the cache in your container build.
"""

from __future__ import annotations

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vps-api")

PROJECT_ROOT = Path(__file__).parent
INDEX_DIR_DEFAULT = PROJECT_ROOT / "models"

_engine = None  # lazy-loaded VisualSearchEngine singleton


def _get_engine():
    """Lazy initialise so importing this module (e.g. for tests) is cheap."""
    global _engine
    if _engine is None:
        from src.predict import VisualSearchEngine

        log.info("Loading VisualSearchEngine from %s", INDEX_DIR_DEFAULT)
        t0 = time.perf_counter()
        _engine = VisualSearchEngine(index_dir=INDEX_DIR_DEFAULT)
        log.info("Engine ready in %.1fs (%d gallery items, %dD features)",
                 time.perf_counter() - t0,
                 len(_engine.product_ids),
                 _engine.gallery_features.shape[1])
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    if (INDEX_DIR_DEFAULT / "gallery.index").exists():
        try:
            _get_engine()
        except Exception as e:
            log.warning("Eager engine load failed (will retry on first request): %s", e)
    else:
        log.warning("No index found at %s — run `python -m src.train` first", INDEX_DIR_DEFAULT)
    yield


app = FastAPI(
    title="Visual Product Search Engine",
    description=(
        "Image-based fashion product retrieval. CLIP ViT-L/14 + color "
        "histograms + spatial color grid + category-filtered FAISS. "
        "Champion R@1=72.9% on DeepFashion In-Shop."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    rank: int = Field(..., description="1-indexed position in result list")
    product_id: str
    category: str
    score: float = Field(..., description="Cosine similarity, higher = more similar")
    gallery_index: int


class SearchResponse(BaseModel):
    query_filename: str
    n_results: int
    results: list[SearchHit]
    latency_ms: float = Field(..., description="End-to-end server time including feature extraction")
    category_filter_applied: bool


class HealthResponse(BaseModel):
    status: str
    engine_loaded: bool
    index_path: str


class InfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_backbone: str
    feature_dim: int
    n_gallery: int
    config: dict
    fusion_weights: dict


class CategoriesResponse(BaseModel):
    categories: list[str]
    counts: dict[str, int]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        engine_loaded=_engine is not None,
        index_path=str(INDEX_DIR_DEFAULT),
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    eng = _get_engine()
    return InfoResponse(
        model_backbone=eng.cfg["model"]["backbone"],
        feature_dim=int(eng.gallery_features.shape[1]),
        n_gallery=len(eng.product_ids),
        config=eng.cfg,
        fusion_weights=eng.cfg["fusion"],
    )


@app.get("/categories", response_model=CategoriesResponse)
async def categories():
    eng = _get_engine()
    counts: dict[str, int] = {}
    for c in eng.categories:
        counts[c] = counts.get(c, 0) + 1
    return CategoriesResponse(categories=sorted(counts.keys()), counts=counts)


@app.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(..., description="Product image (jpg/png)"),
    top_k: int = Form(10, ge=1, le=100, description="Number of results to return"),
    category: Optional[str] = Form(
        None,
        description=(
            "Restrict search to this clothing category — pass when known to "
            "get +6.9pp R@1 (Phase 6 finding). Use /categories to list valid values."
        ),
    ),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image/* content-type, got {image.content_type!r}")

    raw = await image.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not decode image: {e}")

    eng = _get_engine()
    t0 = time.perf_counter()
    try:
        results = eng.search(pil, top_k=top_k, category=category)
    except Exception as e:
        log.exception("Search failed")
        raise HTTPException(500, f"Search failed: {e}")
    latency_ms = (time.perf_counter() - t0) * 1000

    return SearchResponse(
        query_filename=image.filename or "upload",
        n_results=len(results),
        results=[SearchHit(**r) for r in results],
        latency_ms=round(latency_ms, 1),
        category_filter_applied=category is not None,
    )


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Visual Product Search API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)
