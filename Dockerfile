# Visual Product Search Engine — production image
#
# Two-stage build keeps the runtime image lean: the build stage compiles wheels
# (FAISS, torch) once, and the runtime stage installs them. CLIP weights are
# NOT baked in — mount them at /app/.cache/huggingface or let the first
# request download (~1.5 GB, slow on cold start). For a fully self-contained
# image, uncomment the warm-up RUN at the bottom.
#
# Build:    docker build -t vps-api .
# Run:      docker run -p 8000:8000 -v $(pwd)/models:/app/models vps-api
# Health:   curl http://localhost:8000/health

FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="visual-product-search"
LABEL org.opencontainers.image.description="CLIP+color+spatial fashion retrieval API (R@1=72.9%)"
LABEL org.opencontainers.image.source="https://github.com/anthonyrodrigues443/Visual-Product-Search-Engine"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

COPY --from=builder /root/.local /home/app/.local
COPY --chown=app:app src/ ./src/
COPY --chown=app:app config/ ./config/
COPY --chown=app:app api.py ./
COPY --chown=app:app app.py ./

USER app
ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=2 \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# Optional: pre-download CLIP weights at build time (adds ~1.5 GB to the image
# but eliminates cold-start latency). Uncomment to enable.
# RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
