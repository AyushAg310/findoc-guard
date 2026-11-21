# FinDoc Guard: one image for the CLI, the API and the workers (CPU).
#   docker build -t findoc-guard .                          # API + pipeline, no OCR
#   docker build --build-arg EXTRAS=api,ocr -t findoc-guard .   # + EasyOCR (CPU PyTorch, ~1 GB more)

# --- 1. the review UI (React + Vite) ---------------------------------------------------
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- 2. Python runtime -----------------------------------------------------------------
FROM python:3.11-slim AS runtime
ARG EXTRAS=api
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY findoc ./findoc
COPY app ./app
RUN pip install ".[${EXTRAS}]"
COPY --from=ui /ui/dist ./frontend/dist

RUN useradd --create-home appuser && mkdir -p /data /app/models && chown appuser /data /app/models
USER appuser
ENV FINDOC_API_STORAGE_DIR=/data/storage

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)" || exit 1
ENTRYPOINT ["findoc"]
CMD ["--help"]
