# FinDoc Guard - CPU image (a CUDA image is added in Phase 4)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY findoc ./findoc
RUN pip install .

RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["findoc"]
CMD ["--help"]
