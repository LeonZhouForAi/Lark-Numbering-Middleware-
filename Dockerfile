FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils tesseract-ocr tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install . \
    && groupadd --system --gid 10001 rag \
    && useradd --system --uid 10001 --gid rag --home-dir /app rag \
    && mkdir -p /app/data /app/documents \
    && chown -R rag:rag /app

COPY scripts ./scripts
RUN chown -R rag:rag /app/scripts

USER rag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uvicorn", "feishu_rag.web:app", "--host", "0.0.0.0", "--port", "8000"]
