# F.R.I.D.A.Y. — small, unprivileged image. No build tooling in the final layer.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRIDAY_WORKSPACE=/workspace \
    FRIDAY_DATA_DIR=/data \
    FRIDAY_HOST=0.0.0.0 \
    FRIDAY_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY friday ./friday
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 friday \
    && mkdir -p /workspace /data \
    && chown -R friday:friday /workspace /data /app

USER friday
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["friday", "serve"]
