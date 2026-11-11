# syntax=docker/dockerfile:1
#
# Two things this image is careful about:
#   1. The core is stdlib-only; the image installs `[serve]` because a headless
#      gateway with no HUD is a worse default in a container, not a leaner one.
#   2. /data is the ONLY writable state. The build runs as a non-root user and
#      FRIDAY_ROOT points at the volume, so memory/facts/*.md and artifacts/ both
#      survive a container replacement.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRIDAY_ROOT=/data

WORKDIR /app

# Package metadata first, then source: a code change rebuilds the install layer,
# a README change does not invalidate wheel caches.
COPY pyproject.toml README.md LICENSE ./
COPY friday ./friday

RUN pip install --no-cache-dir ".[serve]" \
    && useradd --create-home --uid 10001 friday \
    && mkdir -p /data \
    && chown -R friday:friday /data /app

USER friday
VOLUME ["/data"]
EXPOSE 8642

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import sys,urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8642/api/health', timeout=4).status == 200 else 1)"

# 0.0.0.0 is correct *in a container*: the port mapping is the boundary, and the
# bearer token is still required on every data route. The CLI prints a warning
# naming exactly what that exposes.
CMD ["friday", "serve", "--host", "0.0.0.0", "--port", "8642"]
