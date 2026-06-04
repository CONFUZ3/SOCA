# ---------- Stage 1: frontend build ----------
FROM node:22-bookworm-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---------- Stage 2: runtime ----------
# Python + Node co-installed so supervisord can run both FastAPI and
# the Next.js server in one container, behind a shared nginx on :8080.
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NEXT_TELEMETRY_DISABLED=1 \
    SOCA_BACKEND_URL=http://127.0.0.1:8000 \
    NODE_ENV=production

# System deps: nginx for fan-out, supervisord for process management,
# Node 22 for `next start`, plus the GEOS/PROJ toolchain GeoPandas needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg ca-certificates \
        nginx supervisor \
        build-essential \
        libgeos-dev libproj-dev libgdal-dev gdal-bin \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Python sources (backend wraps these, no duplication).
COPY agent/ ./agent/
COPY solvers/ ./solvers/
COPY utils/ ./utils/
COPY config/ ./config/
COPY backend/ ./backend/

# Next.js standalone output ships its own minimal node_modules.
COPY --from=frontend-build /app/frontend/.next/standalone ./frontend/
COPY --from=frontend-build /app/frontend/.next/static ./frontend/.next/static
COPY --from=frontend-build /app/frontend/public ./frontend/public

# Orchestration
COPY deploy/supervisord.conf /etc/supervisor/conf.d/soca.conf
COPY deploy/nginx.conf /etc/nginx/nginx.conf

EXPOSE 8080

# Healthcheck hits the nginx edge, which must both proxy to FastAPI and
# serve the Next.js app for a healthy container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8080/api/health || exit 1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/soca.conf"]
