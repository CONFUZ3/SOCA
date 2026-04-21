# SOCA container deployment

Single Docker container running the React (Next.js 16) frontend and the
FastAPI backend behind a tiny nginx edge. Supervisord manages the three
long-running processes. The only externally-exposed port is **8080**.

```
┌─────────────────── container ───────────────────┐
│                                                 │
│  nginx :8080 ───┬───► uvicorn  (FastAPI :8000)  │
│                 │                               │
│                 └───► node     (Next.js :3000)  │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Build & run

```bash
# From the repo root
docker build -t soca:latest .

docker run --rm -p 8080:8080 \
  -e GEMINI_API_KEY=sk-... \
  soca:latest
```

Open `http://localhost:8080`.

## Required env vars

| var | purpose |
| --- | --- |
| `GEMINI_API_KEY` | Google Gemini API key used by the ADK agent. |

## Optional env vars

| var | default | notes |
| --- | --- | --- |
| `SOCA_LOG_LEVEL` | `INFO` | passes through to Python `logging`. |
| `SOCA_COOKIE_SECURE` | `0` | set to `1` behind HTTPS. |
| `SOCA_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | only relevant when the browser hits the API cross-origin — inside the container all traffic is same-origin through nginx. |

## SSE

`/api/events/stream` and `/api/chat/stream` are long-lived `text/event-stream`
endpoints. The nginx config disables `proxy_buffering` on those two paths
so tokens reach the client as they arrive. Reverse-proxying the container
with another layer? Make sure that layer also disables buffering for
those paths (e.g. Cloudflare: bypass cache + "no compress") or the live
feed will feel laggy.
