#!/bin/bash
set -e

echo "[STARTUP] Starting FastAPI backend (v2 Soft Voting & Boost) on http://127.0.0.1:8000..."
# Catatan: ganti ke 'app_yoloworld:app' jika ingin kembali ke versi v1
uvicorn app_yoloworld_v1_archive:app --host 127.0.0.1 --port 8000 &

echo "[STARTUP] Starting Streamlit UI demo on http://127.0.0.1:8501..."
streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false &

echo "[STARTUP] Starting Nginx on port 7860..."
exec nginx -c /app/nginx.conf -g "daemon off;"
