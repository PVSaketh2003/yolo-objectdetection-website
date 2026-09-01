#!/bin/bash
# Robust Local Development Launcher with Signal Trap & Port Ready Wait

cleanup() {
    echo -e "\n🛑 Shutting down backend and frontend..."
    lsof -ti:8080 | xargs kill -9 2>/dev/null
    lsof -ti:3000 | xargs kill -9 2>/dev/null
    exit 0
}

trap cleanup INT TERM EXIT

echo "🧹 Freeing ports 8080 and 3000..."
lsof -ti:8080 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null

echo "🚀 Starting Python Backend..."
venv/bin/python backend/app.py &

echo "⏳ Waiting for backend to initialize models and open port 8080..."
while ! curl -s http://127.0.0.1:8080/api/status >/dev/null; do
    sleep 0.3
done

echo "✅ Backend is READY! Starting Next.js Frontend..."
(cd frontend && npm run dev)
