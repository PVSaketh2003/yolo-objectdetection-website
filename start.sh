#!/bin/bash
set -e

echo "=========================================================="
echo "   🚀 Starting YOLO26 Interactive Object Tracker Studio   "
echo "=========================================================="

# 0. Clean any stale processes on ports 8080 and 3000
echo "[Launcher] Cleaning ports 8080 and 3000..."
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
pkill -9 -f yolo_tracker_backend 2>/dev/null || true
pkill -9 -f "next-server" 2>/dev/null || true
pkill -9 -f "cloudflared" 2>/dev/null || true
pkill -9 -f "localtunnel" 2>/dev/null || true
sleep 1

# 1. Build C++ Backend if needed
if [ ! -f "backend/build/yolo_tracker_backend" ]; then
    echo "[Launcher] Building C++ backend..."
    mkdir -p backend/build
    cd backend/build
    cmake ..
    make -j4
    cd ../..
fi

# 2. Start C++ Backend Server
echo "[Launcher] Launching C++ Inference & Tracking Engine on port 8080..."
./backend/build/yolo_tracker_backend &
BACKEND_PID=$!

# 3. Build Production Next.js Bundle if missing
if [ ! -d "frontend/.next" ]; then
    echo "[Launcher] Building Production Next.js Frontend Bundle..."
    cd frontend && npm run build && cd ..
fi

# 4. Start Production Next.js Server
echo "[Launcher] Starting Production Next.js Frontend App on http://localhost:3000..."
cd frontend
npm run start &
FRONTEND_PID=$!
cd ..

# 5. Launch Localtunnel
echo "[Launcher] Creating Custom Name Public Link via Localtunnel..."
npx --yes localtunnel --port 3000 --subdomain sairamsaketh-yolo-studio > public_url.log 2>&1 &
LOCALTUNNEL_PID=$!

trap "echo '[Launcher] Stopping services...'; kill $BACKEND_PID $FRONTEND_PID $LOCALTUNNEL_PID 2>/dev/null || true; pkill -9 -f yolo_tracker_backend 2>/dev/null || true" EXIT

# Wait a moment then fetch public IP
sleep 4
PUBLIC_IP=$(curl -s https://loca.lt/mytunnelpassword || echo "223.185.43.198")

echo "=========================================================="
echo " 🌐 YOUR OFFICIAL PUBLIC WORLDWIDE LINK:"
echo " 👉 https://sairamsaketh-yolo-studio.loca.lt"
echo " 🔑 Access Password (if prompted): $PUBLIC_IP"
echo "=========================================================="

open http://localhost:3000 || true

wait $FRONTEND_PID $BACKEND_PID $LOCALTUNNEL_PID
