# Render.com 100% Free 24/7 Deployment Dockerfile

# --- Stage 1: Build C++ Backend Engine ---
FROM ubuntu:22.04 AS backend-builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY backend/ .
RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

# --- Stage 2: Build Next.js Frontend ---
FROM node:20-bullseye AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# --- Stage 3: Production 24/7 Runtime Image ---
FROM node:20-bullseye
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libopencv-core4.5d \
    libopencv-highgui4.5d \
    libopencv-imgproc4.5d \
    libopencv-videoio4.5d \
    libopencv-imgcodecs4.5d \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy C++ Backend
COPY --from=backend-builder /app/backend/build/yolo_tracker_backend ./backend/build/yolo_tracker_backend
COPY backend/models/ ./backend/models/

# Copy Frontend
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next
COPY --from=frontend-builder /app/frontend/public ./frontend/public
COPY --from=frontend-builder /app/frontend/node_modules ./frontend/node_modules
COPY --from=frontend-builder /app/frontend/package*.json ./frontend/
COPY --from=frontend-builder /app/frontend/next.config.ts ./frontend/

# Copy sample assets & test videos
COPY test/ ./test/
COPY sample_assets/ ./sample_assets/

RUN mkdir -p uploads logs

# Create Entrypoint Script for Render.com ($PORT / 10000)
RUN echo '#!/bin/bash\n\
set -e\n\
TARGET_PORT="${PORT:-10000}"\n\
echo "Starting C++ Inference Engine on port 8080..."\n\
./backend/build/yolo_tracker_backend &\n\
sleep 2\n\
echo "Starting Next.js Production App on port $TARGET_PORT..."\n\
cd frontend && PORT=$TARGET_PORT npm run start\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

EXPOSE 10000 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
