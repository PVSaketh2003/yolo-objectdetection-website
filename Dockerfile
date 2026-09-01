# Render.com 100% Free 24/7 Deployment Dockerfile

# --- Stage 1: Build C++ Backend Engine ---
FROM ubuntu:22.04 AS backend-builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    nlohmann-json3-dev \
    curl \
    ca-certificates \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Download & Install ONNX Runtime C++ v1.17.1 Linux x64
RUN curl -L -O https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz \
    && tar -xzf onnxruntime-linux-x64-1.17.1.tgz \
    && cp -r onnxruntime-linux-x64-1.17.1/include/* /usr/local/include/ \
    && cp -r onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/ \
    && rm -rf onnxruntime-linux-x64-1.17.1*

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

# --- Stage 3: Production 24/7 Runtime Image (Ubuntu 22.04 with Node.js 20 & Python 3) ---
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install OpenCV runtime, Node.js 20, Python 3, ffmpeg, curl
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    gnupg \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y nodejs libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir ultralytics opencv-python-headless fastapi uvicorn python-multipart onnxruntime

WORKDIR /app

# Copy C++ Backend & Libraries
COPY --from=backend-builder /usr/local/lib/libonnxruntime* /usr/local/lib/
COPY --from=backend-builder /app/backend/build/yolo_tracker_backend ./backend/build/yolo_tracker_backend
COPY backend/ ./backend/

# Refresh ldconfig for shared libs
RUN ldconfig

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
echo "Starting Ultralytics Python Inference Engine on port 8080..."\n\
python3 backend/app.py &\n\
sleep 2\n\
echo "Starting Next.js Production App on port $TARGET_PORT..."\n\
cd frontend && PORT=$TARGET_PORT npm run start\n\
' > /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

EXPOSE 10000 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
