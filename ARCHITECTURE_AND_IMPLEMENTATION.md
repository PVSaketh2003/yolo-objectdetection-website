# YOLO26 Tracker Studio — Complete Technical Architecture & Implementation Guide

## Executive Overview

**YOLO26 Tracker Studio** is a high-performance, real-time object tracking and target inspection platform built with a **100% Pure Python FastAPI** backend and a **Next.js 16 (Turbopack)** frontend. The application delivers real-time person detection, camera-driven face recognition, live target Region of Interest (ROI) cropping, and interactive telemetry at 30–60 FPS on Apple Silicon hardware.

```mermaid
graph TD
    A[Client Browser / Next.js UI] -->|HTTP / API Requests| B[FastAPI Backend Server :8080]
    B --> C[Session Manager]
    C --> D[OpenCV Video Capture Engine]
    D --> E[Ultralytics YOLO26 ONNX - Person Detection]
    E --> F[OpenCV YuNet ONNX - Dual-Crop Face Scan]
    F --> G[Target ROI Crop Engine & Main Frame Annotator]
    G -->|MJPEG Video Stream /api/stream| A
    G -->|16-FPS JPEG Polling /api/crop_image| A
    G -->|JSON Telemetry /api/status| A
```

---

## 1. System Architecture

The project is structured into a decoupled frontend-backend architecture:

| Component | Stack / Technologies | Key Responsibilities |
| :--- | :--- | :--- |
| **Backend Engine** | Python 3.11+, FastAPI, Uvicorn, OpenCV, PyTorch, Ultralytics YOLO, ONNX Runtime | Video decoding, real-time YOLO object detection, YuNet face scanning, frame annotation, MJPEG streaming, session isolation. |
| **Frontend UI** | Next.js 16 (Turbopack), React 19, TypeScript, Vanilla Tailwind/CSS, Lucide Icons | Responsive interactive video dashboard, real-time target inspector, strictness controls, live diagnostic logs console. |
| **Hardware Acceleration** | Apple Silicon Metal GPU (`mps`), CoreML Execution Provider | Hardware-accelerated neural network inference on M1/M2/M3/M4 Apple Silicon. |

---

## 2. Neural Model Pipelines & Person Detection

### Ultralytics YOLO ONNX Pipeline (`yolo26s.onnx`)

- **Class Filtering**: Person class ONLY (`class_id == 0`). Non-person classes (cars, chairs, etc.) are strictly filtered out at the inference step.
- **Model Warmup**: To eliminate lazy ONNX CoreML graph compilation delays on the first incoming request, the backend performs a **startup dummy frame warmup** (`640x640` tensor pass).
- **Execution Mode**: Direct `model.predict(source=infer_rgb, conf=0.20, iou=0.45)` operating on RGB matrices resized to $640 \times 640$.

```python
results = model.predict(
    source=infer_rgb,
    conf=session.conf_threshold, # Default: 0.20
    iou=session.nms_threshold,   # Default: 0.45
    verbose=False,
    device="mps"                 # Apple Silicon Metal GPU
)
```

---

## 3. Face Detection & Target ROI Crop Engine

### OpenCV YuNet ONNX Face Detector (`face_detection_yunet_2023mar.onnx`)

Face detection is executed continuously across active tracked person crops using the OpenCV YuNet face detector.

#### Dual-Region Crop Scan Strategy
To handle surveillance and hallway camera angles where full-body crops ($108 \times 198\text{ px}$) distort aspect ratios when scaled to neural inputs, YuNet performs a **dual-pass scan**:
1. **Full-Body Crop Scan**: Scans the entire person bounding box matrix.
2. **Upper Head Crop Scan**: Extracts and scans the top **55% upper region** of the person crop where faces are upright and undistorted.

```python
# Dual-Region Crop Scanning for Maximum Face Detection Accuracy
crops_to_check = [(person_crop, "full", 0)]
if c_h >= 30 and c_w >= 20:
    head_h = max(20, int(c_h * 0.55))
    head_crop = person_crop[0:head_h, 0:c_w].copy()
    crops_to_check.append((head_crop, "head", 0))
```

#### Face-Driven Pink Bounding Box Rule
- **Person Box Color Rule**: A tracked person's bounding box turns **BRIGHT PINK / MAGENTA (`#DD00FF` / `RGB(220, 0, 255)`) ONLY when a face is detected on that person by the camera**.
- **Cyan Face Box**: Inside the pink box on the main video frame and on the right target inspection crop, a **BRIGHT CYAN** rectangle with label `Face XX%` (e.g. `Face 87%`) is drawn directly around the face.
- **Coordinate Translation Mapping**: Face bounding box coordinates detected inside person crops $(f_x, f_y, f_w, f_h)$ are translated back to global main frame space:
  $$\text{main\_fx}_1 = b_x + f_x, \quad \text{main\_fy}_1 = b_y + f_y$$

#### 1.5-Second Target ROI Persistence Memory
To prevent target selection dropouts or black screen flickering when a person momentarily turns their head or blinks, the backend maintains a **1.5-second persistence buffer** (`session.last_face_time`).

---

## 4. Concurrency, Locking & Low-Latency Streaming

### Lock-Free Video Processing Loop (`process_session_frame`)

To prevent Uvicorn worker threads from stalling during `/api/status` or `/api/source` API calls, thread locks (`session.lock`) are acquired strictly for microsecond primitive reads/writes. Heavy operations (OpenCV file decoding, YOLO inference, and YuNet detection) execute **lock-free**:

```python
# Lock acquired ONLY for state snapshot
with session.lock:
    cap_obj = session.cap
    src_fps = session.source_fps
    last_tick = session.last_frame_tick

# Lock-Free Frame Decoding & Neural Inference
ret, frame = cap_obj.read()
```

### High-Speed Zero-Proxy-Buffer Target ROI Inspector (`/api/crop_image`)

Rather than streaming `multipart/x-mixed-replace` (which gets buffered by Next.js dev proxies), the right inspector panel uses rapid 16-FPS single JPEG polling (`/api/crop_image?sid=...&t=...`), bypassing proxy buffers entirely and delivering zero-latency updates.

---

## 5. Process Lifecycle & Robust Startup (`dev.sh`)

The system uses `./dev.sh` to manage backend and frontend process lifecycles:

1. **Port Cleanup**: Kills stale processes bound to ports `8080` (FastAPI) and `3000` (Next.js) using `lsof -ti:8080 | xargs kill -9`.
2. **Signal Trapping**: Intercepts `SIGINT`, `SIGTERM`, and `EXIT` to clean up child background threads gracefully.
3. **HTTP Health Check Polling**: Frontend launches *only* after FastAPI responds to HTTP health checks (`curl -s http://127.0.0.1:8080/api/status`), avoiding `ECONNREFUSED` proxy connection errors.

---

## 6. Smart File Path Resolution (`resolve_video_path`)

To handle user video uploads and file paths with spaces (e.g. `test/ videos_30fps.mp4`), `resolve_video_path` automatically sanitizes whitespace and matches files across `test/` and `uploads/` directories.

---

## 7. Summary of API Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `GET /api/status` | GET | Returns real-time telemetry (FPS, inference latency, active tracks, face state). |
| `GET /api/stream` | GET | Streams live annotated MJPEG video (`multipart/x-mixed-replace`). |
| `GET /api/crop_image` | GET | Returns latest high-resolution JPEG crop of the selected target face/ROI. |
| `POST /api/source` | POST | Dynamic video source switcher (sample videos, user uploads, webcam, RTSP). |
| `POST /api/upload_chunk` | POST | Multi-threaded parallel chunk uploader (4MB chunks). |
| `POST /api/settings` | POST | Adjusts detection strictness (`conf_threshold`, `nms_threshold`). |
| `GET /api/logs` | GET | Retrieves system diagnostic event logs. |
