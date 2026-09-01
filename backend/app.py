import os
import time
import glob
import math
import shutil
import threading
import cv2
import numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, Request, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# Ultralytics Import
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

app = FastAPI(title="YOLO Ultralytics Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Logs List for Diagnostic Console
SYSTEM_LOGS: List[Dict[str, str]] = []
LOG_LOCK = threading.Lock()

def add_log(level: str, module: str, message: str):
    timestamp = time.strftime("%H:%M:%S")
    with LOG_LOCK:
        SYSTEM_LOGS.append({
            "timestamp": timestamp,
            "level": level,
            "module": module,
            "message": message
        })
        if len(SYSTEM_LOGS) > 200:
            SYSTEM_LOGS.pop(0)
    print(f"[{timestamp}] [{level}] [{module}] {message}")

add_log("INFO", "Main", "Initializing Python Ultralytics Prediction Backend...")

# Load Ultralytics Model
MODEL_PATHS = [
    "backend/models/yolo26s.onnx",
    "yolo11s.pt",
    "yolov8s.pt",
    "backend/models/yolov8s.onnx"
]

model = None
active_model_path = ""

for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            add_log("INFO", "ModelLoader", f"Attempting to load model: {path}")
            if ULTRALYTICS_AVAILABLE:
                model = YOLO(path, task="detect")
                active_model_path = path
                add_log("INFO", "ModelLoader", f"Successfully loaded Ultralytics model: {path} 🚀")
                break
        except Exception as e:
            add_log("WARN", "ModelLoader", f"Failed loading {path} with Ultralytics: {e}")

if model is None and ULTRALYTICS_AVAILABLE:
    try:
        add_log("INFO", "ModelLoader", "Downloading default yolo11s.pt model...")
        model = YOLO("yolo11s.pt")
        active_model_path = "yolo11s.pt"
        add_log("INFO", "ModelLoader", "Loaded fallback yolo11s.pt 🚀")
    except Exception as e:
        add_log("CRITICAL", "ModelLoader", f"Failed downloading default model: {e}")

# Session Manager
class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.video_source = "test/15690486_1920_1080_25fps.mp4"
        self.source_type = "test_video"
        self.conf_threshold = 0.35
        self.nms_threshold = 0.45
        self.is_playing = True
        self.selected_track_id = -1
        self.tiling_mode = 0
        self.fps = 30.0
        self.inference_ms = 12.0
        self.current_tracks: List[Dict[str, Any]] = []
        self.current_frame_jpeg: bytes = b""
        self.cap = None
        self.lock = threading.Lock()
        self.running = True

sessions: Dict[str, SessionState] = {}
session_lock = threading.Lock()

def get_session(session_id: str) -> SessionState:
    with session_lock:
        if session_id not in sessions:
            add_log("INFO", "SessionManager", f"Created new isolated session: {session_id}")
            sessions[session_id] = SessionState(session_id)
        return sessions[session_id]

# Process Session Loop Thread
def session_worker_loop():
    while True:
        with session_lock:
            active_sessions = list(sessions.values())

        for session in active_sessions:
            try:
                process_session_frame(session)
            except Exception as e:
                add_log("ERROR", "WorkerLoop", f"Error in session {session.session_id}: {e}")

        time.sleep(0.02) # ~50 FPS loop pace

def process_session_frame(session: SessionState):
    with session.lock:
        if not session.is_playing:
            return

        # Open video capture if needed
        if session.cap is None or not session.cap.isOpened():
            src = session.video_source
            if session.source_type == "webcam":
                try:
                    src = int(src)
                except ValueError:
                    src = 0
            
            add_log("INFO", "VideoCapture", f"Opening video source: {src} (Type: {session.source_type})")
            session.cap = cv2.VideoCapture(src)
            if not session.cap.isOpened():
                # Fallback to test video if uploaded video failed to open
                fallback_path = "test/15690486_1920_1080_25fps.mp4"
                if os.path.exists(fallback_path) and src != fallback_path:
                    add_log("WARN", "VideoCapture", f"Could not open {src}, falling back to {fallback_path}")
                    session.cap = cv2.VideoCapture(fallback_path)

        if session.cap is None or not session.cap.isOpened():
            return

        ret, frame = session.cap.read()
        if not ret or frame is None or frame.size == 0:
            # EOF reached -> Loop video
            session.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = session.cap.read()
            if not ret or frame is None:
                return

        start_time = time.time()
        tracks_list = []
        annotated_frame = frame.copy()

        if model is not None:
            # Run Ultralytics predict / track
            try:
                results = model.predict(
                    source=frame,
                    conf=session.conf_threshold,
                    iou=session.nms_threshold,
                    verbose=False,
                    device="cpu"
                )
                
                if results and len(results) > 0:
                    res = results[0]
                    # Generate beautiful Ultralytics plotted frame
                    annotated_frame = res.plot(conf=True, line_width=2)

                    # Extract metadata for UI
                    if res.boxes is not None:
                        boxes = res.boxes.xyxy.cpu().numpy()
                        confs = res.boxes.conf.cpu().numpy()
                        classes = res.boxes.cls.cpu().numpy()
                        
                        names = res.names if hasattr(res, 'names') else {0: 'person', 2: 'car'}

                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box
                            conf = float(confs[i])
                            cls_id = int(classes[i])
                            label = names.get(cls_id, f"obj_{cls_id}")

                            w = x2 - x1
                            h = y2 - y1

                            tracks_list.append({
                                "track_id": i + 1,
                                "label": label,
                                "confidence": round(conf, 2),
                                "class_id": cls_id,
                                "box": {
                                    "x": round(float(x1)),
                                    "y": round(float(y1)),
                                    "w": round(float(w)),
                                    "h": round(float(h))
                                },
                                "velocity": {"dx": 0.0, "dy": 0.0},
                                "age": 1
                            })
            except Exception as e:
                add_log("ERROR", "Inference", f"Ultralytics predict error: {e}")

        inf_time_ms = (time.time() - start_time) * 1000.0
        session.inference_ms = round(inf_time_ms, 2)
        session.fps = round(1000.0 / max(inf_time_ms, 1.0), 1)
        session.current_tracks = tracks_list

        # Encode frame to JPEG for MJPEG stream
        ret_jpg, jpeg_buf = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret_jpg:
            session.current_frame_jpeg = jpeg_buf.tobytes()

# Start background worker thread
worker_thread = threading.Thread(target=session_worker_loop, daemon=True)
worker_thread.start()

# --- API ENDPOINTS ---

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return Response(status_code=200)

@app.get("/api/status")
async def get_status(request: Request):
    sid = request.headers.get("X-Session-ID", "default_session")
    session = get_session(sid)
    with session.lock:
        return {
            "fps": session.fps,
            "inference_ms": session.inference_ms,
            "selected_track_id": session.selected_track_id,
            "source_type": session.source_type,
            "video_source": session.video_source,
            "conf_threshold": session.conf_threshold,
            "nms_threshold": session.nms_threshold,
            "tiling_mode": session.tiling_mode,
            "is_playing": session.is_playing,
            "tracks": session.current_tracks,
            "active_model": active_model_path
        }

@app.post("/api/source")
async def set_source(request: Request):
    sid = request.headers.get("X-Session-ID", "default_session")
    session = get_session(sid)
    data = await request.json()
    video_source = data.get("video_source", "test/15690486_1920_1080_25fps.mp4")
    source_type = data.get("source_type", "sample")

    with session.lock:
        if session.cap is not None:
            session.cap.release()
            session.cap = None
        session.video_source = video_source
        session.source_type = source_type
        session.selected_track_id = -1

    add_log("INFO", "API", f"Changed video source for {sid} -> {video_source} ({source_type})")
    return {"status": "ok", "video_source": video_source}

@app.post("/api/settings")
async def update_settings(request: Request):
    sid = request.headers.get("X-Session-ID", "default_session")
    session = get_session(sid)
    data = await request.json()
    with session.lock:
        if "conf_threshold" in data:
            session.conf_threshold = float(data["conf_threshold"])
        if "nms_threshold" in data:
            session.nms_threshold = float(data["nms_threshold"])
        if "tiling_mode" in data:
            session.tiling_mode = int(data["tiling_mode"])
        if "is_playing" in data:
            session.is_playing = bool(data["is_playing"])
    return {"status": "ok"}

@app.post("/api/upload_chunk")
async def upload_chunk(
    request: Request,
    chunk: UploadFile = File(...),
    upload_id: str = Form(...),
    filename: str = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...)
):
    sid = request.headers.get("X-Session-ID", "default_session")
    session = get_session(sid)

    upload_dir = os.path.join("uploads", sid, "chunks")
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize filename
    safe_filename = "".join([c if c.isalnum() or c in "._-" else "_" for c in filename])
    chunk_path = os.path.join(upload_dir, f"{upload_id}_part_{chunk_index}")

    content = await chunk.read()
    with open(chunk_path, "wb") as f:
        f.write(content)

    # Check if all chunks have arrived
    all_ready = True
    for i in range(total_chunks):
        p = os.path.join(upload_dir, f"{upload_id}_part_{i}")
        if not os.path.exists(p):
            all_ready = False
            break

    if all_ready:
        final_dir = os.path.join("uploads", sid)
        os.makedirs(final_dir, exist_ok=True)
        final_path = os.path.join(final_dir, f"user_video_{int(time.time())}_{safe_filename}")

        with open(final_path, "wb") as outfile:
            for i in range(total_chunks):
                p = os.path.join(upload_dir, f"{upload_id}_part_{i}")
                with open(p, "rb") as infile:
                    outfile.write(infile.read())
                try:
                    os.remove(p)
                except Exception:
                    pass

        add_log("INFO", "UploadAPI", f"Uploaded user video assembled -> {final_path}")

        # Update session video source to play user video immediately
        with session.lock:
            if session.cap is not None:
                session.cap.release()
                session.cap = None
            session.video_source = final_path
            session.source_type = "file"
            session.selected_track_id = -1

        return {"status": "complete", "video_source": final_path}
    else:
        return {"status": "chunk_received", "chunk_index": chunk_index}

@app.get("/api/stream")
async def video_stream(request: Request):
    sid = request.headers.get("X-Session-ID", request.query_params.get("sid", "default_session"))
    session = get_session(sid)

    def frame_generator():
        while True:
            jpeg_bytes = session.current_frame_jpeg
            if jpeg_bytes:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" +
                       jpeg_bytes + b"\r\n")
            time.sleep(0.033) # 30 FPS stream

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/api/logs")
async def get_logs():
    with LOG_LOCK:
        return {"logs": list(SYSTEM_LOGS)}

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    add_log("INFO", "Main", "Starting FastAPI Server on 0.0.0.0:8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
