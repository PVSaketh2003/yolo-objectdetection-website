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

# Optimize OpenMP and Multi-Threading for CPU Inference
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Ultralytics Import
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

app = FastAPI(title="YOLO Ultralytics High-FPS Engine")

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

add_log("INFO", "Main", "Initializing Python Ultralytics High-FPS Prediction Backend...")

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
                # Warm up model to pre-compile ONNX / CoreML Graph BEFORE Uvicorn starts
                try:
                    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                    model.predict(dummy_img, verbose=False)
                    add_log("INFO", "ModelLoader", "Model compilation warmup complete 🚀")
                except Exception:
                    pass
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

# Initialize OpenCV YuNet Face Detector Model
YUNET_MODEL_PATH = "backend/models/face_detection_yunet_2023mar.onnx"
yunet_detector = None
if os.path.exists(YUNET_MODEL_PATH):
    try:
        yunet_detector = cv2.FaceDetectorYN.create(YUNET_MODEL_PATH, "", (300, 300), 0.25, 0.30, 5000)
        add_log("INFO", "YuNetLoader", "Successfully loaded OpenCV YuNet ONNX Face Detector 🚀")
    except Exception as e:
        add_log("WARN", "YuNetLoader", f"Failed loading YuNet: {e}")

# Fallback Haar Cascade Face Detector
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Mac M4 Hardware Acceleration (Apple Silicon Metal GPU MPS)
import torch
if torch.backends.mps.is_available():
    INFERENCE_DEVICE = "mps"
    add_log("INFO", "HardwareAccel", "Apple Silicon Mac M4 Metal GPU (MPS) Acceleration ENABLED 🚀")
else:
    INFERENCE_DEVICE = "cpu"
    add_log("INFO", "HardwareAccel", "CPU Multi-Threaded Parallel Execution ENABLED ⚡")

# Session Manager
class SessionState:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.video_source = "test/15690486_1920_1080_25fps.mp4"
        self.source_type = "test_video"
        self.conf_threshold = 0.20 # Reduced Person Detection Confidence threshold = 0.20
        self.nms_threshold = 0.45  # NMS IoU threshold = 0.45
        self.is_playing = True
        self.selected_track_id = -1
        self.tiling_mode = 0
        self.fps = 25.0
        self.inference_ms = 12.0
        self.ema_latency = 0.020
        self.frame_counter = 0
        self.face_detected = False
        self.face_count = 0
        self.last_face_time = 0.0
        self.current_tracks: List[Dict[str, Any]] = []
        self.current_frame_jpeg: bytes = b""
        self.current_crop_jpeg: bytes = b""
        self.memo_cache: Dict[bytes, List[Dict[str, Any]]] = {}
        self.memo_keys: List[bytes] = []
        self.cap = None
        self.lock = threading.Lock()
        self.running = True
        
        # Async Inference Engine State & Video FPS Pacing
        self.latest_raw_frame = None
        self.inference_busy = False
        self.source_fps = 30.0
        self.last_frame_tick = time.time()

sessions: Dict[str, SessionState] = {}
session_lock = threading.Lock()

def get_session(session_id: str) -> SessionState:
    with session_lock:
        if session_id not in sessions:
            add_log("INFO", "SessionManager", f"Created new isolated session: {session_id}")
            sessions[session_id] = SessionState(session_id)
        return sessions[session_id]

# Fast Non-Overlapping OpenCV Bounding Box & Label Annotator
def draw_fast_annotations(frame: np.ndarray, tracks: List[Dict[str, Any]], selected_id: int) -> np.ndarray:
    annotated = frame.copy()
    drawn_badges = []

    for trk in tracks:
        box = trk["box"]
        x1, y1 = int(box["x"]), int(box["y"])
        w, h = int(box["w"]), int(box["h"])
        x2, y2 = x1 + w, y1 + h

        is_selected = (trk["track_id"] == selected_id)
        label = trk["label"]
        conf = trk["confidence"]
        t_id = trk["track_id"]

        # Color palette
        if is_selected:
            color = (220, 0, 255) # Bright Pink / Magenta
            thickness = 3
        elif trk["class_id"] == 0:
            color = (255, 180, 0) # Cyan/Blue for Person
            thickness = 2
        elif trk["class_id"] == 2:
            color = (0, 200, 255) # Yellow/Gold for Car
            thickness = 2
        else:
            color = (0, 255, 120) # Green for Object
            thickness = 2

        # Draw Bounding Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Non-overlapping Badge Y Calculation
        text = f"#{t_id} {label} {int(conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

        badge_y = max(y1 - th - 6, 0)
        # Shift badge upwards if it collides with another badge
        for b_x1, b_y1, b_x2, b_y2 in drawn_badges:
            if abs(b_x1 - x1) < (tw + 10) and abs(b_y1 - badge_y) < (th + 6):
                badge_y = max(b_y1 - th - 8, 0)

        drawn_badges.append((x1, badge_y, x1 + tw + 6, badge_y + th + 6))
        
        cv2.rectangle(annotated, (x1, badge_y), (x1 + tw + 6, badge_y + th + 6), color, -1)
        cv2.putText(annotated, text, (x1 + 3, badge_y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated

# Async ONNX Model Tracking & Face Detection Background Worker
def async_inference_worker(session: SessionState):
    t_start = time.perf_counter()
    try:
        frame = session.latest_raw_frame
        if frame is None or model is None:
            return

        h_orig, w_orig = frame.shape[:2]
        infer_mat = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

        tracker_config = "backend/bytetrack_3sec.yaml" if os.path.exists("backend/bytetrack_3sec.yaml") else "bytetrack.yaml"

        results = model.track(
            source=infer_mat,
            persist=True,
            tracker=tracker_config,
            conf=session.conf_threshold,
            iou=session.nms_threshold,
            verbose=False,
            device=INFERENCE_DEVICE
        )

        if results and len(results) > 0:
            res = results[0]
            new_tracks = []

            if res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy()
                ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else None
                
                scale_x = w_orig / 640.0
                scale_y = h_orig / 640.0

                for i, box in enumerate(boxes):
                    conf = float(confs[i])
                    cls_id = int(classes[i])

                    # STRICT REQUIREMENT: PERSON ONLY (class_id == 0)
                    if cls_id != 0:
                        continue

                    x1 = box[0] * scale_x
                    y1 = box[1] * scale_y
                    x2 = box[2] * scale_x
                    y2 = box[3] * scale_y

                    label = "person"
                    track_id = int(ids[i]) if ids is not None else (i + 1)
                    w = x2 - x1
                    h = y2 - y1

                    new_tracks.append({
                        "track_id": track_id,
                        "label": label,
                        "confidence": round(conf, 2),
                        "class_id": 0,
                        "box": {
                            "x": round(float(x1)),
                            "y": round(float(y1)),
                            "w": round(float(w)),
                            "h": round(float(h))
                        },
                        "velocity": {"dx": 0.0, "dy": 0.0},
                        "age": 1
                    })

            with session.lock:
                session.current_tracks = new_tracks
    except Exception as e:
        add_log("ERROR", "AsyncInference", f"Inference worker error: {e}")
    finally:
        t_end = time.perf_counter()
        real_inf_ms = (t_end - t_start) * 1000.0
        session.inference_ms = round(real_inf_ms, 2)
        session.inference_busy = False

def resolve_video_path(src: str) -> str:
    if not src:
        return "test/15690486_1920_1080_25fps.mp4"
    src_clean = str(src).strip()
    if os.path.exists(src_clean):
        return src_clean
    if os.path.exists(src):
        return src
    base = os.path.basename(src_clean)
    for folder in ["test", "uploads"]:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for f in files:
                    if f == base or f.strip() == base or f.strip() == base.strip():
                        return os.path.join(root, f)
    return "test/15690486_1920_1080_25fps.mp4"

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

        time.sleep(0.005)

def process_session_frame(session: SessionState):
    with session.lock:
        if not session.is_playing:
            return
        is_opened = (session.cap is not None and session.cap.isOpened())
        src = session.video_source
        stype = session.source_type

    # 1. Auto-resolve video file path to handle spaces or missing paths
    if not is_opened:
        if stype == "webcam":
            try: resolved_src = int(src)
            except ValueError: resolved_src = 0
        else:
            resolved_src = resolve_video_path(str(src))
        
        add_log("INFO", "VideoCapture", f"Opening video source: {resolved_src} (Type: {stype})")
        new_cap = cv2.VideoCapture(resolved_src)
        if new_cap is not None and new_cap.isOpened():
            v_fps = new_cap.get(cv2.CAP_PROP_FPS)
            fps_val = v_fps if (v_fps and 5.0 < v_fps <= 120.0) else 30.0
            with session.lock:
                session.cap = new_cap
                session.source_fps = fps_val
                session.video_source = str(resolved_src)
        else:
            fallback_path = "test/15690486_1920_1080_25fps.mp4"
            if os.path.exists(fallback_path):
                fb_cap = cv2.VideoCapture(fallback_path)
                with session.lock:
                    session.cap = fb_cap
                    session.source_fps = 25.0
                    session.video_source = fallback_path

    with session.lock:
        cap_obj = session.cap
        src_fps = session.source_fps
        last_tick = session.last_frame_tick

    if cap_obj is None or not cap_obj.isOpened():
        return

    # Precise Wall-Clock FPS Pacing (WITHOUT holding lock)
    target_interval = 1.0 / src_fps
    now = time.time()
    elapsed_since_last = now - last_tick
    if elapsed_since_last < target_interval:
        time.sleep(target_interval - elapsed_since_last)

    ret, frame = cap_obj.read()
    with session.lock:
        session.last_frame_tick = time.time()

    if not ret or frame is None or frame.size == 0:
        cap_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap_obj.read()
        if not ret or frame is None:
            return

    with session.lock:
        session.frame_counter += 1

    img_h, img_w = frame.shape[:2]
    active_tracks = []

    # 2. Direct Ultralytics YOLO Prediction for Person Detection
    if model is not None:
        try:
            t_inf_start = time.perf_counter()
            infer_mat = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            infer_rgb = cv2.cvtColor(infer_mat, cv2.COLOR_BGR2RGB)
            
            results = model.predict(
                source=infer_rgb,
                conf=session.conf_threshold,
                iou=session.nms_threshold,
                verbose=False,
                device=INFERENCE_DEVICE
            )

            if results and len(results) > 0:
                res = results[0]
                if res.boxes is not None:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy()
                    
                    scale_x = img_w / 640.0
                    scale_y = img_h / 640.0

                    for i, box in enumerate(boxes):
                        cls_id = int(classes[i])
                        if cls_id != 0: # STRICT REQUIREMENT: PERSON ONLY (class_id == 0)
                            continue

                        conf = float(confs[i])
                        x1 = box[0] * scale_x
                        y1 = box[1] * scale_y
                        x2 = box[2] * scale_x
                        y2 = box[3] * scale_y

                        w_b = max(1.0, x2 - x1)
                        h_b = max(1.0, y2 - y1)

                        active_tracks.append({
                            "track_id": i + 1,
                            "label": "person",
                            "confidence": round(conf, 2),
                            "class_id": 0,
                            "box": {
                                "x": round(float(x1)),
                                "y": round(float(y1)),
                                "w": round(float(w_b)),
                                "h": round(float(h_b))
                            },
                            "velocity": {"dx": 0.0, "dy": 0.0},
                            "age": 1
                        })

            t_inf_end = time.perf_counter()
            with session.lock:
                session.inference_ms = round((t_inf_end - t_inf_start) * 1000.0, 2)
        except Exception as e:
            add_log("ERROR", "DirectPredict", f"Prediction error: {e}")

    with session.lock:
        session.current_tracks = active_tracks

    # 3. Real-Time Face Detection Scan on active tracked person crops (Head & Full Body)
    face_active_track_id = -1
    best_face_crop = None
    best_face_score = 0.0
    best_face_count = 0
    main_frame_faces = []

    if yunet_detector is not None and len(active_tracks) > 0:
        for trk in active_tracks:
            box = trk["box"]
            bx = max(0, int(box["x"]))
            by = max(0, int(box["y"]))
            bw = max(1, int(box["w"]))
            bh = max(1, int(box["h"]))
            bx2 = min(img_w, bx + bw)
            by2 = min(img_h, by + bh)

            if bx2 > bx and by2 > by:
                person_crop = frame[by:by2, bx:bx2].copy()
                c_h, c_w = person_crop.shape[:2]
                
                # Check both full body crop AND upper head crop (top 55%) for maximum YuNet accuracy
                crops_to_check = [(person_crop, "full", 0)]
                if c_h >= 30 and c_w >= 20:
                    head_h = max(20, int(c_h * 0.55))
                    head_crop = person_crop[0:head_h, 0:c_w].copy()
                    crops_to_check.append((head_crop, "head", 0))

                for crop_mat, crop_type, y_offset in crops_to_check:
                    ch, cw = crop_mat.shape[:2]
                    if cw > 10 and ch > 10:
                        try:
                            yunet_detector.setInputSize((cw, ch))
                            _, faces = yunet_detector.detect(crop_mat)
                            if faces is not None and len(faces) > 0:
                                valid_faces = []
                                for f in faces:
                                    f_score = float(f[-1])
                                    fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                                    if f_score >= 0.20 and fw >= 8 and fh >= 8:
                                        valid_faces.append(f)
                                
                                if len(valid_faces) > 0:
                                    top_sc = max(float(f[-1]) for f in valid_faces)
                                    if top_sc > best_face_score:
                                        best_face_score = top_sc
                                        face_active_track_id = trk["track_id"]
                                        best_face_count = len(valid_faces)

                                        crop_faces_drawn = person_crop.copy()
                                        for f in valid_faces:
                                            fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                                            f_score = float(f[-1])
                                            sc_pct = int(f_score * 100)

                                            # Draw cyan face box on ROI crop
                                            cv2.rectangle(crop_faces_drawn, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
                                            cv2.putText(crop_faces_drawn, f"Face {sc_pct}%",
                                                        (fx, max(fy - 4, 10)),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

                                            # Map face box coordinates to main frame!
                                            m_fx1 = bx + fx
                                            m_fy1 = by + fy + y_offset
                                            m_fx2 = bx + fx + fw
                                            m_fy2 = by + fy + y_offset + fh
                                            main_frame_faces.append((m_fx1, m_fy1, m_fx2, m_fy2, sc_pct))

                                        best_face_crop = crop_faces_drawn
                        except Exception:
                            pass

    # 4. Update session state with 1.5-second Target ROI Persistence Memory
    now_t = time.time()
    with session.lock:
        if face_active_track_id != -1 and best_face_crop is not None:
            session.selected_track_id = face_active_track_id
            session.face_detected = True
            session.face_count = best_face_count
            session.last_face_time = now_t
            ret_c, crop_buf = cv2.imencode(".jpg", best_face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ret_c:
                session.current_crop_jpeg = crop_buf.tobytes()
        elif session.last_face_time > 0 and (now_t - session.last_face_time) < 1.5 and session.selected_track_id != -1:
            session.face_detected = True
        else:
            session.selected_track_id = -1
            session.face_detected = False
            session.face_count = 0
            session.current_crop_jpeg = b""
        
        curr_selected = session.selected_track_id

    # 5. Draw annotations on main video frame
    annotated_frame = draw_fast_annotations(frame, active_tracks, curr_selected)

    # Draw cyan face bounding boxes directly on main video frame!
    for (fx1, fy1, fx2, fy2, sc_pct) in main_frame_faces:
        cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Face {sc_pct}%",
                    (fx1, max(fy1 - 4, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

    # Calculate exact real-time streaming engine FPS per frame
    perf_now = time.perf_counter()
    if hasattr(session, 'last_perf_tick') and session.last_perf_tick > 0:
        dt = perf_now - session.last_perf_tick
        if dt > 0.0001:
            inst_fps = 1.0 / dt
            session.fps = round(session.fps * 0.8 + inst_fps * 0.2, 1)
    else:
        session.fps = round(session.source_fps, 1)
    session.last_perf_tick = perf_now

    # Encode main frame to JPEG for MJPEG stream
    ret_jpg, jpeg_buf = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ret_jpg:
        with session.lock:
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
            "active_model": active_model_path,
            "face_detected": session.face_detected,
            "face_count": session.face_count,
            "hardware_accel": INFERENCE_DEVICE
        }

@app.post("/api/select_track")
async def select_track(request: Request):
    sid = request.headers.get("X-Session-ID", "default_session")
    session = get_session(sid)
    data = await request.json()
    track_id = int(data.get("track_id", -1))
    with session.lock:
        session.selected_track_id = track_id
        session.current_crop_jpeg = b""
    add_log("INFO", "API", f"Selected track target {track_id} for session {sid}")
    return {"status": "ok", "selected_track_id": track_id}

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
        last_sent = -1
        while True:
            if session.frame_counter != last_sent:
                jpeg_bytes = session.current_frame_jpeg
                if jpeg_bytes:
                    last_sent = session.frame_counter
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" +
                           jpeg_bytes + b"\r\n")
            time.sleep(0.008)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/api/crop_image")
async def get_crop_image(request: Request):
    sid = request.headers.get("X-Session-ID", request.query_params.get("sid", "default_session"))
    session = get_session(sid)
    with session.lock:
        jpeg_bytes = session.current_crop_jpeg
        sel_id = session.selected_track_id

    if not jpeg_bytes or sel_id == -1:
        blank_mat = np.zeros((400, 400, 3), dtype=np.uint8)
        _, blank_buf = cv2.imencode(".jpg", blank_mat)
        jpeg_bytes = blank_buf.tobytes()

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/api/crop")
async def crop_stream(request: Request):
    sid = request.headers.get("X-Session-ID", request.query_params.get("sid", "default_session"))
    session = get_session(sid)

    # Pre-encode 400x400 black blank frame for instant clear when person leaves frame
    blank_mat = np.zeros((400, 400, 3), dtype=np.uint8)
    _, blank_buf = cv2.imencode(".jpg", blank_mat)
    blank_jpeg = blank_buf.tobytes()

    def crop_generator():
        last_sent_crop = -1
        while True:
            if session.selected_track_id == -1 or not session.current_crop_jpeg:
                # PERSON IS OUT OF FRAME -> STREAM BLANK RESET FRAME TO CLEAR UI INSTANTLY
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(blank_jpeg)).encode() + b"\r\n\r\n" +
                       blank_jpeg + b"\r\n")
            elif session.frame_counter != last_sent_crop:
                jpeg_bytes = session.current_crop_jpeg
                if jpeg_bytes:
                    last_sent_crop = session.frame_counter
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n" +
                           jpeg_bytes + b"\r\n")
            time.sleep(0.015)

    return StreamingResponse(
        crop_generator(),
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

import socket
import subprocess

def free_port(port: int = 8080):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            res = s.connect_ex(('127.0.0.1', port))
            if res == 0:
                # Port occupied by stale backend -> automatically kill stale process
                subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.3)
    except Exception:
        pass

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    free_port(8080)
    add_log("INFO", "Main", "Starting FastAPI Server on 0.0.0.0:8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
