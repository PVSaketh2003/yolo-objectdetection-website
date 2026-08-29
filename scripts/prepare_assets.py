import os
import urllib.request
import cv2
import numpy as np

def prepare_onnx_model():
    print("Exporting YOLO ONNX model...")
    os.makedirs("backend/models", exist_ok=True)
    target_onnx = "backend/models/yolov8s.onnx"
    if not os.path.exists(target_onnx):
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8s.pt")
            exported_path = model.export(format="onnx", dynamic=False, imgsz=640)
            if os.path.exists(exported_path):
                os.rename(exported_path, target_onnx)
                print(f"Model exported successfully to {target_onnx}")
        except Exception as e:
            print(f"Error exporting model via ultralytics: {e}")

def prepare_sample_video():
    print("Generating sample offline benchmark video...")
    os.makedirs("sample_assets", exist_ok=True)
    video_path = "sample_assets/pedestrian_demo.mp4"
    if os.path.exists(video_path):
        print(f"Sample video already exists at {video_path}")
        return

    # Create synthetic realistic pedestrian benchmark video if offline
    width, height = 1280, 720
    fps = 30
    duration_sec = 15
    total_frames = fps * duration_sec

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Generate animated synthetic scene with walking people silhouettes/rectangles
    np.random.seed(42)
    num_persons = 4
    persons = []
    colors = [(240, 100, 50), (50, 200, 100), (220, 50, 200), (50, 150, 240)]
    for i in range(num_persons):
        persons.append({
            'x': float(np.random.randint(100, width - 200)),
            'y': float(np.random.randint(150, height - 300)),
            'dx': np.random.choice([-2.5, -1.8, 1.8, 2.5]),
            'dy': np.random.choice([-0.5, 0.5]),
            'w': 80,
            'h': 220,
            'color': colors[i % len(colors)]
        })

    for f in range(total_frames):
        # Dark modern styled backdrop canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        for y in range(height):
            v = int(20 + 30 * (y / height))
            frame[y, :] = (v, v + 5, v + 15)

        # Draw grid lines for surveillance visual aesthetic
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, 0), (gx, height), (35, 40, 55), 1)
        for gy in range(0, height, 80):
            cv2.line(frame, (0, gy), (width, gy), (35, 40, 55), 1)

        # Draw timestamp header
        cv2.putText(frame, f"CAM-01 LIVE BENCHMARK | FRAME {f:04d}", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 240), 2)

        # Animate persons
        for i, p in enumerate(persons):
            p['x'] += p['dx']
            p['y'] += p['dy']
            if p['x'] < 50 or p['x'] > width - 150:
                p['dx'] *= -1
            if p['y'] < 100 or p['y'] > height - 250:
                p['dy'] *= -1

            x, y, w, h = int(p['x']), int(p['y']), int(p['w']), int(p['h'])
            # Body torso
            cv2.rectangle(frame, (x + 10, y + 50), (x + w - 10, y + h), p['color'], -1)
            # Head
            cv2.circle(frame, (x + w // 2, y + 25), 22, p['color'], -1)
            # Label overlay
            cv2.putText(frame, f"Person #{i+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Sample video created successfully at {video_path}")

if __name__ == "__main__":
    prepare_onnx_model()
    prepare_sample_video()
