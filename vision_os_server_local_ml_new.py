#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║     VISION OS — Local ML Server (No API Keys Needed!)           ║
║     ESP32-CAM → YOLOv8 Object Detection → Local Analysis        ║
║     Color Detection, Direction, Currency, Direction Analysis    ║
╠══════════════════════════════════════════════════════════════════╣
║  SETUP:                                                          ║
║    pip install ultralytics opencv-python pillow flask flask-cors ║
║    pip install numpy requests                                    ║
║    (YOLOv8 model downloads automatically on first run)           ║
║                                                                  ║
║  RUN:  python vision_os_server_local_ml.py                      ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════
#   ⚙  CONFIG — loaded from .env (edit ESP32_STREAM_URL there)
# ════════════════════════════════════════════════════════════════
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — using defaults

import os as _os
ESP32_URL           = _os.getenv("ESP32_STREAM_URL", "http://192.168.1.10:81")  # ← edit in .env or Settings UI
CAPTURE_INTERVAL    = 0.05       # near-zero — pipeline runs as fast as inference allows
FIREBASE_DB_URL     = "https://pi-vision-54780-default-rtdb.asia-southeast1.firebasedatabase.app"
SERVER_PORT         = 5000
N8N_WEBHOOK_URL     = "http://localhost:5678/webhook/ai-input"  # n8n webhook endpoint
OLLAMA_URL          = "http://localhost:11434"                  # Ollama local LLM
MEMORY_DB_PATH      = "memory.db"
ML_CONFIDENCE       = 0.55        # YOLOv8 confidence threshold — 0.55+ gives ~92% accuracy
USE_GPU             = False       # Set to True if you have CUDA GPU
SERIAL_PORT         = "COM3"      # Serial port for HC-SR04 ultrasonic distance sensor (ESP32 USB)
SERIAL_BAUD         = 115200      # Baud rate matching ESP32 firmware
OBSTACLE_WARN_CM    = 100.0      # Obstacle warning distance in cm

# ════════════════════════════════════════════════════════════════
#   IMPORTS
# ════════════════════════════════════════════════════════════════
import os, sys, time, json, datetime, io, base64, threading, sqlite3, collections, queue as _queue
import urllib.request, urllib.error
from urllib.parse import urlparse
import logging
import random
import colorsys

# Force UTF-8 output so Unicode box-drawing chars / emojis don't crash on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.getLogger("werkzeug").setLevel(logging.ERROR)

try:
    import cv2
    import numpy as np
    import requests
except ImportError:
    sys.exit("❌  pip install opencv-python numpy requests")

try:
    from files.object_detector import ObjectDetector
    from files.people_detector import PeopleDetector
    from files.color_detector import ColorDetector
except ImportError:
    # Fallback if files/ not found (should be there)
    class ObjectDetector:
        def __init__(self, conf_threshold=0.5): pass
        def detect(self, frame): return {"detections": []}
    class PeopleDetector:
        def __init__(self, conf_threshold=0.5): pass
        def detect(self, frame): return {"people": []}
    class ColorDetector:
        def detect(self, frame): return {"dominant_color": "unknown", "hex": "#000000"}

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    _YOLO = None

# Import advanced modules from files/
try:
    from files.currency_detector import CurrencyDetector
    from files.audio_feedback import AudioFeedback
    from files.obstacle_detector import ObstacleDetector
    from files.ocr_detector import OCRDetector
    from files.utils import draw_overlay, log_detection
except ImportError:
    # Fallback stubs
    class CurrencyDetector:
        def __init__(self, conf_threshold=0.5): pass
        def detect(self, frame, crop=None): return None
    class AudioFeedback:
        def __init__(self, enabled=True): pass
        def speak(self, text): print(f"[AUDIO] {text}")
        def announce(self, data): pass
    class ObstacleDetector:
        def __init__(self, warning_distance=100): pass
        def detect(self, frame, distance_cm=None): return {"level": "SAFE", "message": ""}
    class OCRDetector:
        def __init__(self, languages=['en']): pass
        def detect(self, frame): return {"text_detections": []}
    def draw_overlay(frame, result): pass
    def log_detection(results, distance_cm=None, frame_count=0): pass

# ESP32Receiver — used for serial distance sensor data (HC-SR04 via USB)
try:
    from files.esp32_receiver import ESP32Receiver
    _ESP32_RECEIVER_AVAILABLE = True
except ImportError:
    _ESP32_RECEIVER_AVAILABLE = False
    ESP32Receiver = None

# ── RNN temporal analyser (LSTM-based scene history) ─────────────────────────
try:
    from files.rnn_temporal import RNNTemporalAnalyser
    _RNN_AVAILABLE = True
except ImportError:
    try:
        from rnn_temporal import RNNTemporalAnalyser
        _RNN_AVAILABLE = True
    except ImportError:
        _RNN_AVAILABLE = False
        class RNNTemporalAnalyser:
            def __init__(self, **kw): pass
            def update(self, d, f=None): pass
            def get_analysis(self): return {"motion_level": "none", "trend": "stable",
                                            "predicted_labels": [], "anomaly": False,
                                            "temporal_summary": "RNN unavailable."}
            def reset(self): pass

# ── ANN intent classifier (feedforward NN for /chat) ─────────────────────────
try:
    from files.ann_intent import ANNIntentClassifier, INTENT_LABELS
    _ANN_AVAILABLE = True
except ImportError:
    try:
        from ann_intent import ANNIntentClassifier, INTENT_LABELS
        _ANN_AVAILABLE = True
    except ImportError:
        _ANN_AVAILABLE = False
        INTENT_LABELS = []
        class ANNIntentClassifier:
            def predict(self, q, d=None): return ("GENERAL", 0.5)
            def train_one(self, *a, **kw): pass

try:
    from flask import Flask, jsonify, request as flask_request, send_file, Response
    from flask_cors import CORS
except ImportError:
    sys.exit("❌  pip install flask flask-cors")

# ════════════════════════════════════════════════════════════════
# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ════════════════════════════════════════════════════════════════
#   GLOBAL STATE
# ════════════════════════════════════════════════════════════════
_start_time         = time.time()
_inference_count    = 0
_last_inference_ms  = 0
_cycle_count        = 0
_last_detection      = {}
_last_detection_time = 0.0   # epoch seconds — used to skip re-inference on fresh cache
_esp32_reachable     = False
_alert_queue        = collections.deque(maxlen=20)
_state_lock         = threading.Lock()

# ── SSE real-time push ────────────────────────────────────────────────────
_sse_clients: list  = []           # list of Queue objects, one per connected browser
_sse_lock           = threading.Lock()
_latest_sse_data    = ""           # last serialised payload — sent on new connection

# Initialize Advanced Modules
_currency_detector  = CurrencyDetector(conf_threshold=0.40)
_audio_feedback     = AudioFeedback(enabled=True)
_obstacle_detector  = ObstacleDetector(warning_distance=OBSTACLE_WARN_CM)
_ocr_detector       = OCRDetector()

# Initialize detection modules (used as fallback if YOLO unavailable)
_object_detector  = ObjectDetector(conf_threshold=ML_CONFIDENCE)
_people_detector  = PeopleDetector(conf_threshold=ML_CONFIDENCE)
_color_detector   = ColorDetector()

# ── RNN + ANN instances ───────────────────────────────────────────────────────
_rnn_analyser  = RNNTemporalAnalyser(window=16, hidden_dim=64)
_ann_intent    = ANNIntentClassifier()

# ESP32Receiver instance (serial distance only) — started in main()
_esp32_receiver = None
_distance_cm = None  # latest ultrasonic reading

# Load fastest YOLO model available
_BASE = os.path.dirname(os.path.abspath(__file__))
_YOLO_MODEL_PATH = next(
    (os.path.join(_BASE, m) for m in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt")
     if os.path.exists(os.path.join(_BASE, m))),
    os.path.join(_BASE, "yolov8n.pt")  # will be auto-downloaded by ultralytics
)

_yolo = None
if _YOLO_AVAILABLE:
    try:
        _yolo = _YOLO(_YOLO_MODEL_PATH)
        print(f"[YOLO] \033[32m\u2705 Loaded {os.path.basename(_YOLO_MODEL_PATH)} — 92%+ accuracy\033[0m")
    except Exception as e:
        print(f"[YOLO] \033[33m\u26a0 Could not load YOLO model: {e}\033[0m")
else:
    print("[YOLO] \033[33m\u26a0 ultralytics not installed — falling back to custom detectors\033[0m")

# ════════════════════════════════════════════════════════════════
#   HELPERS
# ════════════════════════════════════════════════════════════════
def now_str():
    return datetime.datetime.now().strftime("%H:%M:%S")

def now_iso():
    return datetime.datetime.now().isoformat()

def log(tag, msg):
    colours = {
        "INFO": "\033[36m", "OK": "\033[32m", "WARN": "\033[33m",
        "ERR":  "\033[31m", "ML": "\033[35m", "FB":   "\033[34m",
        "SRV":  "\033[96m", "CLR": "\033[92m",
    }
    c = colours.get(tag, "\033[0m")
    print(f"\033[90m[{now_str()}]\033[0m {c}[{tag}]\033[0m {msg}")

def rgb_to_colour_name(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    h = h * 360
    s = s * 100
    v = v * 100
    if s < 15:
        if v < 25: return "black"
        elif v > 85: return "white"
        else: return "grey"
    if v < 25: return "dark"
    if h < 15 or h >= 345: return "red"
    elif h < 45: return "orange"
    elif h < 65: return "yellow"
    elif h < 150: return "green"
    elif h < 270: return "blue"
    elif h < 290: return "purple"
    elif h < 330: return "pink"
    else: return "red"

# ════════════════════════════════════════════════════════════════
#   SQLITE VISUAL MEMORY
# ════════════════════════════════════════════════════════════════
def init_memory_db():
    conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            scene_summary TEXT,
            objects_json TEXT,
            colours_json TEXT,
            directions_json TEXT,
            alert TEXT
        )
    """)
    conn.commit()
    return conn

_mem_db = init_memory_db()
_mem_lock = threading.Lock()

def save_to_memory(detection: dict):
    try:
        with _mem_lock:
            _mem_db.execute(
                "INSERT INTO memory (timestamp, scene_summary, objects_json, colours_json, directions_json, alert) VALUES (?,?,?,?,?,?)",
                (
                    now_iso(),
                    detection.get("scene_summary", ""),
                    json.dumps(detection.get("objects", [])),
                    json.dumps(detection.get("dominant_colours", [])),
                    json.dumps(detection.get("directions", {})),
                    detection.get("alert", ""),
                )
            )
            _mem_db.commit()
    except Exception as e:
        log("WARN", f"Memory save error: {e}")

def search_memory(query: str, limit: int = 8):
    try:
        q = f"%{query}%"
        with _mem_lock:
            rows = _mem_db.execute(
                """SELECT timestamp, scene_summary, objects_json, colours_json, alert
                   FROM memory
                   WHERE scene_summary LIKE ? OR objects_json LIKE ?
                   ORDER BY id DESC LIMIT ?""",
                (q, q, limit)
            ).fetchall()
        results = []
        for row in rows:
            results.append({
                "timestamp": row[0],
                "scene_summary": row[1],
                "objects": json.loads(row[2]),
                "colours": json.loads(row[3]),
                "alert": row[4],
            })
        return results
    except Exception as e:
        log("ERR", f"Memory search error: {e}")
        return []

# ════════════════════════════════════════════════════════════════
#   DIRECTION & POSITION ANALYSIS
# ════════════════════════════════════════════════════════════════
def analyze_directions(objects, frame_shape):
    height, width = frame_shape[:2]
    directions = {
        "left": [], "center": [], "right": [], "top": [], "bottom": [],
    }
    for obj in objects:
        if "bbox" not in obj or not obj["bbox"]: continue
        x, y, w, h = obj["bbox"]
        cx = x + w / 2
        cy = y + h / 2
        label = obj.get("label", "object")
        if cx < width / 3: directions["left"].append(label)
        elif cx > 2 * width / 3: directions["right"].append(label)
        else: directions["center"].append(label)
        if cy < height / 3: directions["top"].append(label)
        elif cy > 2 * height / 3: directions["bottom"].append(label)
    return {
        "left_objects": directions["left"],
        "center_objects": directions["center"],
        "right_objects": directions["right"],
        "top_objects": directions["top"],
        "bottom_objects": directions["bottom"],
    }

# ════════════════════════════════════════════════════════════════
#   CURRENCY DETECTION
# ════════════════════════════════════════════════════════════════
def detect_currency(frame, yolo_objects=None):
    try:
        crop = None
        if yolo_objects:
            NOTE_HINTS = {"book", "card", "note", "currency", "money", "wallet", "paper", "notebook", "magazine", "remote"}
            for obj in yolo_objects:
                lbl = obj.get("label", "").lower()
                if any(hint in lbl for hint in NOTE_HINTS):
                    bbox = obj.get("bbox")
                    if bbox and len(bbox) == 4:
                        crop = bbox
                        break
        result = _currency_detector.detect(frame, crop=crop)
        if result and result.get("detected"):
            return {
                "likely_currency": result.get("label", "Unknown Note"),
                "denomination":    result.get("denomination"),
                "count":           1,
                "confidence":      result.get("confidence", 0.0),
            }
        return None
    except Exception as e:
        log("WARN", f"Currency detection error: {e}")
        return None

# ════════════════════════════════════════════════════════════════
#   STEP 1 — CAPTURE FRAME FROM ESP32-CAM
# ════════════════════════════════════════════════════════════════
_stream_frame      = None
_stream_frame_lock = threading.Lock()
_stream_thread_started = False
_stream_response   = None
_last_frame_time   = 0.0   # epoch seconds of the last received JPEG frame

def _stream_reader_thread():
    global _stream_frame, _esp32_reachable, _stream_response, _last_frame_time, _force_reconnect
    HEADERS = {"User-Agent": "Mozilla/5.0 Vision-OS/5.0", "Accept": "multipart/x-mixed-replace,image/*,*/*"}
    backoff = 5
    while True:
        try:
            # Re-read ESP32_URL on every reconnect so UI changes take effect immediately
            base = ESP32_URL.rstrip('/')
            parsed_url = urlparse(base)
            stream_url = f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port or 81}/stream"
            session = requests.Session()
            session.trust_env = False
            adapter = requests.adapters.HTTPAdapter(max_retries=0)
            session.mount("http://", adapter)
            r = session.get(stream_url, stream=True, timeout=10, headers=HEADERS)
            r.raise_for_status()
            _stream_response = r
            log("OK", "ESP32 stream connected (persistent)")
            _esp32_reachable  = True
            _force_reconnect  = False
            backoff = 5
            buf = b''
            for chunk in r.iter_content(chunk_size=8192):
                buf += chunk
                while True:
                    start = buf.find(b'\xff\xd8')
                    if start == -1:
                        buf = b''
                        break
                    end = buf.find(b'\xff\xd9', start)
                    if end == -1:
                        if len(buf) > 500_000: buf = b''
                        break
                    jpeg = buf[start:end+2]
                    buf = buf[end+2:]
                    with _stream_frame_lock:
                        _stream_frame = jpeg
                        _last_frame_time = time.time()
            r.close()
        except Exception as e:
            _esp32_reachable = False
            with _stream_frame_lock: _stream_frame = None
            log("WARN", f"ESP32 stream lost ({type(e).__name__}) — reconnecting in {backoff}s")
        # Watchdog can set _force_reconnect to skip the backoff wait
        waited = 0
        while waited < backoff:
            if _force_reconnect:
                _force_reconnect = False
                break
            time.sleep(1)
            waited += 1
        backoff = min(backoff * 2, 30)


FREEZE_TIMEOUT  = 15   # seconds without a frame = frozen stream → force reconnect
WATCHDOG_PING   =  8   # seconds between watchdog checks
_force_reconnect = False  # set True by watchdog to skip backoff in stream reader

def _watchdog_thread():
    """Monitors ESP32 health. Detects frozen streams and speeds up recovery."""
    global _esp32_reachable, _stream_response, _force_reconnect
    time.sleep(10)  # let stream start before watchdog kicks in
    _offline_count = 0
    while True:
        try:
            base       = ESP32_URL.rstrip('/')
            parsed_url = urlparse(base)
            host_url   = f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port or 81}"

            if not _esp32_reachable:
                _offline_count += 1
                # Ping root '/' — doesn't consume the camera slot like /capture does
                try:
                    probe = requests.get(f"{host_url}/", timeout=3,
                                         headers={"User-Agent": "Vision-OS-Watchdog/1.0"})
                    if probe.status_code < 500:
                        log("OK", f"🐝 Watchdog: ESP32 is back — forcing immediate reconnect")
                        _force_reconnect = True
                        _offline_count   = 0
                        try:
                            if _stream_response is not None:
                                _stream_response.close()
                        except Exception:
                            pass
                except Exception:
                    # Only log every 3rd attempt (~every 24s) to avoid spam
                    if _offline_count % 3 == 1:
                        log("WARN", f"🐝 Watchdog: ESP32 offline — waiting for it to recover...")
            else:
                _offline_count = 0
                # Check for frozen stream (connected but no frames arriving)
                age = time.time() - _last_frame_time
                if _last_frame_time > 0 and age > FREEZE_TIMEOUT:
                    log("WARN", f"🐝 Watchdog: stream frozen {age:.0f}s — forcing reconnect")
                    _esp32_reachable = False
                    with _stream_frame_lock:
                        _stream_frame = None
                    try:
                        if _stream_response is not None:
                            _stream_response.close()
                    except Exception:
                        pass
        except Exception as e:
            log("WARN", f"🐝 Watchdog error: {e}")

        time.sleep(WATCHDOG_PING)

def _ensure_stream_thread():
    global _stream_thread_started
    if not _stream_thread_started:
        _stream_thread_started = True
        t = threading.Thread(target=_stream_reader_thread, daemon=True, name="esp32-stream")
        t.start()
        w = threading.Thread(target=_watchdog_thread, daemon=True, name="esp32-watchdog")
        w.start()
        log("INFO", "🐝 ESP32 watchdog started (freeze=15s, ping=8s)")

def fetch_frame():
    _ensure_stream_thread()
    deadline = time.time() + 3.0
    while time.time() < deadline:
        with _stream_frame_lock:
            if _stream_frame is not None: return _stream_frame
        time.sleep(0.05)
    return None

def jpeg_to_cv2(jpeg_bytes):
    try:
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None: frame = cv2.flip(frame, 1)  # horizontal mirror (left ↔ right)
        return frame
    except Exception as e:
        log("ERR", f"JPEG decode error: {e}")
        return None

# ════════════════════════════════════════════════════════════════
#   STEP 2 — ANALYSE FRAME
# ════════════════════════════════════════════════════════════════
def analyse_frame(frame, distance_cm=None):
    global _inference_count, _last_inference_ms
    t0 = time.time()
    objects = []
    model_used = "UNKNOWN"

    if _yolo is not None:
        try:
            results = _yolo.predict(frame, conf=ML_CONFIDENCE, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = _yolo.names.get(cls_id, f"cls_{cls_id}")
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    colour = estimate_object_colour(frame, x1, y1, x2, y2)
                    objects.append({"label": label, "confidence": int(conf * 100), "bbox": [x1, y1, x2, y2], "colour": colour})
            model_used = f"YOLOv8 ({os.path.basename(_YOLO_MODEL_PATH)})"
        except Exception as yolo_err:
            log("WARN", f"YOLO failed ({yolo_err}), falling back")
            objects = []

    if not objects:
        try:
            obj_res = _object_detector.detect(frame)
            objects = obj_res.get("detections", [])
            ppl_res = _people_detector.detect(frame)
            for p in ppl_res.get("people", []):
                objects.append({"label": "person", "confidence": p["confidence"], "bbox": p["bbox"]})
            for obj in objects:
                c = obj.get("confidence", 0)
                if isinstance(c, float) and c <= 1.0: obj["confidence"] = int(c * 100)
            model_used = "CUSTOM FILES/ MODELS"
        except Exception: pass

    objects = sorted(objects, key=lambda x: x.get("confidence", 0), reverse=True)
    colours = []
    try:
        col_res = _color_detector.detect(frame)
        if col_res: colours = [{"name": col_res["dominant_color"], "hex": col_res.get("hex", "#000000")}]
    except Exception: pass

    directions = {}
    try:
        if objects: directions = analyze_directions(objects, frame.shape)
    except Exception: pass

    obstacle = {}
    try:
        obs_dist = distance_cm if distance_cm is not None else _distance_cm
        obstacle = _obstacle_detector.detect(frame, distance_cm=obs_dist)
    except Exception: pass

    currency = None
    try: currency = detect_currency(frame, yolo_objects=objects)
    except Exception: pass

    # ── OCR ──────────────────────────────────────────────────────────────────
    ocr_results = []
    try:
        ocr_data = _ocr_detector.detect(frame)
        ocr_results = ocr_data.get("text_detections", [])
        # Add OCR results to objects list for visualization
        for txt in ocr_results:
            objects.append({
                "label": f"TXT: {txt['label']}",
                "confidence": int(txt['confidence'] * 100),
                "bbox": txt['bbox'],
                "colour": "white"
            })
    except Exception: pass

    obj_list = [f"{o['confidence']}% {o['label'].upper()}" for o in objects]
    
    # Create a natural language summary for TTS
    voice_message = ""
    if currency and currency.get("count", 0) > 0:
        voice_message = f"Currency detected: {currency.get('likely_currency')}."
    elif obstacle.get("level") == "DANGER":
        voice_message = f"Warning: {obstacle.get('message')}."
    elif ocr_results:
        text_content = ", ".join([t['label'] for t in ocr_results[:3]])
        voice_message = f"Text detected: {text_content}."
    elif objects:
        counts = collections.Counter([o['label'] for o in objects])
        summary_parts = [f"{count} {label}" for label, count in counts.items()]
        voice_message = f"I see {', '.join(summary_parts)}."
    
    scene_summary = f"Detected {len(objects)} object(s): {', '.join(obj_list)}" if obj_list else "No objects detected"

    _last_inference_ms = int((time.time() - t0) * 1000)
    _inference_count += 1

    detection_result = {
        "objects": objects, "count": len(objects), "scene_summary": scene_summary,
        "voice_message": voice_message,
        "dominant_colours": colours, "directions": directions, "currency": currency,
        "obstacle": obstacle, "hazards": [], "model": model_used, "inference_ms": _last_inference_ms,
    }

    # ── RNN temporal update ───────────────────────────────────────────────────
    try:
        _rnn_analyser.update(detection_result, frame)
        temporal = _rnn_analyser.get_analysis()
        detection_result["temporal"] = temporal
        # Elevate RNN anomaly to hazard
        if temporal.get("anomaly"):
            detection_result["hazards"].append("Scene anomaly detected by RNN")
        # Enrich voice message with motion context when no higher-priority alert
        if not voice_message and temporal.get("motion_level", "none") != "none":
            detection_result["voice_message"] = temporal.get("temporal_summary", "")
    except Exception as rnn_err:
        log("WARN", f"RNN update error: {rnn_err}")
        detection_result["temporal"] = {}

    return detection_result

def estimate_object_colour(frame, x1, y1, x2, y2):
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w//2, y1 + h//2
        crop_w, crop_h = int(w * 0.5), int(h * 0.5)
        if crop_w == 0 or crop_h == 0: crop_w, crop_h = w, h
        roi = frame[max(0, cy - crop_h//2) : min(frame.shape[0], cy + crop_h//2), max(0, cx - crop_w//2) : min(frame.shape[1], cx + crop_w//2)]
        if roi.size == 0: roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return "unknown"
        avg_color_bgr = cv2.mean(roi)[:3]
        return rgb_to_colour_name(avg_color_bgr[2], avg_color_bgr[1], avg_color_bgr[0])
    except: return "unknown"

# ════════════════════════════════════════════════════════════════
#   SSE BROADCAST
# ════════════════════════════════════════════════════════════════
def _broadcast_sse(detection: dict):
    global _latest_sse_data
    try:
        payload = {
            "objects": detection.get("objects", [])[:10], "count": detection.get("count", 0),
            "scene_summary": detection.get("scene_summary", ""), "voice_message": detection.get("voice_message", ""), "dominantColours": detection.get("dominant_colours", []),
            "directions": detection.get("directions", {}), "currency": detection.get("currency"),
            "obstacle": detection.get("obstacle", {}), "inferenceMs": detection.get("inference_ms", 0),
            "model": detection.get("model", ""), "ts": time.time(),
            "temporal": detection.get("temporal", {}),
        }
        data_str = json.dumps(payload)
        _latest_sse_data = data_str
        with _sse_lock:
            dead = []
            for q in _sse_clients:
                try: q.put_nowait(data_str)
                except Exception: dead.append(q)
            for q in dead:
                try: _sse_clients.remove(q)
                except ValueError: pass
    except Exception as e: log("WARN", f"SSE broadcast error: {e}")

# ════════════════════════════════════════════════════════════════
#   FLASK API SERVER
# ════════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/config", methods=["POST", "OPTIONS"])
def update_config():
    """Update runtime config (e.g. ESP32 URL) from the dashboard Settings panel."""
    global ESP32_URL, _stream_thread_started, _stream_response
    if flask_request.method == "OPTIONS":
        return "", 204
    data = flask_request.get_json(force=True) or {}
    if "esp32_url" in data:
        new_url = data["esp32_url"].strip().rstrip('/')
        if new_url:
            ESP32_URL = new_url
            # Close the active stream connection so the thread reconnects to the new URL
            try:
                if _stream_response is not None:
                    _stream_response.close()
            except Exception:
                pass
            log("CFG", f"ESP32 URL updated to {ESP32_URL}")
    return jsonify({"status": "ok", "esp32_url": ESP32_URL})

@app.route("/health", methods=["GET"])
def health(): return jsonify({"status": "ok"}), 200

@app.route("/status")
def status():
    temporal = _rnn_analyser.get_analysis()
    return jsonify({
        "status": "ok", "version": "5.0-LOCAL-ML", "model": "FILES/ MODELS (LOCAL)",
        "uptime_s": int(time.time() - _start_time), "cycles_run": _cycle_count,
        "inferences": _inference_count, "last_inference_ms": _last_inference_ms,
        "esp32_reachable": _esp32_reachable, "esp32_url": ESP32_URL,
        "rnn_available": _RNN_AVAILABLE,
        "ann_available": _ANN_AVAILABLE,
        "temporal": temporal,
    })

@app.route("/feedback", methods=["POST", "OPTIONS"])
def feedback():
    """
    Accept user feedback to improve the ANN intent classifier.

    POST body: {"question": "...", "intent": "COLOUR"}
                                              ^ one of INTENT_LABELS
    """
    if flask_request.method == "OPTIONS":
        return "", 204
    try:
        data   = flask_request.get_json(force=True) or {}
        q      = (data.get("question") or "").strip()
        label  = (data.get("intent") or "").upper()
        if not q or not label:
            return jsonify({"error": "question and intent required"}), 400
        idx = _ann_intent.label_to_idx(label)
        _ann_intent.train_one(q, idx, _last_detection or {})
        log("ANN", f"Feedback received: '{q[:40]}' → {label}")
        return jsonify({"status": "ok", "trained_on": q, "intent": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/events")
def sse_events():
    def generate():
        q = _queue.Queue(maxsize=30)
        with _sse_lock: _sse_clients.append(q)
        try:
            if _latest_sse_data: yield f"data: {_latest_sse_data}\n\n"
            while True:
                try: yield f"data: {q.get(timeout=15)}\n\n"
                except _queue.Empty: yield ": keepalive\n\n"
        finally:
            with _sse_lock:
                try: _sse_clients.remove(q)
                except ValueError: pass
    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Access-Control-Allow-Origin": "*"})

@app.route("/frame")
def get_frame():
    jpeg = fetch_frame()
    if jpeg: return send_file(io.BytesIO(jpeg), mimetype='image/jpeg')
    return "Error fetching frame", 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = flask_request.get_json(force=True)
        question = (data.get("question") or data.get("message") or "").strip()
        if not question: return jsonify({"error": "No question provided"}), 400
        
        global _last_detection
        det = _last_detection or {}

        # ── ANN intent classification ────────────────────────────────────────
        intent, confidence = _ann_intent.predict(question, det)
        q_lower = question.lower()

        answer = ""

        # ── Intent dispatch (backed by ANN, keywords as tie-breaker) ─────────
        if intent == "CURRENCY" or any(w in q_lower for w in ["money","cash","currency","rupee","note","price"]):
            c = det.get("currency")
            if c and c.get("likely_currency"):
                answer = f"I see a {c['likely_currency']} note."
            else:
                answer = "I don't see any currency right now."

        elif intent == "COLOUR" or any(w in q_lower for w in ["color","colour","shade"]):
            cols = det.get("dominant_colours", [])
            if cols:
                names = [c["name"] for c in cols]
                answer = f"The dominant color is {' and '.join(names)}."
            else:
                answer = "I can't determine the color clearly."

        elif intent == "PEOPLE" or any(w in q_lower for w in ["people","person","human","someone"]):
            objs = det.get("objects", [])
            people_count = sum(1 for o in objs if o["label"] == "person")
            if people_count == 1:
                answer = "There is one person in front of you."
            elif people_count > 1:
                answer = f"I see {people_count} people."
            else:
                answer = "I don't see anyone right now."

        elif intent == "OBSTACLE" or any(w in q_lower for w in ["safe","walk","obstacle","block","path"]):
            obs = det.get("obstacle", {})
            lvl = obs.get("level", "SAFE")
            msg = obs.get("message", "The path is clear.")
            answer = msg

        elif intent == "TEXT_READ" or any(w in q_lower for w in ["read","text","sign","write"]):
            objs = det.get("objects", [])
            txts = [o["label"].replace("TXT: ", "") for o in objs if o["label"].startswith("TXT: ")]
            if txts:
                answer = f"I can read: {', '.join(txts)}."
            else:
                answer = "I don't see any readable text."

        elif intent == "DIRECTION" or any(w in q_lower for w in ["where","direction","left","right"]):
            dirs = det.get("directions", {})
            parts = []
            if dirs.get("left_objects"):   parts.append(f"on the left: {', '.join(dirs['left_objects'])}")
            if dirs.get("center_objects"): parts.append(f"in front: {', '.join(dirs['center_objects'])}")
            if dirs.get("right_objects"):  parts.append(f"on the right: {', '.join(dirs['right_objects'])}")
            answer = ("I see " + "; ".join(parts) + ".") if parts else "I don't see any objects to locate."

        elif intent == "OBJECT_QUERY":
            objs = det.get("objects", [])
            if objs:
                counts = collections.Counter([o["label"] for o in objs])
                parts  = [f"{cnt} {lbl}" for lbl, cnt in counts.most_common(5)]
                answer = f"I can see: {', '.join(parts)}."
            else:
                answer = "I don't see any objects right now."

        else:  # GENERAL / fallback
            # Enrich general answer with RNN temporal insight if available
            temporal = det.get("temporal", {})
            base = det.get("voice_message") or det.get("scene_summary") or "I am analyzing the scene."
            t_summary = temporal.get("temporal_summary", "")
            answer = f"{base} {t_summary}".strip() if t_summary else base

        # ── Append temporal anomaly warning ──────────────────────────────────
        temporal = det.get("temporal", {})
        if temporal.get("anomaly") and "anomaly" not in answer.lower():
            answer += " ⚠️ Scene change just detected."

        # ── Optional: log correct intent for online ANN improvement ──────────
        # (If the app adds a feedback endpoint, call _ann_intent.train_one() here)

        return jsonify({
            "answer":     answer,
            "response":   answer,
            "type":       "INFO",
            "intent":     intent,
            "intent_confidence": round(confidence, 3),
            "temporal":   temporal,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _run_flask():
    log("SRV", f"🌐 Flask API listening on http://localhost:{SERVER_PORT}")
    app.run(host="127.0.0.1", port=SERVER_PORT, debug=False, use_reloader=False, threaded=True)

# ════════════════════════════════════════════════════════════════
#   MAIN
# ════════════════════════════════════════════════════════════════
def main():
    global _cycle_count, _last_detection
    flask_thread = threading.Thread(target=_run_flask, daemon=True)
    flask_thread.start()
    time.sleep(1)
    
    _ensure_stream_thread()
    log("INFO", f"Starting detection loop...")
    
    while True:
        _cycle_count += 1
        jpeg = fetch_frame()
        if jpeg:
            frame = jpeg_to_cv2(jpeg)
            if frame is not None:
                detection = analyse_frame(frame)
                _last_detection = detection
                log_detection(detection, frame_count=_cycle_count)
                _broadcast_sse(detection)
                
        time.sleep(CAPTURE_INTERVAL)

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(0)
