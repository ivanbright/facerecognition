# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:

Haar (multi-face) -> FaceMesh 5pt (per-face ROI) -> align_face_5pt (112x112)
-> ArcFace ONNX embedding -> cosine distance to DB -> label each face.

Includes optional horizontal servo tracking: the primary detected face's
horizontal position is converted to a servo angle and sent to an ESP8266
over serial, panning the camera mount to follow the face.

Run:
    python -m src.recognize

Keys:
    q   : quit
    r   : reload DB from disk (data/face_database.pkl)
    +/- : adjust threshold (distance) live
    d   : toggle debug overlay
    t   : toggle servo tracking on/off

Notes:
- We run FaceMesh on EACH Haar face ROI (not the full frame). This avoids the
  "FaceMesh points not consistent with Haar box" problem and enables multi-face.
- DB is expected from enroll: data/face_database.pkl (name -> embedding vector)
- Distance definition: cosine_distance = 1 - cosine_similarity.
  Since embeddings are L2-normalized, cosine_similarity = dot(a,b).
- Servo tracking requires pyserial: pip install pyserial
"""
from __future__ import annotations

import time
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# Reuse your known-good alignment method (you said alignment is OK now)
from .haar_5pt import Haar5ptDetector, align_face_5pt

# Optional: pyserial for servo tracking (imported gracefully)
try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


# -------------------------
# Servo tracking config
# -------------------------
# Serial port and baud rate — must match the ESP8266 sketch
SERIAL_PORT = "COM4"
SERIAL_BAUD = 115200

# Servo range (degrees)
SERVO_MIN_ANGLE = 0
SERVO_MAX_ANGLE = 180
SERVO_CENTER_ANGLE = 90

# Proportional tracking
# When the face center is at the frame edge (offset = +/-1.0),
# the servo moves this many degrees from center.
SERVO_GAIN = 60.0          # max deflection from center (degrees at full offset)

# Deadzone: ignore offsets smaller than this (normalized -1..1 range)
# Prevents servo jitter when face is nearly centered.
TRACKING_DEADZONE = 0.03   # ~3% of frame width

# Proportional step: each frame nudges current angle toward target by this fraction
# of the remaining distance. Lower = smoother but slower; higher = faster but jerkier.
TRACKING_STEP_ALPHA = 0.35

# Minimum angle change (degrees) before sending to ESP8266.
# Avoids flooding the serial port with tiny updates.
SERVO_SEND_THRESHOLD = 2.0

# Frames without a face before holding the last angle
# (set to 0 to re-center immediately when face is lost)
TRACKING_LOST_HOLD_FRAMES = 30


# -------------------------
# Data
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2) float32 in FULL-frame coords


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# -------------------------
# Math helpers
# -------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a nicer face-like bbox from 5 points with asymmetric padding.
    kps: (5,2) in full-frame coords
    """
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))

    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)

    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    """
    Minimal geometry sanity:
    - eyes not collapsed
    - mouth generally below nose
    """
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist):
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


# -------------------------
# DB helpers
# -------------------------
def load_db_pickle(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    try:
        with open(db_path, "rb") as file:
            data = pickle.load(file)
        if not isinstance(data, dict):
            return {}
        out: Dict[str, np.ndarray] = {}
        for name, embedding in data.items():
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                out[str(name)] = vector / norm
        return out
    except Exception:
        return {}


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    if db_path.suffix == ".npz":
        try:
            data = np.load(str(db_path), allow_pickle=True)
            out: Dict[str, np.ndarray] = {}
            for key in data.files:
                out[key] = np.asarray(data[key], dtype=np.float32).reshape(-1)
            return out
        except Exception:
            return {}
    return load_db_pickle(db_path)


# -------------------------
# Embedder (same as embed_new)
# -------------------------
class ArcFaceEmbedderONNX:
    """
    ArcFace-style ONNX embedder.
    Input: 112x112 BGR -> internally RGB + (x-127.5)/128, then NCHW float32.
    Output: (1,D) or (D,)
    """

    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.model_path = model_path
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)

        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace ONNX model not found at: {model_path}\n"
                f"You need to obtain an ArcFace ONNX model and place it at this path.\n"
                f"Common sources:\n"
                f"- Convert from PyTorch/TensorFlow models\n"
                f"- Download from ONNX model zoo or similar repositories\n"
                f"- Use pre-trained models from face recognition libraries"
            )

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        in_shape = self.sess.get_inputs()[0].shape
        # The w600k_r50.onnx graph does NOT normalize pixels internally.
        # Verified empirically: feeding raw 0..255 saturates the network and all
        # embeddings collapse to the same vector (matched every face). Correct
        # input is NCHW float32 already normalized to (x-127.5)/128.
        self._channels_first = bool(len(in_shape) == 4 and in_shape[1] == 3)

        if self.debug:
            print("[embed] model:", model_path)
            print("[embed] input:", self.sess.get_inputs()[0].name, self.sess.get_inputs()[0].shape,
self.sess.get_inputs()[0].type)
            print("[embed] output:", self.sess.get_outputs()[0].name, self.sess.get_outputs()[0].shape,
self.sess.get_outputs()[0].type)

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        if self._channels_first:
            x = np.ascontiguousarray(rgb.transpose(2, 0, 1))[None, ...]  # (1, 3, 112, 112)
        else:
            x = rgb[None, ...]  # (1, 112, 112, 3)
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32)

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr_112)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._l2_normalize(emb)


# -------------------------
# Multi-face Haar + FaceMesh(ROI) 5pt
# -------------------------
class HaarFaceMesh5pt:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))

        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")

        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe"
            )

        # Create FaceLandmarker using the simpler create_from_model_path method
        # This will use MediaPipe's default face landmarker model
        try:
            self.landmarker = vision.FaceLandmarker.create_from_model_path("")
        except Exception:
            # If that fails, try the full options approach with default model
            try:
                # Download model if needed - MediaPipe 1.0+ can auto-download
                import urllib.request
                import os
                
                # Create models directory if it doesn't exist
                models_dir = "models"
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                
                model_path = os.path.join(models_dir, "face_landmarker.task")
                if not os.path.exists(model_path):
                    print("Downloading face landmarker model...")
                    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
                    urllib.request.urlretrieve(model_url, model_path)
                    print(f"Model downloaded to {model_path}")
                
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize FaceLandmarker: {e}")

        # 5pt indices for facial landmarks (these correspond to the 468-point model)
        # Left eye corner, Right eye corner, Nose tip, Left mouth corner, Right mouth corner
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263  
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32)  # (x,y,w,h)

    def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20:
            return None

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Process the image
        result = self.landmarker.detect(mp_image)
        
        if not result.face_landmarks:
            return None

        # Get the first face's landmarks
        landmarks = result.face_landmarks[0]
        
        # Extract the 5 key points
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP, 
                self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]
        
        pts = []
        for i in idxs:
            landmark = landmarks[i]
            # Convert normalized coordinates to pixel coordinates
            pts.append([landmark.x * W, landmark.y * H])
        
        kps = np.array(pts, dtype=np.float32)

        # enforce left/right ordering
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]

        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []

        # sort by area desc, keep top max_faces
        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]

        out: List[FaceDet] = []
        for (x, y, w, h) in faces:
            # expand ROI a bit for FaceMesh stability
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
            roi = frame_bgr[ry1:ry2, rx1:rx2]

            kps_roi = self._roi_facemesh_5pt(roi)
            if kps_roi is None:
                if self.debug:
                    print("[recognize] FaceMesh none for ROI -> skip")
                continue

            # map ROI kps back to full-frame coords
            kps = kps_roi.copy()
            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)

            # sanity: eye distance relative to Haar width
            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * float(w))):
                if self.debug:
                    print("[recognize] 5pt geometry failed -> skip")
                continue

            # build bbox from kps (centered)
            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)

            out.append(
                FaceDet(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=1.0,
                    kps=kps.astype(np.float32),
                )
            )

        return out


# -------------------------
# Matcher
# -------------------------
class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = float(dist_thresh)

        # pre-stack for speed
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
            # (K,D)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)

        e = emb.reshape(1, -1).astype(np.float32)  # (1,D)
        # cosine similarity since both sides are normalized: sim = dot
        sims = (self._mat @ e.T).reshape(-1)  # (K,)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim

        ok = best_dist <= self.dist_thresh
        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok),
        )


# -------------------------
# Demo
# -------------------------
def main():
    db_path = Path("data/face_database.pkl")

    det = HaarFaceMesh5pt(
        min_size=(70, 70),
        debug=False,
    )
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    db = load_db_npz(db_path)
    matcher = FaceDBMatcher(db=db, dist_thresh=0.40)  # matches similarity > 0.60

    # --- Servo tracking state ---
    tracking_enabled = _SERIAL_AVAILABLE
    ser_conn: Optional["serial.Serial"] = None
    if _SERIAL_AVAILABLE:
        try:
            ser_conn = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2)
            time.sleep(2)  # ESP8266 resets on serial open; wait for boot
            print(f"[tracking] Serial opened: {SERIAL_PORT} @ {SERIAL_BAUD}")
        except serial.SerialException as e:
            print(f"[tracking] Serial port {SERIAL_PORT} unavailable: {e}")
            print("[tracking] Tracking disabled. Recognition continues normally.")
            tracking_enabled = False
    else:
        print("[tracking] pyserial not installed. Tracking disabled.")
        print("[tracking] Install with: pip install pyserial")

    servo_angle = float(SERVO_CENTER_ANGLE)
    last_sent_angle = -999  # last angle sent to ESP8266 (for threshold check)
    frames_without_face = 0
    last_face_time = None

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    print("Recognize (multi-face). q=quit, r=reload DB, +/- threshold, d=debug overlay, t=toggle tracking")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=5)
        recognized_faces = []
        vis = frame.copy()

        # compute fps
        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()

        # draw + recognize each face
        # show aligned thumbnails stacked on the RIGHT, but lower to avoid overlay with green text
        h, w = vis.shape[:2]
        thumb = 112
        pad = 8
        x0 = w - thumb - pad
        y0 = 80  # moved down to avoid your text overlay area
        shown = 0

        for i, f in enumerate(faces):
            # draw bbox + kps
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)

            # align -> embed -> match
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned)
            mr = matcher.match(emb)

            # label
            label = mr.name if mr.name is not None else "Unknown"
            line1 = f"{label}"
            line2 = f"dist={mr.distance:.3f} sim={mr.similarity:.3f}"

            # color: known green, unknown red
            color = (0, 255, 0) if mr.accepted else (0, 0, 255)

            cv2.putText(vis, line1, (f.x1, max(0, f.y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(vis, line2, (f.x1, max(0, f.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if mr.accepted:
                recognized_faces.append((f, mr))
            if y0 + thumb <= h and shown < 4:
                vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                cv2.putText(
                    vis,
                    f"{i+1}:{label}",
                    (x0, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                y0 += thumb + pad
                shown += 1

            if show_debug:
                # show kps coords quickly
                dbg = f"kpsLeye=({f.kps[0,0]:.0f},{f.kps[0,1]:.0f})"
                cv2.putText(vis, dbg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Servo tracking: only recognized enrolled faces are tracked ---
        if tracking_enabled and ser_conn is not None and ser_conn.is_open:
            if recognized_faces:
                primary = recognized_faces[0][0]
                face_center_x = (primary.x1 + primary.x2) / 2.0
                # Normalize to -1.0 (left edge) .. +1.0 (right edge)
                offset = (face_center_x - (w / 2.0)) / (w / 2.0)
                offset = max(-1.0, min(1.0, offset))

                if abs(offset) > TRACKING_DEADZONE:
                    # Target angle: center + gain * offset
                    target_angle = SERVO_CENTER_ANGLE - SERVO_GAIN * offset
                    target_angle = max(float(SERVO_MIN_ANGLE), min(float(SERVO_MAX_ANGLE), target_angle))

                    # Proportional step toward target
                    servo_angle += TRACKING_STEP_ALPHA * (target_angle - servo_angle)
                    servo_angle = max(float(SERVO_MIN_ANGLE), min(float(SERVO_MAX_ANGLE), servo_angle))

                frames_without_face = 0
                last_face_time = time.time()
            else:
                now = time.time()
                if last_face_time is not None and (now - last_face_time) >= 1.0:
                    servo_angle = float(SERVO_CENTER_ANGLE)
                    last_sent_angle = -999
                    last_face_time = None
                else:
                    frames_without_face += 1

            # Only send when change exceeds threshold
            send_angle = int(round(servo_angle))

            if abs(send_angle - last_sent_angle) >= SERVO_SEND_THRESHOLD:
                try:
                    ser_conn.write(f"{send_angle}\r\n".encode("ascii"))
                    ser_conn.flush()
                    last_sent_angle = send_angle
                except serial.SerialException:
                    pass  # don't crash on write failure

            # Draw tracking status on the overlay
            track_state = f"SERVO {int(servo_angle)} deg" if tracking_enabled else "SERVO OFF"
            if len(recognized_faces) == 0 and frames_without_face > 0:
                track_state += f" (holding {frames_without_face}f)"
            cv2.putText(vis, track_state, (10, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # overlay header
        header = f"IDs={len(matcher._names)}  thr(dist)={matcher.dist_thresh:.2f}"
        if fps is not None:
            header += f"  fps={fps:.1f}"
        cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("recognize_new", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload_from(db_path)
            print(f"[recognize] reloaded DB: {len(matcher._names)} identities")
        elif key in (ord("+"), ord("=")):
            matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
            print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
        elif key == ord("-"):
            matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
            print(f"[recognize] thr(dist)={matcher.dist_thresh:.2f} (sim~{1.0-matcher.dist_thresh:.2f})")
        elif key == ord("d"):
            show_debug = not show_debug
            print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")
        elif key == ord("t"):
            tracking_enabled = not tracking_enabled
            if tracking_enabled and ser_conn is None:
                # Try to open serial if it wasn't available at startup
                if _SERIAL_AVAILABLE:
                    try:
                        ser_conn = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2)
                        time.sleep(2)
                        print(f"[tracking] Serial opened: {SERIAL_PORT} @ {SERIAL_BAUD}")
                    except serial.SerialException as e:
                        print(f"[tracking] Cannot open {SERIAL_PORT}: {e}")
                        tracking_enabled = False
            print(f"[tracking] {'ENABLED' if tracking_enabled else 'DISABLED'}")

    cap.release()
    cv2.destroyAllWindows()

    # Clean up serial connection
    if ser_conn is not None and ser_conn.is_open:
        try:
            # Return servo to center before closing
            ser_conn.write(f"{SERVO_CENTER_ANGLE}\r\n".encode("ascii"))
            ser_conn.flush()
            time.sleep(0.1)
            ser_conn.close()
            print("[tracking] Serial closed, servo centered.")
        except Exception:
            ser_conn.close()


if __name__ == "__main__":
    main()
