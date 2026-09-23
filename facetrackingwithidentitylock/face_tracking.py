"""
Part 2: Face tracking with identity lock.

Builds on Part 1's recognition pipeline (src.recognize: HaarFaceMesh5pt,
ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz). One enrolled identity is
locked and followed across frames; every other face is ignored. For the
locked face it reports smile, blink count, eyes-open/closed, and where the
face sits relative to frame center (above/below/left/right/center) as a
normalized error signal pair.

The output is a SOFTWARE signal only - no motor is moved. Use `--signal` to
print one JSON line per frame (the feed Part 3 will consume).

Run (from the repo root, like Part 1):
    python facetrackingwithidentitylock/face_tracking.py --target <name>

Keys (GUI mode): q quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # models/, data/, src.* then resolve exactly like Part 1

from src.recognize import (  # noqa: E402
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)
from src.haar_5pt import align_face_5pt  # noqa: E402
from face_signals import FaceSignalExtractor  # noqa: E402

DEFAULT_MODEL = "models/embedder_arcface.onnx"
DEFAULT_DB = "data/face_database.pkl"
CONFIG_FILE = "camera_config.json"


def load_camera_index() -> int:
    """Camera index from camera_config.json (Part 1 convention, default 0)."""
    try:
        import json as _json
        with open(CONFIG_FILE) as f:
            return int(_json.load(f)["camera_index"])
    except Exception:
        return 0


def list_cameras(max_index: int = 9) -> None:
    """Probe camera indices with both backends so a friend's machine can be
    diagnosed without the project: which index opens, which backend works,
    and at what resolution."""
    print(f"[CAM] Probing camera indices 0..{max_index}")
    for index in range(0, max_index + 1):
        results = []
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            name = "DSHOW" if api == cv2.CAP_DSHOW else "MSMF"
            try:
                cap = cv2.VideoCapture(index, api)
            except Exception as exc:
                results.append(f"{name}: threw {exc}")
                continue
            if not cap.isOpened():
                results.append(f"{name}: could not open")
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                results.append(f"{cap.getBackendName()}: OK "
                               f"{frame.shape[1]}x{frame.shape[0]}")
            else:
                results.append(f"{cap.getBackendName()}: opened but no frame")
            cap.release()
        print(f"[CAM] index {index}: " + " | ".join(results))
    print("[CAM] Hints:")
    print("[CAM]  1. Close apps using the camera (browser, Teams, Zoom, ...).")
    print("[CAM]  2. Windows Settings > Privacy > Camera: allow desktop apps.")
    print("[CAM]  3. Unplug and replug the USB camera, then rerun this.")
    print("[CAM]  4. If the OK index is not in camera_config.json, change it.")


def open_camera(index: int, width: int = 0, height: int = 0,
                apply_quality: bool = True, overrides: Optional[dict] = None):
    """
    Open the webcam (Part 1 style: DSHOW first, then MSMF), request a
    resolution explicitly, then apply focus/quality settings. Without this
    the driver keeps its own defaults, which is why the same camera looks
    soft/low on some PCs and sharp/high on others. 0 = driver default.
    overrides = {brightness|contrast|saturation|gain|exposure: float}.
    """
    for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        try:
            cap = cv2.VideoCapture(index, api)
        except Exception as exc:
            print(f"[CAM] backend {api} threw: {exc}")
            continue
        if not cap.isOpened():
            print(f"[CAM] backend {cap.getBackendName()} could not open index {index}")
            cap.release()
            continue
        if width > 0 and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        good = ok and frame is not None and float(frame.mean()) > 10
        if good:
            print(f"[CAM] Camera index {index} via {cap.getBackendName()}")
            if apply_quality:
                _apply_camera_quality(cap, overrides)
            _camera_report(cap)
            return cap
        mean = float(frame.mean()) if frame is not None else "n/a"
        print(f"[CAM] backend {cap.getBackendName()} opened index {index} "
              f"but the frame is unusable (mean={mean})")
        cap.release()
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        if width > 0 and height > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"[CAM] Camera index {index} (default backend)")
        if apply_quality:
            _apply_camera_quality(cap, overrides)
        _camera_report(cap)
        return cap
    return None


_PROP_BY_NAME = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "gain": cv2.CAP_PROP_GAIN,
    "exposure": cv2.CAP_PROP_EXPOSURE,
}


def _apply_camera_quality(cap, overrides: Optional[dict] = None) -> None:
    """
    Safe focus fixes by default (autofocus on, max sharpness) - these fix
    blur without touching color. Color/exposure are only changed when the
    user asks, because hardcoded values (e.g. brightness 0.5 + gain 50) wash
    the image out on cameras whose properties use a 0-255 scale.
    """
    overrides = overrides or {}
    settings = [
        (cv2.CAP_PROP_AUTOFOCUS, 1),
        (cv2.CAP_PROP_SHARPNESS, 100),
    ]
    applied = ["autofocus", "sharpness"]
    if overrides.get("exposure") is not None:
        settings.append((cv2.CAP_PROP_AUTO_EXPOSURE, 0.25))  # manual exposure
        settings.append((cv2.CAP_PROP_EXPOSURE, float(overrides["exposure"])))
        applied.append("exposure")
    for name in ("brightness", "contrast", "saturation", "gain"):
        value = overrides.get(name)
        if value is not None:
            settings.append((_PROP_BY_NAME[name], float(value)))
            applied.append(name)
    for prop, value in settings:
        try:
            cap.set(prop, value)
        except Exception:
            pass
    print(f"[CAM] Quality applied: {', '.join(applied)}")


def _camera_report(cap) -> None:
    """Let the driver settle, then report resolution and measured sharpness."""
    frame = None
    for _ in range(12):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] Resolution {w}x{h}")
    if frame is None:
        return
    sharpness = float(cv2.Laplacian(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    tag = "SHARP" if sharpness > 100 else ("OK" if sharpness > 50 else "BLURRY")
    print(f"[CAM] Sharpness (Laplacian var) {sharpness:.1f} -> {tag}")


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(
        self,
        target_name: str,
        detector,
        embedder,
        matcher,
        verify_every: int = 10,
        lost_timeout: int = 24,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
        search_faces: int = 3,
        window_pad: float = 0.80,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self.search_faces = search_faces
        self.window_pad = window_pad
        self.state = LockState.SEARCHING
        self.last_box = None
        self.smooth_center = None
        self.lost_frames = 0
        self.frame_index = 0

    @staticmethod
    def box(face):
        return (face.x1, face.y1, face.x2, face.y2)

    def identity(self, frame, face):
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        return self.matcher.match(self.embedder.embed(aligned))

    def target_is_verified(self, frame, face) -> bool:
        match = self.identity(frame, face)
        return match.accepted and match.name == self.target_name

    def acquire(self, frame, faces):
        best = None
        best_similarity = -1.0
        for face in faces:
            match = self.identity(frame, face)
            if (
                match.accepted
                and match.name == self.target_name
                and match.similarity > best_similarity
            ):
                best, best_similarity = face, match.similarity
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None
        last_center = center(self.last_box)
        last_diag = max(np.linalg.norm(np.array([
            self.last_box[2] - self.last_box[0],
            self.last_box[3] - self.last_box[1],
        ], dtype=np.float32)), 1.0)
        ranked = []
        for face in faces:
            box = self.box(face)
            overlap = iou(self.last_box, box)
            displacement = np.linalg.norm(center(box) - last_center) / last_diag
            score = overlap - 0.35 * displacement
            ranked.append((score, face))
        score, candidate = max(ranked, key=lambda item: item[0])
        return candidate if score > -0.30 else None

    def update(self, frame):
        self.frame_index += 1
        faces = self._detect(frame)

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)
            if (
                candidate is not None
                and (self.state == LockState.LOST
                     or self.frame_index % self.verify_every == 0)
                and not self.target_is_verified(frame, candidate)
            ):
                candidate = None

        if candidate is None:
            self.lost_frames += 1
            if self.last_box is not None:
                self.state = LockState.LOST
            if self.lost_frames > self.lost_timeout:
                self.state = LockState.SEARCHING
                self.last_box = None
                self.smooth_center = None
            return None, None

        self.state = LockState.LOCKED
        self.lost_frames = 0
        self.last_box = self.box(candidate)
        raw_center = center(self.last_box)
        if self.smooth_center is None:
            self.smooth_center = raw_center
        else:
            a = self.ema_alpha
            self.smooth_center = a * raw_center + (1.0 - a) * self.smooth_center
        return candidate, self.position_signal(frame.shape)

    def _detect(self, frame):
        """
        Full-frame detection while SEARCHING; once locked/lost we only run
        detection inside a padded window around the last known box. The face
        box, landmarks and keypoints are remapped back to full-frame coords,
        so everything downstream (associate, verify, signal) stays unchanged.
        This cuts the per-frame cost sharply (one small Haar/FaceMesh instead
        of a full 720p pass over up to 8 faces).
        """
        if self.state == LockState.SEARCHING or self.last_box is None:
            return self.detector.detect(frame, max_faces=self.search_faces)
        return self._detect_in_window(frame, self.last_box)

    def _detect_in_window(self, frame, box):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad_x, pad_y = self.window_pad * bw, self.window_pad * bh
        wx1 = max(0, int(round(x1 - pad_x)))
        wy1 = max(0, int(round(y1 - pad_y)))
        wx2 = min(width, int(round(x2 + pad_x)))
        wy2 = min(height, int(round(y2 + pad_y)))
        if wx2 - wx1 < 10 or wy2 - wy1 < 10:
            return []
        window = frame[wy1:wy2, wx1:wx2]
        faces = self.detector.detect(window, max_faces=self.search_faces)
        for face in faces:
            face.x1 += wx1
            face.y1 += wy1
            face.x2 += wx1
            face.y2 += wy1
            face.kps[:, 0] += wx1
            face.kps[:, 1] += wy1
        return faces

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        ex = float((self.smooth_center[0] - width / 2.0) / (width / 2.0))
        ey = float((self.smooth_center[1] - height / 2.0) / (height / 2.0))
        horizontal = "CENTER"
        vertical = "CENTER"
        if ex < -self.dead_zone:
            horizontal = "LEFT"
        elif ex > self.dead_zone:
            horizontal = "RIGHT"
        if ey < -self.dead_zone:
            vertical = "UP"
        elif ey > self.dead_zone:
            vertical = "DOWN"
        return TrackingSignal(ex, ey, horizontal, vertical)


def draw_label(frame, text, xy, color, scale=0.62):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def signal_dict(state, target, locked, frame_n, position=None, face_state=None, blink_total=0):
    d = {
        "part": 2,
        "frame": frame_n,
        "lock_state": state,
        "target": target,
        "identity_locked": locked,
    }
    if locked and position is not None:
        d.update(
            error_x=round(position.error_x, 4),
            error_y=round(position.error_y, 4),
            horizontal=position.horizontal,
            vertical=position.vertical,
        )
    if locked and face_state is not None:
        d.update(
            blink=bool(face_state.blink),
            eyes_closed=bool(face_state.eyes_closed),
            ear=round(face_state.ear, 4),
            smiling=bool(face_state.smiling),
            smile_score=round(face_state.smile_score, 4),
            blink_total=blink_total,
        )
    return d


def main():
    parser = argparse.ArgumentParser(description="Part 2 - Face tracking with identity lock")
    parser.add_argument("--target", default=None,
                        help="enrolled identity to lock (default: first enrolled name)")
    parser.add_argument("--camera", type=int, default=None,
                        help="camera index (default: camera_config.json)")
    parser.add_argument("--threshold", type=float, default=0.34,
                        help="cosine distance threshold (Part 1 sim 0.6 ~ dist 0.4)")
    parser.add_argument("--db", default=DEFAULT_DB, help="enrollment database path")
    parser.add_argument("--signal", action="store_true",
                        help="headless: print one JSON signal line per frame (Part 3 feed)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="quit after N frames in --signal mode (0 = never)")
    parser.add_argument("--width", type=int, default=1280,
                        help="requested camera width (0 = driver default)")
    parser.add_argument("--height", type=int, default=720,
                        help="requested camera height (0 = driver default)")
    parser.add_argument("--no-quality", action="store_true",
                        help="do not apply focus/quality settings")
    parser.add_argument("--brightness", type=float, default=None,
                        help="set camera brightness (unit depends on driver)")
    parser.add_argument("--contrast", type=float, default=None,
                        help="set camera contrast (unit depends on driver)")
    parser.add_argument("--saturation", type=float, default=None,
                        help="set camera saturation (unit depends on driver)")
    parser.add_argument("--gain", type=float, default=None,
                        help="set camera gain (unit depends on driver)")
    parser.add_argument("--exposure", type=float, default=None,
                        help="manual exposure value (disables auto exposure)")
    parser.add_argument("--smile-on", type=float, default=0.05,
                        help="smile trigger: rise of mouth/face ratio above neutral")
    parser.add_argument("--smile-off", type=float, default=0.035,
                        help="smile release: drop below neutral + this value")
    parser.add_argument("--list-cameras", action="store_true",
                        help="probe camera indices/backends and exit (diagnostics)")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    detector = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path=DEFAULT_MODEL,
        input_size=(112, 112),
        debug=False,
    )

    db = load_db_npz(Path(args.db))
    if not db:
        print(f"[DB] No enrolled identities in {args.db}")
        print("[DB] Enroll someone first: python working_face_recognition.py  (mode 1)")
        return
    names = sorted(db)
    target = args.target or names[0]
    if target not in db:
        print(f"[TARGET] '{target}' is not enrolled. Enrolled: {names}")
        return

    matcher = FaceDBMatcher(db, dist_thresh=args.threshold)
    tracker = LockedFaceTracker(target, detector, embedder, matcher)
    signals = FaceSignalExtractor(
        smile_delta_on=args.smile_on,
        smile_delta_off=args.smile_off,
    )

    index = args.camera if args.camera is not None else load_camera_index()
    overrides = {}
    for name in ("brightness", "contrast", "saturation", "gain", "exposure"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    cap = open_camera(index, args.width, args.height, not args.no_quality, overrides)
    if cap is None:
        print(f"[CAM] Camera index {index} not available")
        print("[CAM] Run 'python face_tracking.py --list-cameras' to see which")
        print("[CAM] index/backend works, then update camera_config.json or use --camera.")
        print("[CAM] Also: close apps using the camera, allow desktop apps under")
        print("[CAM] Windows Settings > Privacy > Camera, and replug the USB camera.")
        signals.close()
        return

    print(f"[LOCK] Identity lock target: '{target}'")
    print(f"[LOCK] Enrolled: {names}")
    print(f"[LOCK] threshold(dist)={args.threshold:.2f}  camera={index}")
    print(f"[LOCK] smile on>={args.smile_on:.3f} off<{args.smile_off:.3f} above neutral")
    if not args.signal:
        print("Press 'q' to quit")

    blink_total = 0
    frame_n = 0
    fps_start = time.time()
    fps_frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_n += 1

            locked_face, position = tracker.update(frame)
            view = frame.copy()

            state_text = f"{tracker.state.name}: {target}"
            state_color = (0, 180, 0) if locked_face is not None else (0, 140, 255)

            face_state = None
            if locked_face is not None:
                box = tracker.box(locked_face)
                x1, y1, x2, y2 = box
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 170, 0), 3)
                face_state = signals.analyze(frame, box)
                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1
                    expression = "SMILE" if face_state.smiling else "NEUTRAL"
                    eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"
                    draw_label(view, expression, (x1, max(55, y1 - 50)), (0, 255, 255))
                    draw_label(view, f"{eye_text}  blinks={blink_total}",
                               (x1, max(78, y1 - 25)), (255, 255, 0))
                    draw_label(view, f"EAR={face_state.ear:.3f} "
                                     f"smile={face_state.smile_score:.3f}",
                               (12, view.shape[0] - 18), (255, 255, 255), 0.52)
                draw_label(view,
                           f"H={position.horizontal} V={position.vertical} "
                           f"error=({position.error_x:+.2f},{position.error_y:+.2f})",
                           (12, 56), (255, 170, 0), 0.60)
            else:
                signals.reset()

            if args.signal:
                out = signal_dict(tracker.state.name, target,
                                  locked_face is not None, frame_n,
                                  position=position, face_state=face_state,
                                  blink_total=blink_total)
                print(json.dumps(out, sort_keys=True), flush=True)
                if args.max_frames and frame_n >= args.max_frames:
                    break
            else:
                fps_frames += 1
                now = time.time()
                if now - fps_start >= 1.0:
                    fps = fps_frames / (now - fps_start)
                    fps_frames = 0
                    fps_start = now
                draw_label(view, state_text, (12, 28), state_color, 0.72)
                h, w = view.shape[:2]
                dz = tracker.dead_zone
                cv2.rectangle(view,
                              (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                              (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                              (120, 120, 120), 1)
                if locked_face is None:
                    draw_label(view, "SEARCHING for target...",
                               (12, view.shape[0] - 45), state_color, 0.60)
                draw_label(view, f"{fps:.0f} fps", (w - 100, 28),
                           (255, 255, 255), 0.55)
                cv2.imshow("Locked Face Tracking", view)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        cap.release()
        signals.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()