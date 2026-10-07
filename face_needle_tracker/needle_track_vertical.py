"""
Part 3 - vertical (Y) face tracking over MQTT.

Separate companion to needle_track.py: instead of mapping the locked face's
horizontal center X to the needle, it maps the VERTICAL center Y so that

    face rises  -> needle sweeps to the LEFT
    face dips   -> needle sweeps to the RIGHT

Same identity-lock, MQTT topic (`face_track`) and ESP firmware as the horizontal
pipeline - the ESP is untouched. This file only resamples Y into the camera_x
domain the firmware already maps (inverted through the calibrated points), so no
re-flash is needed and the existing calibration keeps driving the same rig.

The selfie/mirror display rule is kept: the view is flipped so right is right on
screen, and drawn geometry is mirrored to match. Vertical up/down is unaffected
by that horizontal mirror.

Run (from the repo root):
    python face_needle_tracker/needle_track_vertical.py --target kelia --mqtt

Optional config in mqtt_config.json -> "calibration_vertical":
    { "y_top": 40, "y_bottom": 680 }   (face-Y pixel range mapped to the needle)

Keys (GUI mode): q quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent            # face_needle_tracker/
REPO = ROOT.parent                                # facerecognition/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OLD = REPO / "facetrackingwithidentitylock"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
os.chdir(REPO)

from src.recognize import (  # noqa: E402
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)
from facetrackingwithidentitylock.face_tracking import (  # noqa: E402
    LockedFaceTracker,
    draw_label,
    list_cameras,
    load_camera_index,
    open_camera,
    signal_dict,
)
from track_position import (  # noqa: E402
    MQTTPublisher,
    load_config,
)

DEFAULT_MODEL = "models/embedder_arcface.onnx"
DEFAULT_DB = "data/face_database.pkl"


class VerticalMapper:
    """Maps the locked face's vertical Y into the camera_x value to send.

    The ESP maps its camera_x calibration straight to motor position, so to make
    the needle sweep LEFT when the face rises (and RIGHT when it dips) we invert
    the calibration here: pick the motor position proportional to Y and resample
    that motor back to the x the firmware already understands.
    """

    def __init__(self, xs=None, ms=None, y_top=0.0, y_bottom=720.0):
        self.xs = [float(v) for v in (xs or [100.0, 545.0])]
        self.ms = [float(v) for v in (ms or [100.0, 1948.0])]
        self.y_top = float(y_top)
        self.y_bottom = float(y_bottom)
        self._xs = np.asarray(self.xs)
        self._ms = np.asarray(self.ms)

    @classmethod
    def from_config(cls, cfg, default_height: int) -> "VerticalMapper":
        pts = ((cfg or {}).get("calibration") or {}).get("points") or []
        if pts:
            xs = [float(p["camera_x"]) for p in pts]
            ms = [float(p["motor_position"]) for p in pts]
        else:
            xs, ms = None, None
        vert = (cfg or {}).get("calibration_vertical") or {}
        y_top = float(vert.get("y_top", 0.0))
        y_bottom = float(vert.get("y_bottom", default_height))
        return cls(xs, ms, y_top, y_bottom)

    def target_x(self, y: float) -> float:
        """Returns the camera_x value to publish for face-center Y (pixels)."""
        y = min(max(float(y), self.y_top), self.y_bottom)
        span = max(1e-9, self.y_bottom - self.y_top)
        t = (y - self.y_top) / span                # 0 at top, 1 at bottom
        # ms[-1] parks the needle LEFT, ms[0] parks it RIGHT (flipped rig), so:
        motor = self.ms[-1] + (self.ms[0] - self.ms[-1]) * t
        motor = min(max(motor, self.ms[0]), self.ms[-1])
        return float(np.interp(motor, self._ms, self._xs))


def main():
    parser = argparse.ArgumentParser(
        description="Part 3 - Face-identity VERTICAL needle tracking over MQTT")
    parser.add_argument("--target", default=None,
                        help="enrolled identity to lock (default: first enrolled name)")
    parser.add_argument("--camera", type=int, default=None,
                        help="camera index (default: camera_config.json)")
    parser.add_argument("--threshold", type=float, default=0.34,
                        help="cosine distance threshold (Part 1 sim 0.6 ~ dist 0.4)")
    parser.add_argument("--db", default=DEFAULT_DB, help="enrollment database path")
    parser.add_argument("--signal", action="store_true",
                        help="headless: print one JSON signal line per frame")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="quit after N frames in --signal mode (0 = never)")
    parser.add_argument("--width", type=int, default=1280,
                        help="requested camera width")
    parser.add_argument("--height", type=int, default=720,
                        help="requested camera height")
    parser.add_argument("--no-quality", action="store_true",
                        help="do not apply focus/quality settings")
    parser.add_argument("--list-cameras", action="store_true",
                        help="probe camera indices/backends and exit (diagnostics)")
    parser.add_argument("--mqtt", action="store_true",
                        help="publish tracking JSON to Mosquitto (topic face_track)")
    parser.add_argument("--mqtt-config", default=str(ROOT / "mqtt_config.json"),
                        help="MQTT/calibration config file")
    parser.add_argument("--broker", default=None,
                        help="override broker address from --mqtt-config")
    parser.add_argument("--y-top", type=float, default=None,
                        help="override vertical map top (pixels)")
    parser.add_argument("--y-bottom", type=float, default=None,
                        help="override vertical map bottom (pixels)")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    detector = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(model_path=DEFAULT_MODEL, input_size=(112, 112),
                                   debug=False)
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

    try:
        part3_cfg = load_config(args.mqtt_config)
    except FileNotFoundError:
        part3_cfg = None
    smoothing_alpha = float((part3_cfg or {}).get("smoothing_alpha", 0.30))
    dead_zone_ratio = float((part3_cfg or {}).get("dead_zone_ratio", 0.07))

    tracker = LockedFaceTracker(target, detector, embedder, matcher,
                                ema_alpha=smoothing_alpha,
                                dead_zone=dead_zone_ratio)

    index = args.camera if args.camera is not None else load_camera_index()
    cap = open_camera(index, args.width, args.height, not args.no_quality)
    if cap is None:
        print(f"[CAM] Camera index {index} not available")
        print("[CAM] Run 'python face_needle_tracker/needle_track_vertical.py --list-cameras'")
        return

    print(f"[LOCK] Identity lock target: '{target}'  Enrolled: {names}")
    print(f"[LOCK] threshold(dist)={args.threshold:.2f}  camera={index}")
    print(f"[LOCK] smoothing_alpha={smoothing_alpha:.2f}  "
          f"dead_zone(ratio)={dead_zone_ratio:.3f}")
    if not args.signal:
        print("Vertical rule: face UP -> needle LEFT, face DOWN -> needle RIGHT")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height or 720

    vmap = VerticalMapper.from_config(part3_cfg, H)
    if args.y_top is not None:
        vmap.y_top = float(args.y_top)
    if args.y_bottom is not None:
        vmap.y_bottom = float(args.y_bottom)
    if vmap.y_bottom <= vmap.y_top:
        print(f"[V] BAD vertical range [{vmap.y_top}, {vmap.y_bottom}] - exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    pub = None
    if args.mqtt:
        if part3_cfg is not None:
            broker = args.broker or part3_cfg.get("broker", "localhost")
            port = int(part3_cfg.get("port", 1883))
            topic = part3_cfg.get("topic", "face_track")
            interval = float(part3_cfg.get("publish_interval_s", 0.10))
            lost = part3_cfg.get("face_lost_timeout_s", 2.0)
        else:
            broker = args.broker or "localhost"
            port, topic, interval, lost = 1883, "face_track", 0.10, 2.0
        pub = MQTTPublisher(broker=broker, port=port, topic=topic, interval_s=interval)
        if not pub.wait_connected(timeout=10):
            print(f"[MQTT] WARNING: broker {broker}:{port} not reachable after 10s - "
                  f"publishes will be dropped until it connects", file=sys.stderr)
        print(f"[MQTT] publishing to {broker}:{port} on '{topic}' "
              f"every {interval:.2f}s (face_lost={lost}s)")
    print(f"[V] vertical map: camera_y [{vmap.y_top:.0f}, {vmap.y_bottom:.0f}] -> "
          f"camera_x [{vmap.xs[0]:.0f}, {vmap.xs[-1]:.0f}] "
          f"(motor [{vmap.ms[0]:.0f}, {vmap.ms[-1]:.0f}])")

    e_y = None                       # EMA of the face center Y
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
            view = cv2.flip(frame, 1)

            locked_now = locked_face is not None
            x_sent = None
            y_raw = None
            v_state = "CENTER"
            if locked_now:
                box = tracker.box(locked_face)
                y_raw = (box[1] + box[3]) / 2.0
                e_y = y_raw if e_y is None else (
                    smoothing_alpha * y_raw + (1.0 - smoothing_alpha) * e_y)
                x_sent = vmap.target_x(e_y)
                mid = H / 2.0
                if e_y < mid - dead_zone_ratio * mid:
                    v_state = "UP"
                elif e_y > mid + dead_zone_ratio * mid:
                    v_state = "DOWN"

            if pub is not None:
                if locked_now:
                    pub.publish(True, x=x_sent, position=v_state,
                                confidence=1.0, frame_width=W)
                else:
                    pub.publish(False, frame_width=W)

            if locked_now:
                box = tracker.box(locked_face)
                # mirrored display: only the horizontal geometry is flipped
                cv2.rectangle(view, (W - box[2], box[1]), (W - box[0], box[3]),
                              (255, 170, 0), 3)
                draw_label(view, f"Y RAW={y_raw:.0f}  Y SM={e_y:.0f}  [{v_state}]  "
                                 f"x_sent={x_sent:.0f}",
                           (12, 30), (0, 255, 0), 0.60)
            else:
                e_y = None
                draw_label(view, f"SEARCHING for '{target}' - lock first", (12, 30),
                           (0, 140, 255), 0.60)

            # vertical center reference + tracked-Y marker
            cv2.line(view, (0, H // 2), (W, H // 2), (0, 200, 255), 1)
            if e_y is not None:
                cy = int(e_y)
                cv2.line(view, (0, cy), (W, cy), (0, 255, 0), 2)

            if args.signal:
                out = signal_dict(tracker.state.name, target,
                                  locked_now, frame_n, position=position)
                out["y_smoothed"] = round(e_y, 1) if e_y is not None else None
                out["v_state"] = v_state
                out["x_sent"] = round(x_sent, 1) if x_sent is not None else None
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
                state_text = f"{tracker.state.name}: {target}"
                state_color = (0, 180, 0) if locked_now else (0, 140, 255)
                draw_label(view, state_text, (12, 28), state_color, 0.72)
                draw_label(view, f"{fps:.0f} fps", (W - 100, 28),
                           (255, 255, 255), 0.55)
                cv2.imshow("Face Needle Tracker (VERTICAL)", view)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        cap.release()
        if pub is not None:
            pub.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()