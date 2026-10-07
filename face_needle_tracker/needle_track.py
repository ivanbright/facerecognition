"""
Part 3: face_needle_tracker - physical needle drive over MQTT.

A separate project built ON TOP of Part 2. All recognition/tracking code stays
in the existing `facetrackingwithidentitylock` project (mapped in through the
filesystem junction `face_needle_tracker/facetrackingwithidentitylock` ->
`../facetrackingwithidentitylock`); this package only adds the Layer-3 pieces:

  * track_position.HorizontalTracker - horizontal state from the EXISTING
    LockedFaceTracker (smoothed center X, LEFT/CENTER/RIGHT, confidence).
  * track_position.PositionMapper    - calibrated camera_x -> motor-position
    mapping, piecewise-linear, clamped (no naive x/W*180).
  * track_position.MQTTPublisher     - compact JSON on topic `face_track` via
    Paho (non-blocking: the camera loop is never blocked by the network).
  * calibrate_position.py            - interactive camera<->needle calibration.
  * firmware/mqtt_stepper/           - ESP8266 subscriber + 28BYJ-48 driver.

Only the trained target moves the needle: unknown faces publish
`recognized: false` and the firmware holds.

Run (from the repo root):
    python face_needle_tracker/needle_track.py --target bright --mqtt

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

ROOT = Path(__file__).resolve().parent            # face_needle_tracker/
REPO = ROOT.parent                                # facerecognition/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# The old project imports its own folder as a script (plain `from face_signals`
# imports); map it onto sys.path so it can be reused as a package untouched.
OLD = REPO / "facetrackingwithidentitylock"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
os.chdir(REPO)  # models/, data/, camera_config.json and src.* resolve like Part 1/2

from src.recognize import (  # noqa: E402
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)
from facetrackingwithidentitylock.face_tracking import (  # noqa: E402
    FaceSignalExtractor,
    LockedFaceTracker,
    draw_label,
    list_cameras,
    load_camera_index,
    open_camera,
    signal_dict,
)
from track_position import (  # noqa: E402
    HorizontalTracker,
    MQTTPublisher,
    PositionMapper,
    load_config,
)

DEFAULT_MODEL = "models/embedder_arcface.onnx"
DEFAULT_DB = "data/face_database.pkl"


def main():
    parser = argparse.ArgumentParser(
        description="Part 3 - Face-identity needle tracking over MQTT")
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
                        help="requested camera width (0 = driver default)")
    parser.add_argument("--height", type=int, default=720,
                        help="requested camera height (0 = driver default)")
    parser.add_argument("--no-quality", action="store_true",
                        help="do not apply focus/quality settings")
    parser.add_argument("--brightness", type=float, default=None)
    parser.add_argument("--contrast", type=float, default=None)
    parser.add_argument("--saturation", type=float, default=None)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--smile-on", type=float, default=0.05)
    parser.add_argument("--smile-off", type=float, default=0.035)
    parser.add_argument("--smile-frames", type=int, default=4)
    parser.add_argument("--list-cameras", action="store_true",
                        help="probe camera indices/backends and exit (diagnostics)")
    parser.add_argument("--mqtt", action="store_true",
                        help="publish tracking JSON to Mosquitto (topic face_track)")
    parser.add_argument("--mqtt-config", default=str(ROOT / "mqtt_config.json"),
                        help="MQTT/calibration/motor config file")
    parser.add_argument("--broker", default=None,
                        help="override broker address from --mqtt-config")
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

    try:
        part3_cfg = load_config(args.mqtt_config)
    except FileNotFoundError:
        part3_cfg = None
    smoothing_alpha = float((part3_cfg or {}).get("smoothing_alpha", 0.30))
    dead_zone_ratio = float((part3_cfg or {}).get("dead_zone_ratio", 0.07))

    tracker = LockedFaceTracker(target, detector, embedder, matcher,
                                ema_alpha=smoothing_alpha,
                                dead_zone=dead_zone_ratio)
    signals = FaceSignalExtractor(
        smile_delta_on=args.smile_on,
        smile_delta_off=args.smile_off,
        smile_on_frames=args.smile_frames,
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
        print("[CAM] Run 'python face_needle_tracker/needle_track.py --list-cameras'")
        print("[CAM] to see which index/backend works, then update camera_config.json")
        print("[CAM] or use --camera. Close apps using the camera and allow desktop")
        print("[CAM] apps under Windows Settings > Privacy > Camera.")
        signals.close()
        return

    print(f"[LOCK] Identity lock target: '{target}'  Enrolled: {names}")
    print(f"[LOCK] threshold(dist)={args.threshold:.2f}  camera={index}")
    print(f"[LOCK] smile on>={args.smile_on:.3f} off<={args.smile_off:.3f} "
          f"hold>={args.smile_frames} frames")
    print(f"[LOCK] smoothing_alpha={smoothing_alpha:.2f}  "
          f"dead_zone(ratio)={dead_zone_ratio:.3f}")
    if not args.signal:
        print("Press 'q' to quit")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    horiz = HorizontalTracker(frame_width=W, dead_zone_ratio=dead_zone_ratio)
    mapper = PositionMapper.from_config(part3_cfg) if part3_cfg else PositionMapper()
    if part3_cfg is not None:
        cal = part3_cfg.get("calibration") or {}
        if cal.get("points"):
            mapper.load(ROOT / cal.get("file", "data/calibration.json"))
            print(f"[CAL] Calibration: {len(mapper.points)} points "
                  f"({mapper.min_motor:.0f}..{mapper.max_motor:.0f} steps)")

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
            view = cv2.flip(frame, 1)

            locked_now = locked_face is not None
            if locked_now:
                horiz.update(locked_face, tracker)
                target_steps = mapper.map_int(horiz.smoothed_x)
            else:
                target_steps = None
            if pub is not None:
                if locked_now:
                    pub.publish(True, x=horiz.smoothed_x, position=horiz.state,
                                confidence=horiz.confidence, frame_width=W)
                else:
                    pub.publish(False, frame_width=W)

            state_text = f"{tracker.state.name}: {target}"
            state_color = (0, 180, 0) if locked_face is not None else (0, 140, 255)

            face_state = None
            if locked_face is not None:
                box = tracker.box(locked_face)
                x1, y1, x2, y2 = box
                # The display is mirrored (selfie view) but analysis uses the raw
                # frame; only the drawn geometry is flipped to match the image.
                mx1 = W - x2
                cv2.rectangle(view, (mx1, y1), (W - x1, y2), (255, 170, 0), 3)
                face_state = signals.analyze(frame, box)
                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1
                    expression = "SMILE" if face_state.smiling else "NEUTRAL"
                    eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"
                    draw_label(view, expression, (mx1, max(55, y1 - 50)), (0, 255, 255))
                    draw_label(view, f"{eye_text}  blinks={blink_total}",
                               (mx1, max(78, y1 - 25)), (255, 255, 0))
                    draw_label(view, f"EAR={face_state.ear:.3f} "
                                     f"smile={face_state.smile_score:.3f} "
                                     f"neu={face_state.smile_neutral:.3f} "
                                     f"d={face_state.smile_delta:+.3f}",
                               (12, view.shape[0] - 18), (255, 255, 255), 0.52)
                draw_label(view,
                           f"H={position.horizontal} V={position.vertical} "
                           f"error=({position.error_x:+.2f},{position.error_y:+.2f})",
                           (12, 56), (255, 170, 0), 0.60)
                if target_steps is not None and mapper.is_calibrated():
                    draw_label(view,
                               f"X={horiz.raw_x:.0f} SM={horiz.smoothed_x:.0f} "
                               f"[{horiz.state}] needle={target_steps}",
                               (12, view.shape[0] - 45), (0, 255, 120), 0.60)
                else:
                    draw_label(view,
                               f"X={horiz.raw_x:.0f} SM={horiz.smoothed_x:.0f} "
                               f"[{horiz.state}] (uncalibrated)",
                               (12, view.shape[0] - 45), (0, 255, 120), 0.60)
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
                cv2.line(view, (w // 2, 0), (w // 2, h), (0, 200, 255), 1)
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
                cv2.imshow("Face Needle Tracker", view)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        cap.release()
        signals.close()
        if pub is not None:
            pub.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()