"""
Part 3 calibration: camera X position <-> physical needle position.

Reuses the existing Part 2 pipeline (detector + identity lock) - mapped in
through the filesystem junction face_needle_tracker/facetrackingwithidentitylock
-> ../facetrackingwithidentitylock - to show the live smoothed face-center X
with the trained target locked, and lets the operator record calibration points
WITHOUT touching source code:

  * move your face so the needle (or a marker) sits at reference position i
  * press the number key i (1..N) -> current smoothed camera X is recorded
  * press 's' to save to data/calibration.json   (reused by needle_track.py)
  * press 'r' to discard this session's measurements
  * press 'q' to quit

The motor position for each reference is taken from mqtt_config.json
(calibration.points[i].motor_position), so the operator maps real camera-X
values onto the physical reference positions chosen on the hardware.

The 'data' junction points at the real ../data, so calibration.json lands in
the shared data folder with the enrollment database.

Run:
    python face_needle_tracker/calibrate_position.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Old project maps its folder onto sys.path for its script-style imports.
OLD = REPO / "facetrackingwithidentitylock"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))

from facetrackingwithidentitylock.face_tracking import (  # noqa: E402
    LockedFaceTracker,
    draw_label,
    load_camera_index,
    open_camera,
)
from src.recognize import (  # noqa: E402
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)
from track_position import (  # noqa: E402
    MQTTPublisher,
    load_config,
)

DEFAULT_MODEL = "models/embedder_arcface.onnx"
DEFAULT_DB = "data/face_database.pkl"


def print_calibration(points) -> None:
    if not points:
        print("  (no points measured yet)")
        return
    print("  idx  name           camera_x    motor_position")
    for i, name in enumerate(points):
        p = points[name]
        cam = p["camera_x"]
        cam_txt = f"{cam:.1f}" if p["measured"] else f"{cam:.1f} (default)"
        print(f"  {i + 1:<3d} {name:<14s} {cam_txt:>12s} {p['motor_position']:>12d}")


def ascii_plot(points) -> None:
    """Small in-terminal calibration plot (optional visual aid)."""
    if len(points) < 2:
        return
    # np.interp needs ascending xp - measurements are not necessarily taken
    # left-to-right, so sort the (camera_x, motor) pairs first.
    pairs = sorted((float(p["camera_x"]), float(p["motor_position"])) for p in points)
    xs = [a for a, _ in pairs]
    ms = [b for _, b in pairs]
    lo_x, hi_x = min(xs), max(xs)
    lo_m, hi_m = min(ms), max(ms)
    width = 40
    print("\n  camera_x -> motor position (clamped interpolation):")
    for i in range(width + 1):
        frac = i / width
        cam = lo_x + (hi_x - lo_x) * frac
        mot = np.interp(cam, xs, ms)
        col = int((mot - lo_m) / max(1e-9, hi_m - lo_m) * width)
        print(f"  {' ' * col}*")
    print(f"  camera_x {lo_x:.0f} .. {hi_x:.0f}   motor {lo_m:.0f} .. {hi_m:.0f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 3 camera-X calibration")
    parser.add_argument("--target", default=None)
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--config", default=str(ROOT / "mqtt_config.json"))
    parser.add_argument("--mqtt-config", default=str(ROOT / "mqtt_config.json"),
                        help="MQTT broker/topic for live needle feedback during calibration")
    parser.add_argument("--no-mqtt", action="store_true",
                        help="do not publish positions (needle won't follow your face)")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"[CAL] Config not found: {exc}")
        return
    points_cfg = cfg.get("calibration", {}).get("points", [])
    if not points_cfg:
        print("[CAL] mqtt_config.json has no calibration.points - nothing to do.")
        return

    detector = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(model_path=DEFAULT_MODEL, input_size=(112, 112))
    db = load_db_npz(Path(DEFAULT_DB))
    if not db:
        print("[CAL] No enrolled identities. Enroll first via working_face_recognition.py.")
        return
    target = args.target or sorted(db)[0]
    if target not in db:
        print(f"[CAL] '{target}' not enrolled. Enrolled: {sorted(db)}")
        return
    matcher = FaceDBMatcher(db, dist_thresh=0.34)
    tracker = LockedFaceTracker(target, detector, embedder, matcher)

    index = args.camera if args.camera is not None else load_camera_index()
    cap = open_camera(index, args.width, args.height, apply_quality=True)
    if cap is None:
        print(f"[CAL] Camera index {index} not available.")
        return

    examples = "  keys: " + " ".join(
        f"{i + 1}={p['name']}" for i, p in enumerate(points_cfg)
    )
    print(f"[CAL] Calibrating target: '{target}'  ({len(points_cfg)} reference points)")
    print(examples)
    print("[CAL] Move face so the needle is at a reference, press its number key.")
    print("[CAL] Then s=save  r=reset  q=quit")

    session = {
        p["name"]: {"motor_position": int(p["motor_position"]),
                    "camera_x": float(p["camera_x"]),
                    "measured": False}
        for p in points_cfg
    }

    pub = None
    if not args.no_mqtt:
        try:
            mcfg = load_config(args.mqtt_config)
        except FileNotFoundError:
            mcfg = None
        if mcfg and mcfg.get("broker"):
            pub = MQTTPublisher(
                broker=mcfg.get("broker", "localhost"),
                port=int(mcfg.get("port", 1883)),
                topic=mcfg.get("topic", "face_track"),
                interval_s=float(mcfg.get("publish_interval_s", 0.10)),
            )
            if not pub.wait_connected(timeout=5):
                print(f"[CAL] WARNING: broker {mcfg.get('broker')} unreachable - "
                      f"the needle won't follow your face during calibration")

    reason = "running"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Some webcams hiccup on the first frame(s) after open_camera's
                # quality pass. Retry before bailing so the GUI doesn't exit
                # silently on the very first read.
                print("[CAL] camera read failed - retrying ...")
                time.sleep(0.25)
                for _ in range(5):
                    ok, frame = cap.read()
                    if ok:
                        break
                if not ok:
                    reason = "camera stopped returning frames (after retries)"
                    print(f"[CAL] {reason} - exiting.")
                    break
            view = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            locked_face, _ = tracker.update(frame)
            if locked_face is not None:
                box = tracker.box(locked_face)
                # Display is mirrored (selfie view); raw box coords stay raw for
                # signal analysis, only the drawn box is flipped.
                cv2.rectangle(view, (w - box[2], box[1]), (w - box[0], box[3]),
                              (255, 170, 0), 3)
                smoothed = float(tracker.smooth_center[0])
                raw = float((box[0] + box[2]) / 2.0)
                draw_label(view, f"RAW X={raw:.1f}  SMOOTHED X={smoothed:.1f}",
                           (12, 30), (0, 255, 0), 0.60)
                if pub is not None:
                    ratio = float(mcfg.get("dead_zone_ratio", 0.07))
                    center = w / 2.0
                    if smoothed < center - ratio * center:
                        pos = "LEFT"
                    elif smoothed > center + ratio * center:
                        pos = "RIGHT"
                    else:
                        pos = "CENTER"
                    pub.publish(True, x=smoothed, position=pos,
                                confidence=1.0, frame_width=w)
            else:
                smoothed = None
                draw_label(view, f"SEARCHING for '{target}' - lock first", (12, 30),
                           (0, 140, 255), 0.60)
                if pub is not None:
                    pub.publish(False, frame_width=w)

            cv2.line(view, (w // 2, 0), (w // 2, h), (0, 200, 255), 1)
            if smoothed is not None:
                cx = w - int(smoothed)
                cv2.line(view, (cx, h - 40), (cx, h), (0, 255, 0), 2)
            recorded = sum(1 for n in session if session[n]["measured"])
            draw_label(view, f"recorded {recorded}/{len(session)}  (s=save q=quit)",
                       (12, 60), (255, 255, 255), 0.55)
            try:
                cv2.imshow("Part 3 Calibration", view)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as exc:
                reason = f"window error: {exc}"
                break
            if key == ord("q"):
                reason = "q pressed"
                break
            if ord("1") <= key <= ord("9"):
                i = key - ord("1")
                if i >= len(points_cfg):
                    continue
                if smoothed is None:
                    print("[CAL] No locked face - position face first.")
                    continue
                name = points_cfg[i]["name"]
                session[name]["camera_x"] = smoothed
                session[name]["measured"] = True
                print(f"[CAL] Point {i + 1} '{name}': camera_x = {smoothed:.1f}")
            elif key == ord("s"):
                missing = [n for n in session if not session[n]["measured"]]
                if missing:
                    print(f"[CAL] Not saving: {len(missing)} point(s) still unmeasured: "
                          f"{', '.join(missing)}")
                    continue
                out_path = ROOT / cfg["calibration"].get("file", "data/calibration.json")
                out = {"points": [
                    {"name": n, "camera_x": round(session[n]["camera_x"], 1),
                     "motor_position": session[n]["motor_position"]}
                    for n in session
                ]}
                out_path.parent.mkdir(exist_ok=True)
                out_path.write_text(json.dumps(out, indent=2))
                print(f"[CAL] Saved {len(out['points'])} points to {out_path}")
                print_calibration(session)
                ascii_plot(out["points"])
            elif key == ord("r"):
                for n in session:
                    session[n]["camera_x"] = float(
                        next(p["camera_x"] for p in points_cfg if p["name"] == n)
                    )
                    session[n]["measured"] = False
                print("[CAL] Measurements reset.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if pub is not None:
            pub.close()
    print(f"[CAL] exited: {reason}")


if __name__ == "__main__":
    main()