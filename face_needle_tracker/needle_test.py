"""
Part 3 motor test: sweeps the needle over MQTT without the camera.

Publishes `recognized: true` messages with x sweeping back and forth across the
calibrated camera range, so the ESP8266/28BYJ-48 has to chase the target the
whole way (current vs target -> the needle visibly swings). Ends with
`recognized: false` so the firmware stops/holds the needle.

Use this to confirm the stepper + wiring + Wi-Fi + broker link BEFORE doing
camera calibration or face tracking:

    python face_needle_tracker/needle_test.py

Options: --broker/--port/--topic/--config, --period (s per x position),
         --cycles (full back-and-forth sweeps), --min-x/--max-x to force the
         swing range, --n-points, --debug (print every message).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from track_position import MQTTPublisher, load_config  # noqa: E402  # noqa: F401

DEFAULT_X_MIN, DEFAULT_X_MAX = 100.0, 545.0


def sweep_xs(x_min: float, x_max: float, n_points: int):
    """camera-X positions for one cycle: start..end..start (continuous)."""
    seq = [
        x_min + (x_max - x_min) * i / (n_points - 1)
        for i in range(n_points)
    ]
    return seq + [x for x in reversed(seq[:-1])]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 3 - sweep the needle over MQTT (motor/wiring test)")
    parser.add_argument("--broker", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--config", default=str(ROOT / "mqtt_config.json"))
    parser.add_argument("--period", type=float, default=0.15,
                        help="seconds between published X positions (default 0.15)")
    parser.add_argument("--cycles", type=int, default=2,
                        help="full back-and-forth sweeps (default 2)")
    parser.add_argument("--n-points", type=int, default=40,
                        help="X positions per sweep direction")
    parser.add_argument("--min-x", type=float, default=None,
                        help="override sweep start (camera X)")
    parser.add_argument("--max-x", type=float, default=None,
                        help="override sweep end (camera X)")
    parser.add_argument("--debug", action="store_true",
                        help="print every published message")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        cfg = {}
    broker = args.broker or cfg.get("broker", "localhost")
    port = args.port or int(cfg.get("port", 1883))
    topic = args.topic or cfg.get("topic", "face_track")

    cal = (cfg.get("calibration") or {}).get("points", [])
    if cal:
        cam_xs = [float(p["camera_x"]) for p in cal]
        default_min, default_max = min(cam_xs), max(cam_xs)
    else:
        default_min, default_max = DEFAULT_X_MIN, DEFAULT_X_MAX

    x_min = args.min_x if args.min_x is not None else default_min
    x_max = args.max_x if args.max_x is not None else default_max
    xs = sweep_xs(x_min, x_max, args.n_points)

    print(f"[TEST] sweeping needle via '{topic}' on {broker}:{port}")
    print(f"[TEST] camera_x {x_min:.0f} .. {x_max:.0f}"
          + (f"  ({len(cal)} calibration points)" if cal else "  (default range)"))
    print(f"[TEST] {args.cycles} cycle(s) x {len(xs)} positions, "
          f"{args.period:.2f}s between each - Ctrl+C to abort")

    pub = MQTTPublisher(broker=broker, port=port, topic=topic, interval_s=0.0)
    if not pub.wait_connected(timeout=10):
        print(f"[TEST] WARNING: broker {broker}:{port} not reachable - "
              f"the ESP won't receive anything", file=sys.stderr)
    try:
        for cycle in range(1, args.cycles + 1):
            for x in xs:
                pos = "LEFT" if x < (x_min + x_max) / 2 else "RIGHT"
                pub.publish(True, x=x, position=pos,
                            confidence=1.0, frame_width=1280)
                if args.debug:
                    print(f"[TEST] cycle {cycle}/{args.cycles}  "
                          f"x={x:.1f}  position={pos}")
                time.sleep(args.period)
        print("[TEST] sweep finished - holding (recognized=false)")
        pub.publish(False, frame_width=1280)
    except KeyboardInterrupt:
        print("\n[TEST] aborted - holding (recognized=false)")
        pub.publish(False, frame_width=1280)
    finally:
        pub.close()


if __name__ == "__main__":
    main()