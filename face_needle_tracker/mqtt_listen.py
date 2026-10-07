"""
Independent MQTT subscriber - verifies the Python -> Mosquitto link (Test 5)
and the ESP8266's view of the messages (Test 6). Subscribe to `face_track`
and print each compact JSON payload. Run it on the same PC as needle_track.py
(`localhost`) or from anywhere on the network (`--broker <ip>`).

    python face_needle_tracker/mqtt_listen.py --broker localhost
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paho.mqtt.client as mqtt  # noqa: E402

from track_position import load_config  # noqa: E402


def on_message(client, userdata, msg) -> None:
    try:
        data = json.loads(msg.payload.decode("utf-8", errors="replace"))
    except Exception as exc:
        print(f"[LISTEN] malformed JSON on {msg.topic}: {msg.payload!r} ({exc})")
        return
    rec = data.get("recognized")
    x = data.get("x")
    pos = data.get("position")
    conf = data.get("confidence")
    w = data.get("frame_width")
    ts = data.get("timestamp")
    print(f"[LISTEN] received on '{msg.topic}': recognized={rec} x={x} "
          f"position={pos} confidence={conf} frame_width={w} ts={ts}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Subscribe to face_track and print messages")
    parser.add_argument("--broker", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--config", default=str(ROOT / "mqtt_config.json"))
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        broker = args.broker or cfg["broker"]
        port = args.port or cfg.get("port", 1883)
        topic = args.topic or cfg.get("topic", "face_track")
    except FileNotFoundError:
        broker, port, topic = args.broker, args.port or 1883, args.topic or "face_track"

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"facelock_listen_{os.getpid()}")
    client.on_message = on_message
    print(f"[LISTEN] connecting to {broker}:{port}, subscribing '{topic}' ...")
    client.connect(broker, port)
    client.subscribe(topic)
    print(f"[LISTEN] connected + subscribed to '{topic}' (Ctrl+C to quit)")
    client.loop_forever()


if __name__ == "__main__":
    main()