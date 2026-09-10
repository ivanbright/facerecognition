"""
tools/serial_test.py

Standalone serial test for ESP8266 servo tracker.

Sends a sequence of fixed angles over serial to confirm the ESP8266
receives and acts on them correctly. Run this AFTER the servo sweep
test confirms wiring is good, and BEFORE integrating into recognize.py.

Usage:
    python tools/serial_test.py

Controls:
    Enter an angle (0-180) and press ENTER to send it.
    Type 'sweep' to run a full 0-180-0 sweep.
    Type 'q' or 'quit' to exit.

Requirements:
    pip install pyserial
"""
from __future__ import annotations

import sys
import time

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("pyserial not installed. Run:  pip install pyserial")
    sys.exit(1)


# -------------------------
# Config
# -------------------------
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
TIMEOUT_S = 2.0       # serial read timeout
SETTLE_MS = 300       # ms to wait after sending an angle (servo needs time to move)


# -------------------------
# Helpers
# -------------------------
def list_ports() -> None:
    """Print available serial ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("[serial] No serial ports found.")
    else:
        print("[serial] Available ports:")
        for p in ports:
            print(f"  {p.device}  —  {p.description}")


def open_port(port: str, baud: int) -> serial.Serial:
    """Open serial port. Raises on failure."""
    s = serial.Serial(port, baud, timeout=TIMEOUT_S)
    time.sleep(2)  # ESP8266 resets on serial open; wait for it to boot
    print(f"[serial] Opened {port} @ {baud}")
    return s


def send_angle(ser: serial.Serial, angle: int) -> None:
    """Send a single angle as ASCII integer + newline."""
    angle = max(0, min(180, angle))
    msg = f"{angle}\n"
    ser.write(msg.encode("ascii"))
    ser.flush()
    print(f"[serial] Sent: {angle}")

    # Read back the echo (ESP8266 echoes "[tracker] angle=N")
    time.sleep(SETTLE_MS / 1000.0)
    while ser.in_waiting:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line:
            print(f"[serial] Recv: {line}")


def read_pending(ser: serial.Serial) -> None:
    """Read and print any pending serial data."""
    while ser.in_waiting:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line:
            print(f"[serial] Recv: {line}")


# -------------------------
# Demo sequences
# -------------------------
def test_sequence(ser: serial.Serial) -> None:
    """Send a predefined sequence of angles."""
    angles = [90, 45, 135, 0, 180, 90]
    print(f"\n[serial] Running test sequence: {angles}")
    for a in angles:
        send_angle(ser, a)
    print("[serial] Test sequence complete.\n")


def sweep_sequence(ser: serial.Serial) -> None:
    """Sweep 0 -> 180 -> 0 in 10-degree steps."""
    print("\n[serial] Running sweep: 0 -> 180 -> 0")
    for a in range(0, 181, 10):
        send_angle(ser, a)
    for a in range(180, -1, -10):
        send_angle(ser, a)
    print("[serial] Sweep complete.\n")


# -------------------------
# Main
# -------------------------
def main() -> None:
    print("=" * 50)
    print(" ESP8266 Servo Serial Test")
    print("=" * 50)

    list_ports()
    print(f"\n[serial] Attempting to open {SERIAL_PORT}...")

    try:
        ser = open_port(SERIAL_PORT, BAUD_RATE)
    except serial.SerialException as e:
        print(f"\n[serial] FAILED to open {SERIAL_PORT}: {e}")
        print("[serial] Possible causes:")
        print("  - Another program (Arduino Serial Monitor) has the port open")
        print("  - Wrong port — check Device Manager / ls /dev/tty*")
        print("  - ESP8266 not connected")
        print("\n[serial] Available ports listed above. Try a different port.\n")
        sys.exit(1)

    # Read any boot message from ESP8266
    time.sleep(1)
    read_pending(ser)

    print("\n[serial] Connected. Commands:")
    print("  <angle>    — send angle (0-180), e.g. '90'")
    print("  sweep      — full 0->180->0 sweep")
    print("  test       — predefined test sequence")
    print("  ports      — list serial ports")
    print("  q / quit   — exit\n")

    try:
        while True:
            try:
                raw = input("angle> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[serial] Exiting.")
                break

            if not raw:
                continue

            if raw.lower() in ("q", "quit", "exit"):
                print("[serial] Exiting.")
                break

            if raw.lower() == "ports":
                list_ports()
                continue

            if raw.lower() == "test":
                test_sequence(ser)
                continue

            if raw.lower() == "sweep":
                sweep_sequence(ser)
                continue

            # Try parsing as integer angle
            try:
                angle = int(raw)
            except ValueError:
                print(f"[serial] Unknown command: '{raw}'")
                continue

            if angle < 0 or angle > 180:
                print("[serial] Angle must be 0-180")
                continue

            send_angle(ser, angle)

    finally:
        if ser.is_open:
            ser.close()
            print("[serial] Port closed.")


if __name__ == "__main__":
    main()
