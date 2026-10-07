# Firmware

Two rigs live here:

1. `servo_tracker/` - the older serial SERVO rig (below).
2. The Part 3 needle rig moved into its own project: `face_needle_tracker/firmware/mqtt_stepper/` (MQTT + 28BYJ-48 tachometer needle, section at the end).

---

# Servo Tracker Firmware

ESP8266 firmware for the face-tracking camera rig. The servo pans a camera mount horizontally to follow a detected face.

## Components

- ESP8266MOD (e.g. NodeMCU v2/v3) — USB-connected to PC
- Single hobby servo (e.g. SG90, MG90S)
- USB cable for power/programming

## Wiring

| Servo Wire | Connects To               |
|------------|---------------------------|
| Signal     | ESP8266 **GPIO D2 (GPIO4)** |
| VCC (red)  | ESP8266 **5V** pin         |
| GND (brown)| ESP8266 **GND** pin        |

> **Pin confirmation required:** Both `.ino` files use `const int SERVO_PIN = 4` (GPIO4 = D2 on most NodeMCU boards). Check your board's silk screen or pinout diagram to confirm D2/GPIO4 is correct. If your board labels pins differently, update the `SERVO_PIN` constant in the sketch before uploading.

> **Power note:** Most small servos (SG90) work from the ESP8266's 5V pin via USB. Larger servos may draw too much current and cause the ESP8266 to brown out. If the board resets when the servo moves, use an external 5V supply with a common ground.

## Baud Rate

**115200** — both the Arduino sketch and the PC-side Python code must use the same baud rate. This is configured in:
- `Serial.begin(115200)` in the `.ino` sketches
- `BAUD_RATE = 115200` in `tools/serial_test.py`
- `SERIAL_BAUD = 11200` in `src/recognize.py`

## Serial Protocol

PC sends a plain ASCII integer angle (0–180) terminated by newline:

```
97\n
```

The ESP8266 parses it, clamps to 0–180, and calls `servo.write(angle)`. Malformed input is silently ignored.

## Testing Order

Follow these three steps **in order**. Do not skip or reorder.

### Step 1 — Servo Sweep (wiring test)

**File:** `firmware/servo_sweep_test/servo_sweep_test.ino`

Upload with Arduino IDE. The servo sweeps 0 → 180 → 0 continuously with no serial input.

**Verify:** The servo moves smoothly through the full range. If it stutters or the board resets, check wiring and consider external power.

### Step 2 — Serial Test (communication test)

**File:** `firmware/servo_tracker/servo_tracker.ino`

Upload with Arduino IDE. The servo sits at 90° center.

Then from the PC:

```bash
pip install pyserial
python tools/serial_test.py
```

Type angles (e.g. `0`, `90`, `180`) and press Enter. Or type `sweep` for a full range test.

**Verify:** The servo moves to the requested angle and the ESP8266 echoes back the angle.

### Step 3 — Recognition + Tracking (integration)

**File:** `src/recognize.py`

Prerequisites:
- Step 1 and Step 2 passed
- At least one person enrolled (`python -m src.enroll`)
- ArcFace ONNX model in `models/embedder_arcface.onnx`

```bash
pip install pyserial
python -m src.recognize
```

The servo follows the detected face horizontally. Press `t` to toggle tracking on/off.

**Verify:** Moving left/right in front of the camera causes the servo to pan. The on-screen overlay shows the current servo angle.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `[tracking] Serial port COM4 unavailable` | Port open in another program | Close Arduino Serial Monitor or other terminal |
| Servo twitches then board resets | Insufficient power | Use external 5V supply for servo |
| Servo doesn't move at all | Wrong GPIO pin | Confirm `SERVO_PIN` matches your board's pinout |
| Tracking stutters | Gain or deadzone too aggressive | Adjust `SERVO_GAIN`, `TRACKING_DEADZONE`, `TRACKING_STEP_ALPHA` in `src/recognize.py` |

---

# Part 3: mqtt_stepper.ino (MQTT + 28BYJ-48 tachometer needle)

Lives in the **new project folder**: `face_needle_tracker/firmware/mqtt_stepper/mqtt_stepper.ino`.

The needle rig. ESP8266 subscribes to `face_track` on a Mosquitto broker,
maps the recognized face's smoothed camera X to an absolute motor position
(piecewise-linear, clamped), and drives a 28BYJ-48 through a ULN2003 driver.
Stateful, non-blocking: current vs target position, bounded steps per loop.

## Wiring (28BYJ-48 / ULN2003)

| Motor wire (ULN board) | ESP8266 GPIO | NodeMCU label |
|------------------------|-------------|---------------|
| IN1                    | GPIO0       | D3            |
| IN2                    | GPIO2       | D4            |
| IN3                    | GPIO4       | D2            |
| IN4                    | GPIO5       | D1            |
| Motor 5-12V / GND      | external PSU | common GND   |

Verify pin labels against your board's silkscreen; they are `#define`s at the
top of the sketch. The stepper's `VCC` should come from its driver supply, not
the ESP's 3.3V.

## Before flashing

Edit the top of `mqtt_stepper.ino`:

- `WIFI_SSID` / `WIFI_PASS`
- `MQTT_BROKER` = the PC's LAN IP (never `localhost` on the ESP)
- Calibration arrays `CAM_X[]` / `MOTOR_STEPS[]` — paste the values that the
  PC-side calibration produces (see main README, Part 3 calibration), and set
  `CAL_N` to match.

## Motor step model

- 28BYJ-48: 64 motor steps x 64:1 gear = **4096 half-steps per revolution**
  (~2048 full-steps). The firmware drives a standard 8-phase half-step
  sequence.
- `MAX_MOTOR_POSITION` etc. are absolute *software* step counts from the
  documented reference (the sketch starts at `MIN_MOTOR_POSITION`; there is no
  limit switch / homing, so the reference is calibrated instead).
- If the needle mirrors the face, set `DIRECTION_FLIP = true`.

## Libraries

`PubSubClient` and `ArduinoJson` (v6) via the Arduino Library Manager.
