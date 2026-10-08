# Face Tracking Thing (A.K.A. "please just follow me, servo")

> The face-locking project (identity lock + smile/blink/position signal, no
> hardware) also has its own repo: **https://github.com/ivanbright/FaceLocking**.
> This repo is the full rig — enrollment, recognition, servo, and firmware.
>
> Related projects (each easy to clone on its own):
> - **https://github.com/ivanbright/FaceLocking** — Part 2, identity lock only
>   (the code here is the source of truth; that repo mirrors it).
> - **https://github.com/ivanbright/face-needle-tracker** — Part 3, the
>   face→stepper needle rig (horizontal + vertical tracking) built on top of
>   Part 1 (`src/`) and Part 2 (`facetrackingwithidentitylock/`).

A small project where a camera on a little servo pans around to follow *my*
face — not strangers, not the cat, just people I actually enrolled.

## The hardware situation

- A USB webcam. **On the device it is camera index `2`.** Camera `0` on a
  normal PC. On this rig, `0` is black and it confused me for a whole day.
- An ESP8266 (Adafruit Feather Huzzah) on `COM4`, 115200 baud.
- A servo on GPIO `14`.

## Getting it running

1. Install the usual suspects:

   ```bash
   pip install opencv-python onnxruntime numpy pyserial mediapipe paho-mqtt
   ```

2. Put the ArcFace model at `models/embedder_arcface.onnx`
   (`download_arcface_model.py` can grab it).

3. Flash `firmware/servo_tracker/servo_tracker.ino` with the Arduino IDE
   (ESP8266 board, `COM4`), then open the Serial Monitor once to see
   `[tracker] Ready`. **Close the Serial Monitor after** — if it stays open,
   Python can't open the port (`Access is denied`, I learned that one too).

## Using it

Enroll someone (this overwrites/adds to `data/face_database.pkl`):

```bash
python working_face_recognition.py   # pick mode 1
```

Follow that person:

```bash
python working_face_recognition.py   # pick mode 2
```

In mode 1: `SPACE` grabs one aligned sample, `s` saves, `q` quits. The box
won't capture until it gets a solid five-point lock. Try a few angles.

## Part 2: Face tracking with identity lock

Same recognition pipeline, but the output is a **software signal** — no motor
moves. It locks one enrolled identity, ignores every other face, and for the
locked face reports smile, blink count, eyes open/closed, and where the face
sits relative to frame center as a normalized error signal pair
(`error_x`, `error_y`). Part 3 consumes the signal.

```bash
python facetrackingwithidentitylock/face_tracking.py --target ivan
```

- GUI mode: boxes the locked face, shows SMILE/NEUTRAL, EYES OPEN/CLOSED,
  `blinks=`, EAR, smile score, the error signal, and live fps. `q` quits.
- `--signal` prints one JSON line per frame (the Part 3 feed):
  `python facetrackingwithidentitylock/face_tracking.py --target ivan --signal --max-frames 300`
- Tuning: `--threshold` (cosine dist, ~0.34), `--smile-on/--smile-off` (mouth/face
  ratio rise above the person's tracked neutral value), camera color via
  `--brightness/--contrast/--saturation/--gain/--exposure`.

It needs two model files in `models/` (ArcFace is shared with Part 1, the
landmarker is new):

| Model | Where to get it |
| --- | --- |
| `models/embedder_arcface.onnx` (w600k_r50) | GitHub release zip: `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip` (extract `w600k_r50.onnx`) — or the public HuggingFace mirror `https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/w600k_r50.onnx` |
| `models/face_landmarker.task` | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` |

Note: the `deepinsight/insightface` HuggingFace repo is login-gated — use the
GitHub release or the `deepghs` mirror above.

## Part 3: MQTT → ESP8266 → 28BYJ-48 tachometer needle

Same recognition pipeline, but the locked face's **horizontal** position now
drives a physical needle. Only the *trained* face moves it — unknown faces
publish `recognized: false` and the needle holds.

```
webcam -> existing identity lock -> smoothed face center X
  -> calibrated camera_x -> motor-position map (piecewise, clamped)
  -> Paho MQTT (topic face_track) -> Mosquitto -> Wi-Fi
  -> ESP8266 (mqtt_stepper.ino) -> 28BYJ-48 (ULN2003) -> needle
```

Part 3 lives in its **own project folder**, `face_needle_tracker/`, mapped to
the existing code through the filesystem (NTFS junctions):
`face_needle_tracker/facetrackingwithidentitylock`, `src`, `models` and `data`
are links into the same repo, so the old project is reused without copying a
single line. `facetrackingwithidentitylock/` itself stays the Part 2 software
only. Config lives in `face_needle_tracker/mqtt_config.json`.

Python side:

```bash
# calibrate camera X <-> needle reference positions first (GUI, no code edits)
python face_needle_tracker/calibrate_position.py --target bright

# then run the tracker publishing over MQTT
python face_needle_tracker/needle_track.py --target bright --mqtt
```

- `calibrate_position.py`: lock the target, then for each reference point move
  your face so the needle is physically at that reference and press its number
  key `1..N`; `s` saves `data/calibration.json` (the `data` junction puts it in
  the shared folder, reloaded next run), `r` resets, `q` quits. Prints the
  table + an ASCII camera→motor plot.
- `needle_track.py --mqtt`: publishes compact JSON to `face_track`
  (`recognized, x, position, confidence, frame_width, timestamp`), throttled by
  `publish_interval_s`. A center line, `X/SM/[LEFT|CENTER|RIGHT]` and the
  calibrated needle steps are drawn in the overlay for debugging.
- Independent subscriber (verifies Python→Mosquitto, and is exactly what the
  ESP8266 sees):
  ```bash
  python face_needle_tracker/mqtt_listen.py --broker localhost
  ```
- Firmware: `face_needle_tracker/firmware/mqtt_stepper/mqtt_stepper.ino` —
  paste the calibration arrays from `data/calibration.json` into
  `CAM_X[]`/`MOTOR_STEPS[]`, set Wi-Fi and `MQTT_BROKER` (the PC's LAN IP,
  never localhost), flash with Arduino IDE.

Behavior:
- Only `recognized: true` moves the needle. `recognized: false` → hold.
- Motor is stateful: `current` vs `target` position, direction = diff sign,
  absolute step counts, clamped to `MIN/MAX_MOTOR_POSITION`.
- Non-blocking: `MAX_STEPS_PER_LOOP` steps per loop, then MQTT is serviced, so
  the needle follows a moving face instead of finishing old commands first.
- Lost face → hold; movement stops after `FACE_LOST_GRACE_MS` without a
  recognized message. No homing (no limit switch) — the calibrated position is
  the software reference.
- No Y axis, no naive `x / width * 180`: the pixel→steps mapping comes from
  your calibration points, interpolated and clamped.

### Suggested test order

1. **Recognition** — target still locks (previous sections).
2. **Horizontal** — move face L→C→R, watch `X` change, `Y` has no effect.
3. **Dead zone** — small moves near center must not flicker LEFT/RIGHT.
4. **Smoothing** — jitter is filtered, tracking still feels live.
5. **MQTT** — run `mqtt_listen.py` and see the JSON stream.
6. **ESP receive** — flash `mqtt_stepper.ino`, Serial Monitor prints x/target/current.
7. **Motor** — direction, limits, current-position tracking.
8. **Calibrate** — several real camera→needle points via `calibrate_position.py`.
9. **End-to-end** — move face LEFT → CENTER → RIGHT → CENTER → LEFT; needle follows smoothly.

## How it thinks

```
webcam -> Haar finds a face -> MediaPipe gets 5 points (eyes, nose, mouth)
 -> align to 112x112 -> ArcFace embedding -> compare to database
 -> if it matches someone enrolled, point the servo at them
```

## The parts

| File | What it does |
| --- | --- |
| `working_face_recognition.py` | The main thing: enroll + track + servo |
| `src/enroll.py` | Fancier enrollment with auto-capture |
| `src/recognize.py` | Multi-face recognition + servo |
| `src/landmarks.py` | Just shows the 5 points so you can debug the box |
| `facetrackingwithidentitylock/face_tracking.py` | Part 2: identity lock, error signal, smile/blink (GUI + `--signal`) |
| `facetrackingwithidentitylock/face_signals.py` | EAR/blink/eyes-closed + adaptive smile from the locked face |
| `face_needle_tracker/needle_track.py` | Part 3: runner that reuses `face_tracking.py` via the junction + publishes over MQTT |
| `face_needle_tracker/track_position.py` | Part 3: horizontal state, calibrated camera→motor map, Paho publisher |
| `face_needle_tracker/calibrate_position.py` | Part 3: interactive camera↔needle calibration GUI |
| `face_needle_tracker/mqtt_listen.py` | Part 3: independent MQTT subscriber (test tool) |
| `face_needle_tracker/mqtt_config.json` | Part 3: broker, topic, smoothing/dead-zone, calibration, motor limits |
| `test/test_servo_port.py` | Quick check that the ESP talks back |
| `firmware/servo_tracker/servo_tracker.ino` | The servo firmware |
| `face_needle_tracker/firmware/mqtt_stepper/mqtt_stepper.ino` | Part 3: MQTT subscriber + 28BYJ-48 needle driver |

## Things that bit me

- Only enrolled people move the servo. Strangers get drawn in red but the
  servo ignores them on purpose.
- The servo pan direction was backwards once (it mirrored my motion). If yours
  does that, flip the `offset * 60` sign in `working_face_recognition.py`.
- Angles are sent as integers on purpose. The old firmware would read a
  decimal like `94.2` as `942`, clamp it to `180`, and park the servo there.
- If the servo feels twitchy, lower `max_step_per_command`; if it feels drunk
  and laggy, raise it a little. 4° is a decent starting point.
- The same camera looks sharp on one PC and blurry on another — Part 2 turns
  on autofocus and max sharpness by default (`--no-quality` to skip) and
  reports measured sharpness on startup.
- Smile detection is adaptive: it learns your neutral mouth width, so a fixed
  threshold won't work, and the rise must hold `--smile-frames` frames so
  talking/jitters don't fire it. Watch `smile neu d` in the overlay. If it
  reads SMILE while you're calm, raise `--smile-on` and/or `--smile-frames`.
  If it reads NEUTRAL while you grin, lower `--smile-on`.
- Part 2 works in a window around the last known face once locked, so it stays
  fast even at 720p — the fps readout is at the top right.