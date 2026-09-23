# Face Tracking Thing (A.K.A. "please just follow me, servo")

> The face-locking project (identity lock + smile/blink/position signal, no
> hardware) also has its own repo: **https://github.com/ivanbright/FaceLocking**.
> This repo is the full rig — enrollment, recognition, servo, and firmware.

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
   pip install opencv-python onnxruntime numpy pyserial mediapipe
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
| `test/test_servo_port.py` | Quick check that the ESP talks back |
| `firmware/servo_tracker/servo_tracker.ino` | The servo firmware |

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
  threshold won't work. If it reads NEUTRAL while you grin, lower `--smile-on`.
- Part 2 works in a window around the last known face once locked, so it stays
  fast even at 720p — the fps readout is at the top right.