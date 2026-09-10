# Face Tracking Thing (A.K.A. "please just follow me, servo")

A small project where a camera on a little servo pans around to follow *my*
face — not strangers, not the cat, just people I actually enrolled.

## The hardware situation

- A USB webcam. **On the device it is camera index `2`.** Camera `0` on a
  normal PC. On this rig, `0` is black and it confused me for a whole day.
- An ESP8266 (Adafruit Feather Huzzah) on `COM3`, 115200 baud.
- A servo on GPIO `14`.

## Getting it running

1. Install the usual suspects:

   ```bash
   pip install opencv-python onnxruntime numpy pyserial mediapipe
   ```

2. Put the ArcFace model at `models/embedder_arcface.onnx`
   (`download_arcface_model.py` can grab it).

3. Flash `firmware/servo_tracker/servo_tracker.ino` with the Arduino IDE
   (ESP8266 board, `COM3`), then open the Serial Monitor once to see
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