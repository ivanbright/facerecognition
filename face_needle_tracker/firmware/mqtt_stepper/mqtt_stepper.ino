#include <dummy.h>

/*
 * mqtt_stepper.ino
 *
 * Part 3 ESP8266 firmware: subscribe to Mosquitto topic `face_track`, parse
 * the compact JSON tracking message, convert the smoothed camera X into an
 * absolute motor position with the calibrated piecewise-linear mapping, and
 * drive a 28BYJ-48 stepper (via ULN2003 driver) toward that position.
 *
 * Design notes (from the assignment):
 *  - Only `recognized = true` messages move the needle. Unknown faces are
 *    ignored. `recognized = false` stops movement immediately (hold).
 *  - Motor position is STATEFUL: current_position vs target_position, the
 *    difference decides direction and step count. No arbitrary "move N steps"
 *    commands over MQTT.
 *  - Non-blocking: at most MAX_STEPS_PER_LOOP steps are issued per loop
 *    iteration, then MQTT is serviced again, so the ESP stays responsive while
 *    the needle follows a moving face.
 *  - The motor never goes outside [MIN_MOTOR_POSITION, MAX_MOTOR_POSITION].
 *  - If no recognized message arrives for FACE_LOST_GRACE_MS, movement stops.
 *
 * Libraries: PubSubClient, ArduinoJson (v6).
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ------------------------- config -------------------------
// Wi-Fi / broker. Use the PC's LAN address for MQTT_BROKER - NOT localhost.
const char *WIFI_SSID     = "RCA-A";
const char *WIFI_PASS     = "RCA@2024";
const char *MQTT_BROKER   = "10.12.73.27";
const int   MQTT_PORT     = 1883;
const char *MQTT_TOPIC    = "face_track";

// 28BYJ-48 via ULN2003 (default NodeMCU pins - verify against your board).
const uint8_t IN1 = 0;   // D3
const uint8_t IN2 = 2;   // D4
const uint8_t IN3 = 4;   // D2
const uint8_t IN4 = 5;   // D1
// Flip direction if the needle mirrors the face.
const bool    DIRECTION_FLIP = true;

// Motor / control parameters.
const long   MIN_MOTOR_POSITION   = 0;      // software reference, safe left
const long   MAX_MOTOR_POSITION   = 2048;   // safe right (28BYJ-48 ~2048
                                            // full-steps / rev, 4096 half-steps)
const long   POSITION_TOLERANCE   = 2;      // steps; smaller = no trig on 1 step
const uint8_t MAX_STEPS_PER_LOOP  = 2;      // non-blocking step budget per loop
const unsigned int STEP_DELAY_MS  = 2;      // min pause between step phases
const unsigned long FACE_LOST_GRACE_MS = 2000; // stop if no recognized=true

// Heartbeat to the broker (PubSubClient needs it).
const unsigned long KEEPALIVE_MS = 10000;

// ------------------------- calibration -------------------------
// PASTE from mqtt_config.json -> calibration.points (camera_x, motor_position).
// Camera X is in pixels; motor_position is absolute stepper steps.
static const int CAL_N = 5;
static const float  CAM_X[CAL_N]     = {100.0f, 200.0f, 320.0f, 430.0f, 545.0f};
static const long   MOTOR_STEPS[CAL_N] = {100L,  500L,  1024L, 1500L, 1948L};

// ------------------------- motor -------------------------
class Stepper28BYJ {
  public:
    Stepper28BYJ(uint8_t a, uint8_t b, uint8_t c, uint8_t d, bool flip)
      : _pins{a, b, c, d}, _flip(flip), _phase(0), _position(0) {}

    void begin() {
      for (int i = 0; i < 4; i++) pinMode(_pins[i], OUTPUT);
    }

    long position() const { return _position; }

    long stepsRemaining(long target) const { return labs(target - _position); }

    // Advance one half-step toward `target`, or stay put at a limit.
    bool stepToward(long target) {
      if (target > _position) {
        if (_position >= MAX_MOTOR_POSITION) return false;
        advance(true);
      } else if (target < _position) {
        if (_position <= MIN_MOTOR_POSITION) return false;
        advance(false);
      } else {
        return false;
      }
      return true;
    }

  private:
    // 8-phase half-step sequence for the 28BYJ-48 (IN1..IN4), high = coil on.
    static const uint8_t SEQUENCE[8];
    uint8_t _pins[4];
    bool    _flip;
    uint8_t _phase;
    long    _position;

    void advance(bool forward) {
      bool dir = forward;
      if (_flip) dir = !dir;
      _phase = (dir ? _phase + 1 : _phase + 7) & 0x07;
      drive(_phase);
      _position += (forward ? 1 : -1);
      if (STEP_DELAY_MS) delay(STEP_DELAY_MS);
    }

    void drive(uint8_t idx) {
      uint8_t bits = SEQUENCE[idx & 0x07];
      for (int i = 0; i < 4; i++) {
        digitalWrite(_pins[i], (bits >> (3 - i)) & 0x01 ? HIGH : LOW);
      }
    }
};

const uint8_t Stepper28BYJ::SEQUENCE[8] = {
  0b1000, 0b1100, 0b0100, 0b0110, 0b0010, 0b0011, 0b0001, 0b1001
};

// ------------------------- globals -------------------------
WiFiClient            wifiClient;
PubSubClient          mqttClient(wifiClient);
Stepper28BYJ          stepper(IN1, IN2, IN3, IN4, DIRECTION_FLIP);

long                  targetPosition = MIN_MOTOR_POSITION;
unsigned long         lastRecognizedAt = 0;   // ms of last recognized=true msg
unsigned long         lastKeepAlive = 0;

// ------------------------- calibration mapping -------------------------
long mapXToMotor(float x) {
  if (CAL_N < 2) return MIN_MOTOR_POSITION;
  if (x <= CAM_X[0])                 return constrain(MOTOR_STEPS[0],
                                                       MIN_MOTOR_POSITION,
                                                       MAX_MOTOR_POSITION);
  if (x >= CAM_X[CAL_N - 1])         return constrain(MOTOR_STEPS[CAL_N - 1],
                                                       MIN_MOTOR_POSITION,
                                                       MAX_MOTOR_POSITION);
  for (int i = 0; i < CAL_N - 1; i++) {
    if (x >= CAM_X[i] && x < CAM_X[i + 1]) {
      float frac = (x - CAM_X[i]) / (CAM_X[i + 1] - CAM_X[i]);
      long out = MOTOR_STEPS[i] + (long)(frac * (MOTOR_STEPS[i + 1] - MOTOR_STEPS[i]));
      return constrain(out, MIN_MOTOR_POSITION, MAX_MOTOR_POSITION);
    }
  }
  return MIN_MOTOR_POSITION;
}

// ------------------------- MQTT -------------------------
void onMessage(char *topic, byte *payload, unsigned int length) {
  if (strcmp(topic, MQTT_TOPIC) != 0) return;

  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.print(F("[mqtt] malformed JSON ignored: "));
    Serial.println(err.c_str());
    return;
  }

  bool recognized = doc["recognized"] | false;
  if (!recognized) {
    // Unknown / no face: do NOT move toward it. Hold current position.
    Serial.println(F("[mqtt] recognized=false -> holding position"));
    return;
  }

  float x = doc["x"] | -1.0f;
  const char *pos = doc["position"] | "?";
  float conf = doc["confidence"] | 0.0f;
  if (x < 0) {
    Serial.println(F("[mqtt] missing/invalid x -> ignored"));
    return;
  }

  long newTarget = mapXToMotor(x);
  if (newTarget != targetPosition) {
    Serial.printf("[mqtt] received recognized=true x=%.0f position=%s confidence=%.2f\n",
                  x, pos, conf);
    Serial.printf("[mqtt] target motor position = %ld, current = %ld\n",
                  newTarget, stepper.position());
  }
  targetPosition = newTarget;
  lastRecognizedAt = millis();
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("[net] connecting to SSID '%s'", WIFI_SSID);
  unsigned long deadline = millis() + 15000;
  while (WiFi.status() != WL_CONNECTED && millis() < deadline) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[net] connected, IP = %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println(F("[net] NOT connected after 15s - MQTT will keep retrying, check SSID/pass"));
  }
}

void connectMQTT() {
  while (!mqttClient.connected()) {
    Serial.printf("[mqtt] connecting to %s:%d ...\n", MQTT_BROKER, MQTT_PORT);
    String id = String("facelock_esp886_") + String(millis());
    if (mqttClient.connect(id.c_str())) {
      mqttClient.subscribe(MQTT_TOPIC);
      Serial.println("[mqtt] connected :)");
      Serial.printf("[mqtt] subscribed: %s\n", MQTT_TOPIC);
    } else {
      Serial.printf("[mqtt] failed rc=%d, retry in 3s\n", mqttClient.state());
      delay(3000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  Serial.println(F("--- Part 3 mqtt_stepper ---"));
  stepper.begin();
  targetPosition = MIN_MOTOR_POSITION;   // documented software reference, no homing
  connectWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(onMessage);
  setupMotorDiag();
}

void setupMotorDiag() {
  Serial.printf("[motor] limits %ld..%ld, position=%ld, target=%ld\n",
                MIN_MOTOR_POSITION, MAX_MOTOR_POSITION,
                stepper.position(), targetPosition);
}

void loop() {
  // A router power-cycle / reboot drops the Wi-Fi carrier underneath MQTT.
  // connectMQTT() only retries the socket, so re-establish Wi-Fi first or the
  // socket can never succeed after a mid-run drop.
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println(F("[net] Wi-Fi lost - reconnecting ..."));
    connectWiFi();
  }
  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();

  if (millis() - lastKeepAlive > KEEPALIVE_MS) { lastKeepAlive = millis(); }

  // Only move while we have received a (fresh) recognized=true. Otherwise hold.
  bool fresh = (millis() - lastRecognizedAt) < FACE_LOST_GRACE_MS;
  if (!fresh) return;

  // Non-blocking: bounded steps per loop, then let MQTT run again.
  long remaining = stepper.stepsRemaining(targetPosition);
  if (remaining <= POSITION_TOLERANCE) return;
  uint8_t budget = MAX_STEPS_PER_LOOP;
  while (budget-- && stepper.stepsRemaining(targetPosition) > POSITION_TOLERANCE) {
    stepper.stepToward(targetPosition);
  }
}