/*
 * servo_sweep_test.ino
 *
 * Standalone servo sweep test — NO serial input, NO computer communication.
 * Moves the servo from 0 to 180 degrees and back in steps, repeating forever.
 *
 * PURPOSE: Confirm wiring, power, and servo operation are correct BEFORE
 * attempting any serial-controlled behavior.
 *
 * Wiring:
 *   Servo signal wire  -> ESP8266 GPIO D2 (GPIO4)   [see SERVO_PIN below]
 *   Servo VCC          -> ESP8266 5V (or external 5V if servo is large)
 *   Servo GND          -> ESP8266 GND
 *
 * NOTE: If the servo stutters or resets the board, you need an external
 *       5V power supply — the USB/ESP8266 regulator cannot reliably
 *       drive most hobby servos under load.
 *
 * Upload: Arduino IDE -> Board: "NodeMCU 1.0 (ESP-12E Module)" -> Upload
 */

#include <Servo.h>

// ---- Configuration ----
const int SERVO_PIN = 4;   // GPIO4 = D2 on most NodeMCU/ESP8266MOD boards
                            // CONFIRM this matches YOUR board's silk screen
const int STEP_DELAY_MS = 15;   // milliseconds between steps (controls sweep speed)
const int STEP_SIZE = 2;        // degrees per step

Servo myServo;

void setup() {
    Serial.begin(115200);
    delay(500);

    myServo.attach(SERVO_PIN);

    // Sweep to center first as a quick sanity check
    myServo.write(90);
    delay(500);

    Serial.println("[sweep] Servo sweep test started");
    Serial.println("[sweep] Sweep: 0 -> 180 -> 0, repeating");
    Serial.println("[sweep] Press RESET to stop");
}

void loop() {
    // Sweep 0 -> 180
    for (int angle = 0; angle <= 180; angle += STEP_SIZE) {
        myServo.write(angle);
        delay(STEP_DELAY_MS);
    }

    // Sweep 180 -> 0
    for (int angle = 180; angle >= 0; angle -= STEP_SIZE) {
        myServo.write(angle);
        delay(STEP_DELAY_MS);
    }
}
