/*
 * servo_tracker_v2.ino
 * 
 * Serial-controlled servo with improved error handling and buffering.
 * 
 * Serial protocol:
 *   PC sends:  "90\n" (ASCII integer + newline)
 *   ESP8266:   parses, clamps to 0-180, writes to servo, echoes back
 */

#include <Servo.h>

const int SERVO_PIN = 5;
const int CENTER_ANGLE = 90;
Servo myServo;

char inputBuffer[16];
int inputIndex = 0;

void setup() {
    Serial.begin(115200);
    delay(500);
    
    myServo.attach(SERVO_PIN);
    myServo.write(CENTER_ANGLE);
    delay(300);
    
    Serial.println("[tracker] Ready");
}

void loop() {
    if (Serial.available() > 0) {
        char c = Serial.read();
        
        if (c == '\n' || c == '\r') {
            if (inputIndex > 0) {
                inputBuffer[inputIndex] = '\0';
                int angle = atoi(inputBuffer);
                
                if (angle < 0) angle = 0;
                if (angle > 180) angle = 180;
                
                myServo.write(angle);
                delay(20);
                
                Serial.print("[tracker] angle=");
                Serial.println(angle);
                
                inputIndex = 0;
            }
        } else if (inputIndex < 14) {
            if (c >= '0' && c <= '9') {
                inputBuffer[inputIndex++] = c;
            }
        }
    }
}
