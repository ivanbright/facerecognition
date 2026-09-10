/*
 * servo_continuous_test.ino
 * 
 * Continuously moves servo back and forth to test power and wiring.
 * Use this to verify servo hardware before face tracking.
 */

#include <Servo.h>

const int SERVO_PIN = 4;  // GPIO4 = D2
Servo myServo;

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("=== SERVO CONTINUOUS TEST ===");
    Serial.println("Servo should sweep continuously");
    Serial.println("If no movement:");
    Serial.println("1. Check servo power (needs 5V, not 3.3V)");
    Serial.println("2. Check wiring connections");
    Serial.println("3. Try external power supply");
    
    myServo.attach(SERVO_PIN);
    
    // Initial position
    myServo.write(90);
    delay(1000);
    Serial.println("Starting continuous sweep...");
}

void loop() {
    // Slow sweep 0 to 180
    Serial.println("Sweeping 0 -> 180 degrees");
    for (int angle = 0; angle <= 180; angle += 5) {
        myServo.write(angle);
        Serial.print("Angle: ");
        Serial.print(angle);
        Serial.println("°");
        delay(200);  // Slow movement for easy observation
    }
    
    delay(1000);  // Pause at 180°
    
    // Slow sweep 180 to 0  
    Serial.println("Sweeping 180 -> 0 degrees");
    for (int angle = 180; angle >= 0; angle -= 5) {
        myServo.write(angle);
        Serial.print("Angle: ");
        Serial.print(angle);
        Serial.println("°");
        delay(200);
    }
    
    delay(1000);  // Pause at 0°
}