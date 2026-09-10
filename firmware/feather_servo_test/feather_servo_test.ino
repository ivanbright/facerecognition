/*
 * feather_servo_test.ino
 * 
 * Servo test specifically for Adafruit Feather HUZZAH ESP8266
 * Tests multiple pins to find the correct servo connection
 */

#include <Servo.h>

// Adafruit Feather HUZZAH pin definitions
const int PINS_TO_TEST[] = {0, 2, 4, 5, 12, 13, 14, 15, 16};
const int NUM_PINS = sizeof(PINS_TO_TEST) / sizeof(PINS_TO_TEST[0]);

Servo testServo;

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("");
    Serial.println("========================================");
    Serial.println("ADAFRUIT FEATHER HUZZAH SERVO TEST");
    Serial.println("Testing all possible servo pins");
    Serial.println("Watch/listen for servo movement on ANY pin");
    Serial.println("========================================");
    
    // Test each pin
    for (int i = 0; i < NUM_PINS; i++) {
        int pin = PINS_TO_TEST[i];
        testPin(pin);
    }
    
    Serial.println("========================================");
    Serial.println("Pin test complete. Did you see movement on any pin?");
    Serial.println("If YES, note which GPIO number worked");
    Serial.println("========================================");
}

void testPin(int pin) {
    Serial.println("");
    Serial.print("*** TESTING GPIO ");
    Serial.print(pin);
    Serial.println(" ***");
    Serial.println("Watch for servo movement...");
    
    testServo.attach(pin);
    delay(500);
    
    // Test sequence for this pin
    Serial.println("Moving to 0 degrees...");
    testServo.write(0);
    delay(2000);
    
    Serial.println("Moving to 90 degrees...");
    testServo.write(90);
    delay(2000);
    
    Serial.println("Moving to 180 degrees...");
    testServo.write(180);
    delay(2000);
    
    Serial.println("Back to 90 degrees...");
    testServo.write(90);
    delay(2000);
    
    testServo.detach();
    
    Serial.print("GPIO ");
    Serial.print(pin);
    Serial.println(" test complete.");
    delay(1000);
}

void loop() {
    // After testing all pins, do continuous test on GPIO 4 (most likely)
    static int angle = 0;
    static int direction = 5;
    static bool firstRun = true;
    
    if (firstRun) {
        Serial.println("");
        Serial.println("Starting continuous test on GPIO 4...");
        Serial.println("(This is the most likely correct pin)");
        testServo.attach(4);
        firstRun = false;
    }
    
    testServo.write(angle);
    Serial.print("GPIO 4 - Angle: ");
    Serial.println(angle);
    
    angle += direction;
    if (angle >= 180 || angle <= 0) {
        direction = -direction;
    }
    
    delay(100);  // Fast movement for easy observation
}