/*
 * GPIO Identification for Adafruit Feather HUZZAH ESP8266
 * 
 * This script systematically tests all available GPIO pins to identify
 * which one controls your servo. It will test each pin and ask for
 * user confirmation via serial monitor.
 *
 * Adafruit Feather HUZZAH ESP8266 Available GPIO pins:
 * - GPIO 0  (NodeMCU D3)
 * - GPIO 2  (NodeMCU D4, built-in LED)  
 * - GPIO 4  (NodeMCU D2)
 * - GPIO 5  (NodeMCU D1)
 * - GPIO 12 (NodeMCU D6)
 * - GPIO 13 (NodeMCU D7)
 * - GPIO 14 (NodeMCU D5)
 * - GPIO 15 (NodeMCU D8)
 * - GPIO 16 (NodeMCU D0, wake pin)
 */

#include <Servo.h>

Servo testServo;

// Available GPIO pins on Adafruit Feather HUZZAH ESP8266
int gpioPins[] = {0, 2, 4, 5, 12, 13, 14, 15, 16};
int numPins = sizeof(gpioPins) / sizeof(gpioPins[0]);

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println();
    Serial.println("==========================================");
    Serial.println("*** ADAFRUIT FEATHER HUZZAH ESP8266 ***");
    Serial.println("***     SERVO GPIO IDENTIFICATION     ***");
    Serial.println("==========================================");
    Serial.println();
    Serial.println("This script will test each GPIO pin systematically.");
    Serial.println("Watch your servo and listen for movement sounds.");
    Serial.println("Type 'y' when you see/hear servo movement!");
    Serial.println();
    delay(2000);
    
    testAllGPIOPins();
}

void loop() {
    // Nothing in loop - all testing done in setup
    delay(1000);
}

void testAllGPIOPins() {
    for (int i = 0; i < numPins; i++) {
        int pin = gpioPins[i];
        
        Serial.println();
        Serial.println("******************************************");
        Serial.print("*** TESTING GPIO ");
        Serial.print(pin);
        Serial.println(" ***");
        Serial.println("Watch for servo movement...");
        Serial.println("******************************************");
        
        // Attach servo to current pin
        testServo.attach(pin);
        delay(500);
        
        // Test sequence for this pin
        testPin(pin);
        
        // Detach servo before testing next pin
        testServo.detach();
        delay(1000);
    }
    
    Serial.println();
    Serial.println("******************************************");
    Serial.println("*** ALL GPIO TESTS COMPLETE ***");
    Serial.println("Did you hear servo sounds during any test?");
    Serial.println("******************************************");
}

void testPin(int pin) {
    // Move servo through test positions
    Serial.print("GPIO ");
    Serial.print(pin);
    Serial.println(" - Moving to 0°");
    testServo.write(0);
    delay(1000);
    
    Serial.print("GPIO ");
    Serial.print(pin);
    Serial.println(" - Moving to 180°");
    testServo.write(180);
    delay(1000);
    
    Serial.print("GPIO ");
    Serial.print(pin);
    Serial.println(" - Moving to 90°");
    testServo.write(90);
    delay(1000);
    
    // Multiple cycles for clear identification
    for (int cycle = 1; cycle <= 3; cycle++) {
        Serial.print("GPIO ");
        Serial.print(pin);
        Serial.print(" - Cycle ");
        Serial.print(cycle);
        Serial.println(" - Moving to 0°");
        testServo.write(0);
        delay(800);
        
        Serial.print("GPIO ");
        Serial.print(pin);
        Serial.print(" - Cycle ");
        Serial.print(cycle);
        Serial.println(" - Moving to 180°");
        testServo.write(180);
        delay(800);
    }
    
    Serial.print("GPIO ");
    Serial.print(pin);
    Serial.println(" - Center position (90°)");
    testServo.write(90);
    delay(1000);
    
    Serial.println("******************************************");
    Serial.print("*** GPIO ");
    Serial.print(pin);
    Serial.println(" TEST COMPLETE ***");
    Serial.println("Did you hear servo sounds during this test?");
    Serial.println("******************************************");
    
    // Wait for user input or timeout
    Serial.println("Type 'y' if servo moved, or wait 5 seconds to continue...");
    
    unsigned long startTime = millis();
    bool responseReceived = false;
    
    while (millis() - startTime < 5000 && !responseReceived) {
        if (Serial.available() > 0) {
            char response = Serial.read();
            if (response == 'y' || response == 'Y') {
                Serial.println();
                Serial.println("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉");
                Serial.print("*** SERVO FOUND ON GPIO ");
                Serial.print(pin);
                Serial.println(" ***");
                Serial.println("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉");
                Serial.println();
                Serial.print("Update your servo_tracker.ino:");
                Serial.print("const int SERVO_PIN = ");
                Serial.print(pin);
                Serial.println(";");
                Serial.println();
                responseReceived = true;
                
                // Do a celebration dance
                for (int i = 0; i < 5; i++) {
                    testServo.write(45);
                    delay(200);
                    testServo.write(135);
                    delay(200);
                }
                testServo.write(90);
            }
        }
        delay(50);
    }
    
    if (!responseReceived) {
        Serial.print("No response - continuing to next GPIO...");
    }
    
    Serial.println();
}