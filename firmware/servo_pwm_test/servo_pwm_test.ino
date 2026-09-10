/*
 * servo_pwm_test.ino
 * 
 * Test servo with proper PWM signals and different timings.
 * This will help identify if servo needs different signal characteristics.
 */

#include <Servo.h>

const int SERVO_PIN = 4;  // GPIO4 = D2
Servo myServo;

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("");
    Serial.println("========================================");
    Serial.println("SERVO PWM SIGNAL TEST");
    Serial.println("Testing different servo configurations");
    Serial.println("Watch for ANY servo movement");
    Serial.println("========================================");
    
    // Test 1: Standard servo library
    Serial.println("TEST 1: Standard Servo Library");
    myServo.attach(SERVO_PIN);
    
    // Move to center first
    Serial.println("Moving to 90 degrees (center)...");
    myServo.write(90);
    delay(2000);
    
    // Test extreme positions
    Serial.println("Moving to 0 degrees...");
    myServo.write(0);
    delay(2000);
    
    Serial.println("Moving to 180 degrees...");
    myServo.write(180);
    delay(2000);
    
    Serial.println("Back to center (90 degrees)...");
    myServo.write(90);
    delay(2000);
    
    myServo.detach();
    
    // Test 2: Manual PWM with different pulse widths
    Serial.println("TEST 2: Manual PWM signals");
    testManualPWM();
    
    // Test 3: Continuous rotation servo commands
    Serial.println("TEST 3: Continuous rotation servo test");
    testContinuousServo();
    
    Serial.println("All tests completed. Did you see ANY movement?");
}

void testManualPWM() {
    Serial.println("Sending manual PWM signals...");
    
    // Test different pulse widths (500us to 2500us)
    int pulseWidths[] = {500, 1000, 1500, 2000, 2500};
    
    for (int i = 0; i < 5; i++) {
        int pulseWidth = pulseWidths[i];
        Serial.print("Testing pulse width: ");
        Serial.print(pulseWidth);
        Serial.println(" microseconds");
        
        // Send PWM for 2 seconds
        for (int j = 0; j < 100; j++) {  // 2 seconds at 50Hz
            digitalWrite(SERVO_PIN, HIGH);
            delayMicroseconds(pulseWidth);
            digitalWrite(SERVO_PIN, LOW);
            delay(20 - (pulseWidth/1000));  // Complete 20ms cycle
        }
        delay(1000);
    }
}

void testContinuousServo() {
    // In case it's a continuous rotation servo
    myServo.attach(SERVO_PIN);
    
    Serial.println("Testing as continuous rotation servo...");
    
    Serial.println("Stop (90 degrees)");
    myServo.write(90);
    delay(2000);
    
    Serial.println("Rotate one direction (0 degrees)");
    myServo.write(0);
    delay(3000);
    
    Serial.println("Stop (90 degrees)");
    myServo.write(90);
    delay(2000);
    
    Serial.println("Rotate other direction (180 degrees)");
    myServo.write(180);
    delay(3000);
    
    Serial.println("Stop (90 degrees)");
    myServo.write(90);
    delay(2000);
    
    myServo.detach();
}

void loop() {
    // Continuous slow sweep for observation
    static int angle = 0;
    static int direction = 1;
    
    myServo.attach(SERVO_PIN);
    
    myServo.write(angle);
    Serial.print("Continuous test - Angle: ");
    Serial.println(angle);
    
    angle += direction * 10;
    
    if (angle >= 180) {
        angle = 180;
        direction = -1;
    } else if (angle <= 0) {
        angle = 0;
        direction = 1;
    }
    
    delay(500);  // Slow movement for easy observation
}