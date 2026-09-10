/*
 * test_esp8266.ino
 * 
 * Simple test to verify ESP8266 is working and can be programmed.
 * Just blinks the built-in LED and sends messages over serial.
 */

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);
    
    delay(1000);
    Serial.println("");
    Serial.println("=== ESP8266 Test Program ===");
    Serial.println("If you see this, ESP8266 is working!");
    Serial.println("Built-in LED should be blinking");
}

void loop() {
    // Blink LED
    digitalWrite(LED_BUILTIN, LOW);   // Turn LED on (LOW = on for ESP8266)
    delay(500);
    digitalWrite(LED_BUILTIN, HIGH);  // Turn LED off
    delay(500);
    
    // Send serial message
    static int counter = 0;
    Serial.print("ESP8266 alive! Count: ");
    Serial.println(counter++);
    
    delay(1000);
}