/*
 * esp_led_test.ino
 * 
 * Simple test to verify ESP8266 is working and running our code.
 * Built-in LED should blink and serial messages should appear.
 */

void setup() {
    Serial.begin(115200);
    pinMode(LED_BUILTIN, OUTPUT);  // Built-in LED (usually GPIO2)
    pinMode(4, OUTPUT);            // GPIO4 (D2) - servo pin as regular output
    
    delay(2000);  // Wait for serial to stabilize
    
    Serial.println("");
    Serial.println("========================================");
    Serial.println("ESP8266 LED & GPIO TEST");
    Serial.println("Built-in LED should blink every second");
    Serial.println("GPIO4 (D2) should also toggle");
    Serial.println("========================================");
}

void loop() {
    static int counter = 0;
    
    // Blink built-in LED (visible on ESP8266 board)
    digitalWrite(LED_BUILTIN, LOW);   // Turn ON (LOW = on for ESP8266)
    digitalWrite(4, HIGH);            // GPIO4 high
    
    Serial.print("LED ON, GPIO4 HIGH - Count: ");
    Serial.println(counter++);
    
    delay(1000);
    
    // Turn off LED
    digitalWrite(LED_BUILTIN, HIGH);  // Turn OFF (HIGH = off for ESP8266)  
    digitalWrite(4, LOW);             // GPIO4 low
    
    Serial.print("LED OFF, GPIO4 LOW - Count: ");
    Serial.println(counter++);
    
    delay(1000);
}