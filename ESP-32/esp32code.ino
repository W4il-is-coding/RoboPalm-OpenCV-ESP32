#include <ESP32Servo.h>

Servo servo[5];
int pins[5] = {13, 12, 14, 27, 26};

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 5; i++) {
    servo[i].setPeriodHertz(50);      // MG90 standard
    servo[i].attach(pins[i], 500, 2500);
  }
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    int angles[5];
    int idx = 0;

    char buf[data.length() + 1];
    data.toCharArray(buf, sizeof(buf));

    char *token = strtok(buf, ",");
    while (token && idx < 5) {
      angles[idx++] = constrain(atoi(token), 0, 180);
      token = strtok(NULL, ",");
    }

    if (idx == 5) {
      for (int i = 0; i < 5; i++) {
        servo[i].write(angles[i]);
      }
    }
  }
}