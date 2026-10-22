/*
  HC-SR04 object height measurement with median filtering + smoothing.
  Sensor-to-ground reference height: 15.0 cm
  TRIG -> D9, ECHO -> D10, VCC -> 5V, GND -> GND

  Serial output stays compatible with serial_uploader.py:
  Distance: 9.0 cm | Object Height: 6.0 cm
*/

const byte TRIG_PIN = 9;
const byte ECHO_PIN = 10;
const float REFERENCE_HEIGHT_CM = 15.0;

const byte SAMPLE_COUNT = 9;
const unsigned long ECHO_TIMEOUT_US = 25000UL;
const float SMOOTH_ALPHA = 0.25;  // lower = steadier but slower response

float smoothedDistance = -1.0;

float readOneDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(3);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);
  if (duration == 0) return -1.0;

  float distance = duration * 0.0343 / 2.0;
  if (distance < 2.0 || distance > 400.0) return -1.0;
  return distance;
}

void sortValues(float values[], byte count) {
  for (byte i = 0; i < count; i++) {
    for (byte j = i + 1; j < count; j++) {
      if (values[j] < values[i]) {
        float temp = values[i];
        values[i] = values[j];
        values[j] = temp;
      }
    }
  }
}

float readMedianDistanceCm() {
  float samples[SAMPLE_COUNT];
  byte validCount = 0;

  for (byte i = 0; i < SAMPLE_COUNT; i++) {
    float value = readOneDistanceCm();
    if (value > 0) {
      samples[validCount++] = value;
    }
    delay(35);  // avoid overlapping ultrasonic echoes
  }

  if (validCount == 0) return -1.0;
  sortValues(samples, validCount);
  return samples[validCount / 2];
}

void setup() {
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  float medianDistance = readMedianDistanceCm();

  if (medianDistance > 0) {
    if (smoothedDistance < 0) {
      smoothedDistance = medianDistance;
    } else {
      smoothedDistance =
        SMOOTH_ALPHA * medianDistance +
        (1.0 - SMOOTH_ALPHA) * smoothedDistance;
    }

    float objectHeight = REFERENCE_HEIGHT_CM - smoothedDistance;
    if (objectHeight < 0) objectHeight = 0;

    Serial.print("Distance: ");
    Serial.print(smoothedDistance, 1);
    Serial.print(" cm | Object Height: ");
    Serial.print(objectHeight, 1);
    Serial.println(" cm");
  } else {
    Serial.println("No valid echo");
  }

  delay(100);
}
