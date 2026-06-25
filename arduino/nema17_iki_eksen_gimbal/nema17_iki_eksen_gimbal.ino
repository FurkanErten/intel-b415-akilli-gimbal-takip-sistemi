#include <AccelStepper.h>

// =====================================================
// 2 EKSEN NEMA17 GIMBAL - 800 PULSE HIZLI KONTROL
// PAN  = Sağ - Sol
// TILT = Yukarı - Aşağı
//
// Bağlantı tipi:
// PUL+ -> Arduino 5V
// DIR+ -> Arduino 5V
// PUL- -> Arduino STEP pini
// DIR- -> Arduino DIR pini
//
// 800 pulse/devir
// =====================================================

#define PAN_STEP_PIN   2
#define PAN_DIR_PIN    5

#define TILT_STEP_PIN  3
#define TILT_DIR_PIN   6

// ENA kullanıyorsan:
// ENA+ -> Arduino 5V
// ENA- -> Arduino D8
// ENA kullanmıyorsan boş kalabilir.
#define ENABLE_PIN     8

const int PULSES_PER_REV = 800;
const float STEPS_PER_DEG = PULSES_PER_REV / 360.0;

const float MAX_PAN_SPEED  = 6000.0;   // step/s
const float MAX_TILT_SPEED = 5000.0;   // step/s

const float PAN_ACCEL  = 60000.0;      // step/s^2
const float TILT_ACCEL = 50000.0;      // step/s^2

const unsigned long COMMAND_TIMEOUT_MS = 500;

AccelStepper panStepper(AccelStepper::DRIVER, PAN_STEP_PIN, PAN_DIR_PIN);
AccelStepper tiltStepper(AccelStepper::DRIVER, TILT_STEP_PIN, TILT_DIR_PIN);

float targetPanSpeed = 0.0;
float targetTiltSpeed = 0.0;

float currentPanSpeed = 0.0;
float currentTiltSpeed = 0.0;

unsigned long lastUpdateMicros = 0;
unsigned long lastCommandTimeMs = 0;

char rxBuffer[64];
uint8_t rxIndex = 0;

float clampFloat(float v, float mn, float mx) {
  if (v < mn) return mn;
  if (v > mx) return mx;
  return v;
}

float approach(float current, float target, float maxDelta) {
  if (current < target) {
    current += maxDelta;
    if (current > target) current = target;
  }
  else if (current > target) {
    current -= maxDelta;
    if (current < target) current = target;
  }
  return current;
}

void stopMotors() {
  targetPanSpeed = 0.0;
  targetTiltSpeed = 0.0;
}

void processLine(char *line) {
  // Komut formatı:
  // V panSpeed tiltSpeed
  // Örnek:
  // V 3500 -2500

  if (line[0] == 'V') {
    long panCmd = 0;
    long tiltCmd = 0;

    int ok = sscanf(line + 1, "%ld %ld", &panCmd, &tiltCmd);

    if (ok == 2) {
      targetPanSpeed = clampFloat((float)panCmd, -MAX_PAN_SPEED, MAX_PAN_SPEED);
      targetTiltSpeed = clampFloat((float)tiltCmd, -MAX_TILT_SPEED, MAX_TILT_SPEED);
      lastCommandTimeMs = millis();
    }
  }
  else if (strcmp(line, "STOP") == 0) {
    stopMotors();
    lastCommandTimeMs = millis();
  }
}

void readSerialNonBlocking() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (rxIndex > 0) {
        rxBuffer[rxIndex] = '\0';
        processLine(rxBuffer);
        rxIndex = 0;
      }
    }
    else {
      if (rxIndex < sizeof(rxBuffer) - 1) {
        rxBuffer[rxIndex++] = c;
      }
    }
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);

  panStepper.setMaxSpeed(MAX_PAN_SPEED);
  tiltStepper.setMaxSpeed(MAX_TILT_SPEED);

  // TB6600 tarzı sürücüler için pulse genişliği
  panStepper.setMinPulseWidth(5);
  tiltStepper.setMinPulseWidth(5);

  // Senin bağlantında PUL- Arduino pinine bağlı olduğu için STEP invert.
  panStepper.setPinsInverted(false, true, false);
  tiltStepper.setPinsInverted(false, true, false);

  panStepper.setSpeed(0);
  tiltStepper.setSpeed(0);

  lastUpdateMicros = micros();
  lastCommandTimeMs = millis();

  Serial.println("GIMBAL_2_EKSEN_HAZIR_800_HIZLI");
}

void loop() {
  readSerialNonBlocking();

  if (millis() - lastCommandTimeMs > COMMAND_TIMEOUT_MS) {
    stopMotors();
  }

  unsigned long now = micros();
  float dt = (now - lastUpdateMicros) / 1000000.0;
  lastUpdateMicros = now;

  if (dt <= 0.0 || dt > 0.05) {
    dt = 0.001;
  }

  float panMaxDelta = PAN_ACCEL * dt;
  float tiltMaxDelta = TILT_ACCEL * dt;

  currentPanSpeed = approach(currentPanSpeed, targetPanSpeed, panMaxDelta);
  currentTiltSpeed = approach(currentTiltSpeed, targetTiltSpeed, tiltMaxDelta);

  panStepper.setSpeed(currentPanSpeed);
  tiltStepper.setSpeed(currentTiltSpeed);

  panStepper.runSpeed();
  tiltStepper.runSpeed();
}
