# Bed Bug Detector Assembly Guide

This guide provides step-by-step instructions for assembling your DIY bed bug detector. Follow these instructions carefully to ensure proper functionality.

## Prerequisites

Before beginning assembly, ensure you have:

1. All components listed in the [Components List](components.md)
2. Basic soldering equipment (soldering iron, solder, helping hands)
3. Small Phillips screwdriver
4. Wire cutters/strippers
5. 3D printed case (print files from the `hardware/3d-models/` directory)
6. Micro USB cable (for programming and charging)

## Safety Precautions

- Work in a well-ventilated area when soldering
- Be careful handling the battery to avoid short circuits
- Disconnect power when making connections
- Follow proper ESD (Electrostatic Discharge) precautions when handling electronic components

## Assembly Steps

### Step 1: Prepare the ESP32 Development Board

1. Inspect the ESP32 board to ensure all pins are straight and no components are damaged
2. If your ESP32 board doesn't have pre-soldered headers, solder the included male header pins

```
   Flash the firmware before completing hardware assembly to ensure the ESP32 is functioning correctly:

   # For Linux/Termux
   git clone https://github.com/your-username/bed-bug-detector.git
   cd bed-bug-detector/firmware
   arduino-cli compile --fqbn esp32:esp32:esp32 detector.ino
   arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 detector.ino

   # For Windows
   git clone https://github.com/your-username/bed-bug-detector.git
   cd bed-bug-detector\firmware
   arduino-cli compile --fqbn esp32:esp32:esp32 detector.ino
   arduino-cli upload -p COM3 --fqbn esp32:esp32:esp32 detector.ino
```

### Step 2: Connect the Temperature Sensor (DHT22)

1. Connect the DHT22 pins to the ESP32 as follows:
   - DHT22 VCC → ESP32 3.3V
   - DHT22 DATA → ESP32 D4
   - DHT22 GND → ESP32 GND

2. For a more permanent connection, solder the wires or use a small piece of breadboard

![DHT22 Connection Diagram](https://github.com/your-username/bed-bug-detector/raw/main/docs/charts/dht22-wiring.png)

### Step 3: Connect the CO2 Sensor (MH-Z19)

1. Connect the MH-Z19 pins to the ESP32 as follows:
   - MH-Z19 VCC → ESP32 5V
   - MH-Z19 GND → ESP32 GND
   - MH-Z19 TX → ESP32 RX2 (D16)
   - MH-Z19 RX → ESP32 TX2 (D17)

2. Note: The TX/RX connections are crossed (TX connects to RX and vice versa)

### Step 4: Connect the PIR Motion Sensor

1. Connect the PIR sensor pins to the ESP32 as follows:
   - PIR VCC → ESP32 5V
   - PIR GND → ESP32 GND
   - PIR OUT → ESP32 D2

2. Adjust the sensitivity and delay potentiometers on the PIR sensor:
   - Turn the sensitivity potentiometer clockwise to increase sensitivity
   - Set the delay potentiometer to minimum (counterclockwise) for faster response

### Step 5: Connect the Camera Module (OV2640)

1. Connect the camera module pins to the ESP32 as follows:
   - Camera VCC → ESP32 3.3V
   - Camera GND → ESP32 GND
   - Camera SDA → ESP32 D21
   - Camera SCL → ESP32 D22
   - Camera CS → ESP32 D13
   - Camera MOSI → ESP32 D12
   - Camera MISO → ESP32 D14
   - Camera SCK → ESP32 D27

2. Ensure the camera ribbon cable is securely inserted in the connector

### Step 6: Set Up the Battery Power System

1. Prepare the TP4056 charging module:
   - Connect the TP4056 OUT+ to ESP32 VIN
   - Connect the TP4056 OUT- to ESP32 GND

2. Connect the 18650 battery holder:
   - Connect battery holder positive (+) to TP4056 B+
   - Connect battery holder negative (-) to TP4056 B-

3. Double-check all connections to avoid short circuits

### Step 7: Final Assembly in the 3D Printed Case

1. Place the ESP32 in the main compartment of the 3D printed case
2. Mount the camera module in the front camera slot, ensuring the lens is properly aligned
3. Position the DHT22 temperature sensor near an air vent in the case
4. Install the CO2 sensor ensuring its air inlet is aligned with case vents
5. Mount the PIR sensor behind the designated opening in the case
6. Place the battery and TP4056 module in the battery compartment
7. Carefully route all wires to avoid pinching when closing the case
8. Close the case and secure with the provided screws

## Testing the Assembled Device

Before deploying your bed bug detector, perform these tests to ensure proper functionality:

1. **Power Test**:
   - Insert a fully charged 18650 battery
   - Verify the power LED on the ESP32 illuminates
   - Check the charging LED on the TP4056 module

2. **Connectivity Test**:
   - Power on the detector
   - Look for the "BedBugDetector" WiFi network on your phone
   - Connect to the network (password: "detector123")
   - Open a web browser and navigate to http://192.168.4.1
   - Verify you see the detector homepage

3. **Sensor Test**:
   - Navigate to http://192.168.4.1/sensor-data in your browser
   - Verify all sensors are reporting values
   - Breathe on the temperature sensor and verify temperature/humidity changes
   - Wave your hand near the PIR sensor and check if motion is detected

4. **Camera Test**:
   - Navigate to http://192.168.4.1/capture in your browser
   - Verify the camera captures and displays an image

## Mobile App Setup

1. Install the mobile app on your device:
   ```bash
   # For Android
   cd bed-bug-detector/app
   npm install
   npx react-native run-android
   ```

2. Open the app and connect to your detector
3. Complete the initial calibration process:
   - Place the detector in a clean, bed bug-free area
   - Press the "Calibrate" button in the app
   - Wait for the calibration process to complete

## Usage Tips

- When testing for bed bugs, move the detector slowly across suspected areas
- Focus on mattress seams, box springs, bed frames, and nearby furniture
- Allow the detector to remain still for 30-60 seconds in each location
- Use the camera function for visual confirmation of suspected bed bugs
- Calibrate the detector in a known clean environment before each use session
- Check the battery level before extended detection sessions
- Clean sensors with compressed air if they become dusty

## Troubleshooting

| Problem | Possible Causes | Solution |
|---------|----------------|----------|
| Detector won't power on | Dead battery, loose connections | Charge battery, check all connections |
| No WiFi network appears | ESP32 firmware issue | Reflash firmware, check power connections |
| Sensors show no readings | Connection issues, component damage | Verify connections, replace component if necessary |
| False positives | Improper calibration, interference | Recalibrate in clean area, move away from heat sources |
| Camera not working | Loose cable, incorrect pins | Check camera ribbon cable, verify pin connections |
| Short battery life | Power-hungry settings, battery issues | Use deep sleep mode, replace battery |

For additional support, check the issues section of the GitHub repository or create a new issue with a detailed description of your problem.
