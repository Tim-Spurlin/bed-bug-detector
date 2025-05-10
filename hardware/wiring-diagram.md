# Wiring Diagram for Bed Bug Detector

This document provides detailed instructions for connecting all electronic components of the bed bug detector.

## Overview Diagram

```
                  ┌─────────────┐
                  │             │
                  │    ESP32    │
                  │             │
                  └─────────────┘
                        │
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│    Camera   │ │Temperature  │ │    CO2      │
│   Module    │ │   Sensor    │ │   Sensor    │
└─────────────┘ └─────────────┘ └─────────────┘
                        │
                        │
                        ▼
                ┌─────────────┐
                │   Battery   │
                │ Power System│
                └─────────────┘
```

## Detailed Connections

### ESP32 to Camera Module (OV2640)

| ESP32 Pin | Camera Pin | Wire Color |
|-----------|------------|------------|
| 3.3V      | VCC        | Red        |
| GND       | GND        | Black      |
| D21 (SDA) | SDA        | Yellow     |
| D22 (SCL) | SCL        | Green      |
| D13       | CS         | White      |
| D12       | MOSI       | Purple     |
| D14       | MISO       | Gray       |
| D27       | SCK        | Orange     |

### ESP32 to Temperature Sensor (DHT22)

| ESP32 Pin | DHT22 Pin | Wire Color |
|-----------|-----------|------------|
| 3.3V      | VCC       | Red        |
| GND       | GND       | Black      |
| D4        | DATA      | Blue       |

### ESP32 to CO2 Sensor (MH-Z19)

| ESP32 Pin | MH-Z19 Pin | Wire Color |
|-----------|------------|------------|
| 5V        | VCC        | Red        |
| GND       | GND        | Black      |
| D16 (RX2) | TX         | Yellow     |
| D17 (TX2) | RX         | Green      |

### ESP32 to PIR Motion Sensor

| ESP32 Pin | PIR Sensor Pin | Wire Color |
|-----------|----------------|------------|
| 5V        | VCC            | Red        |
| GND       | GND            | Black      |
| D2        | OUT            | Blue       |

### ESP32 to Battery System

| ESP32 Pin | TP4056 Pin | Battery Holder | Wire Color |
|-----------|------------|----------------|------------|
| VIN       | OUT+       | -              | Red        |
| GND       | OUT-       | -              | Black      |
| -         | B+         | Positive (+)   | Red        |
| -         | B-         | Negative (-)   | Black      |

## Detailed Visual Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│ ESP32                                                           │
│                                                                 │
│  3V3 ──┬─[RED]────────────▶ VCC (Camera)                        │
│        │                                                        │
│        ├─[RED]────────────▶ VCC (DHT22)                         │
│                                                                 │
│  5V  ──┬─[RED]────────────▶ VCC (MH-Z19)                        │
│        │                                                        │
│        └─[RED]────────────▶ VCC (PIR)                           │
│                                                                 │
│  GND ──┬─[BLACK]──────────▶ GND (Camera)                        │
│        │                                                        │
│        ├─[BLACK]──────────▶ GND (DHT22)                         │
│        │                                                        │
│        ├─[BLACK]──────────▶ GND (MH-Z19)                        │
│        │                                                        │
│        └─[BLACK]──────────▶ GND (PIR)                           │
│                                                                 │
│  D21 ──[YELLOW]───────────▶ SDA (Camera)                        │
│  D22 ──[GREEN]────────────▶ SCL (Camera)                        │
│  D13 ──[WHITE]────────────▶ CS (Camera)                         │
│  D12 ──[PURPLE]───────────▶ MOSI (Camera)                       │
│  D14 ──[GRAY]─────────────▶ MISO (Camera)                       │
│  D27 ──[ORANGE]───────────▶ SCK (Camera)                        │
│                                                                 │
│  D4  ──[BLUE]─────────────▶ DATA (DHT22)                        │
│                                                                 │
│  D16 ──[YELLOW]───────────▶ TX (MH-Z19)                         │
│  D17 ──[GREEN]────────────▶ RX (MH-Z19)                         │
│                                                                 │
│  D2  ──[BLUE]─────────────▶ OUT (PIR)                           │
│                                                                 │
│  VIN ──[RED]──────────────▶ OUT+ (TP4056)                       │
│  GND ──[BLACK]────────────▶ OUT- (TP4056)                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│ TP4056                                                          │
│                                                                 │
│  B+ ───[RED]──────────────▶ POSITIVE (18650 Battery Holder)     │
│  B- ───[BLACK]────────────▶ NEGATIVE (18650 Battery Holder)     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3D Case Layout

Top view of component arrangement inside the 3D printed case:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│      ┌─────────┐           ┌─────────────┐      │
│      │ Camera  │           │ Temperature │      │
│      │ Module  │           │ Sensor      │      │
│      └─────────┘           └─────────────┘      │
│                                                 │
│          ┌─────────────────────┐                │
│          │                     │                │
│          │        ESP32        │                │
│          │                     │                │
│          └─────────────────────┘                │
│                                                 │
│      ┌─────────┐           ┌─────────────┐      │
│      │  CO2    │           │ PIR Motion  │      │
│      │ Sensor  │           │   Sensor    │      │
│      └─────────┘           └─────────────┘      │
│                                                 │
│          ┌─────────────────────┐                │
│          │     TP4056 &        │                │
│          │   Battery Holder    │                │
│          └─────────────────────┘                │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Assembly Tips

1. **Color-coding**: Follow the wire color recommendations for easier troubleshooting
2. **Soldering**: After testing on a breadboard, solder all connections for reliability
3. **Heat management**: Position the CO2 sensor away from the ESP32 to avoid temperature interference
4. **Power management**: Ensure the battery is securely connected to the TP4056 module
5. **Sensor positioning**:
   - The camera should face the front of the case
   - The PIR motion sensor should have an unobstructed view
   - The temperature sensor should be positioned away from heat-generating components
   - The CO2 sensor needs adequate airflow

## Connection Verification

After wiring all components, verify connections by:

1. Double-checking all wiring against this diagram
2. Measuring continuity between connected points with a multimeter
3. Looking for any exposed wires that could cause shorts
4. Ensuring the battery is properly seated and charged

## Powering Up for Testing

1. Install a fully charged 18650 battery
2. Verify the red LED on the TP4056 module illuminates
3. Check that the power LED on the ESP32 lights up
4. If using Termux/Kali Linux, connect a USB cable to upload firmware

## Troubleshooting Common Wiring Issues

1. **No power to ESP32**: Check battery connections and TP4056 module
2. **Camera not detected**: Verify all SDA/SCL and CS/MOSI/MISO/SCK connections
3. **Temperature readings incorrect**: Check DHT22 data pin connection
4. **CO2 sensor not responding**: Verify TX/RX connections are correct (note they are crossed)
5. **Motion detection not working**: Ensure PIR sensor OUT pin is connected to ESP32 D2

For detailed assembly instructions, see the [Assembly Guide](../docs/assembly-guide.md).
