# Bed Bug Detector - Python Server

The Python Server provides a powerful backend for the Bed Bug Detector system, handling data collection, storage, analysis, and machine learning-based detection.

## Features

- **REST API** for communication with ESP32 devices and client applications
- **Machine Learning Detection** for improved accuracy over the ESP32's basic detection
- **Image Analysis** to identify bed bugs visually
- **Data Storage** for detection history and images
- **Analytics** to identify patterns and trends
- **Multi-Device Support** to manage multiple detectors

## Requirements

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/bed-bug-detector.git
cd bed-bug-detector/python/server
```

### 2. Create Virtual Environment (Recommended)

```bash
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Server

Create a `.env` file in the server directory with the following settings:

```
# Server settings
HOST=0.0.0.0
PORT=5000
DEBUG=False

# Data directories
DATA_DIR=data
IMAGE_DIR=data/images
DETECTION_DIR=data/detections

# Detection settings
TEMP_THRESHOLD=1.5
CO2_THRESHOLD=200
CONFIDENCE_THRESHOLD=70.0

# ML settings
ML_MODEL_PATH=../ml/models/bedbug_detector_model.h5
USE_GPU=False

# Security settings (optional)
REQUIRE_API_KEY=False
API_KEY=your_secure_api_key
```

## Running the Server

### Development Mode

```bash
# Start development server
python app.py
```

### Production Mode

For production deployment, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

To run as a background service on Linux:

```bash
# Create a systemd service
sudo nano /etc/systemd/system/bedbug-server.service
```

With the following content:

```
[Unit]
Description=Bed Bug Detector Server
After=network.target

[Service]
User=yourusername
WorkingDirectory=/path/to/bed-bug-detector/python/server
Environment="PATH=/path/to/bed-bug-detector/python/server/venv/bin"
ExecStart=/path/to/bed-bug-detector/python/server/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:

```bash
sudo systemctl enable bedbug-server
sudo systemctl start bedbug-server
```

## API Endpoints

The server provides the following API endpoints:

### Health Check

```
GET /api/health
```

Returns the server status and version.

### Device Registration

```
POST /api/device/register
```

Registers a new ESP32 device with the server.

Request body:
```json
{
  "device_id": "esp32-1234",
  "name": "Bedroom Detector"
}
```

### Sensor Data

```
POST /api/sensor_data
```

Receives and stores sensor data from an ESP32 device.

Request body:
```json
{
  "device_id": "esp32-1234",
  "temperature": 1.8,
  "humidity": 55.3,
  "co2": 250,
  "motion": true,
  "calibrated": true,
  "confidence": 45
}
```

### Detection

```
POST /api/detection
```

Performs bed bug detection based on sensor data and optional image.

Multipart form data:
- `device_id`: ESP32 device ID
- `temperature`: Temperature reading
- `humidity`: Humidity reading
- `co2`: CO2 reading
- `motion`: Motion detection (true/false)
- `calibrated`: Calibration status (true/false)
- `confidence`: Raw confidence from ESP32
- `image`: (Optional) Image file for visual analysis

### Detection History

```
GET /api/history/<device_id>
```

Retrieves detection history for a specific device.

### Calibration

```
POST /api/calibrate
```

Stores baseline calibration data for a device.

Request body:
```json
{
  "device_id": "esp32-1234",
  "baselineTemp": 23.5,
  "baselineCO2": 450
}
```

### Device Settings

```
POST /api/settings/<device_id>
```

Updates settings for a specific device.

Request body:
```json
{
  "settings": {
    "tempThreshold": 1.5,
    "co2Threshold": 200,
    "deepSleepEnabled": false,
    "sleepDurationMinutes": 5
  }
}
```

### Image Retrieval

```
GET /api/image/<device_id>/<timestamp>
```

Retrieves a specific image by device ID and timestamp.

## Integration with ESP32

The server can communicate with the ESP32 in two ways:

1. **Direct Mode**: The ESP32 makes API calls to the server to send data and receive commands.
2. **Proxy Mode**: Mobile/desktop apps communicate with the ESP32 through the server.

### Direct Mode Setup

Add the following to your ESP32 firmware's `config.h`:

```cpp
// Server API settings
#define SERVER_URL "http://your-server-ip:5000"
#define SERVER_API_ENABLED true
#define DEVICE_ID "esp32-unique-id"  // Make this unique for each device
```

Then modify the ESP32 firmware to make API calls to the server.

## Integration with Mobile/Desktop Apps

The server can be used with both the mobile app and desktop application:

1. Set the server URL in the app settings
2. Connect to the server to access advanced detection and analytics

## Data Management

By default, the server stores:

- Detection data as JSON files in `data/detections/`
- Images in `data/images/`

The server automatically cleans up old data:
- Images are kept for 30 days
- Detection data is kept for 90 days

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure the server is running and the port is not blocked by firewall
2. **Image Processing Errors**: Verify OpenCV is properly installed
3. **ML Model Not Found**: Ensure the model path is correct in the configuration

### Logs

Check the server logs for detailed error information:

```bash
cat server.log
```

## Development and Customization

### Adding Custom Detection Logic

To customize the detection algorithm, modify:
- `predict.py` in the ML module for changing the detection logic
- `image_processing.py` in the utils module for image analysis

### Extending the API

To add new API endpoints, modify `app.py`:

```python
@app.route('/api/new_endpoint', methods=['POST'])
def new_endpoint():
    # Your endpoint logic here
    return jsonify({"status": "success"})
```

## Performance Optimization

For systems with high load or limited resources:

1. Reduce image size before processing:
   - Set `BEDBUG_MAX_IMAGE_SIZE=640` in your environment
   
2. Disable ML for faster but less accurate detection:
   - Set `BEDBUG_USE_ML=False` in your environment

3. Enable caching for frequent API calls:
   - Install Redis
   - Set `BEDBUG_ENABLE_CACHE=True`

## Security Considerations

By default, the server does not require authentication. For production use:

1. Enable API key authentication:
   - Set `REQUIRE_API_KEY=True` and a strong `API_KEY` in your configuration
   
2. Use HTTPS:
   - Set up a reverse proxy with Nginx or Apache
   - Configure SSL/TLS certificates

3. Restrict network access:
   - Run the server on a local network
   - Use firewall rules to limit access

## License

This software is provided under the MIT License. See the LICENSE file for details.
