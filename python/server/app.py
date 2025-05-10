"""
Bed Bug Detector - Python Backend Server

This server provides a REST API for:
1. Data collection from ESP32 devices
2. Integration with machine learning detection
3. Historical data storage and retrieval
4. Image processing and enhancement
"""

from flask import Flask, request, jsonify, send_file
import os
import json
import logging
from datetime import datetime
import numpy as np
import cv2
from io import BytesIO
import base64
import threading
import time

# Import custom modules
from config import CONFIG
import sys
sys.path.append('../ml')
sys.path.append('../utils')
from predict import predict_bed_bugs
from image_processing import enhance_image
from data_analysis import analyze_detection_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# In-memory database (replace with SQLite or other DB in production)
detections_db = []
devices_db = {}

# Ensure data directories exist
os.makedirs('data/images', exist_ok=True)
os.makedirs('data/detections', exist_ok=True)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": CONFIG['version']
    })


@app.route('/api/device/register', methods=['POST'])
def register_device():
    """Register a new ESP32 device"""
    try:
        data = request.json
        device_id = data.get('device_id')
        name = data.get('name', f"BedBugDetector-{device_id[-6:]}")
        
        if not device_id:
            return jsonify({"error": "device_id is required"}), 400
            
        # Store device information
        devices_db[device_id] = {
            "name": name,
            "ip_address": request.remote_addr,
            "last_seen": datetime.now().isoformat(),
            "firmware_version": data.get('firmware_version', 'unknown'),
            "calibrated": data.get('calibrated', False),
            "settings": data.get('settings', {}),
        }
        
        logger.info(f"Device registered: {device_id} - {name}")
        return jsonify({"status": "success", "device_id": device_id})
    
    except Exception as e:
        logger.error(f"Error registering device: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/sensor_data', methods=['POST'])
def receive_sensor_data():
    """Receive and store sensor data from ESP32"""
    try:
        data = request.json
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({"error": "device_id is required"}), 400
        
        # Update device last_seen timestamp
        if device_id in devices_db:
            devices_db[device_id]["last_seen"] = datetime.now().isoformat()
        
        # Process sensor data
        sensor_data = {
            "device_id": device_id,
            "timestamp": datetime.now().isoformat(),
            "temperature": data.get('temperature'),
            "humidity": data.get('humidity'),
            "co2": data.get('co2'),
            "motion": data.get('motion'),
            "calibrated": data.get('calibrated', False),
            "raw_confidence": data.get('confidence', 0)
        }
        
        # Store sensor reading
        with open(f"data/detections/{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(sensor_data, f)
        
        # Return simple success without running ML (which will be done asynchronously)
        return jsonify({"status": "success", "received": sensor_data})
    
    except Exception as e:
        logger.error(f"Error processing sensor data: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/detection', methods=['POST'])
def process_detection():
    """Perform bed bug detection based on sensor data and image (if provided)"""
    try:
        # Get form data and files
        data = request.json if request.is_json else request.form.to_dict()
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({"error": "device_id is required"}), 400
        
        # Extract sensor data
        sensor_data = {
            "temperature": float(data.get('temperature', 0)),
            "humidity": float(data.get('humidity', 0)),
            "co2": int(data.get('co2', 0)),
            "motion": data.get('motion') == 'True',
            "calibrated": data.get('calibrated') == 'True',
            "raw_confidence": float(data.get('confidence', 0))
        }
        
        # Process image if provided
        image_data = None
        enhanced_image = None
        has_image = False
        
        if 'image' in request.files:
            image_file = request.files['image']
            img_bytes = image_file.read()
            
            # Save original image
            img_filename = f"{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            with open(f"data/images/{img_filename}", 'wb') as f:
                f.write(img_bytes)
            
            # Convert to OpenCV format for processing
            nparr = np.frombuffer(img_bytes, np.uint8)
            image_data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Enhance image for better detection
            enhanced_image = enhance_image(image_data)
            has_image = True
        
        # Run ML detection
        detection_result = predict_bed_bugs(
            sensor_data=sensor_data,
            image_data=enhanced_image if has_image else None
        )
        
        # Combine results and sensor data
        result = {
            "timestamp": datetime.now().isoformat(),
            "device_id": device_id,
            "sensor_data": sensor_data,
            "has_image": has_image,
            "detection": {
                "detected": detection_result["detected"],
                "confidence": detection_result["confidence"],
                "bounding_boxes": detection_result.get("bounding_boxes", []) if has_image else [],
                "algorithm_version": detection_result["algorithm_version"]
            },
            "advice": generate_advice(detection_result, sensor_data)
        }
        
        # Save detection result
        detection_filename = f"{device_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_detection.json"
        with open(f"data/detections/{detection_filename}", 'w') as f:
            # Convert numpy values to Python native types for JSON serialization
            json_result = {k: v for k, v in result.items()}
            json.dump(json_result, f)
        
        # Add to in-memory database for quick retrieval
        detections_db.append(result)
        
        # Keep only the latest 1000 detections in memory
        if len(detections_db) > 1000:
            detections_db.pop(0)
        
        return jsonify({
            "status": "success",
            "detection": result["detection"],
            "advice": result["advice"]
        })
    
    except Exception as e:
        logger.error(f"Error processing detection: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/history/<device_id>', methods=['GET'])
def get_detection_history(device_id):
    """Retrieve detection history for a specific device"""
    try:
        # Filter in-memory database for the specified device
        device_history = [d for d in detections_db if d["device_id"] == device_id]
        
        # If not enough in memory, load from files
        if len(device_history) < 10:
            # List detection JSON files for this device
            detection_files = [f for f in os.listdir("data/detections") 
                             if f.startswith(device_id) and f.endswith('_detection.json')]
            
            # Load each file and add to history if not already present
            for filename in sorted(detection_files, reverse=True)[:100]:  # Limit to latest 100
                file_path = os.path.join("data/detections", filename)
                with open(file_path, 'r') as f:
                    try:
                        detection_data = json.load(f)
                        # Check if already in device_history
                        if not any(d["timestamp"] == detection_data["timestamp"] for d in device_history):
                            device_history.append(detection_data)
                    except json.JSONDecodeError:
                        logger.warning(f"Couldn't parse {filename}")
        
        # Sort by timestamp (newest first)
        device_history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Analyze the history data
        analysis = analyze_detection_history(device_history)
        
        return jsonify({
            "status": "success",
            "device_id": device_id,
            "history": device_history[:50],  # Return the latest 50 entries
            "analysis": analysis
        })
    
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/calibrate', methods=['POST'])
def calibrate_detector():
    """Store baseline calibration data for a device"""
    try:
        data = request.json
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({"error": "device_id is required"}), 400
        
        # Get baseline readings
        baseline_temp = float(data.get('baselineTemp', 0))
        baseline_co2 = int(data.get('baselineCO2', 0))
        
        # Update device calibration status
        if device_id in devices_db:
            devices_db[device_id]["calibrated"] = True
            devices_db[device_id]["last_calibration"] = datetime.now().isoformat()
            devices_db[device_id]["baseline"] = {
                "temperature": baseline_temp,
                "co2": baseline_co2
            }
        
        # Save calibration data
        calibration_data = {
            "device_id": device_id,
            "timestamp": datetime.now().isoformat(),
            "baselineTemp": baseline_temp,
            "baselineCO2": baseline_co2
        }
        
        with open(f"data/detections/{device_id}_calibration.json", 'w') as f:
            json.dump(calibration_data, f)
        
        return jsonify({
            "status": "success",
            "baselineTemp": baseline_temp,
            "baselineCO2": baseline_co2
        })
    
    except Exception as e:
        logger.error(f"Error during calibration: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/settings/<device_id>', methods=['POST'])
def update_settings(device_id):
    """Update settings for a specific device"""
    try:
        if device_id not in devices_db:
            return jsonify({"error": "Device not found"}), 404
        
        data = request.json
        settings = data.get('settings', {})
        
        # Update device settings
        devices_db[device_id]["settings"] = settings
        
        return jsonify({
            "status": "success",
            "device_id": device_id,
            "settings": settings
        })
    
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/image/<device_id>/<timestamp>', methods=['GET'])
def get_image(device_id, timestamp):
    """Retrieve a specific image by device ID and timestamp"""
    try:
        # Find image file matching the pattern
        image_files = [f for f in os.listdir("data/images") 
                     if f.startswith(f"{device_id}_{timestamp}")]
        
        if not image_files:
            return jsonify({"error": "Image not found"}), 404
        
        image_path = os.path.join("data/images", image_files[0])
        return send_file(image_path, mimetype='image/jpeg')
    
    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        return jsonify({"error": str(e)}), 500


def generate_advice(detection_result, sensor_data):
    """Generate tailored advice based on detection results"""
    detected = detection_result["detected"]
    confidence = detection_result["confidence"]
    
    if not detected:
        return "No significant bed bug activity detected. Continue monitoring regularly as early infestations can be difficult to detect. For peace of mind, consider scanning multiple areas of concern."
    
    if confidence > 90:
        return "High confidence bed bug detection! Immediate action recommended. Thoroughly inspect the area, focusing on mattress seams, bedding, and nearby furniture. Consider professional treatment."
    
    if confidence > 70:
        return "Signs of bed bug activity detected. Recommend thorough inspection of the area, focusing on mattress seams, bedding, and nearby furniture. Consider professional treatment if multiple areas show positive detection."
    
    return "Possible bed bug activity detected with low confidence. Re-check this area with additional scans. Focus inspection on mattress seams, box springs, and furniture crevices."


# Background task to clean up old data
def cleanup_old_data():
    """Periodically clean up old data files"""
    while True:
        try:
            now = datetime.now()
            # Keep images for 30 days
            cutoff = now.replace(day=now.day-30)
            cutoff_str = cutoff.strftime('%Y%m%d')
            
            # Clean up old images
            for filename in os.listdir("data/images"):
                if filename < f"{cutoff_str}_000000.jpg":
                    os.remove(os.path.join("data/images", filename))
            
            # Clean up old detection data (keep for 90 days)
            cutoff = now.replace(day=now.day-90)
            cutoff_str = cutoff.strftime('%Y%m%d')
            
            for filename in os.listdir("data/detections"):
                if not filename.endswith('calibration.json') and filename < f"{cutoff_str}_000000.json":
                    os.remove(os.path.join("data/detections", filename))
                    
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
        # Run every 24 hours
        time.sleep(86400)


if __name__ == '__main__':
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_data, daemon=True)
    cleanup_thread.start()
    
    # Start server
    app.run(
        host=CONFIG['host'],
        port=CONFIG['port'],
        debug=CONFIG['debug']
    )
