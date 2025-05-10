"""
Bed Bug Detector - Device Manager

This module provides a Python interface for direct communication with
the ESP32-based bed bug detector. It handles connection, data retrieval,
and command execution.
"""

import requests
import logging
import json
import time
import os
import base64
from io import BytesIO
from datetime import datetime
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BedBugDetector:
    """Class for managing communication with a bed bug detector device"""
    
    def __init__(self, device_ip="192.168.4.1", server_url=None, device_id=None):
        """
        Initialize connection to a bed bug detector
        
        Args:
            device_ip (str): IP address of the ESP32 detector
            server_url (str, optional): URL of the Python server for indirect communication
            device_id (str, optional): Unique ID of the device
        """
        self.device_ip = device_ip
        self.server_url = server_url
        self.device_id = device_id or f"esp32-{device_ip.replace('.', '')}"
        self.direct_mode = server_url is None
        self.connected = False
        self.last_error = None
        self.timeout = 5  # Request timeout in seconds
        
        # Device data
        self.calibrated = False
        self.baseline_temp = None
        self.baseline_co2 = None
        self.last_sensor_data = None
        self.firmware_version = None
        
        # Connection stats
        self.connection_attempts = 0
        self.successful_connections = 0
        self.last_connection_time = None
    
    def connect(self):
        """
        Connect to the detector device
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        self.connection_attempts += 1
        
        try:
            if self.direct_mode:
                # Direct connection to ESP32
                url = f"http://{self.device_ip}/sensor-data"
                response = requests.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    # Extract device info
                    data = response.json()
                    self.calibrated = data.get("calibrated", False)
                    
                    # Set firmware version if available
                    if "version" in data:
                        self.firmware_version = data.get("version")
                    
                    self.connected = True
                    self.successful_connections += 1
                    self.last_connection_time = datetime.now()
                    logger.info(f"Successfully connected to detector at {self.device_ip}")
                    return True
                else:
                    self.last_error = f"HTTP error: {response.status_code}"
                    logger.error(f"Connection failed: {self.last_error}")
                    self.connected = False
                    return False
            else:
                # Connection through server
                url = f"{self.server_url}/api/device/register"
                data = {
                    "device_id": self.device_id,
                    "name": f"Detector at {self.device_ip}"
                }
                
                response = requests.post(url, json=data, timeout=self.timeout)
                
                if response.status_code == 200:
                    self.connected = True
                    self.successful_connections += 1
                    self.last_connection_time = datetime.now()
                    logger.info(f"Successfully registered detector with server")
                    return True
                else:
                    self.last_error = f"Server error: {response.status_code}"
                    logger.error(f"Server registration failed: {self.last_error}")
                    self.connected = False
                    return False
                
        except requests.exceptions.ConnectionError as e:
            self.last_error = f"Connection error: {str(e)}"
            logger.error(f"Connection failed: {self.last_error}")
            self.connected = False
            return False
        
        except requests.exceptions.Timeout as e:
            self.last_error = "Connection timed out"
            logger.error(f"Connection timed out: {str(e)}")
            self.connected = False
            return False
        
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Unexpected error during connection: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Disconnect from the detector device
        
        Returns:
            bool: True if successful
        """
        # There's no active connection to close in HTTP, so just update the status
        self.connected = False
        logger.info(f"Disconnected from detector at {self.device_ip}")
        return True
    
    def get_sensor_data(self):
        """
        Retrieve current sensor data from the detector
        
        Returns:
            dict: Sensor data dictionary or None if failed
        """
        if not self.connected and not self.connect():
            return None
        
        try:
            if self.direct_mode:
                # Direct connection to ESP32
                url = f"http://{self.device_ip}/sensor-data"
                response = requests.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    self.last_sensor_data = data
                    return data
                else:
                    self.last_error = f"HTTP error: {response.status_code}"
                    logger.error(f"Failed to get sensor data: {self.last_error}")
                    return None
            else:
                # Get data through server
                url = f"{self.server_url}/api/sensor_data"
                params = {"device_id": self.device_id}
                
                response = requests.get(url, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    self.last_sensor_data = data
                    return data
                else:
                    self.last_error = f"Server error: {response.status_code}"
                    logger.error(f"Failed to get sensor data from server: {self.last_error}")
                    return None
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting sensor data: {str(e)}")
            return None
    
    def calibrate(self):
        """
        Calibrate the detector
        
        Returns:
            dict: Calibration results or None if failed
        """
        if not self.connected and not self.connect():
            return None
        
        try:
            if self.direct_mode:
                # Direct connection to ESP32
                url = f"http://{self.device_ip}/calibrate"
                response = requests.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    self.calibrated = True
                    self.baseline_temp = data.get("baselineTemp")
                    self.baseline_co2 = data.get("baselineCO2")
                    logger.info(f"Calibration successful: Temp={self.baseline_temp}°C, CO2={self.baseline_co2}ppm")
                    return data
                else:
                    self.last_error = f"HTTP error: {response.status_code}"
                    logger.error(f"Calibration failed: {self.last_error}")
                    return None
            else:
                # Calibrate through server
                url = f"{self.server_url}/api/calibrate"
                # First get current sensor data
                sensor_data = self.get_sensor_data()
                
                if not sensor_data:
                    return None
                
                # Send calibration request
                data = {
                    "device_id": self.device_id,
                    "baselineTemp": sensor_data.get("temperature", 0),
                    "baselineCO2": sensor_data.get("co2", 0)
                }
                
                response = requests.post(url, json=data, timeout=self.timeout)
                
                if response.status_code == 200:
                    result = response.json()
                    self.calibrated = True
                    self.baseline_temp = result.get("baselineTemp")
                    self.baseline_co2 = result.get("baselineCO2")
                    logger.info(f"Calibration successful via server: Temp={self.baseline_temp}°C, CO2={self.baseline_co2}ppm")
                    return result
                else:
                    self.last_error = f"Server error: {response.status_code}"
                    logger.error(f"Calibration via server failed: {self.last_error}")
                    return None
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error during calibration: {str(e)}")
            return None
    
    def capture_image(self):
        """
        Capture an image from the detector's camera
        
        Returns:
            numpy.ndarray: Image data as numpy array or None if failed
        """
        if not self.connected and not self.connect():
            return None
        
        try:
            if self.direct_mode:
                # Direct connection to ESP32
                url = f"http://{self.device_ip}/capture"
                response = requests.get(url, timeout=10)  # Longer timeout for image capture
                
                if response.status_code == 200:
                    # Convert image bytes to numpy array
                    nparr = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    logger.info(f"Image captured successfully")
                    return img
                else:
                    self.last_error = f"HTTP error: {response.status_code}"
                    logger.error(f"Image capture failed: {self.last_error}")
                    return None
            else:
                # Capture through server
                url = f"{self.server_url}/api/image/{self.device_id}/latest"
                
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    # Convert image bytes to numpy array
                    nparr = np.frombuffer(response.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    logger.info(f"Image captured via server successfully")
                    return img
                else:
                    self.last_error = f"Server error: {response.status_code}"
                    logger.error(f"Image capture via server failed: {self.last_error}")
                    return None
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error capturing image: {str(e)}")
            return None
    
    def run_detection(self):
        """
        Run bed bug detection using current sensor data and camera image
        
        Returns:
            dict: Detection results or None if failed
        """
        if not self.connected and not self.connect():
            return None
        
        try:
            if self.direct_mode:
                # In direct mode, we'll implement a simplified detection algorithm
                # Get sensor data
                sensor_data = self.get_sensor_data()
                
                if not sensor_data:
                    return None
                
                # Capture image if camera is available
                image = self.capture_image()
                
                # Run simplified detection algorithm
                return self._local_detection_algorithm(sensor_data, image)
            else:
                # Run detection through server
                url = f"{self.server_url}/api/detection"
                
                # Get current sensor data
                sensor_data = self.get_sensor_data()
                
                if not sensor_data:
                    return None
                
                # Prepare multipart form data
                files = {}
                data = {
                    "device_id": self.device_id,
                    "temperature": sensor_data.get("temperature", 0),
                    "humidity": sensor_data.get("humidity", 0),
                    "co2": sensor_data.get("co2", 0),
                    "motion": sensor_data.get("motion", False),
                    "calibrated": sensor_data.get("calibrated", False),
                    "confidence": sensor_data.get("raw_confidence", 0)
                }
                
                # Try to capture image if available
                try:
                    image = self.capture_image()
                    if image is not None:
                        # Convert numpy array to image bytes
                        success, img_encoded = cv2.imencode('.jpg', image)
                        if success:
                            files["image"] = ("capture.jpg", img_encoded.tobytes())
                except Exception as e:
                    logger.warning(f"Could not capture image for detection: {str(e)}")
                
                # Send detection request
                response = requests.post(
                    url, 
                    data=data, 
                    files=files, 
                    timeout=15  # Longer timeout for detection
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Detection successful: {result.get('detection', {}).get('detected', False)}")
                    return result
                else:
                    self.last_error = f"Server error: {response.status_code}"
                    logger.error(f"Detection via server failed: {self.last_error}")
                    return None
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error during detection: {str(e)}")
            return None
    
    def _local_detection_algorithm(self, sensor_data, image=None):
        """
        Simple local detection algorithm when server is not available
        
        Args:
            sensor_data (dict): Sensor readings
            image (numpy.ndarray, optional): Image data
            
        Returns:
            dict: Detection results
        """
        # Extract sensor values
        temperature = sensor_data.get("temperature", 0)
        humidity = sensor_data.get("humidity", 0)
        co2 = sensor_data.get("co2", 0)
        motion = sensor_data.get("motion", False)
        calibrated = sensor_data.get("calibrated", False)
        
        # Default thresholds
        temp_threshold = 1.5  # °C
        co2_threshold = 200   # ppm
        
        # If device is calibrated, we should have baseline values
        if calibrated and self.baseline_temp is not None and self.baseline_co2 is not None:
            # Calculate differences from baseline
            temp_diff = abs(temperature - self.baseline_temp)
            co2_diff = co2 - self.baseline_co2
        else:
            # If not calibrated, use absolute values
            temp_diff = abs(temperature)
            co2_diff = co2
        
        # Calculate confidence score
        confidence = 0
        
        # Check temperature anomaly
        if temp_diff > temp_threshold:
            # Proportional to how much it exceeds threshold
            temp_factor = min(temp_diff / temp_threshold, 2.0)
            confidence += 30 * temp_factor
        
        # Check CO2 levels
        if co2_diff > co2_threshold:
            # Proportional to how much it exceeds threshold
            co2_factor = min(co2_diff / co2_threshold, 2.0)
            confidence += 30 * co2_factor
        
        # Check motion
        if motion:
            confidence += 40
        
        # If we have an image, try basic visual detection
        if image is not None:
            try:
                image_confidence = self._basic_image_detection(image)
                # Combine confidences (image-based detection gets 25% weight)
                confidence = 0.75 * confidence + 0.25 * image_confidence
            except Exception as e:
                logger.error(f"Error in image detection: {str(e)}")
        
        # Cap confidence at 100
        confidence = min(confidence, 100)
        
        # Determine if bed bugs are detected
        detected = confidence >= 70  # 70% confidence threshold
        
        # Generate advice based on detection
        if not detected:
            advice = "No significant bed bug activity detected. Continue monitoring regularly as early infestations can be difficult to detect."
        elif confidence > 90:
            advice = "High confidence bed bug detection! Immediate action recommended. Thoroughly inspect the area, focusing on mattress seams, bedding, and nearby furniture. Consider professional treatment."
        elif confidence > 70:
            advice = "Signs of bed bug activity detected. Recommend thorough inspection of the area, focusing on mattress seams, bedding, and nearby furniture. Consider professional treatment if multiple areas show positive detection."
        else:
            advice = "Possible bed bug activity detected with low confidence. Re-check this area with additional scans. Focus inspection on mattress seams, box springs, and furniture crevices."
        
        # Return detection results
        return {
            "status": "success",
            "detection": {
                "detected": detected,
                "confidence": confidence,
                "algorithm_version": "1.0.0 (local)"
            },
            "advice": advice
        }
    
    def _basic_image_detection(self, image):
        """
        Basic image-based bed bug detection
        
        Args:
            image (numpy.ndarray): Image data
            
        Returns:
            float: Confidence score (0-100)
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Bed bugs are typically reddish-brown
            # Define HSV range for reddish-brown colors
            lower_brown = np.array([0, 50, 50])
            upper_brown = np.array([20, 255, 255])
            
            # Create mask and find contours
            mask = cv2.inRange(hsv, lower_brown, upper_brown)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Bed bug size is typically 4-5mm, which could be roughly 20-30 pixels
            # in an average close-up image
            min_area = 200  # Minimum contour area
            max_area = 3000  # Maximum contour area
            
            # Calculate confidence based on contours
            confidence = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (bed bugs are roughly oval-shaped)
                    aspect_ratio = float(w) / h
                    if 0.5 < aspect_ratio < 2.0:
                        # Extract region of interest
                        roi = image[y:y+h, x:x+w]
                        
                        # Calculate color match
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        h, s, v = cv2.split(hsv_roi)
                        
                        avg_hue = np.mean(h)
                        avg_saturation = np.mean(s)
                        
                        # If color matches bed bug color range
                        if 0 <= avg_hue <= 20 and avg_saturation > 50:
                            confidence += 10  # Add to confidence for each matching contour
            
            # Cap confidence at 100
            return min(confidence, 100)
            
        except Exception as e:
            logger.error(f"Error in basic image detection: {str(e)}")
            return 0
    
    def update_settings(self, settings):
        """
        Update detector settings
        
        Args:
            settings (dict): Dictionary of settings to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected and not self.connect():
            return False
        
        try:
            if self.direct_mode:
                # Direct connection to ESP32
                url = f"http://{self.device_ip}/set-thresholds"
                response = requests.post(url, json=settings, timeout=self.timeout)
                
                if response.status_code == 200:
                    logger.info(f"Settings updated successfully")
                    return True
                else:
                    self.last_error = f"HTTP error: {response.status_code}"
                    logger.error(f"Settings update failed: {self.last_error}")
                    return False
            else:
                # Update through server
                url = f"{self.server_url}/api/settings/{self.device_id}"
                
                data = {
                    "settings": settings
                }
                
                response = requests.post(url, json=data, timeout=self.timeout)
                
                if response.status_code == 200:
                    logger.info(f"Settings updated via server successfully")
                    return True
                else:
                    self.last_error = f"Server error: {response.status_code}"
                    logger.error(f"Settings update via server failed: {self.last_error}")
                    return False
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error updating settings: {str(e)}")
            return False
    
    def get_detection_history(self, limit=10):
        """
        Get detection history
        
        Args:
            limit (int): Maximum number of history items to return
            
        Returns:
            list: List of detection history items or None if failed
        """
        if self.direct_mode:
            # Not supported in direct mode
            logger.warning("Detection history not available in direct mode")
            return []
        
        if not self.connected and not self.connect():
            return None
        
        try:
            # Get history through server
            url = f"{self.server_url}/api/history/{self.device_id}"
            
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                history = data.get("history", [])
                logger.info(f"Retrieved {len(history)} history items")
                return history[:limit]
            else:
                self.last_error = f"Server error: {response.status_code}"
                logger.error(f"History retrieval failed: {self.last_error}")
                return None
                
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error retrieving history: {str(e)}")
            return None
    
    def save_image(self, image, filename=None):
        """
        Save an image to file
        
        Args:
            image (numpy.ndarray): Image data
            filename (str, optional): Filename to save as, or auto-generate if None
            
        Returns:
            str: Path to saved image or None if failed
        """
        if image is None:
            return None
        
        try:
            # Create directory if needed
            os.makedirs("images", exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"images/capture_{timestamp}.jpg"
            
            # Ensure filename has jpg extension
            if not filename.endswith(".jpg"):
                filename += ".jpg"
            
            # Save image
            cv2.imwrite(filename, image)
            logger.info(f"Image saved to {filename}")
            return filename
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error saving image: {str(e)}")
            return None
    
    def get_status(self):
        """
        Get detector status information
        
        Returns:
            dict: Status information
        """
        return {
            "connected": self.connected,
            "device_ip": self.device_ip,
            "device_id": self.device_id,
            "direct_mode": self.direct_mode,
            "calibrated": self.calibrated,
            "baseline_temp": self.baseline_temp,
            "baseline_co2": self.baseline_co2,
            "firmware_version": self.firmware_version,
            "connection_attempts": self.connection_attempts,
            "successful_connections": self.successful_connections,
            "last_connection_time": self.last_connection_time.isoformat() if self.last_connection_time else None,
            "last_error": self.last_error
        }


# Example usage
if __name__ == "__main__":
    print("Bed Bug Detector - Device Manager")
    
    # Connect to a detector
    detector = BedBugDetector(device_ip="192.168.4.1")
    
    if detector.connect():
        print("Connected to detector")
        
        # Get sensor data
        sensor_data = detector.get_sensor_data()
        if sensor_data:
            print("Sensor data:")
            print(f"  Temperature: {sensor_data.get('temperature', 0):.1f}°C")
            print(f"  Humidity: {sensor_data.get('humidity', 0):.1f}%")
            print(f"  CO2: {sensor_data.get('co2', 0)} ppm")
            print(f"  Motion: {sensor_data.get('motion', False)}")
        
        # Calibrate
        print("\nCalibrating...")
        calibration = detector.calibrate()
        if calibration:
            print(f"Calibration successful")
            print(f"  Baseline temperature: {calibration.get('baselineTemp', 0):.1f}°C")
            print(f"  Baseline CO2: {calibration.get('baselineCO2', 0)} ppm")
        
        # Run detection
        print("\nRunning detection...")
        result = detector.run_detection()
        if result:
            detection = result.get("detection", {})
            detected = detection.get("detected", False)
            confidence = detection.get("confidence", 0)
            
            print(f"Detection result: {'Detected' if detected else 'Not detected'}")
            print(f"Confidence: {confidence:.1f}%")
            print(f"Advice: {result.get('advice', '')}")
        
        # Disconnect
        detector.disconnect()
        print("\nDisconnected from detector")
    else:
        print(f"Connection failed: {detector.last_error}")
