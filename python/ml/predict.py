"""
Bed Bug Detector - Machine Learning Prediction Module

This module handles the detection of bed bugs based on:
1. Sensor data (temperature, humidity, CO2, motion)
2. Image data (optional)

It combines both traditional rule-based detection and machine learning approaches.
"""

import os
import sys
import logging
import numpy as np
import json
from datetime import datetime
import cv2

# Conditionally import TensorFlow for image detection
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.config import CONFIG, DETECTION_WEIGHTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load ML model if available
model = None
if TF_AVAILABLE:
    try:
        model_path = CONFIG['ml_model_path']
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded ML model from {model_path}")
        else:
            logger.warning(f"ML model not found at {model_path}, will use rule-based detection only")
    except Exception as e:
        logger.error(f"Error loading ML model: {str(e)}")


def predict_bed_bugs(sensor_data, image_data=None):
    """
    Performs bed bug detection using a combination of rule-based and ML approaches
    
    Args:
        sensor_data (dict): Dictionary containing sensor readings
                           (temperature, humidity, co2, motion)
        image_data (numpy.ndarray, optional): Image data as a numpy array
    
    Returns:
        dict: Detection result with confidence score and bounding boxes (if image provided)
    """
    # Initialize result
    result = {
        "detected": False,
        "confidence": 0,
        "algorithm_version": "1.0.0",
    }
    
    # First, run rule-based detection on sensor data
    sensor_confidence = rule_based_detection(sensor_data)
    result["sensor_confidence"] = sensor_confidence
    
    # Then, if image is provided, run image-based detection
    image_confidence = 0
    bounding_boxes = []
    
    if image_data is not None:
        image_result = image_based_detection(image_data)
        image_confidence = image_result["confidence"]
        bounding_boxes = image_result.get("bounding_boxes", [])
        result["bounding_boxes"] = bounding_boxes
    
    # Combine confidences using configured weights
    if image_data is not None:
        # If we have both sensor and image data
        total_confidence = (
            sensor_confidence * (DETECTION_WEIGHTS["temperature"] + 
                                DETECTION_WEIGHTS["co2"] + 
                                DETECTION_WEIGHTS["motion"]) +
            image_confidence * DETECTION_WEIGHTS["image"]
        )
    else:
        # If we only have sensor data, normalize the weights
        sensor_weights_sum = (DETECTION_WEIGHTS["temperature"] + 
                             DETECTION_WEIGHTS["co2"] + 
                             DETECTION_WEIGHTS["motion"])
        
        # Adjust weights to account for missing image data
        total_confidence = sensor_confidence * (sensor_weights_sum / 
                                              (sensor_weights_sum + DETECTION_WEIGHTS["image"]))
    
    # Round to 2 decimal places
    result["confidence"] = round(total_confidence, 2)
    
    # Determine if bed bugs are detected based on confidence threshold
    result["detected"] = result["confidence"] >= CONFIG["detection_confidence_threshold"]
    
    return result


def rule_based_detection(sensor_data):
    """
    Performs rule-based detection using sensor data
    
    Args:
        sensor_data (dict): Dictionary containing sensor readings
    
    Returns:
        float: Confidence score (0-100)
    """
    # Extract sensor values
    temperature = sensor_data.get("temperature", 0)
    humidity = sensor_data.get("humidity", 0)
    co2 = sensor_data.get("co2", 0)
    motion = sensor_data.get("motion", False)
    calibrated = sensor_data.get("calibrated", False)
    raw_confidence = sensor_data.get("raw_confidence", 0)
    
    # If not calibrated, we use the raw confidence from the ESP32
    if not calibrated:
        return raw_confidence
    
    # Start with 0 confidence
    confidence = 0
    
    # Default thresholds (these should ideally be calibrated per device)
    temp_threshold = CONFIG.get("default_temp_threshold", 1.5)
    co2_threshold = CONFIG.get("default_co2_threshold", 200)
    
    # Check temperature anomaly (bed bugs create heat spots)
    # This assumes baseline temperature was established during calibration
    if abs(temperature) > temp_threshold:
        # Proportional to how much it exceeds threshold
        temp_factor = min(abs(temperature) / temp_threshold, 2.0)
        confidence += 30 * temp_factor
    
    # Check CO2 levels (bed bugs emit CO2)
    if co2 > co2_threshold:
        # Proportional to how much it exceeds threshold
        co2_factor = min(co2 / co2_threshold, 2.0)
        confidence += 30 * co2_factor
    
    # Check motion (bed bugs create small movements)
    if motion:
        confidence += 40
    
    # Cap confidence at 100
    confidence = min(confidence, 100)
    
    return confidence


def image_based_detection(image_data):
    """
    Performs image-based bed bug detection
    
    Args:
        image_data (numpy.ndarray): Image data as a numpy array
    
    Returns:
        dict: Detection result with confidence score and bounding boxes
    """
    result = {
        "confidence": 0,
        "bounding_boxes": []
    }
    
    # If no image data, return zero confidence
    if image_data is None:
        return result
    
    # If TensorFlow and model are not available, use basic OpenCV detection
    if not TF_AVAILABLE or model is None:
        return basic_image_detection(image_data)
    
    try:
        # Preprocess the image for the model
        processed_image = preprocess_image(image_data)
        
        # Make prediction using the model
        predictions = model.predict(np.expand_dims(processed_image, axis=0))
        
        # Process predictions (depends on model architecture)
        # This assumes a model that outputs confidence and bounding box coordinates
        if isinstance(predictions, list) and len(predictions) > 1:
            # YOLO-like model with separate outputs for class and bounding boxes
            class_preds = predictions[0][0]
            bbox_preds = predictions[1][0]
            
            # Find bed bug detections (class index 0)
            confidence = float(class_preds[0]) * 100
            
            # Process bounding boxes if confidence is high enough
            if confidence > 30:
                # Assuming bbox_preds contains [x, y, width, height, confidence]
                for i in range(len(bbox_preds)):
                    bbox = bbox_preds[i]
                    if bbox[4] > 0.2:  # Confidence threshold for boxes
                        result["bounding_boxes"].append({
                            "x": int(bbox[0]),
                            "y": int(bbox[1]),
                            "width": int(bbox[2]),
                            "height": int(bbox[3]),
                            "confidence": float(bbox[4])
                        })
        else:
            # Simple classification model
            confidence = float(predictions[0][0]) * 100
        
        result["confidence"] = confidence
        
    except Exception as e:
        logger.error(f"Error in ML-based detection: {str(e)}")
        # Fallback to basic detection
        return basic_image_detection(image_data)
    
    return result


def basic_image_detection(image_data):
    """
    Basic image-based detection using OpenCV when ML model is not available
    
    Args:
        image_data (numpy.ndarray): Image data as a numpy array
    
    Returns:
        dict: Detection result with confidence
    """
    result = {
        "confidence": 0,
        "bounding_boxes": []
    }
    
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)
        
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
        
        bed_bug_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (bed bugs are roughly oval-shaped)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:
                    # Calculate some confidence based on shape and color
                    color_confidence = analyze_color_profile(image_data[y:y+h, x:x+w])
                    shape_confidence = analyze_shape(contour)
                    
                    contour_confidence = (color_confidence * 0.7 + shape_confidence * 0.3)
                    
                    if contour_confidence > 40:
                        bed_bug_candidates.append({
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h),
                            "confidence": contour_confidence
                        })
        
        # Calculate overall confidence based on candidates
        if len(bed_bug_candidates) > 0:
            # Sort by confidence
            bed_bug_candidates.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Use average of top 3 confidences or all if fewer
            top_candidates = bed_bug_candidates[:min(3, len(bed_bug_candidates))]
            result["confidence"] = sum(c["confidence"] for c in top_candidates) / len(top_candidates)
            result["bounding_boxes"] = top_candidates
        
    except Exception as e:
        logger.error(f"Error in basic image detection: {str(e)}")
    
    return result


def analyze_color_profile(roi):
    """
    Analyze color profile of a region of interest
    
    Args:
        roi (numpy.ndarray): Region of interest from the image
    
    Returns:
        float: Confidence based on color (0-100)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate average hue, saturation, value
    h, s, v = cv2.split(hsv)
    avg_hue = np.mean(h)
    avg_saturation = np.mean(s)
    avg_value = np.mean(v)
    
    # Bed bugs are typically reddish-brown
    # Hue: 0-20 (red to orange-brown in OpenCV HSV)
    # Saturation: Medium to high (100-255)
    # Value: Medium (80-200)
    
    hue_score = 100 - min(abs(avg_hue - 10) * 5, 100)  # Highest at hue = 10
    saturation_score = min(avg_saturation / 1.5, 100)  # Higher saturation = higher score
    value_score = 100 - min(abs(avg_value - 140) / 1.4, 100)  # Highest at value = 140
    
    # Weighted average
    color_confidence = (hue_score * 0.5 + saturation_score * 0.3 + value_score * 0.2)
    
    return color_confidence


def analyze_shape(contour):
    """
    Analyze shape characteristics for bed bug detection
    
    Args:
        contour (numpy.ndarray): Contour from OpenCV
    
    Returns:
        float: Confidence based on shape (0-100)
    """
    # Calculate shape metrics
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Bed bugs have an oval shape
    # Calculate circularity: 4*pi*area/perimeter^2
    # Perfect circle = 1.0, oval ~= 0.7-0.9
    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Bed bugs have circularity around 0.7-0.8
    circularity_score = 100 - min(abs(circularity - 0.75) * 200, 100)
    
    # Calculate convexity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = 0
    if hull_area > 0:
        solidity = float(area) / hull_area
    
    # Bed bugs are mostly convex (solidity around 0.9)
    solidity_score = solidity * 100
    
    # Calculate shape confidence
    shape_confidence = (circularity_score * 0.6 + solidity_score * 0.4)
    
    return shape_confidence


def preprocess_image(image):
    """
    Preprocess image for the ML model
    
    Args:
        image (numpy.ndarray): Image data
    
    Returns:
        numpy.ndarray: Processed image
    """
    # Resize to expected input size (e.g., 224x224 for many models)
    target_size = (224, 224)
    resized = cv2.resize(image, target_size)
    
    # Convert to RGB if needed (TensorFlow models often expect RGB)
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        # OpenCV uses BGR, convert to RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to 0-1
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized


if __name__ == "__main__":
    # Simple test code
    print("Testing bed bug detection")
    
    # Example sensor data
    test_sensor_data = {
        "temperature": 2.0,
        "humidity": 60,
        "co2": 250,
        "motion": True,
        "calibrated": True
    }
    
    # Test the rule-based detection
    confidence = rule_based_detection(test_sensor_data)
    print(f"Rule-based detection confidence: {confidence}%")
    
    # Test the full detection pipeline
    result = predict_bed_bugs(test_sensor_data)
    print(f"Detection result: {json.dumps(result, indent=2)}")
    
    # If we have an image, test image-based detection
    test_image_path = os.path.join(os.path.dirname(__file__), "dataset", "test_image.jpg")
    if os.path.exists(test_image_path):
        print(f"Testing with image: {test_image_path}")
        image = cv2.imread(test_image_path)
        if image is not None:
            result = predict_bed_bugs(test_sensor_data, image)
            print(f"Detection result with image: {json.dumps(result, indent=2)}")
