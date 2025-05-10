"""
Configuration settings for the Bed Bug Detector Python Server
"""

import os

# Default configuration (can be overridden by environment variables)
CONFIG = {
    # Server settings
    "host": os.getenv("BEDBUG_HOST", "0.0.0.0"),
    "port": int(os.getenv("BEDBUG_PORT", 5000)),
    "debug": os.getenv("BEDBUG_DEBUG", "False").lower() == "true",
    "version": "1.0.0",
    
    # Path settings
    "data_dir": os.getenv("BEDBUG_DATA_DIR", "data"),
    "image_dir": os.getenv("BEDBUG_IMAGE_DIR", "data/images"),
    "detection_dir": os.getenv("BEDBUG_DETECTION_DIR", "data/detections"),
    
    # Detection settings
    "default_temp_threshold": float(os.getenv("BEDBUG_TEMP_THRESHOLD", "1.5")),
    "default_co2_threshold": int(os.getenv("BEDBUG_CO2_THRESHOLD", "200")),
    "detection_confidence_threshold": float(os.getenv("BEDBUG_CONFIDENCE_THRESHOLD", "70.0")),
    
    # ML model settings
    "ml_model_path": os.getenv("BEDBUG_ML_MODEL_PATH", "../ml/models/bedbug_detector_model.h5"),
    "use_gpu": os.getenv("BEDBUG_USE_GPU", "False").lower() == "true",
    
    # Security settings
    "require_api_key": os.getenv("BEDBUG_REQUIRE_API_KEY", "False").lower() == "true",
    "api_key": os.getenv("BEDBUG_API_KEY", ""),
    
    # Advanced settings
    "max_image_size": int(os.getenv("BEDBUG_MAX_IMAGE_SIZE", "1280")),  # Max width/height in pixels
    "max_history_length": int(os.getenv("BEDBUG_MAX_HISTORY", "1000")),  # Max detection history entries in memory
    "cleanup_interval_days": int(os.getenv("BEDBUG_CLEANUP_INTERVAL", "1")),  # Days between cleanup runs
    "image_retention_days": int(os.getenv("BEDBUG_IMAGE_RETENTION", "30")),  # Days to keep images
    "data_retention_days": int(os.getenv("BEDBUG_DATA_RETENTION", "90")),  # Days to keep detection data
}

# Ensure data directories exist
os.makedirs(CONFIG["image_dir"], exist_ok=True)
os.makedirs(CONFIG["detection_dir"], exist_ok=True)

# Detection algorithm weights
# These weights determine how much each sensor contributes to the final detection score
DETECTION_WEIGHTS = {
    "temperature": float(os.getenv("BEDBUG_WEIGHT_TEMP", "0.3")),  # 30% weight
    "co2": float(os.getenv("BEDBUG_WEIGHT_CO2", "0.3")),          # 30% weight
    "motion": float(os.getenv("BEDBUG_WEIGHT_MOTION", "0.15")),   # 15% weight
    "image": float(os.getenv("BEDBUG_WEIGHT_IMAGE", "0.25")),     # 25% weight
}

# Normalize weights to ensure they sum to 1.0
weight_sum = sum(DETECTION_WEIGHTS.values())
for key in DETECTION_WEIGHTS:
    DETECTION_WEIGHTS[key] /= weight_sum
