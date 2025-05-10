"""
Bed Bug Detector - Data Analysis Utilities

This module provides functions for analyzing detection data,
identifying patterns, and generating insights from detection history.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_detection_history(history_data):
    """
    Analyze detection history to identify patterns and trends
    
    Args:
        history_data (list): List of detection result dictionaries
        
    Returns:
        dict: Analysis results
    """
    if not history_data:
        return {
            "detection_rate": 0,
            "average_confidence": 0,
            "detection_trend": "neutral",
            "high_risk_areas": [],
            "peak_activity_time": None,
            "recommendations": [
                "No detection history available for analysis. Perform some scans to gather data."
            ]
        }
    
    try:
        # Convert to pandas DataFrame for easier analysis
        records = []
        for entry in history_data:
            # Extract basic info
            timestamp = entry.get("timestamp")
            device_id = entry.get("device_id")
            
            # Extract detection info
            detection = entry.get("detection", {})
            detected = detection.get("detected", False)
            confidence = detection.get("confidence", 0)
            
            # Extract sensor data
            sensor_data = entry.get("sensor_data", {})
            temperature = sensor_data.get("temperature", 0)
            humidity = sensor_data.get("humidity", 0)
            co2 = sensor_data.get("co2", 0)
            motion = sensor_data.get("motion", False)
            
            # Create record
            record = {
                "timestamp": timestamp,
                "device_id": device_id,
                "detected": detected,
                "confidence": confidence,
                "temperature": temperature,
                "humidity": humidity,
                "co2": co2,
                "motion": motion,
                "location": entry.get("location", "Unknown")
            }
            
            records.append(record)
        
        if not records:
            return {
                "detection_rate": 0,
                "average_confidence": 0,
                "detection_trend": "neutral",
                "high_risk_areas": [],
                "peak_activity_time": None,
                "recommendations": [
                    "No valid detection records found. Ensure proper data format."
                ]
            }
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"])
        
        # Calculate basic statistics
        total_scans = len(df)
        positive_detections = df["detected"].sum()
        detection_rate = positive_detections / total_scans if total_scans > 0 else 0
        average_confidence = df["confidence"].mean()
        
        # Calculate time-based statistics
        df["hour"] = df["datetime"].dt.hour
        hourly_detections = df[df["detected"]].groupby("hour").size()
        peak_hour = hourly_detections.idxmax() if not hourly_detections.empty else None
        
        # Format peak time
        peak_time = f"{peak_hour:02d}:00 - {(peak_hour+1)%24:02d}:00" if peak_hour is not None else None
        
        # Calculate trend over time (is it getting better or worse?)
        if len(df) >= 5:
            # Split into equally sized chunks
            chunk_size = max(len(df) // 5, 1)
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            
            # Calculate detection rate for each chunk
            chunk_rates = [chunk["detected"].mean() for chunk in chunks]
            
            # Linear regression to determine trend
            x = np.arange(len(chunk_rates))
            if len(x) > 1:
                slope = np.polyfit(x, chunk_rates, 1)[0]
                
                if slope > 0.05:
                    trend = "increasing"
                elif slope < -0.05:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "neutral"
        else:
            trend = "neutral"
        
        # Identify high-risk areas
        location_data = {}
        for location in df["location"].unique():
            location_df = df[df["location"] == location]
            location_data[location] = {
                "total_scans": len(location_df),
                "positive_detections": location_df["detected"].sum(),
                "detection_rate": location_df["detected"].mean(),
                "average_confidence": location_df["confidence"].mean()
            }
        
        # Sort locations by detection rate
        high_risk_locations = sorted(
            location_data.items(),
            key=lambda x: x[1]["detection_rate"],
            reverse=True
        )
        
        # Keep only locations with at least one detection
        high_risk_locations = [loc for loc in high_risk_locations 
                             if loc[1]["positive_detections"] > 0]
        
        # Limit to top 3
        high_risk_locations = high_risk_locations[:3]
        
        # Format high risk areas
        high_risk_areas = [
            {
                "location": loc[0],
                "detection_rate": loc[1]["detection_rate"],
                "positive_detections": int(loc[1]["positive_detections"]),
                "total_scans": loc[1]["total_scans"],
                "average_confidence": loc[1]["average_confidence"]
            }
            for loc in high_risk_locations
        ]
        
        # Generate recommendations based on analysis
        recommendations = generate_recommendations(
            detection_rate, 
            trend, 
            high_risk_areas, 
            peak_time
        )
        
        # Additional indicators: correlation between sensors and detections
        sensor_correlations = {}
        for sensor in ["temperature", "humidity", "co2", "motion"]:
            if df[sensor].std() > 0:  # Only calculate if there's variation
                corr = df[sensor].corr(df["detected"])
                sensor_correlations[sensor] = corr
        
        # Return analysis results
        return {
            "total_scans": int(total_scans),
            "positive_detections": int(positive_detections),
            "detection_rate": float(detection_rate),
            "average_confidence": float(average_confidence),
            "detection_trend": trend,
            "high_risk_areas": high_risk_areas,
            "peak_activity_time": peak_time,
            "sensor_correlations": sensor_correlations,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error analyzing detection history: {str(e)}")
        return {
            "error": str(e),
            "recommendations": [
                "An error occurred during analysis. Check log files for details."
            ]
        }


def generate_recommendations(detection_rate, trend, high_risk_areas, peak_time):
    """
    Generate recommendations based on analysis results
    
    Args:
        detection_rate (float): Overall detection rate
        trend (str): Detection trend (increasing, decreasing, stable)
        high_risk_areas (list): List of high-risk areas
        peak_time (str): Peak activity time
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    # Recommendations based on detection rate
    if detection_rate > 0.5:
        recommendations.append(
            "HIGH ALERT: Significant bed bug activity detected. Consider professional treatment."
        )
    elif detection_rate > 0.2:
        recommendations.append(
            "Moderate bed bug activity detected. Continue monitoring and consider treatment options."
        )
    elif detection_rate > 0:
        recommendations.append(
            "Low-level bed bug activity detected. Increase monitoring frequency."
        )
    else:
        recommendations.append(
            "No bed bug activity detected. Continue routine monitoring."
        )
    
    # Recommendations based on trend
    if trend == "increasing":
        recommendations.append(
            "Warning: Bed bug detections are increasing over time. Take action soon."
        )
    elif trend == "decreasing":
        recommendations.append(
            "Good news: Bed bug detections are decreasing. Current measures may be working."
        )
    
    # Recommendations based on high-risk areas
    if high_risk_areas:
        area_names = [area["location"] for area in high_risk_areas]
        recommendations.append(
            f"Focus attention on high-risk areas: {', '.join(area_names)}"
        )
    
    # Recommendations based on peak time
    if peak_time:
        recommendations.append(
            f"Schedule additional checks during peak activity time: {peak_time}"
        )
    
    # General recommendations
    recommendations.append(
        "Regularly wash and heat-dry bedding, clothing, and other items."
    )
    
    recommendations.append(
        "Check luggage and clothing after traveling."
    )
    
    return recommendations


def analyze_sensor_correlations(history_data):
    """
    Analyze correlations between sensor readings and detection results
    
    Args:
        history_data (list): List of detection result dictionaries
        
    Returns:
        dict: Correlation analysis
    """
    try:
        # Extract sensor data and detection results
        data = []
        for entry in history_data:
            sensor_data = entry.get("sensor_data", {})
            detection = entry.get("detection", {})
            
            record = {
                "temperature": sensor_data.get("temperature", 0),
                "humidity": sensor_data.get("humidity", 0),
                "co2": sensor_data.get("co2", 0),
                "motion": 1 if sensor_data.get("motion", False) else 0,
                "detected": 1 if detection.get("detected", False) else 0,
                "confidence": detection.get("confidence", 0)
            }
            
            data.append(record)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Calculate correlations
        correlation_matrix = df.corr()
        
        # Extract correlations with detection
        detection_correlations = correlation_matrix["detected"].drop("detected")
        confidence_correlations = correlation_matrix["confidence"].drop("confidence")
        
        # Convert to Python dict
        result = {
            "detection_correlations": detection_correlations.to_dict(),
            "confidence_correlations": confidence_correlations.to_dict()
        }
        
        # Add interpretation
        result["interpretation"] = {}
        
        for sensor, corr in detection_correlations.items():
            if abs(corr) > 0.7:
                strength = "strong"
            elif abs(corr) > 0.4:
                strength = "moderate"
            elif abs(corr) > 0.2:
                strength = "weak"
            else:
                strength = "very weak"
                
            direction = "positive" if corr > 0 else "negative"
            
            result["interpretation"][sensor] = f"{strength} {direction} correlation"
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing sensor correlations: {str(e)}")
        return {"error": str(e)}


def generate_detection_report(history_data, file_path=None):
    """
    Generate a comprehensive detection report
    
    Args:
        history_data (list): List of detection result dictionaries
        file_path (str, optional): Path to save the report
        
    Returns:
        dict: Report data
    """
    try:
        # Perform main analysis
        analysis = analyze_detection_history(history_data)
        
        # Perform correlation analysis
        correlations = analyze_sensor_correlations(history_data)
        
        # Combine results
        report = {
            "timestamp": datetime.now().isoformat(),
            "period": {
                "start": min([entry.get("timestamp") for entry in history_data]) if history_data else None,
                "end": max([entry.get("timestamp") for entry in history_data]) if history_data else None
            },
            "summary": analysis,
            "correlations": correlations,
            "data_points": len(history_data)
        }
        
        # Save to file if path provided
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Report saved to {file_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating detection report: {str(e)}")
        return {"error": str(e)}


def calculate_infestation_probability(detection_history, recent_window_days=7):
    """
    Calculate probability of bed bug infestation based on detection history
    
    Args:
        detection_history (list): List of detection result dictionaries
        recent_window_days (int): Number of days to consider for recent activity
        
    Returns:
        dict: Infestation probability and factors
    """
    try:
        if not detection_history:
            return {
                "overall_probability": 0,
                "confidence": "low",
                "factors": {
                    "detection_rate": 0,
                    "recent_activity": 0,
                    "consistency": 0
                },
                "interpretation": "No detection data available"
            }
        
        # Extract detection results
        detections = []
        for entry in detection_history:
            timestamp = entry.get("timestamp")
            detected = entry.get("detection", {}).get("detected", False)
            confidence = entry.get("detection", {}).get("confidence", 0)
            
            detections.append({
                "timestamp": timestamp,
                "datetime": datetime.fromisoformat(timestamp.replace('Z', '+00:00')),
                "detected": detected,
                "confidence": confidence
            })
        
        # Sort by time
        detections.sort(key=lambda x: x["datetime"])
        
        # Calculate detection rate
        total_scans = len(detections)
        positive_detections = sum(1 for d in detections if d["detected"])
        detection_rate = positive_detections / total_scans if total_scans > 0 else 0
        
        # Calculate recent activity (last 7 days)
        now = datetime.now()
        cutoff = now - timedelta(days=recent_window_days)
        
        recent_detections = [d for d in detections if d["datetime"] > cutoff]
        recent_positive = sum(1 for d in recent_detections if d["detected"])
        recent_activity = recent_positive / len(recent_detections) if recent_detections else 0
        
        # Calculate consistency (variation in detection rate over time)
        if len(detections) >= 5:
            # Split into equally sized chunks
            chunk_size = max(len(detections) // 5, 1)
            chunks = [detections[i:i+chunk_size] for i in range(0, len(detections), chunk_size)]
            
            # Calculate detection rate for each chunk
            chunk_rates = [
                sum(1 for d in chunk if d["detected"]) / len(chunk)
                for chunk in chunks
            ]
            
            # Calculate standard deviation of rates
            std_dev = np.std(chunk_rates)
            
            # Higher consistency = lower std_dev
            consistency = 1 - min(std_dev * 2, 1)  # Scale to 0-1
        else:
            consistency = 0.5  # Neutral if not enough data
        
        # Combine factors with weights
        factor_weights = {
            "detection_rate": 0.4,
            "recent_activity": 0.4,
            "consistency": 0.2
        }
        
        factors = {
            "detection_rate": detection_rate,
            "recent_activity": recent_activity,
            "consistency": consistency
        }
        
        # Calculate overall probability
        overall_probability = sum(
            factors[factor] * weight
            for factor, weight in factor_weights.items()
        )
        
        # Scale to 0-100%
        overall_probability = min(overall_probability * 100, 100)
        
        # Determine confidence in the estimate
        if total_scans < 3:
            confidence = "very low"
        elif total_scans < 10:
            confidence = "low"
        elif total_scans < 20:
            confidence = "moderate"
        else:
            confidence = "high"
        
        # Generate interpretation
        if overall_probability < 10:
            interpretation = "Very low probability of infestation"
        elif overall_probability < 30:
            interpretation = "Low probability of infestation"
        elif overall_probability < 60:
            interpretation = "Moderate probability of infestation"
        elif overall_probability < 80:
            interpretation = "High probability of infestation"
        else:
            interpretation = "Very high probability of infestation"
        
        return {
            "overall_probability": overall_probability,
            "confidence": confidence,
            "factors": factors,
            "interpretation": interpretation
        }
        
    except Exception as e:
        logger.error(f"Error calculating infestation probability: {str(e)}")
        return {
            "overall_probability": 0,
            "confidence": "error",
            "factors": {},
            "interpretation": f"Error: {str(e)}"
        }


if __name__ == "__main__":
    # Example usage with dummy data
    history = [
        {
            "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "device_id": "test123",
            "sensor_data": {
                "temperature": 1.8,
                "humidity": 60,
                "co2": 300,
                "motion": True
            },
            "detection": {
                "detected": True,
                "confidence": 85
            },
            "location": "Bedroom"
        },
        {
            "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
            "device_id": "test123",
            "sensor_data": {
                "temperature": 1.2,
                "humidity": 55,
                "co2": 250,
                "motion": False
            },
            "detection": {
                "detected": True,
                "confidence": 65
            },
            "location": "Bedroom"
        },
        {
            "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
            "device_id": "test123",
            "sensor_data": {
                "temperature": 0.5,
                "humidity": 50,
                "co2": 180,
                "motion": False
            },
            "detection": {
                "detected": False,
                "confidence": 30
            },
            "location": "Living Room"
        }
    ]
    
    print("Testing data analysis functions")
    
    # Test main analysis
    analysis_result = analyze_detection_history(history)
    print("Analysis result:")
    print(json.dumps(analysis_result, indent=2))
    
    # Test correlation analysis
    correlation_result = analyze_sensor_correlations(history)
    print("\nCorrelation analysis:")
    print(json.dumps(correlation_result, indent=2))
    
    # Test infestation probability
    probability_result = calculate_infestation_probability(history)
    print("\nInfestation probability:")
    print(json.dumps(probability_result, indent=2))
    
    # Test report generation
    report = generate_detection_report(history, "example_report.json")
    print("\nReport generated and saved to example_report.json")
