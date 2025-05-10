"""
Bed Bug Detector - Image Processing Utilities

This module provides functions for enhancing and analyzing images for bed bug detection.
"""

import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def enhance_image(image):
    """
    Enhance image for better bed bug detection
    
    Args:
        image (numpy.ndarray): Raw image data
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    try:
        # Check if image is valid
        if image is None or image.size == 0:
            logger.error("Invalid image provided to enhance_image")
            return None
        
        # Create a copy to avoid modifying the original
        enhanced = image.copy()
        
        # Convert to appropriate color format if needed
        if len(enhanced.shape) == 2:
            # Convert grayscale to BGR
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        elif len(enhanced.shape) == 3 and enhanced.shape[2] == 4:
            # Convert BGRA to BGR
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGRA2BGR)
        
        # Step 1: Denoise the image
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Step 2: Adjust contrast and brightness
        # Convert to LAB color space for better brightness/contrast adjustment
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge enhanced L channel with original A and B channels
        enhanced_lab = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 3: Sharpen the image
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
        
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image  # Return original image if enhancement fails


def detect_bed_bug_features(image):
    """
    Detect features that might indicate bed bugs
    
    Args:
        image (numpy.ndarray): Image data
        
    Returns:
        tuple: (enhanced_image, regions_of_interest)
    """
    try:
        # Enhance the image first
        enhanced = enhance_image(image)
        if enhanced is None:
            return None, []
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        
        # Create masks for different colors associated with bed bugs
        
        # Reddish-brown mask (main bed bug color)
        lower_brown = np.array([0, 50, 50])
        upper_brown = np.array([20, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Darker brown mask (for larger/older bed bugs)
        lower_dark = np.array([0, 50, 20])
        upper_dark = np.array([20, 255, 120])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(brown_mask, dark_mask)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and shape
        regions_of_interest = []
        min_area = 200
        max_area = 5000
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio (bed bugs typically have aspect ratio between 0.5 and 2.0)
                aspect_ratio = float(w) / h
                if 0.4 < aspect_ratio < 2.5:
                    # Extract region for further analysis
                    roi = enhanced[y:y+h, x:x+w]
                    
                    # Calculate texture features
                    texture_score = calculate_texture_score(roi)
                    
                    # Calculate shape features
                    shape_score = calculate_shape_score(contour)
                    
                    # Calculate color consistency
                    color_score = calculate_color_score(roi)
                    
                    # Calculate overall score (weighted average)
                    overall_score = (
                        texture_score * 0.3 +
                        shape_score * 0.4 +
                        color_score * 0.3
                    )
                    
                    if overall_score > 50:  # Threshold for potential bed bug
                        regions_of_interest.append({
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h),
                            "score": overall_score,
                            "area": area,
                            "aspect_ratio": aspect_ratio
                        })
        
        # Sort regions by score (highest first)
        regions_of_interest.sort(key=lambda x: x["score"], reverse=True)
        
        return enhanced, regions_of_interest
        
    except Exception as e:
        logger.error(f"Error detecting bed bug features: {str(e)}")
        return image, []


def calculate_texture_score(roi):
    """
    Calculate texture score for potential bed bug region
    
    Args:
        roi (numpy.ndarray): Region of interest
        
    Returns:
        float: Texture score (0-100)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate Haralick texture features
        # First, calculate GLCM (Gray-Level Co-occurrence Matrix)
        glcm = np.zeros((256, 256), dtype=np.uint32)
        h, w = gray.shape
        
        # Simple GLCM calculation with horizontal offset
        for i in range(h):
            for j in range(w-1):
                i_val = gray[i, j]
                j_val = gray[i, j+1]
                glcm[i_val, j_val] += 1
        
        # Normalize GLCM
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
        
        # Calculate contrast (high for bed bugs due to segmented body)
        contrast = 0
        for i in range(256):
            for j in range(256):
                contrast += glcm[i, j] * ((i-j) ** 2)
        
        # Calculate homogeneity (lower for bed bugs)
        homogeneity = 0
        for i in range(256):
            for j in range(256):
                homogeneity += glcm[i, j] / (1 + abs(i-j))
        
        # Calculate energy (lower for bed bugs)
        energy = np.sqrt(np.sum(glcm * glcm))
        
        # Scale and combine metrics
        contrast_score = min(contrast * 10, 100)  # Higher is better
        homogeneity_score = 100 - min(homogeneity * 100, 100)  # Lower is better
        energy_score = 100 - min(energy * 100, 100)  # Lower is better
        
        # Weighted combination
        texture_score = (
            contrast_score * 0.5 +
            homogeneity_score * 0.3 +
            energy_score * 0.2
        )
        
        return texture_score
        
    except Exception as e:
        logger.error(f"Error calculating texture score: {str(e)}")
        return 50  # Return neutral score on error


def calculate_shape_score(contour):
    """
    Calculate shape score for potential bed bug contour
    
    Args:
        contour (numpy.ndarray): Contour points
        
    Returns:
        float: Shape score (0-100)
    """
    try:
        # Calculate shape metrics
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity (4π × area / perimeter²)
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Bed bugs have circularity around 0.6-0.8 (not perfectly circular)
        circularity_score = 100 - min(abs(circularity - 0.7) * 200, 100)
        
        # Calculate convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = 0
        if hull_area > 0:
            convexity = float(area) / hull_area
        
        # Bed bugs are mostly convex but have some concavities
        convexity_score = 100 - min(abs(convexity - 0.9) * 200, 100)
        
        # Calculate moments and Hu moments for shape analysis
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # First Hu moment (relates to elongation)
        hu1_score = 100 - min(abs(hu_moments[0] - 0.2) * 500, 100)
        
        # Combine scores
        shape_score = (
            circularity_score * 0.4 +
            convexity_score * 0.4 +
            hu1_score * 0.2
        )
        
        return shape_score
        
    except Exception as e:
        logger.error(f"Error calculating shape score: {str(e)}")
        return 50  # Return neutral score on error


def calculate_color_score(roi):
    """
    Calculate color score for potential bed bug region
    
    Args:
        roi (numpy.ndarray): Region of interest
        
    Returns:
        float: Color score (0-100)
    """
    try:
        # Convert to HSV color space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate average HSV values
        h_avg = np.mean(hsv[:,:,0])
        s_avg = np.mean(hsv[:,:,1])
        v_avg = np.mean(hsv[:,:,2])
        
        # Bed bugs are typically reddish-brown
        # Hue: 0-20 (reddish brown in OpenCV's HSV)
        # Saturation: medium-high (100-255)
        # Value: medium (80-200)
        
        # Score based on how close to ideal bed bug color
        h_score = 100 - min(abs(h_avg - 10) * 5, 100)  # Ideal around hue = 10
        s_score = min(max(s_avg - 50, 0) / 1.5, 100)  # Higher saturation better
        v_score = 100 - min(abs(v_avg - 140) / 1.4, 100)  # Ideal around value = 140
        
        # Calculate color consistency (bed bugs have fairly consistent coloration)
        h_std = np.std(hsv[:,:,0])
        s_std = np.std(hsv[:,:,1])
        v_std = np.std(hsv[:,:,2])
        
        # Lower standard deviation = more consistent color
        consistency_score = 100 - min((h_std + s_std + v_std) / 3, 100)
        
        # Combine scores
        color_score = (
            h_score * 0.3 +
            s_score * 0.2 +
            v_score * 0.2 +
            consistency_score * 0.3
        )
        
        return color_score
        
    except Exception as e:
        logger.error(f"Error calculating color score: {str(e)}")
        return 50  # Return neutral score on error


def draw_detection_overlay(image, regions):
    """
    Draw bounding boxes and information around detected bed bug regions
    
    Args:
        image (numpy.ndarray): Original or enhanced image
        regions (list): List of region dictionaries with x, y, width, height, and score
        
    Returns:
        numpy.ndarray: Image with overlay
    """
    try:
        # Make a copy to avoid modifying the original
        overlay = image.copy()
        
        # Draw regions
        for region in regions:
            x = region["x"]
            y = region["y"]
            w = region["width"]
            h = region["height"]
            score = region["score"]
            
            # Color based on confidence (green to red)
            # High confidence: more red
            # Low confidence: more green
            confidence = min(score / 100, 1.0)
            blue = 0
            green = int(255 * (1 - confidence))
            red = int(255 * confidence)
            color = (blue, green, red)
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence text
            text = f"{score:.1f}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
            
            # Ensure text background fits within image
            text_x = x
            text_y = y - 5 if y > 20 else y + h + 20
            
            # Draw text background
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                overlay,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return overlay
        
    except Exception as e:
        logger.error(f"Error drawing detection overlay: {str(e)}")
        return image


if __name__ == "__main__":
    # Example usage and testing
    import os
    import matplotlib.pyplot as plt
    
    print("Testing image processing functions")
    
    # Test image path
    test_image_path = "../ml/dataset/test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        exit(1)
    
    # Load test image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Failed to load image from {test_image_path}")
        exit(1)
    
    # Test enhancement
    enhanced_image = enhance_image(image)
    print("Image enhancement complete")
    
    # Test feature detection
    enhanced, regions = detect_bed_bug_features(image)
    print(f"Detected {len(regions)} potential bed bug regions")
    
    # Draw detection overlay
    overlay = draw_detection_overlay(enhanced, regions)
    print("Drew detection overlay")
    
    # Display results using matplotlib
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title("Enhanced Image")
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 3)
    plt.title("Detection Overlay")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    
    plt.tight_layout()
    plt.show()
