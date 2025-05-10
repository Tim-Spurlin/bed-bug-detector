"""
Bed Bug Detector - Machine Learning Training Script

This script trains a machine learning model to detect bed bugs based on:
1. Sensor data (temperature, humidity, CO2, motion)
2. Images (optional)

The trained model is saved and can be used by the prediction module.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import json
import cv2
import random
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Conditionally import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_processing import enhance_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("ml_training.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)  # Input size for image model
SENSOR_FEATURES = ["temperature", "humidity", "co2", "motion"]
MODEL_VERSION = "1.0.0"


def load_sensor_data(data_file):
    """
    Load sensor training data from CSV file
    
    Args:
        data_file (str): Path to CSV file with sensor data
        
    Returns:
        pandas.DataFrame: Loaded data or None if failed
    """
    try:
        # Check if file exists
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return None
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Verify required columns
        required_columns = SENSOR_FEATURES + ["detected"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Convert boolean columns to int
        if "motion" in df.columns and df["motion"].dtype == bool:
            df["motion"] = df["motion"].astype(int)
        
        if "detected" in df.columns and df["detected"].dtype == bool:
            df["detected"] = df["detected"].astype(int)
        
        logger.info(f"Loaded {len(df)} sensor data records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading sensor data: {str(e)}")
        return None


def load_image_data(image_dir, labels_file):
    """
    Load image training data
    
    Args:
        image_dir (str): Directory containing image files
        labels_file (str): JSON file mapping image filenames to labels
        
    Returns:
        tuple: (images, labels) or (None, None) if failed
    """
    try:
        # Check if directory exists
        if not os.path.exists(image_dir):
            logger.error(f"Image directory not found: {image_dir}")
            return None, None
        
        # Check if labels file exists
        if not os.path.exists(labels_file):
            logger.error(f"Labels file not found: {labels_file}")
            return None, None
        
        # Load labels
        with open(labels_file, 'r') as f:
            labels_dict = json.load(f)
        
        # Get all image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Filter to only images with labels
        labeled_images = [f for f in image_files if f in labels_dict]
        
        if not labeled_images:
            logger.error("No labeled images found")
            return None, None
        
        # Load images and labels
        images = []
        labels = []
        
        for img_file in labeled_images:
            # Load and preprocess image
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                # Resize to standard size
                img = cv2.resize(img, IMAGE_SIZE)
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize to 0-1
                img = img.astype(np.float32) / 255.0
                
                # Get label
                label = 1 if labels_dict[img_file] else 0
                
                images.append(img)
                labels.append(label)
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        logger.info(f"Loaded {len(images)} labeled images")
        return images, labels
        
    except Exception as e:
        logger.error(f"Error loading image data: {str(e)}")
        return None, None


def create_sensor_model():
    """
    Create a neural network model for sensor data
    
    Returns:
        tensorflow.keras.Model: Sensor model
    """
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    try:
        # Create a sequential model
        model = Sequential([
            Input(shape=(len(SENSOR_FEATURES),)),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating sensor model: {str(e)}")
        return None


def create_image_model():
    """
    Create a convolutional neural network model for image data
    
    Returns:
        tensorflow.keras.Model: Image model
    """
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    try:
        # Create a sequential CNN model
        model = Sequential([
            Input(shape=(*IMAGE_SIZE, 3)),
            
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating image model: {str(e)}")
        return None


def create_combined_model():
    """
    Create a combined model using both sensor and image data
    
    Returns:
        tensorflow.keras.Model: Combined model
    """
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    try:
        # Sensor input branch
        sensor_input = Input(shape=(len(SENSOR_FEATURES),), name='sensor_input')
        x1 = Dense(64, activation='relu')(sensor_input)
        x1 = Dropout(0.3)(x1)
        x1 = Dense(32, activation='relu')(x1)
        x1 = Dropout(0.2)(x1)
        x1 = Dense(16, activation='relu')(x1)
        
        # Image input branch
        image_input = Input(shape=(*IMAGE_SIZE, 3), name='image_input')
        
        x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x2)
        x2 = MaxPooling2D((2, 2))(x2)
        x2 = Dropout(0.25)(x2)
        
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
        x2 = MaxPooling2D((2, 2))(x2)
        x2 = Dropout(0.25)(x2)
        
        x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
        x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
        x2 = MaxPooling2D((2, 2))(x2)
        x2 = Dropout(0.25)(x2)
        
        x2 = Flatten()(x2)
        x2 = Dense(512, activation='relu')(x2)
        x2 = Dropout(0.5)(x2)
        
        # Combine the branches
        combined = Concatenate()([x1, x2])
        
        # Output layer
        output = Dense(1, activation='sigmoid')(combined)
        
        # Create and compile model
        model = Model(inputs=[sensor_input, image_input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        logger.error(f"Error creating combined model: {str(e)}")
        return None


def train_sensor_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train a model using sensor data
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation features
        y_val (numpy.ndarray): Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tensorflow.keras.Model: Trained model or None if failed
    """
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    try:
        # Create model
        model = create_sensor_model()
        
        if model is None:
            return None
        
        # Create callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                'models/sensor_model_checkpoint.h5', 
                save_best_only=True, 
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        logger.info("Training sensor model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating sensor model...")
        loss, accuracy = model.evaluate(X_val, y_val)
        logger.info(f"Validation accuracy: {accuracy:.4f}")
        
        # Plot training history
        plot_training_history(history, "sensor_model_history.png")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training sensor model: {str(e)}")
        return None


def train_image_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train a model using image data
    
    Args:
        X_train (numpy.ndarray): Training images
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation images
        y_val (numpy.ndarray): Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tensorflow.keras.Model: Trained model or None if failed
    """
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    try:
        # Create model
        model = create_image_model()
        
        if model is None:
            return None
        
        # Create callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                'models/image_model_checkpoint.h5', 
                save_best_only=True, 
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        logger.info("Training image model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating image model...")
        loss, accuracy = model.evaluate(X_val, y_val)
        logger.info(f"Validation accuracy: {accuracy:.4f}")
        
        # Plot training history
        plot_training_history(history, "image_model_history.png")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training image model: {str(e)}")
        return None


def train_combined_model(sensor_X_train, image_X_train, y_train, 
                          sensor_X_val, image_X_val, y_val, 
                          epochs=50, batch_size=32):
    """
    Train a combined model using both sensor and image data
    
    Args:
        sensor_X_train (numpy.ndarray): Training sensor features
        image_X_train (numpy.ndarray): Training images
        y_train (numpy.ndarray): Training labels
        sensor_X_val (numpy.ndarray): Validation sensor features
        image_X_val (numpy.ndarray): Validation images
        y_val (numpy.ndarray): Validation labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tensorflow.keras.Model: Trained model or None if failed
    """
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available")
        return None
    
    try:
        # Create model
        model = create_combined_model()
        
        if model is None:
            return None
        
        # Create callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-6),
            ModelCheckpoint(
                'models/combined_model_checkpoint.h5', 
                save_best_only=True, 
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        logger.info("Training combined model...")
        history = model.fit(
            [sensor_X_train, image_X_train], y_train,
            validation_data=([sensor_X_val, image_X_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating combined model...")
        loss, accuracy = model.evaluate([sensor_X_val, image_X_val], y_val)
        logger.info(f"Validation accuracy: {accuracy:.4f}")
        
        # Plot training history
        plot_training_history(history, "combined_model_history.png")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training combined model: {str(e)}")
        return None


def plot_training_history(history, filename):
    """
    Plot training history and save to file
    
    Args:
        history (tensorflow.keras.callbacks.History): Training history
        filename (str): Output filename
    """
    try:
        # Create output directory if needed
        os.makedirs("charts", exist_ok=True)
        
        # Create figure with subplots
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join("charts", filename))
        plt.close()
        
        logger.info(f"Training history plotted and saved to charts/{filename}")
        
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")


def plot_confusion_matrix(y_true, y_pred, filename):
    """
    Plot confusion matrix and save to file
    
    Args:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels
        filename (str): Output filename
    """
    try:
        # Create output directory if needed
        os.makedirs("charts", exist_ok=True)
        
        # Convert predictions to binary
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Bed Bugs', 'Bed Bugs'],
                   yticklabels=['No Bed Bugs', 'Bed Bugs'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join("charts", filename))
        plt.close()
        
        logger.info(f"Confusion matrix plotted and saved to charts/{filename}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_binary, 
                                   target_names=['No Bed Bugs', 'Bed Bugs']))
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")


def save_model(model, model_type):
    """
    Save the trained model to file
    
    Args:
        model (tensorflow.keras.Model): Trained model
        model_type (str): Type of model ('sensor', 'image', or 'combined')
        
    Returns:
        str: Path to saved model or None if failed
    """
    if not TF_AVAILABLE or model is None:
        logger.error("Cannot save model: TensorFlow not available or model is None")
        return None
    
    try:
        # Create output directory if needed
        os.makedirs("models", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"models/bedbug_{model_type}_model_{timestamp}.h5"
        
        # Save model
        model.save(filename)
        
        # Also save as latest version
        latest_filename = f"models/bedbug_{model_type}_model.h5"
        model.save(latest_filename)
        
        logger.info(f"Model saved to {filename} and {latest_filename}")
        
        # Save model info
        model_info = {
    
