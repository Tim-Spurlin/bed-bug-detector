"""
Bed Bug Detector - Desktop Application

This is a desktop alternative to the mobile app, providing a user interface
for controlling the bed bug detector and viewing results.
"""

import sys
import os
import logging
import json
import threading
import time
from datetime import datetime, timedelta
import requests
import numpy as np
import cv2
from PIL import Image, ImageTk

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.image_processing import enhance_image, draw_detection_overlay
from utils.data_analysis import analyze_detection_history, calculate_infestation_probability

# GUI library
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("desktop_app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Default server URL
DEFAULT_SERVER = "http://localhost:5000"

class BedBugDetectorApp(tk.Tk):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Bed Bug Detector - Desktop Client")
        self.geometry("1200x800")
        self.minsize(800, 600)
        
        # Application state
        self.detector_ip = tk.StringVar(value="192.168.4.1")
        self.server_url = tk.StringVar(value=DEFAULT_SERVER)
        self.device_id = tk.StringVar(value="")
        self.connection_status = tk.StringVar(value="Disconnected")
        self.last_detection_result = None
        self.detection_history = []
        self.camera_image = None
        
        # Create UI
        self.create_menu()
        self.create_layout()
        
        # Check server connection
        self.check_server_connection()
    
    def create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Connect", command=self.connect_to_detector)
        file_menu.add_command(label="Disconnect", command=self.disconnect_from_detector)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Detection menu
        detection_menu = tk.Menu(menubar, tearoff=0)
        detection_menu.add_command(label="Start Detection", command=self.start_detection)
        detection_menu.add_command(label="Calibrate", command=self.calibrate_detector)
        detection_menu.add_command(label="Capture Image", command=self.capture_image)
        menubar.add_cascade(label="Detection", menu=detection_menu)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Detector Settings", command=self.show_detector_settings)
        settings_menu.add_command(label="Server Settings", command=self.show_server_settings)
        settings_menu.add_command(label="App Settings", command=self.show_app_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
    
    def create_layout(self):
        """Create main application layout"""
        # Main frame with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        detection_tab = ttk.Frame(notebook)
        history_tab = ttk.Frame(notebook)
        analysis_tab = ttk.Frame(notebook)
        settings_tab = ttk.Frame(notebook)
        
        notebook.add(detection_tab, text="Detection")
        notebook.add(history_tab, text="History")
        notebook.add(analysis_tab, text="Analysis")
        notebook.add(settings_tab, text="Settings")
        
        # Set up each tab
        self.setup_detection_tab(detection_tab)
        self.setup_history_tab(history_tab)
        self.setup_analysis_tab(analysis_tab)
        self.setup_settings_tab(settings_tab)
        
        # Status bar at bottom
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status indicators
        ttk.Label(status_frame, text="Connection:").pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.connection_status).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(status_frame, text="Device ID:").pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.device_id).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(status_frame, text="Server:").pack(side=tk.LEFT, padx=5)
        ttk.Label(status_frame, textvariable=self.server_url).pack(side=tk.LEFT, padx=5)
    
    def setup_detection_tab(self, parent):
        """Set up the detection tab with camera view and sensor readings"""
        # Split into left (camera) and right (controls) sections
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for camera/visualization
        left_frame = ttk.LabelFrame(paned, text="Camera View")
        paned.add(left_frame, weight=3)
        
        # Camera canvas
        self.camera_canvas = tk.Canvas(left_frame, bg="black", highlightthickness=0)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right frame for controls and readings
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        # Connection frame
        connection_frame = ttk.LabelFrame(right_frame, text="Connection")
        connection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(connection_frame, text="Detector IP:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(connection_frame, textvariable=self.detector_ip).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        btn_frame = ttk.Frame(connection_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        ttk.Button(btn_frame, text="Connect", command=self.connect_to_detector).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Disconnect", command=self.disconnect_from_detector).pack(side=tk.LEFT, padx=5)
        
        # Sensor readings frame
        readings_frame = ttk.LabelFrame(right_frame, text="Sensor Readings")
        readings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Sensor value labels
        self.temp_value = tk.StringVar(value="--")
        self.humidity_value = tk.StringVar(value="--")
        self.co2_value = tk.StringVar(value="--")
        self.motion_value = tk.StringVar(value="--")
        
        ttk.Label(readings_frame, text="Temperature:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(readings_frame, textvariable=self.temp_value).grid(row=0, column=1, padx=5, pady=2, sticky=tk.E)
        ttk.Label(readings_frame, text="°C").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(readings_frame, text="Humidity:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(readings_frame, textvariable=self.humidity_value).grid(row=1, column=1, padx=5, pady=2, sticky=tk.E)
        ttk.Label(readings_frame, text="%").grid(row=1, column=2, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(readings_frame, text="CO2:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(readings_frame, textvariable=self.co2_value).grid(row=2, column=1, padx=5, pady=2, sticky=tk.E)
        ttk.Label(readings_frame, text="ppm").grid(row=2, column=2, padx=5, pady=2, sticky=tk.W)
        
        ttk.Label(readings_frame, text="Motion:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(readings_frame, textvariable=self.motion_value).grid(row=3, column=1, padx=5, pady=2, sticky=tk.E)
        
        # Action buttons
        actions_frame = ttk.LabelFrame(right_frame, text="Actions")
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(actions_frame, text="Start Detection", command=self.start_detection).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(actions_frame, text="Calibrate", command=self.calibrate_detector).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(actions_frame, text="Capture Image", command=self.capture_image).pack(fill=tk.X, padx=5, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(right_frame, text="Detection Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=10)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.config(state=tk.DISABLED)
    
    def setup_history_tab(self, parent):
        """Set up the history tab with detection records"""
        # Controls at top
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(controls_frame, text="Filter:").pack(side=tk.LEFT, padx=5)
        self.filter_var = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(controls_frame, textvariable=self.filter_var, 
                                    values=["All", "Positive Only", "Last 24 Hours", "Last Week"])
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind("<<ComboboxSelected>>", self.filter_history)
        
        ttk.Button(controls_frame, text="Refresh", command=self.refresh_history).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export", command=self.export_history).pack(side=tk.LEFT, padx=5)
        
        # Treeview for history
        columns = ("timestamp", "location", "detected", "confidence")
        self.history_tree = ttk.Treeview(parent, columns=columns, show="headings")
        
        # Define headings
        self.history_tree.heading("timestamp", text="Date & Time")
        self.history_tree.heading("location", text="Location")
        self.history_tree.heading("detected", text="Detected")
        self.history_tree.heading("confidence", text="Confidence")
        
        # Define columns
        self.history_tree.column("timestamp", width=180)
        self.history_tree.column("location", width=150)
        self.history_tree.column("detected", width=100)
        self.history_tree.column("confidence", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=5)
        
        # Bind selection event
        self.history_tree.bind("<<TreeviewSelect>>", self.on_history_select)
    
    def setup_analysis_tab(self, parent):
        """Set up the analysis tab with charts and statistics"""
        # Main container
        analysis_frame = ttk.Frame(parent)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Split into left (statistics) and right (charts) sections
        paned = ttk.PanedWindow(analysis_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for statistics
        self.stats_frame = ttk.LabelFrame(paned, text="Detection Statistics")
        paned.add(self.stats_frame, weight=1)
        
        # Stats content will be generated dynamically
        self.generate_statistics_view()
        
        # Right frame for charts and recommendations
        charts_frame = ttk.Frame(paned)
        paned.add(charts_frame, weight=2)
        
        # Top section for charts
        self.charts_container = ttk.LabelFrame(charts_frame, text="Analysis Charts")
        self.charts_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add placeholder for charts (will be generated from history data)
        placeholder = ttk.Label(self.charts_container, text="No data available for charts.\nPerform some detections to generate analysis.")
        placeholder.pack(expand=True, pady=50)
        
        # Bottom section for recommendations
        recommendations_frame = ttk.LabelFrame(charts_frame, text="Recommendations")
        recommendations_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.recommendations_text = scrolledtext.ScrolledText(recommendations_frame, wrap=tk.WORD, height=6)
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.recommendations_text.config(state=tk.DISABLED)
    
    def setup_settings_tab(self, parent):
        """Set up the settings tab with configuration options"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create settings tabs
        detector_tab = ttk.Frame(notebook)
        server_tab = ttk.Frame(notebook)
        app_tab = ttk.Frame(notebook)
        
        notebook.add(detector_tab, text="Detector")
        notebook.add(server_tab, text="Server")
        notebook.add(app_tab, text="Application")
        
        # Detector settings
        self.setup_detector_settings(detector_tab)
        
        # Server settings
        self.setup_server_settings(server_tab)
        
        # App settings
        self.setup_app_settings(app_tab)
    
    def setup_detector_settings(self, parent):
        """Set up detector settings panel"""
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Detection thresholds
        thresholds_frame = ttk.LabelFrame(settings_frame, text="Detection Thresholds")
        thresholds_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.temp_threshold = tk.DoubleVar(value=1.5)
        self.co2_threshold = tk.IntVar(value=200)
        self.confidence_threshold = tk.DoubleVar(value=70.0)
        
        ttk.Label(thresholds_frame, text="Temperature Threshold (°C):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(thresholds_frame, from_=0.1, to=5.0, increment=0.1, textvariable=self.temp_threshold).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(thresholds_frame, text="CO2 Threshold (ppm):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(thresholds_frame, from_=50, to=500, increment=10, textvariable=self.co2_threshold).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(thresholds_frame, text="Confidence Threshold (%):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(thresholds_frame, from_=50, to=95, increment=5, textvariable=self.confidence_threshold).grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # WiFi settings
        wifi_frame = ttk.LabelFrame(settings_frame, text="WiFi Settings")
        wifi_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.wifi_ssid = tk.StringVar(value="BedBugDetector")
        self.wifi_password = tk.StringVar(value="detector123")
        
        ttk.Label(wifi_frame, text="SSID:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(wifi_frame, textvariable=self.wifi_ssid).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        ttk.Label(wifi_frame, text="Password:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(wifi_frame, textvariable=self.wifi_password, show="*").grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Power settings
        power_frame = ttk.LabelFrame(settings_frame, text="Power Management")
        power_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.deep_sleep = tk.BooleanVar(value=False)
        self.sleep_minutes = tk.IntVar(value=5)
        
        ttk.Checkbutton(power_frame, text="Enable Deep Sleep", variable=self.deep_sleep).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(power_frame, text="Sleep Duration (minutes):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Spinbox(power_frame, from_=1, to=60, increment=1, textvariable=self.sleep_minutes).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Buttons
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(btn_frame, text="Save Settings", command=self.save_detector_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Reset to Defaults", command=self.reset_detector_settings).pack(side=tk.RIGHT, padx=5)
    
    def setup_server_settings(self, parent):
        """Set up server settings panel"""
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Server connection
        server_frame = ttk.LabelFrame(settings_frame, text="Server Connection")
        server_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(server_frame, text="Server URL:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(server_frame, textvariable=self.server_url).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        
        self.use_api_key = tk.BooleanVar(value=False)
        self.api_key = tk.StringVar(value="")
        
        ttk.Checkbutton(server_frame, text="Use API Key", variable=self.use_api_key).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(server_frame, textvariable=self.api_key, show="*").grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        
        # Test connection button
        ttk.Button(server_frame, text="Test Connection", command=self.test_server_connection).grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # Data storage
        storage_frame = ttk.LabelFrame(settings_frame, text="Data Storage")
        storage_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.data_dir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "BedBugDetector"))
        
        ttk.Label(storage_frame, text="Data Directory:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(storage_frame, textvariable=self.data_dir).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(storage_frame, text="Browse...", command=self.browse_data_dir).grid(row=0, column=2, padx=5, pady=5)
        
        self.keep_images = tk.BooleanVar(value=True)
        ttk.Checkbutton(storage_frame, text="Store Images Locally", variable=self.keep_images).grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)
        
        # Buttons
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(btn_frame, text="Save Settings", command=self.save_server_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Reset to Defaults", command=self.reset_server_settings).pack(side=tk.RIGHT, padx=5)
    
    def setup_app_settings(self, parent):
        """Set up application settings panel"""
        settings_frame = ttk.Frame(parent)
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Display settings
        display_frame = ttk.LabelFrame(settings_frame, text="Display Settings")
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.use_celsius = tk.BooleanVar(value=True)
        ttk.Radiobutton(display_frame, text="Celsius (°C)", variable=self.use_celsius, value=True).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Radiobutton(display_frame, text="Fahrenheit (°F)", variable=self.use_celsius, value=False).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        self.auto_refresh = tk.BooleanVar(value=True)
        self.refresh_interval = tk.IntVar(value=5)
        
        ttk.Checkbutton(display_frame, text="Auto-refresh Data", variable=self.auto_refresh).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        refresh_frame = ttk.Frame(display_frame)
        refresh_frame.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(refresh_frame, text="Interval:").pack(side=tk.LEFT)
        ttk.Spinbox(refresh_frame, from_=1, to=30, increment=1, width=5, textvariable=self.refresh_interval).pack(side=tk.LEFT, padx=5)
        ttk.Label(refresh_frame, text="seconds").pack(side=tk.LEFT)
        
        # Notification settings
        notify_frame = ttk.LabelFrame(settings_frame, text="Notification Settings")
        notify_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.enable_notifications = tk.BooleanVar(value=True)
        self.notify_on_detection = tk.BooleanVar(value=True)
        self.play_sound = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(notify_frame, text="Enable Notifications", variable=self.enable_notifications).grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        ttk.Checkbutton(notify_frame, text="Notify on Detection", variable=self.notify_on_detection).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Checkbutton(notify_frame, text="Play Sound on Detection", variable=self.play_sound).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Buttons
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(btn_frame, text="Save Settings", command=self.save_app_settings).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Reset to Defaults", command=self.reset_app_settings).pack(side=tk.RIGHT, padx=5)
    
    # ---- Functionality methods ----
    
    def check_server_connection(self):
        """Check if server is available"""
        try:
            response = requests.get(f"{self.server_url.get()}/api/health", timeout=3)
            if response.status_code == 200:
                logger.info("Server connection successful")
                self.connection_status.set("Server Connected")
                return True
            else:
                logger.warning(f"Server returned status code {response.status_code}")
                self.connection_status.set("Server Error")
                return False
        except Exception as e:
            logger.error(f"Server connection failed: {str(e)}")
            self.connection_status.set("Server Unavailable")
            return False
    
    def connect_to_detector(self):
        """Connect to the bed bug detector"""
        ip = self.detector_ip.get()
        
        try:
            # Try to connect directly to ESP32
            self.connection_status.set("Connecting...")
            self.update_idletasks()
            
            # Test direct connection to ESP32
            response = requests.get(f"http://{ip}/sensor-data", timeout=5)
            
            if response.status_code == 200:
                # Direct connection successful
                self.connection_status.set("Connected (Direct)")
                
                # Get device ID
                try:
                    data = response.json()
                    device_id = data.get("device_id", f"esp32-{ip.replace('.', '')}")
                    self.device_id.set(device_id)
                except:
                    # Generate a device ID based on IP if not available
                    self.device_id.set(f"esp32-{ip.replace('.', '')}")
                
                # Start sensor polling
                self.poll_sensor_data()
                
                messagebox.showinfo("Connection", f"Successfully connected to detector at {ip}")
                return True
                
            else:
                # Try through server API
                if self.check_server_connection():
                    # Register device through server
                    register_data = {
                        "device_id": f"esp32-{ip.replace('.', '')}",
                        "name": f"Detector at {ip}"
                    }
                    
                    register_response = requests.post(
                        f"{self.server_url.get()}/api/device/register",
                        json=register_data,
                        timeout=5
                    )
                    
                    if register_response.status_code == 200:
                        self.connection_status.set("Connected (Server)")
                        self.device_id.set(register_data["device_id"])
                        
                        # Start sensor polling through server
                        self.poll_sensor_data()
                        
                        messagebox.showinfo("Connection", f"Connected to detector via server at {ip}")
                        return True
                    else:
                        raise Exception("Failed to register device with server")
                else:
                    raise Exception("Server unavailable and direct connection failed")
                
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            self.connection_status.set("Disconnected")
            messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")
            return False
    
    def disconnect_from_detector(self):
        """Disconnect from the bed bug detector"""
        # TODO: Implement proper disconnect cleanup
        self.connection_status.set("Disconnected")
        messagebox.showinfo("Disconnected", "Disconnected from detector")
    
    def poll_sensor_data(self):
        """Poll sensor data periodically"""
        if self.connection_status.get() == "Disconnected":
            return
        
        try:
            ip = self.detector_ip.get()
            device_id = self.device_id.get()
            
            # Try direct connection first
            try:
                response = requests.get(f"http://{ip}/sensor-data", timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    self.update_sensor_display(data)
            except:
                # If direct connection fails, try through server
                if self.check_server_connection():
                    response = requests.get(
                        f"{self.server_url.get()}/api/sensor_data/{device_id}",
                        timeout=2
                    )
                    if response.status_code == 200:
                        data = response.json()
                        self.update_sensor_display(data["sensor_data"])
        except Exception as e:
            logger.error(f"Error polling sensor data: {str(e)}")
        
        # Schedule next poll if auto-refresh is enabled
        if self.auto_refresh.get():
            self.after(self.refresh_interval.get() * 1000, self.poll_sensor_data)
    
    def update_sensor_display(self, data):
        """Update sensor display with new data"""
        # Update temperature
        temp = data.get("temperature", 0)
        if not self.use_celsius.get():
            # Convert to Fahrenheit if needed
            temp = (temp * 9/5) + 32
            self.temp_value.set(f"{temp:.1f}")
        else:
            self.temp_value.set(f"{temp:.1f}")
        
        # Update other values
        self.humidity_value.set(f"{data.get('humidity', 0):.1f}")
        self.co2_value.set(str(data.get("co2", 0)))
        self.motion_value.set("Detected" if data.get("motion", False) else "None")
    
    def start_detection(self):
        """Start the bed bug detection process"""
        if self.connection_status.get() == "Disconnected":
            messagebox.showerror("Error", "Not connected to detector")
            return
        
        try:
            ip = self.detector_ip.get()
            device_id = self.device_id.get()
            
            # Update UI to show we're scanning
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Scanning for bed bugs...\n")
            self.result_text.config(state=tk.DISABLED)
            self.update_idletasks()
            
            # Try direct connection first
            try:
                # Get current sensor readings
                sensor_response = requests.get(f"http://{ip}/sensor-data", timeout=5)
                sensor_data = sensor_response.json()
                
                # Get camera image
                image_response = requests.get(f"http://{ip}/capture", timeout=5)
                image_data = image_response.content
                
                # Process locally or send to server
                if self.check_server_connection():
                    # Send to server for processing
                    files = {}
                    if image_data:
                        files["image"] = ("capture.jpg", image_data)
                    
                    detection_response = requests.post(
                        f"{self.server_url.get()}/api/detection",
                        data=sensor_data,
                        files=files,
                        timeout=10
                    )
                    
                    if detection_response.status_code == 200:
                        result = detection_response.json()
                        self.display_detection_result(result, image_data)
                    else:
                        raise Exception(f"Server detection failed: {detection_response.status_code}")
                else:
                    # Process locally with simplified algorithm
                    confidence = self.simple_local_detection(sensor_data)
                    detected = confidence >= self.confidence_threshold.get()
                    
                    result = {
                        "status": "success",
                        "detection": {
                            "detected": detected,
                            "confidence": confidence
                        },
                        "advice": self.generate_simple_advice(detected, confidence)
                    }
                    
                    self.display_detection_result(result, image_data)
            
            except Exception as e:
                logger.error(f"Detection error: {str(e)}")
                
                # Try through server instead
                if self.check_server_connection():
                    # Use server to communicate with detector
                    response = requests.post(
                        f"{self.server_url.get()}/api/trigger_detection/{device_id}",
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        self.display_detection_result(result, None)
                    else:
                        raise Exception(f"Server detection failed: {response.status_code}")
                else:
                    raise Exception("Detection failed and server unavailable")
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Detection failed: {str(e)}\n")
            self.result_text.config(state=tk.DISABLED)
            
            messagebox.showerror("Detection Error", f"Detection failed: {str(e)}")
    
    def display_detection_result(self, result, image_data=None):
        """Display detection results in the UI"""
        detection = result.get("detection", {})
        detected = detection.get("detected", False)
        confidence = detection.get("confidence", 0)
        advice = result.get("advice", "")
        
        # Store the result for history
        timestamp = datetime.now().isoformat()
        history_entry = {
            "timestamp": timestamp,
            "device_id": self.device_id.get(),
            "detection": detection,
            "advice": advice,
            "location": "Unknown"  # TODO: Add location input
        }
        
        self.detection_history.append(history_entry)
        self.refresh_history()
        
        # Update results text
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        if detected:
            self.result_text.insert(tk.END, "BED BUG ACTIVITY DETECTED\n", "detected")
        else:
            self.result_text.insert(tk.END, "No bed bug activity detected\n", "not-detected")
        
        self.result_text.insert(tk.END, f"Confidence: {confidence:.1f}%\n\n")
        self.result_text.insert(tk.END, f"Advice:\n{advice}\n")
        
        self.result_text.tag_configure("detected", foreground="red", font=("TkDefaultFont", 10, "bold"))
        self.result_text.tag_configure("not-detected", foreground="green", font=("TkDefaultFont", 10, "bold"))
        
        self.result_text.config(state=tk.DISABLED)
        
        # Display image if available
        if image_data:
            try:
                # Convert image data to format usable by Tkinter
                img = Image.open(BytesIO(image_data))
                img = img.resize((self.camera_canvas.winfo_width(), self.camera_canvas.winfo_height()), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Display on canvas
                self.camera_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.camera_image = photo  # Keep reference to prevent garbage collection
            except Exception as e:
                logger.error(f"Error displaying image: {str(e)}")
    
    def simple_local_detection(self, sensor_data):
        """Simplified local detection algorithm"""
        # Extract sensor values
        temperature = sensor_data.get("temperature", 0)
        co2 = sensor_data.get("co2", 0)
        motion = sensor_data.get("motion", False)
        
        # Get thresholds from settings
        temp_threshold = self.temp_threshold.get()
        co2_threshold = self.co2_threshold.get()
        
        # Calculate confidence score
        confidence = 0
        
        # Check temperature
        if abs(temperature) > temp_threshold:
            # Proportional to how much it exceeds threshold
            temp_factor = min(abs(temperature) / temp_threshold, 2.0)
            confidence += 30 * temp_factor
        
        # Check CO2
        if co2 > co2_threshold:
            co2_factor = min(co2 / co2_threshold, 2.0)
            confidence += 30 * co2_factor
        
        # Check motion
        if motion:
            confidence += 40
        
        # Cap confidence at 100
        confidence = min(confidence, 100)
        
        return confidence
    
    def generate_simple_advice(self, detected, confidence):
        """Generate simple advice based on detection result"""
        if not detected:
            return "No significant bed bug activity detected. Continue monitoring regularly as early infestations can be difficult to detect."
        
        if confidence > 90:
            return "High confidence bed bug detection! Immediate action recommended. Thoroughly inspect the area, focusing on mattress seams, bedding, and nearby furniture. Consider professional treatment."
        
        if confidence > 70:
            return "Signs of bed bug activity detected. Recommend thorough inspection of the area, focusing on mattress seams, bedding, and nearby furniture. Consider professional treatment if multiple areas show positive detection."
        
        return "Possible bed bug activity detected with low confidence. Re-check this area with additional scans. Focus inspection on mattress seams, box springs, and furniture crevices."
    
    def calibrate_detector(self):
        """Calibrate the bed bug detector"""
        if self.connection_status.get() == "Disconnected":
            messagebox.showerror("Error", "Not connected to detector")
            return
        
        # Confirm calibration
        if not messagebox.askyesno("Calibrate", 
                                   "Calibration should be performed in a known bed bug-free area.\n\n"
                                   "Are you sure you want to calibrate now?"):
            return
        
        try:
            ip = self.detector_ip.get()
            
            # Try direct connection
            response = requests.get(f"http://{ip}/calibrate", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                messagebox.showinfo("Calibration", 
                                   f"Calibration successful!\n\n"
                                   f"Baseline temperature: {data.get('baselineTemp', 0):.1f}°C\n"
                                   f"Baseline CO2: {data.get('baselineCO2', 0)} ppm")
            else:
                raise Exception(f"Calibration failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Calibration error: {str(e)}")
            messagebox.showerror("Calibration Error", f"Calibration failed: {str(e)}")
    
    def capture_image(self):
        """Capture an image from the detector camera"""
        if self.connection_status.get() == "Disconnected":
            messagebox.showerror("Error", "Not connected to detector")
            return
        
        try:
            ip = self.detector_ip.get()
            
            # Try direct connection
            response = requests.get(f"http://{ip}/capture", timeout=5)
            
            if response.status_code == 200:
                # Convert image data to format usable by Tkinter
                img = Image.open(BytesIO(response.content))
                img = img.resize((self.camera_canvas.winfo_width(), self.camera_canvas.winfo_height()), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Display on canvas
                self.camera_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.camera_image = photo  # Keep reference to prevent garbage collection
                
                # Save image if enabled
                if self.keep_images.get():
                    os.makedirs(os.path.join(self.data_dir.get(), "images"), exist_ok=True)
                    filename = os.path.join(
                        self.data_dir.get(), 
                        "images", 
                        f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    )
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Image saved to {filename}")
            else:
                raise Exception(f"Image capture failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Image capture error: {str(e)}")
            messagebox.showerror("Capture Error", f"Image capture failed: {str(e)}")
    
    def refresh_history(self):
        """Refresh the history view"""
        # Clear current items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Apply filter
        filter_type = self.filter_var.get()
        filtered_history = []
        
        if filter_type == "All":
            filtered_history = self.detection_history
        elif filter_type == "Positive Only":
            filtered_history = [h for h in self.detection_history if h.get("detection", {}).get("detected", False)]
        elif filter_type == "Last 24 Hours":
            cutoff = datetime.now() - timedelta(hours=24)
            filtered_history = [h for h in self.detection_history 
                               if datetime.fromisoformat(h.get("timestamp", "").replace('Z', '+00:00')) > cutoff]
        elif filter_type == "Last Week":
            cutoff = datetime.now() - timedelta(days=7)
            filtered_history = [h for h in self.detection_history 
                               if datetime.fromisoformat(h.get("timestamp", "").replace('Z', '+00:00')) > cutoff]
        
        # Sort by timestamp (newest first)
        filtered_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Add to tree
        for item in filtered_history:
            timestamp = item.get("timestamp", "")
            try:
                # Convert ISO timestamp to local datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp
                
            location = item.get("location", "Unknown")
            detected = "Yes" if item.get("detection", {}).get("detected", False) else "No"
            confidence = f"{item.get('detection', {}).get('confidence', 0):.1f}%"
            
            # Add to tree with tag for color
            tag = "detected" if detected == "Yes" else "normal"
            self.history_tree.insert("", "end", values=(formatted_time, location, detected, confidence), tags=(tag,))
        
        # Configure tags
        self.history_tree.tag_configure("detected", background="#ffdddd")
        
        # Update analysis if we have data
        if self.detection_history:
            self.update_analysis()
    
    def filter_history(self, event=None):
        """Filter history based on selected criteria"""
        self.refresh_history()
    
    def on_history_select(self, event=None):
        """Handle history item selection"""
        selected_item = self.history_tree.selection()
        if not selected_item:
            return
        
        # Get the index in the filtered history
        index = self.history_tree.index(selected_item[0])
        
        # Get the filter type
        filter_type = self.filter_var.get()
        filtered_history = []
        
        if filter_type == "All":
            filtered_history = self.detection_history
        elif filter_type == "Positive Only":
            filtered_history = [h for h in self.detection_history if h.get("detection", {}).get("detected", False)]
        elif filter_type == "Last 24 Hours":
            cutoff = datetime.now() - timedelta(hours=24)
            filtered_history = [h for h in self.detection_history 
                               if datetime.fromisoformat(h.get("timestamp", "").replace('Z', '+00:00')) > cutoff]
        elif filter_type == "Last Week":
            cutoff = datetime.now() - timedelta(days=7)
            filtered_history = [h for h in self.detection_history 
                               if datetime.fromisoformat(h.get("timestamp", "").replace('Z', '+00:00')) > cutoff]
        
        # Sort by timestamp (newest first)
        filtered_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        if index < len(filtered_history):
            # Display the selected detection
            selected_detection = filtered_history[index]
            
            # Update result display
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            
            detection = selected_detection.get("detection", {})
            detected = detection.get("detected", False)
            confidence = detection.get("confidence", 0)
            advice = selected_detection.get("advice", "")
            timestamp = selected_detection.get("timestamp", "")
            location = selected_detection.get("location", "Unknown")
            
            try:
                # Convert ISO timestamp to local datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp
            
            self.result_text.insert(tk.END, f"Location: {location}\n")
            self.result_text.insert(tk.END, f"Time: {formatted_time}\n\n")
            
            if detected:
                self.result_text.insert(tk.END, "BED BUG ACTIVITY DETECTED\n", "detected")
            else:
                self.result_text.insert(tk.END, "No bed bug activity detected\n", "not-detected")
            
            self.result_text.insert(tk.END, f"Confidence: {confidence:.1f}%\n\n")
            self.result_text.insert(tk.END, f"Advice:\n{advice}\n")
            
            self.result_text.tag_configure("detected", foreground="red", font=("TkDefaultFont", 10, "bold"))
            self.result_text.tag_configure("not-detected", foreground="green", font=("TkDefaultFont", 10, "bold"))
            
            self.result_text.config(state=tk.DISABLED)
            
            # TODO: Load image if available
    
    def update_analysis(self):
        """Update the analysis tab with current data"""
        if not self.detection_history:
            return
        
        # Perform analysis
        analysis_result = analyze_detection_history(self.detection_history)
        
        # Update statistics
        self.generate_statistics_view(analysis_result)
        
        # Update recommendations
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        
        for recommendation in analysis_result.get("recommendations", []):
            self.recommendations_text.insert(tk.END, f"• {recommendation}\n")
        
        self.recommendations_text.config(state=tk.DISABLED)
        
        # TODO: Generate and update charts
    
    def generate_statistics_view(self, analysis_result=None):
        """Generate statistics view"""
        # Clear current stats
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        if not analysis_result:
            # No data available
            ttk.Label(self.stats_frame, text="No detection data available").pack(pady=20)
            return
        
        # Basic statistics
        basic_frame = ttk.Frame(self.stats_frame)
        basic_frame.pack(fill=tk.X, padx=5, pady=5)
        
        total_scans = analysis_result.get("total_scans", 0)
        positive_detections = analysis_result.get("positive_detections", 0)
        detection_rate = analysis_result.get("detection_rate", 0) * 100
        avg_confidence = analysis_result.get("average_confidence", 0)
        
        ttk.Label(basic_frame, text="Total Scans:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(basic_frame, text=str(total_scans)).grid(row=0, column=1, padx=5, pady=2, sticky=tk.E)
        
        ttk.Label(basic_frame, text="Positive Detections:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(basic_frame, text=str(positive_detections)).grid(row=1, column=1, padx=5, pady=2, sticky=tk.E)
        
        ttk.Label(basic_frame, text="Detection Rate:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(basic_frame, text=f"{detection_rate:.1f}%").grid(row=2, column=1, padx=5, pady=2, sticky=tk.E)
        
        ttk.Label(basic_frame, text="Average Confidence:").grid(row=3, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(basic_frame, text=f"{avg_confidence:.1f}%").grid(row=3, column=1, padx=5, pady=2, sticky=tk.E)
        
        # Trend
        trend = analysis_result.get("detection_trend", "neutral")
        trend_text = "Stable"
        if trend == "increasing":
            trend_text = "⚠️ Increasing"
        elif trend == "decreasing":
            trend_text = "✓ Decreasing"
        
        ttk.Label(basic_frame, text="Detection Trend:").grid(row=4, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(basic_frame, text=trend_text).grid(row=4, column=1, padx=5, pady=2, sticky=tk.E)
        
        # Peak activity time
        peak_time = analysis_result.get("peak_activity_time")
        if peak_time:
            ttk.Label(basic_frame, text="Peak Activity Time:").grid(row=5, column=0, padx=5, pady=2, sticky=tk.W)
            ttk.Label(basic_frame, text=peak_time).grid(row=5, column=1, padx=5, pady=2, sticky=tk.E)
        
        # High risk areas
        high_risk_frame = ttk.LabelFrame(self.stats_frame, text="High Risk Areas")
        high_risk_frame.pack(fill=tk.X, padx=5, pady=5)
        
        high_risk_areas = analysis_result.get("high_risk_areas", [])
        
        if high_risk_areas:
            for i, area in enumerate(high_risk_areas):
                location = area.get("location", "Unknown")
                rate = area.get("detection_rate", 0) * 100
                
                ttk.Label(high_risk_frame, text=f"{location}:").grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)
                ttk.Label(high_risk_frame, text=f"{rate:.1f}%").grid(row=i, column=1, padx=5, pady=2, sticky=tk.E)
        else:
            ttk.Label(high_risk_frame, text="No high risk areas identified").grid(padx=5, pady=5)
        
        # Probability assessment
        prob_result = calculate_infestation_probability(self.detection_history)
        
        prob_frame = ttk.LabelFrame(self.stats_frame, text="Infestation Probability")
        prob_frame.pack(fill=tk.X, padx=5, pady=5)
        
        probability = prob_result.get("overall_probability", 0)
        confidence = prob_result.get("confidence", "low")
        interpretation = prob_result.get("interpretation", "Unknown")
        
        ttk.Label(prob_frame, text="Probability:").grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(prob_frame, text=f"{probability:.1f}%").grid(row=0, column=1, padx=5, pady=2, sticky=tk.E)
        
        ttk.Label(prob_frame, text="Estimate Confidence:").grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(prob_frame, text=confidence.capitalize()).grid(row=1, column=1, padx=5, pady=2, sticky=tk.E)
        
        ttk.Label(prob_frame, text="Interpretation:").grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(prob_frame, text=interpretation).grid(row=2, column=1, padx=5, pady=2, sticky=tk.E)
    
    def export_data(self):
        """Export detection data"""
        if not self.detection_history:
            messagebox.showwarning("Export", "No detection data available to export")
            return
        
        # Ask for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Detection Data"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                json.dump(self.detection_history, f, indent=2)
            
            messagebox.showinfo("Export", f"Successfully exported data to {file_path}")
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def export_history(self):
        """Export detection history to file"""
        self.export_data()
    
    def show_detector_settings(self):
        """Show detector settings dialog"""
        # TODO: Implement or use settings tab
        pass
    
    def show_server_settings(self):
        """Show server settings dialog"""
        # TODO: Implement or use settings tab
        pass
    
    def show_app_settings(self):
        """Show application settings dialog"""
        # TODO: Implement or use settings tab
        pass
    
    def save_detector_settings(self):
        """Save detector settings"""
        # TODO: Implement
        messagebox.showinfo("Settings", "Detector settings saved")
    
    def reset_detector_settings(self):
        """Reset detector settings to defaults"""
        # TODO: Implement
        messagebox.showinfo("Settings", "Detector settings reset to defaults")
    
    def save_server_settings(self):
        """Save server settings"""
        # TODO: Implement
        messagebox.showinfo("Settings", "Server settings saved")
    
    def reset_server_settings(self):
        """Reset server settings to defaults"""
        # TODO: Implement
        self.server_url.set(DEFAULT_SERVER)
        messagebox.showinfo("Settings", "Server settings reset to defaults")
    
    def save_app_settings(self):
        """Save application settings"""
        # TODO: Implement
        messagebox.showinfo("Settings", "Application settings saved")
    
    def reset_app_settings(self):
        """Reset application settings to defaults"""
        # TODO: Implement
        messagebox.showinfo("Settings", "Application settings reset to defaults")
    
    def browse_data_dir(self):
        """Browse for data directory"""
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.data_dir.set(directory)
    
    def test_server_connection(self):
        """Test connection to server"""
        if self.check_server_connection():
            messagebox.showinfo("Connection", "Successfully connected to server")
        else:
            messagebox.showerror("Connection Error", "Failed to connect to server")
    
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
Bed Bug Detector - User Guide

1. CONNECTION
   - Enter the IP address of your detector (default: 192.168.4.1)
   - Click "Connect" to establish connection
   - Make sure your computer is connected to the detector's WiFi network

2. DETECTION
   - Click "Start Detection" to begin scanning for bed bugs
   - The results will appear in the results panel
   - Positive detections will be highlighted in red

3. CALIBRATION
   - Important: Calibrate in a known bed bug-free area
   - Click "Calibrate" to set baseline readings
   - This improves detection accuracy

4. CAMERA
   - Click "Capture Image" to take a photo
   - Images can help with visual confirmation
   - Images are saved locally if enabled in settings

5. HISTORY
   - View past detection results in the History tab
   - Filter results by different criteria
   - Select an entry to view full details

6. ANALYSIS
   - The Analysis tab shows statistics and trends
   - Review recommendations for your situation
   - Track detection patterns over time

7. SETTINGS
   - Customize detection thresholds
   - Configure server connection
   - Set application preferences

For more information, visit the project repository:
https://github.com/your-username/bed-bug-detector
"""
        
        # Create popup window
        guide_window = tk.Toplevel(self)
        guide_window.title("User Guide")
        guide_window.geometry("600x500")
        guide_window.minsize(500, 400)
        
        # Add scrollable text area
        text_area = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD)
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Insert guide text
        text_area.insert(tk.END, guide_text)
        text_area.config(state=tk.DISABLED)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Bed Bug Detector v1.0

This application provides a desktop interface for the Bed Bug Detector system.

It allows you to:
- Connect to the detector hardware
- Perform bed bug detection scans
- Calibrate the detection system
- Capture and analyze images
- Review detection history
- Analyze trends and patterns
- Configure detection settings

The detector uses multiple sensors and image analysis to identify the presence 
of bed bugs with good accuracy at a fraction of the cost of professional services.

This is an open-source project. For more information, visit:
https://github.com/your-username/bed-bug-detector

License: MIT
"""
        
        messagebox.showinfo("About Bed Bug Detector", about_text)


if __name__ == "__main__":
    # Create and run application
    app = BedBugDetectorApp()
    app.mainloop()
