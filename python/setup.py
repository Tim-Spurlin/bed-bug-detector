#!/usr/bin/env python

"""
Bed Bug Detector - Python Components Setup Script

This script installs the Python components of the Bed Bug Detector system.
It sets up the directory structure, installs dependencies, and configures
the environment.
"""

import os
import sys
import subprocess
import argparse
import shutil
import platform
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("setup")

# Define directory structure
DIRECTORIES = [
    "server",
    "server/data",
    "server/data/images",
    "server/data/detections",
    "ml",
    "ml/models",
    "ml/dataset",
    "ml/dataset/images",
    "utils",
    "desktop"
]

# Define dependencies for different components
DEPENDENCIES = {
    "core": [
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "requests",
        "python-dotenv",
        "opencv-python"
    ],
    "server": [
        "flask",
        "gunicorn",
        "scikit-learn",
        "pillow"
    ],
    "ml": [
        "tensorflow",
        "scikit-learn",
        "scikit-image",
        "seaborn"
    ],
    "desktop": [
        "pillow",
        "tk"  # Tkinter is usually included with Python, but some systems need it
    ]
}

# OS-specific package manager commands
PKG_MANAGERS = {
    "apt": {
        "check": "apt --version",
        "install": "apt install -y",
        "packages": {
            "opencv": "libopencv-dev python3-opencv",
            "tk": "python3-tk"
        }
    },
    "dnf": {
        "check": "dnf --version",
        "install": "dnf install -y",
        "packages": {
            "opencv": "opencv-devel python3-opencv",
            "tk": "python3-tkinter"
        }
    },
    "pacman": {
        "check": "pacman --version",
        "install": "pacman -S --noconfirm",
        "packages": {
            "opencv": "opencv python-opencv",
            "tk": "tk"
        }
    },
    "brew": {
        "check": "brew --version",
        "install": "brew install",
        "packages": {
            "opencv": "opencv",
            "tk": "python-tk"
        }
    }
}


def run_command(command, shell=False):
    """
    Run a shell command and return the output
    
    Args:
        command (str or list): Command to run
        shell (bool): Whether to run command in shell mode
    
    Returns:
        tuple: (success, output)
    """
    try:
        if isinstance(command, str) and not shell:
            command = command.split()
        
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            universal_newlines=True
        )
        
        stdout, stderr = proc.communicate()
        
        if proc.returncode != 0:
            return False, stderr
        
        return True, stdout
    except Exception as e:
        return False, str(e)


def setup_directories():
    """
    Create the directory structure
    
    Returns:
        bool: Success or failure
    """
    logger.info("Setting up directory structure...")
    
    try:
        for directory in DIRECTORIES:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        return False


def detect_package_manager():
    """
    Detect the system package manager
    
    Returns:
        str or None: Name of package manager or None if not found
    """
    for manager in PKG_MANAGERS:
        success, _ = run_command(PKG_MANAGERS[manager]["check"])
        if success:
            return manager
    
    return None


def install_system_dependencies(components=None):
    """
    Install required system dependencies
    
    Args:
        components (list): List of components to install or None for all
        
    Returns:
        bool: Success or failure
    """
    if components is None:
        components = ["core", "server", "ml", "desktop"]
    
    logger.info("Checking system dependencies...")
    
    # Detect OS
    system = platform.system()
    
    if system == "Windows":
        logger.info("Windows detected, skipping system package installation")
        return True
    
    # Detect package manager
    pkg_manager = detect_package_manager()
    
    if pkg_manager is None:
        logger.warning("Could not detect package manager, skipping system package installation")
        return True
    
    logger.info(f"Detected package manager: {pkg_manager}")
    
    # Install OpenCV dependencies if needed
    if "ml" in components or "desktop" in components:
        logger.info("Installing OpenCV dependencies...")
        success, output = run_command(
            f"sudo {PKG_MANAGERS[pkg_manager]['install']} {PKG_MANAGERS[pkg_manager]['packages']['opencv']}",
            shell=True
        )
        
        if not success:
            logger.warning(f"Failed to install OpenCV dependencies: {output}")
    
    # Install Tkinter if needed for desktop app
    if "desktop" in components:
        logger.info("Installing Tkinter dependencies...")
        success, output = run_command(
            f"sudo {PKG_MANAGERS[pkg_manager]['install']} {PKG_MANAGERS[pkg_manager]['packages']['tk']}",
            shell=True
        )
        
        if not success:
            logger.warning(f"Failed to install Tkinter dependencies: {output}")
    
    return True


def install_python_dependencies(components=None, use_venv=True):
    """
    Install Python dependencies
    
    Args:
        components (list): List of components to install or None for all
        use_venv (bool): Whether to create and use a virtual environment
        
    Returns:
        bool: Success or failure
    """
    if components is None:
        components = ["core", "server", "ml", "desktop"]
    
    logger.info("Installing Python dependencies...")
    
    # Determine pip command
    pip_cmd = "pip"
    if sys.version_info.major == 3:
        pip_cmd = "pip3"
    
    # Create virtual environment if requested
    if use_venv:
        logger.info("Creating virtual environment...")
        
        success, output = run_command(f"{sys.executable} -m venv venv")
        
        if not success:
            logger.error(f"Failed to create virtual environment: {output}")
            return False
        
        # Activate virtual environment
        if platform.system() == "Windows":
            pip_cmd = os.path.join("venv", "Scripts", "pip")
        else:
            pip_cmd = os.path.join("venv", "bin", "pip")
    
    # Upgrade pip
    logger.info("Upgrading pip...")
    success, output = run_command(f"{pip_cmd} install --upgrade pip")
    
    if not success:
        logger.warning(f"Failed to upgrade pip: {output}")
    
    # Install dependencies for each component
    for component in components:
        if component in DEPENDENCIES:
            logger.info(f"Installing {component} dependencies...")
            
            # Install each dependency
            for package in DEPENDENCIES[component]:
                logger.info(f"Installing {package}...")
                success, output = run_command(f"{pip_cmd} install {package}")
                
                if not success:
                    logger.warning(f"Failed to install {package}: {output}")
    
    # Generate requirements.txt files for each component
    generate_requirements_files(pip_cmd, components)
    
    return True


def generate_requirements_files(pip_cmd, components):
    """
    Generate requirements.txt files for each component
    
    Args:
        pip_cmd (str): Pip command to use
        components (list): List of components
    """
    logger.info("Generating requirements.txt files...")
    
    # Get all installed packages
    success, output = run_command(f"{pip_cmd} freeze")
    
    if not success:
        logger.warning(f"Failed to get installed packages: {output}")
        return
    
    installed_packages = output.splitlines()
    
    # Generate requirements file for each component
    for component in components:
        if component == "core":
            continue  # Core dependencies are included in other components
        
        component_packages = []
        
        # Find all packages for this component
        for package in DEPENDENCIES.get(component, []):
            for installed in installed_packages:
                if installed.lower().startswith(package.lower()):
                    component_packages.append(installed)
        
        # Add core dependencies
        for package in DEPENDENCIES.get("core", []):
            for installed in installed_packages:
                if installed.lower().startswith(package.lower()):
                    if installed not in component_packages:
                        component_packages.append(installed)
        
        # Write requirements file
        requirements_path = f"{component}/requirements.txt"
        
        with open(requirements_path, "w") as f:
            f.write("\n".join(component_packages))
        
        logger.info(f"Generated {requirements_path}")


def copy_code_files():
    """
    Copy code files from repository to appropriate directories
    
    Returns:
        bool: Success or failure
    """
    logger.info("Copying code files...")
    
    try:
        # Create map of source files to destinations
        file_map = {
            # Server files
            "python/server/app.py": "server/app.py",
            "python/server/config.py": "server/config.py",
            "python/server/README.md": "server/README.md",
            
            # ML files
            "python/ml/train.py": "ml/train.py",
            "python/ml/predict.py": "ml/predict.py",
            
            # Utils files
            "python/utils/image_processing.py": "utils/image_processing.py",
            "python/utils/data_analysis.py": "utils/data_analysis.py",
            "python/utils/device_manager.py": "utils/device_manager.py",
            
            # Desktop app files
            "python/desktop/main.py": "desktop/main.py"
        }
        
        # Copy each file
        for source, dest in file_map.items():
            if os.path.exists(source):
                # Create destination directory if needed
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                
                # Copy file
                shutil.copy2(source, dest)
                logger.info(f"Copied {source} to {dest}")
            else:
                logger.warning(f"Source file not found: {source}")
        
        return True
    except Exception as e:
        logger.error(f"Error copying code files: {str(e)}")
        return False


def create_env_file():
    """
    Create .env file with default settings
    
    Returns:
        bool: Success or failure
    """
    logger.info("Creating .env file...")
    
    try:
        env_content = """# Bed Bug Detector - Environment Configuration

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
USE_ML=True

# Security settings
REQUIRE_API_KEY=False
API_KEY=change_this_to_your_secret_key
"""
        
        # Write to server directory
        with open("server/.env", "w") as f:
            f.write(env_content)
        
        logger.info("Created server/.env file")
        return True
    except Exception as e:
        logger.error(f"Error creating .env file: {str(e)}")
        return False


def create_startup_scripts():
    """
    Create startup scripts for server and desktop app
    
    Returns:
        bool: Success or failure
    """
    logger.info("Creating startup scripts...")
    
    try:
        # Create server startup script
        if platform.system() == "Windows":
            with open("start_server.bat", "w") as f:
                f.write("@echo off\n")
                f.write("cd server\n")
                f.write("if exist ..\\venv\\Scripts\\activate.bat (\n")
                f.write("  call ..\\venv\\Scripts\\activate.bat\n")
                f.write(")\n")
                f.write("python app.py\n")
            
            logger.info("Created start_server.bat")
            
            # Create desktop app startup script
            with open("start_desktop.bat", "w") as f:
                f.write("@echo off\n")
                f.write("cd desktop\n")
                f.write("if exist ..\\venv\\Scripts\\activate.bat (\n")
                f.write("  call ..\\venv\\Scripts\\activate.bat\n")
                f.write(")\n")
                f.write("python main.py\n")
            
            logger.info("Created start_desktop.bat")
        else:
            with open("start_server.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write("cd server\n")
                f.write("if [ -f ../venv/bin/activate ]; then\n")
                f.write("  source ../venv/bin/activate\n")
                f.write("fi\n")
                f.write("python app.py\n")
            
            with open("start_desktop.sh", "w") as f:
                f.write("#!/bin/bash\n")
                f.write("cd desktop\n")
                f.write("if [ -f ../venv/bin/activate ]; then\n")
                f.write("  source ../venv/bin/activate\n")
                f.write("fi\n")
                f.write("python main.py\n")
            
            # Make scripts executable
            os.chmod("start_server.sh", 0o755)
            os.chmod("start_desktop.sh", 0o755)
            
            logger.info("Created start_server.sh and start_desktop.sh")
        
        return True
    except Exception as e:
        logger.error(f"Error creating startup scripts: {str(e)}")
        return False


def main():
    """Main setup function"""
    print("Bed Bug Detector - Python Setup")
    print("===============================")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Setup script for Bed Bug Detector Python components")
    parser.add_argument("--no-venv", action="store_true", help="Don't create a virtual environment")
    parser.add_argument("--components", choices=["server", "ml", "desktop", "all"], default="all", help="Components to install")
    args = parser.parse_args()
    
    # Determine components to install
    if args.components == "all":
        components = ["core", "server", "ml", "desktop"]
    else:
        components = ["core", args.components]
    
    # Get start time
    start_time = datetime.now()
    logger.info(f"Setup started at {start_time}")
    
    # Setup directory structure
    if not setup_directories():
        logger.error("Failed to set up directory structure")
        return False
    
    # Install system dependencies
    if not install_system_dependencies(components):
        logger.warning("Some system dependencies could not be installed")
    
    # Install Python dependencies
    if not install_python_dependencies(components, not args.no_venv):
        logger.error("Failed to install Python dependencies")
        return False
    
    # Copy code files
    if not copy_code_files():
        logger.warning("Some code files could not be copied")
    
    # Create environment file
    if "server" in components:
        if not create_env_file():
            logger.warning("Failed to create .env file")
    
    # Create startup scripts
    if not create_startup_scripts():
        logger.warning("Failed to create startup scripts")
    
    # Get end time
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Setup completed at {end_time} (duration: {duration})")
    
    # Print final instructions
    print("\nSetup completed successfully!")
    print("\nTo start the server:")
    if platform.system() == "Windows":
        print("  run start_server.bat")
    else:
        print("  ./start_server.sh")
    
    if "desktop" in components:
        print("\nTo start the desktop application:")
        if platform.system() == "Windows":
            print("  run start_desktop.bat")
        else:
            print("  ./start_desktop.sh")
    
    print("\nFor more information, see the README files in each component directory.")
    
    return True


if __name__ == "__main__":
    main()
