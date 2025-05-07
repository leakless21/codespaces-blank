import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List, Tuple, Union, Optional

"""
Traffic Monitoring System Configuration
---------------------------------------
Configuration hierarchy (highest to lowest priority):
1. Environment variables (.env file or system environment)
2. YAML configuration (config.yaml)
3. Default values defined in this file

Use environment variables for deployment-specific settings and
YAML for application settings that rarely change between deployments.
"""

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Load YAML configuration
config_path = Path(__file__).parent / "settings" / "config.yaml"
with open(config_path, 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

# Helper functions for type conversion
def env_bool(name: str, default: bool = False) -> bool:
    """Convert environment variable to boolean with fallback"""
    value = os.getenv(name, str(default)).lower()
    return value in ('true', 'yes', '1', 'y', 't')

def env_int(name: str, default: int) -> int:
    """Convert environment variable to integer with fallback"""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

def env_float(name: str, default: float) -> float:
    """Convert environment variable to float with fallback"""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default

def env_list(name: str, default: List[Any], separator: str = ',') -> List[Any]:
    """Convert comma-separated environment variable to list with fallback"""
    value = os.getenv(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]

# === Environment-specific settings (from .env) ===

# Model configurations
VEHICLE_DETECTION_MODEL = os.getenv("VEHICLE_DETECTION_MODEL", str(MODELS_DIR / "vehicle_detection.onnx"))
PLATE_DETECTION_MODEL = os.getenv("PLATE_DETECTION_MODEL", str(MODELS_DIR / "plate_detection.onnx"))

# Video settings
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # 0 for webcam, or file path/RTSP URL

# Raw coordinates flag
USE_RAW_COORDINATES = env_bool("USE_RAW_COORDINATES", False)

# MQTT connection settings
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = env_int("MQTT_PORT", 1883)
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")  # Optional username
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")  # Optional password

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = env_int("API_PORT", 8000)

# Database settings
DB_URL = os.getenv("DB_URL", f"sqlite:///{BASE_DIR}/data/traffic_data.db")

# === Application settings (from YAML with env override capability) ===

# Hardware acceleration settings
USE_GPU = env_bool("USE_GPU", 
                  yaml_config['hardware'].get('use_gpu', False))
HARDWARE_PROVIDER = os.getenv("HARDWARE_PROVIDER", 
                             yaml_config['hardware'].get('provider', 'auto'))
HARDWARE_PRECISION = os.getenv("HARDWARE_PRECISION", 
                              yaml_config['hardware'].get('precision', 'fp32'))

# Detection settings
DETECTION_CONFIDENCE = env_float("DETECTION_CONFIDENCE", 
                                yaml_config['detection'].get('confidence', 0.5))
DETECTION_IOU_THRESHOLD = env_float("DETECTION_IOU_THRESHOLD", 
                                   yaml_config['detection'].get('iou_threshold', 0.45))

# Model version settings
VEHICLE_MODEL_VERSION = os.getenv("VEHICLE_MODEL_VERSION",
                                yaml_config['detection']['model_versions'].get('vehicle_model', 'yolov8'))
PLATE_MODEL_VERSION = os.getenv("PLATE_MODEL_VERSION",
                              yaml_config['detection']['model_versions'].get('plate_model', 'yolov8'))

# Vehicle class settings
ENABLED_VEHICLE_CLASSES = env_list("ENABLED_VEHICLE_CLASSES", 
                                  yaml_config['detection']['vehicle_classes']['enabled_classes'])
VEHICLE_CLASS_NAMES = yaml_config['detection']['vehicle_classes']['class_names']
COUNT_VEHICLE_CLASSES = yaml_config['detection']['vehicle_classes']['count_classes']

# Tracking settings
TRACKER_TYPE = os.getenv("TRACKER_TYPE", 
                        yaml_config['tracking'].get('tracker_type', 'bytetrack'))
TRACKING_CONFIDENCE = env_float("TRACKING_CONFIDENCE", 
                               yaml_config['tracking'].get('confidence', 0.3))

# ReID model path for trackers that support it
REID_WEIGHTS_PATH = os.getenv("REID_WEIGHTS_PATH",
                             os.path.join(str(MODELS_DIR), 
                                         yaml_config['tracking'].get('reid_weights', 'osnet_x0_25_msmt17.pt')))

# Common tracking parameters
TRACKER_MAX_AGE = env_int("TRACKER_MAX_AGE",
                         yaml_config['tracking'].get('max_age', 30))
TRACKER_MIN_HITS = env_int("TRACKER_MIN_HITS",
                          yaml_config['tracking'].get('min_hits', 3))
TRACKER_FRAME_RATE = env_int("TRACKER_FRAME_RATE",
                            yaml_config['tracking'].get('frame_rate', 30))

# Tracker type-specific configurations (accessible via get_config function)
# Example: get_config('tracking.bytetrack.track_thresh', 0.6)

# OCR settings
OCR_LANGUAGES = env_list("OCR_LANGUAGES", 
                        yaml_config['ocr'].get('languages', ['en']))
OCR_GPU = env_bool("OCR_GPU", 
                  yaml_config['ocr'].get('use_gpu', False))

# Video processing settings
FRAME_SKIP = env_int("FRAME_SKIP", 
                    yaml_config['video'].get('frame_skip', 1))
OUTPUT_FPS = env_float("OUTPUT_FPS", 
                      yaml_config['video'].get('output_fps', 30.0))
PROCESS_RESOLUTION = tuple(map(int, 
                              env_list("PROCESS_RESOLUTION", 
                                      yaml_config['video'].get('process_resolution', [640, 480]))))

# MQTT topic settings
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", 
                             yaml_config['mqtt'].get('topic_prefix', 'traffic'))

# Counting line settings
# Load use_raw_coordinates from YAML if present, otherwise from env var
USE_RAW_COORDINATES = env_bool("USE_RAW_COORDINATES", 
                              yaml_config['counting'].get('use_raw_coordinates', False))

# Store both raw and normalized coordinates for flexibility
RAW_COUNTING_LINE = [
    yaml_config['counting']['raw_coordinates'].get('start', [0, 400]),
    yaml_config['counting']['raw_coordinates'].get('end', [800, 400])
]

NORMALIZED_COUNTING_LINE = [
    yaml_config['counting']['normalized_coordinates'].get('start', [0.25, 0.5]),
    yaml_config['counting']['normalized_coordinates'].get('end', [0.75, 0.5])
]

# Set the primary counting line based on the use_raw_coordinates flag
COUNTING_LINE = RAW_COUNTING_LINE if USE_RAW_COORDINATES else NORMALIZED_COUNTING_LINE

def get_config(path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation with type conversion
    
    Args:
        path: Dot-separated path to the configuration value (e.g., 'counting.raw_coordinates.start')
        default: Default value if the path doesn't exist
        
    Returns:
        The configuration value at the specified path, or the default value if not found
        
    Example: 
        get_config('counting.raw_coordinates.start', [0, 400])
    """
    parts = path.split('.')
    current = yaml_config
    
    try:
        for part in parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default

def reload_config() -> None:
    """
    Reload configuration from files
    
    This function can be called to reload configuration at runtime
    if configuration files or environment variables change.
    """
    global yaml_config
    
    # Reload environment variables
    load_dotenv(override=True)
    
    # Reload YAML configuration
    with open(config_path, 'r') as config_file:
        yaml_config = yaml.safe_load(config_file)
        
    # Note: Global variables won't be automatically updated
    # You would need to re-import the module to get updated values