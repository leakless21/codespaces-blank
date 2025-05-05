import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

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

# === Environment-specific settings (from .env) ===

# Model configurations
VEHICLE_DETECTION_MODEL = os.getenv("VEHICLE_DETECTION_MODEL", str(MODELS_DIR / "vehicle_detection.onnx"))
PLATE_DETECTION_MODEL = os.getenv("PLATE_DETECTION_MODEL", str(MODELS_DIR / "plate_detection.onnx"))

# Video settings
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # 0 for webcam, or file path/RTSP URL

# Raw coordinates flag
USE_RAW_COORDINATES = os.getenv("USE_RAW_COORDINATES", "False").lower() == "true"

# MQTT connection settings
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Database settings
DB_URL = os.getenv("DB_URL", f"sqlite:///{BASE_DIR}/data/traffic_data.db")

# === Application settings (from YAML) ===

# Detection settings
DETECTION_CONFIDENCE = yaml_config['detection']['confidence']
DETECTION_IOU_THRESHOLD = yaml_config['detection']['iou_threshold']

# Vehicle class settings - New configuration
ENABLED_VEHICLE_CLASSES = yaml_config['detection']['vehicle_classes']['enabled_classes']
VEHICLE_CLASS_NAMES = yaml_config['detection']['vehicle_classes']['class_names']
COUNT_VEHICLE_CLASSES = yaml_config['detection']['vehicle_classes']['count_classes']

# Tracking settings
TRACKER_TYPE = yaml_config['tracking']['tracker_type']
TRACKING_CONFIDENCE = yaml_config['tracking']['confidence']

# OCR settings
OCR_LANGUAGES = yaml_config['ocr']['languages']
OCR_GPU = yaml_config['ocr']['use_gpu']

# Video processing settings
FRAME_SKIP = yaml_config['video']['frame_skip']
OUTPUT_FPS = yaml_config['video']['output_fps']
PROCESS_RESOLUTION = tuple(yaml_config['video']['process_resolution'])

# MQTT topic settings
MQTT_TOPIC_PREFIX = yaml_config['mqtt']['topic_prefix']

# Counting line settings
if USE_RAW_COORDINATES:
    COUNTING_LINE = [
        yaml_config['counting']['raw_coordinates']['start'],
        yaml_config['counting']['raw_coordinates']['end']
    ]
else:
    COUNTING_LINE = [
        yaml_config['counting']['normalized_coordinates']['start'],
        yaml_config['counting']['normalized_coordinates']['end']
    ]

# Add a helper function to get configuration values
def get_config(path, default=None):
    """
    Get a configuration value using dot notation
    
    Example: get_config('counting.raw_coordinates.start')
    """
    parts = path.split('.')
    current = yaml_config
    
    try:
        for part in parts:
            current = current[part]
        return current
    except (KeyError, TypeError):
        return default