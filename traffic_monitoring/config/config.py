import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Model configurations
VEHICLE_DETECTION_MODEL = os.getenv("VEHICLE_DETECTION_MODEL", str(MODELS_DIR / "vehicle_detection.onnx"))
PLATE_DETECTION_MODEL = os.getenv("PLATE_DETECTION_MODEL", str(MODELS_DIR / "plate_detection.onnx"))

# Detection settings
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.25"))
DETECTION_IOU_THRESHOLD = float(os.getenv("DETECTION_IOU_THRESHOLD", "0.45"))

# Tracking settings
TRACKER_TYPE = os.getenv("TRACKER_TYPE", "bytetrack")
TRACKING_CONFIDENCE = float(os.getenv("TRACKING_CONFIDENCE", "0.3"))

# OCR settings
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "en").split(",")
OCR_GPU = os.getenv("OCR_GPU", "False").lower() == "true"

# Counting settings
COUNTING_LINE = [
    [float(x) for x in os.getenv("COUNTING_LINE_START", "0.25,0.6").split(",")],
    [float(x) for x in os.getenv("COUNTING_LINE_END", "0.75,0.6").split(",")]
]

# Video settings
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # 0 for webcam, or file path/RTSP URL
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "1"))
OUTPUT_FPS = int(os.getenv("OUTPUT_FPS", "20"))
PROCESS_RESOLUTION = tuple(map(int, os.getenv("PROCESS_RESOLUTION", "640,480").split(",")))

# MQTT settings for messaging
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "traffic_monitoring")

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Database settings
DB_URL = os.getenv("DB_URL", f"sqlite:///{BASE_DIR}/data/traffic_data.db")