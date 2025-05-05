# Traffic Monitoring System

A modular, scalable vehicle detection and tracking system designed for edge devices.

![Traffic Monitoring System](docs/images/system_overview.png)

## Features

- **Real-time Vehicle Detection**: Detects vehicles using ONNX-optimized models
- **License Plate Detection and OCR**: Reads and recognizes license plates
- **Multi-Object Tracking**: Tracks vehicles across video frames using ByteTrack
- **Vehicle Counting**: Counts vehicles crossing a virtual line
- **Database Storage**: Records detection results and statistics
- **MQTT Integration**: Communicates between services using message queues
- **Modular Architecture**: Easy to extend or replace individual components

## Requirements

- Python 3.8+
- OpenCV 4.6+
- ONNX Runtime
- BoxMOT
- EasyOCR
- MQTT Client
- YOLO models (for conversion to ONNX)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic_monitoring.git
   cd traffic_monitoring
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Convert YOLO models to ONNX format:
   ```bash
   python utils/model_converter.py --model /path/to/vehicle_model.pt
   python utils/model_converter.py --model /path/to/plate_model.pt
   ```

4. Configure the system by editing `.env` file (create if not exists):
   ```bash
   cp .env.example .env
   nano .env
   ```

## Usage

### Basic Usage

```bash
python main.py --source /path/to/video.mp4
```

### Command Line Arguments

- `--source`: Video source (file path, RTSP URL, or device ID)
- `--no-ui`: Disable UI display
- `--record`: Record output video

### User Interface Controls

- Press `q` to quit
- Press `r` to reset counters

## System Architecture

The system consists of the following components:

1. **Video Ingestion Service**: Captures video frames from source
2. **Detection Service**: Detects vehicles and license plates using ONNX models
3. **Tracking Service**: Tracks vehicles across frames using BoxMOT
4. **Counting Service**: Counts vehicles crossing a user-defined line
5. **OCR Service**: Reads license plates using EasyOCR
6. **Storage Service**: Stores data in a SQLite database
7. **Main Application**: Orchestrates the services and provides visualization

## Configuration

Configuration parameters are defined in `.env` file and loaded through `config/config.py`.

```bash
# Model configurations
VEHICLE_DETECTION_MODEL=models/vehicle_detection.onnx
PLATE_DETECTION_MODEL=models/plate_detection.onnx

# Detection settings
DETECTION_CONFIDENCE=0.25
DETECTION_IOU_THRESHOLD=0.45

# Tracking settings
TRACKER_TYPE=bytetrack
TRACKING_CONFIDENCE=0.3

# OCR settings
OCR_LANGUAGES=en
OCR_GPU=False

# Video settings
VIDEO_SOURCE=0
FRAME_SKIP=1
PROCESS_RESOLUTION=640,480

# MQTT settings
MQTT_BROKER=localhost
MQTT_PORT=1883
MQTT_TOPIC_PREFIX=traffic_monitoring
```

## Directory Structure

```
traffic_monitoring/
├── main.py                 # Main application
├── requirements.txt        # Dependencies
├── README.md               # Documentation
├── config/                 # Configuration files
├── data/                   # Data storage
├── models/                 # ONNX models
├── services/               # Service modules
│   ├── video_ingestion/    # Video capture service
│   ├── detection/          # ONNX detection service
│   ├── tracking/           # BoxMOT tracking service
│   ├── counting/           # Vehicle counting service
│   ├── ocr/                # License plate OCR service
│   └── storage/            # Database storage service
├── utils/                  # Utility scripts
└── tests/                  # Unit tests
```

## Extending the System

See the [Developer Documentation](docs/developer_guide.md) for details on extending the system with new services or modifying existing ones.

## Performance Optimization

Performance can be optimized by:

1. Adjusting `FRAME_SKIP` to process fewer frames
2. Reducing `PROCESS_RESOLUTION` for faster processing
3. Using GPU-accelerated inference where available
4. Adjusting confidence thresholds for faster detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [BoxMOT](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [ONNX Runtime](https://onnxruntime.ai/)