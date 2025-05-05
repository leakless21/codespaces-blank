# Traffic Monitoring System

A user-friendly system for detecting, tracking, and counting vehicles in video streams. Designed to work well even on low-powered devices.

![Traffic Monitoring System](docs/images/system_overview.png)

## What This System Does

This system can:

- **Detect Vehicles in Real-Time**: Spot cars, trucks, and other vehicles in video feeds
- **Read License Plates**: Automatically identify and record license plate numbers
- **Track Moving Vehicles**: Follow each vehicle as it moves through the video
- **Count Traffic**: Count vehicles as they cross a line you define on the screen
- **Store Results**: Save all detection data in a database for later analysis
- **Easy to Extend**: Built with modular components you can customize

## What You'll Need

- **Python 3.8 or newer**: The programming language this system uses
- **OpenCV 4.6+**: Library for computer vision (processing images and video)
- **ONNX Runtime**: Engine that runs the AI models efficiently
- **BoxMOT**: Library for tracking objects across video frames
- **EasyOCR**: Tool for reading text from images (like license plates)
- **MQTT Client**: Messaging system that helps components communicate
- **YOLO models**: Pre-trained AI models for detecting objects

## Getting Started

### Step 1: Get the Code

Download the system files to your computer:
```bash
git clone https://github.com/yourusername/traffic_monitoring.git
cd traffic_monitoring
```

### Step 2: Install Required Software

Install all the libraries and tools the system needs:
```bash
pip install -r requirements.txt
```

### Step 3: Download AI Models

Get the pre-trained AI models the system needs to detect vehicles:
```bash
python utils/download_model.py --model yolo11s  # Downloads the vehicle detection model
```

### Step 4: Convert Models to the Right Format

Convert the models to a format that runs efficiently:
```bash
python utils/model_converter.py --model /path/to/vehicle_model.pt
python utils/model_converter.py --model /path/to/plate_model.pt
```

### Step 5: Set Up Configuration

Create a settings file to customize how the system works:
```bash
cp .env.example .env
nano .env  # Use any text editor you prefer
```

## Using the System

### Basic Commands

To start the system with a video file:
```bash
python main.py --source /path/to/video.mp4
```

To use your webcam as the video source:
```bash
python main.py --source 0
```

### Command Options

- `--source`: Where to get video from (file path, camera URL, or device number)
- `--no-ui`: Run without showing the video window (good for servers)
- `--record`: Save the output as a new video file

### Keyboard Controls

When the system is running:
- Press `q` to quit the program
- Press `r` to reset the vehicle counters to zero

## How It Works

The system uses several components that work together:

1. **Video Ingestion**: Gets frames from your video source
2. **Detection**: Uses AI to find vehicles and license plates
3. **Tracking**: Keeps track of each vehicle as it moves
4. **Counting**: Counts vehicles when they cross your defined line
5. **OCR**: Reads license plate text when detected
6. **Storage**: Saves results to a database
7. **Main Application**: Coordinates all the components and shows results

## Customizing the System

You can easily change how the system works by editing the configuration files. The main settings are in the `.env` file and the YAML configuration.

### Configuration Settings

```bash
# Model file locations
VEHICLE_DETECTION_MODEL=models/vehicle_detection.onnx
PLATE_DETECTION_MODEL=models/plate_detection.onnx

# How confident the AI needs to be (0-1)
DETECTION_CONFIDENCE=0.25
DETECTION_IOU_THRESHOLD=0.45

# Tracking settings
TRACKER_TYPE=bytetrack
TRACKING_CONFIDENCE=0.3

# License plate reading settings
OCR_LANGUAGES=en
OCR_GPU=False

# Vehicle counting settings
USE_RAW_COORDINATES=False

# For counting line using relative coordinates (0-1):
COUNTING_LINE_START=0.25,0.6
COUNTING_LINE_END=0.75,0.6

# For counting line using exact pixel positions:
# USE_RAW_COORDINATES=True
# COUNTING_LINE_START=320,360
# COUNTING_LINE_END=960,360

# Video processing settings
VIDEO_SOURCE=0  # 0 means first webcam
FRAME_SKIP=1    # Process every frame (2 would process every other frame)
PROCESS_RESOLUTION=640,480  # Width,height to resize video for processing
```

### Counting Line Explained

The counting line is an invisible line that counts vehicles when they cross it. You can define it in two ways:

1. **Normalized Coordinates** (default): Values from 0-1 representing percentages of the screen width/height. Works with any video size.
   
2. **Pixel Coordinates**: Exact pixel positions on the screen. More precise but depends on video resolution.

To switch between them, set `USE_RAW_COORDINATES` to `True` or `False` in your configuration.

## System Organization

```
traffic_monitoring/
├── main.py                 # The main program file
├── requirements.txt        # List of required libraries
├── README.md               # This guide
├── config/                 # Configuration files
│   ├── config.py           # Loads settings
│   └── settings/           # YAML configuration files
├── data/                   # Where data gets stored
├── models/                 # AI model files
├── services/               # The main components
│   ├── video_ingestion/    # Gets video from sources
│   ├── detection/          # Detects vehicles and plates
│   ├── tracking/           # Tracks vehicles across frames
│   ├── counting/           # Counts vehicles crossing a line
│   ├── ocr/                # Reads license plates
│   └── storage/            # Saves results to database
├── utils/                  # Helper tools
└── tests/                  # Test programs
```

## Making It Run Faster

If the system is running slowly, you can:

1. Skip frames by increasing `FRAME_SKIP` in the configuration
2. Reduce the processing resolution (e.g., `PROCESS_RESOLUTION=320,240`)
3. Use a GPU if available (set `OCR_GPU=True` for license plate reading)
4. Increase confidence thresholds to detect fewer objects

## For Developers

If you're a developer and want to extend or modify the system, check out the [Developer Guide](docs/developer_guide.md) for detailed technical information.

## License

This project is available under the MIT License - see the LICENSE file for details.

## Credits

This system builds upon these amazing projects:
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- [BoxMOT](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [ONNX Runtime](https://onnxruntime.ai/)