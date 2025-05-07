# Traffic Monitoring System

A user-friendly system for detecting, tracking, and counting vehicles in video streams. Designed to work well even on low-powered devices.

![Traffic Monitoring System](docs/images/system_overview.png)

## What This System Does

This system can:

- **Detect Vehicles in Real-Time**: Spot cars, trucks, and other vehicles in video feeds
- **Read License Plates**: Automatically identify and record license plate numbers
- **Track Moving Vehicles**: Follow each vehicle as it moves through the video
- **Count Traffic**: Count vehicles as they cross a line you define on the screen
- **Visualize Results**: Display bounding boxes, vehicle IDs, license plates, and timestamp
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

To process a video file and generate output without displaying UI (batch processing):
```bash
python main.py --source /path/to/input.mp4 --render-video --output /path/to/output.mp4
```

This batch processing mode is especially useful for:
- Processing videos on headless servers
- Generating results for multiple videos without manual intervention
- Converting surveillance footage to annotated videos with detection results

### Command Options

- `--source`: Where to get video from (file path, camera URL, or device number)
- `--no-ui`: Run without showing the video window (good for servers)
- `--record`: Save the output as a new video file
- `--output`: Specify the path where the output video will be saved
- `--render-video`: Process a video file and generate output without displaying the UI (batch processing mode)

### Keyboard Controls

When the system is running:
- Press `q` to quit the program
- Press `r` to reset the vehicle counter to zero

## System Visualization Features

The system provides a rich visual interface that displays:

- **Vehicle Bounding Boxes**: Each detected vehicle is surrounded by a colored box
  - Orange boxes: Vehicles that haven't crossed the counting line
  - Green boxes: Vehicles that have been counted (crossed the line)
- **Vehicle ID**: Each vehicle gets a unique tracking ID
- **License Plates**: When detected, license plate text is displayed
- **Counting Line**: A red line that counts vehicles when they cross it
- **Total Count**: Prominent display showing total vehicles counted
- **Timestamp**: Current date and time displayed on the video
- **FPS Counter**: Shows the processing speed (frames per second)

These visualizations work in both live view mode and in recorded output videos.

## How It Works

The system uses several components that work together:

1. **Video Ingestion**: Gets frames from your video source, preserving original resolution
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

# Model versions - specify which YOLO model format is being used
VEHICLE_MODEL_VERSION=yolo11  # Options: yolov5, yolov8, yolo11
PLATE_MODEL_VERSION=yolov8    # Options: yolov5, yolov8, yolo11

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
USE_RAW_COORDINATES=True  # Use raw pixel coordinates by default

# For counting line using exact pixel positions (default):
COUNTING_LINE_START=320,360
COUNTING_LINE_END=960,360

# For counting line using relative coordinates (0-1):
# USE_RAW_COORDINATES=False
# COUNTING_LINE_START=0.25,0.6
# COUNTING_LINE_END=0.75,0.6

# Video processing settings
VIDEO_SOURCE=0  # 0 means first webcam
FRAME_SKIP=1    # Process every frame (2 would process every other frame)
PROCESS_RESOLUTION=640,480  # Width,height to resize video for processing
```

### Counting Line Explained

The counting line is an invisible line that counts vehicles when they cross it. You can define it in two ways:

1. **Pixel Coordinates** (default): Exact pixel positions on the screen. More precise and intuitive to work with when setting up a line on a specific video.
   
2. **Normalized Coordinates**: Values from 0-1 representing percentages of the screen width/height. Works with any video size but less intuitive to set up.

To switch between them, set `USE_RAW_COORDINATES` to `True` or `False` in your configuration.

## Hardware Acceleration

The system supports GPU acceleration to improve performance. By default, GPU acceleration is disabled. You can enable it in the configuration with minimal changes:

### Quick Start with GPU Acceleration

To enable GPU for all components:

1. Edit `config/settings/config.yaml` and set:
   ```yaml
   hardware:
     use_gpu: true   # Master switch for GPU acceleration
     provider: "auto" # Let the system detect available GPU
     precision: "fp32" # Use FP32 precision (or "fp16" for more speed but less accuracy)
   ```

2. Or simply run with environment variable:
   ```bash
   USE_GPU=true python main.py --source /path/to/video.mp4
   ```

### Platform-Specific Options

The system automatically detects and uses the best available GPU acceleration option for your hardware:

#### NVIDIA GPUs
- Automatically uses CUDA if available
- For manual configuration, use `provider: "cuda"` or `HARDWARE_PROVIDER=cuda`
- For TensorRT acceleration, use `provider: "tensorrt"` (requires TensorRT installation)

#### AMD GPUs
- Automatically uses ROCm if available
- For manual configuration, use `provider: "rocm"` or `HARDWARE_PROVIDER=rocm`

#### Intel GPUs
- Automatically uses OpenVINO if available
- For manual configuration, use `provider: "openvino"` or `HARDWARE_PROVIDER=openvino`

#### Microsoft DirectML (Windows)
- Automatically uses DirectML if available
- For manual configuration, use `provider: "directml"` or `HARDWARE_PROVIDER=directml`

### Advanced Configuration

For more detailed GPU configuration, you can adjust these settings:

1. In `config.yaml`:
   ```yaml
   hardware:
     use_gpu: true
     provider: "cuda"   # Options: "auto", "cuda", "tensorrt", "openvino", "directml", "rocm"
     precision: "fp16"  # Options: "fp32", "fp16" (fp16 is faster but less accurate)
   
   ocr:
     use_gpu: true     # Separate control for OCR GPU usage
   ```

2. Or through environment variables:
   ```bash
   USE_GPU=true HARDWARE_PROVIDER=cuda HARDWARE_PRECISION=fp16 OCR_GPU=true python main.py
   ```

### Troubleshooting GPU Acceleration

If you encounter issues with GPU acceleration:

1. Check if your GPU is detected:
   ```bash
   python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```

2. Make sure you have the necessary drivers installed:
   - NVIDIA: CUDA and cuDNN
   - AMD: ROCm
   - Intel: OpenVINO toolkit
   - Windows: DirectML

3. Try running with specific provider:
   ```bash
   HARDWARE_PROVIDER=cpu python main.py  # Force CPU for troubleshooting
   ```

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