# Developer Guide for Traffic Monitoring System

This guide helps developers who want to extend, modify, or customize the Traffic Monitoring System. While the README provides basic usage instructions, this guide dives deeper into the technical aspects of how the system works and how to modify it.

## 1. Setting Up Your Development Environment

### 1.1 What You'll Need Before Starting

- **Python 3.8 or newer**: The core programming language
- **Git**: For version control and downloading the code
- **OpenCV with GPU support** (optional but recommended): For faster image processing
- **CUDA Toolkit** (optional): For GPU acceleration if you have an NVIDIA graphics card
- **MQTT broker** (e.g., Mosquitto): For message passing between system components

### 1.2 Step-by-Step Setup

1. **Get the code**:
   ```bash
   git clone https://github.com/yourusername/traffic_monitoring.git
   cd traffic_monitoring
   ```

2. **Create a virtual environment** (keeps dependencies organized):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional tools for development**:
   ```bash
   pip install pytest pytest-cov black flake8
   ```

5. **Set up code quality tools** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```
   This helps maintain code quality by automatically checking your code when you commit changes.

## 2. Understanding the Code Structure

The system uses a modular design where each part handles a specific task and can be replaced or modified independently.

### 2.1 Project Layout

```
traffic_monitoring/
├── main.py                 # Entry point that ties everything together
├── config/
│   ├── config.py           # Loads and manages configuration
│   └── settings/           # YAML configuration files 
├── services/
│   ├── video_ingestion/    # Gets video frames from sources
│   ├── detection/          # Finds objects using AI models
│   ├── tracking/           # Follows objects across frames
│   ├── counting/           # Counts objects crossing a line
│   ├── ocr/                # Reads text from images
│   └── storage/            # Saves data to database
├── utils/                  # Helper functions and tools
│   └── model_converter.py  # Converts AI models to ONNX format
├── models/                 # Where AI models are stored
├── data/                   # Where data is stored
└── tests/                  # Automated tests
```

### 2.2 How the Services Work Together

Each service follows a similar pattern:

1. **Initialization**: Sets up resources and configuration when created
2. **Starting/Stopping**: Methods to begin and end operation properly
3. **Processing**: The main function that processes input data
4. **Helper Methods**: Internal functions that support the main processing

The services form a pipeline where:
- Video frames come in from the video ingestion service
- Each frame passes through detection, tracking, counting, and OCR
- Results are stored in the database
- The main application coordinates and visualizes everything

### 2.3 Recent Updates to the System

The system has been updated to address several issues:

1. **Fixed Video Resolution and Speed Issues**:
   - Modified video ingestion to preserve original frame resolution
   - Updated video writing to match source video FPS
   - Improved visualization to scale properly with any resolution

2. **Enhanced Visualization**:
   - Added vehicle bounding boxes with colored indicators
   - Added unique vehicle ID display
   - Added timestamp display
   - Added license plate detection and display
   - Improved readability of text elements with background blocks

3. **Simplified Counting Logic**:
   - Removed directional counting (up/down) in favor of total count only
   - Simplified the counting interface in both code and UI

## 3. Service Design Guidelines

When working with existing services or creating new ones, follow these patterns to maintain consistency.

### 3.1 Service Class Template

Here's a simplified template showing how services are structured:

```python
class MyService:
    """
    Service for [describe what it does]
    """
    def __init__(self, param1=None, param2=None):
        """
        Initialize the service
        
        Args:
            param1: What this parameter does
            param2: What this parameter does
        """
        # Get values from parameters or fall back to config
        self.param1 = param1 or config.PARAM1
        self.param2 = param2 or config.PARAM2
        
        # Initialize any resources needed
        self.resource = None
        
        # Set up MQTT messaging if needed
        self.client = mqtt.Client()
        self.mqtt_topic = f"{config.MQTT_TOPIC_PREFIX}/my_topic"
        
        # Try to connect to MQTT broker
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            self.client = None
    
    def start(self):
        """Start the service and initialize resources"""
        # Any startup steps go here
        pass
    
    def process_data(self, input_data):
        """
        Process input data and return results
        
        Args:
            input_data: The data to process
            
        Returns:
            dict: Results of processing
        """
        # Main processing logic
        results = {}
        
        # Send results via MQTT if available
        if self.client:
            self.client.publish(self.mqtt_topic, json.dumps(results))
        
        return results
    
    def stop(self):
        """Properly shut down the service and free resources"""
        # Clean up any resources
        
        # Disconnect from MQTT
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        
        print("Service stopped")
```

### 3.2 Key Design Principles

When building or modifying services:

1. **Use Configuration Properly**: Get defaults from config.py, but allow overriding when creating the service
2. **Handle Messaging Correctly**: Use MQTT for loose coupling between services
3. **Manage Errors Gracefully**: Use try/except blocks to prevent crashes
4. **Clean Up Resources**: Always provide a stop() method that properly releases resources
5. **Document Everything**: Add clear comments to explain what your code does

## 4. How to Extend the System

### 4.1 Adding a New Service

To create an entirely new service:

1. **Create a Directory**: Make a new folder under `services/` with a descriptive name
2. **Create the Implementation**: Create a `service.py` file with your service class
3. **Follow the Template**: Structure your service similar to the existing ones
4. **Add It to the Pipeline**: Update the main application to include your service:

   ```python
   # In main.py - Import your service
   from services.your_service.service import YourService
   
   # Inside the TrafficMonitoringApp.__init__ method
   self.your_service = YourService()
   
   # Start your service in the appropriate place
   self.your_service.start()
   
   # Use it in the processing pipeline
   your_results = self.your_service.process_data(input_data)
   
   # Stop it when the application stops
   self.your_service.stop()
   ```

### 4.2 Enhancing Existing Services

To improve or change existing services:

1. **Maintain Compatibility**: Keep the existing interface so other components can still use it
2. **Use Inheritance** for major changes:

   ```python
   class BetterDetectionService(DetectionService):
       """Enhanced version of the detection service"""
       
       def __init__(self, *args, **kwargs):
           # First call the original initialization
           super().__init__(*args, **kwargs)
           # Then add your own new features
           self.additional_feature = True
       
       def detect(self, frame_data):
           # Call the original detection method
           results = super().detect(frame_data)
           
           # Add your enhancements
           results['enhanced_data'] = self._enhance_detection(results)
           
           return results
       
       def _enhance_detection(self, results):
           # Your new logic here
           return enhanced_data
   ```

3. **Add Tests**: Create unit tests for your changes to ensure they work correctly

### 4.3 Implementing New Features

When adding new functionality:

1. **Identify Target Services**: Figure out which service(s) need to change
2. **Make Minimal Changes**: Modify only what's necessary
3. **Update Configuration**: Add any new settings to `config.py` and config files
4. **Update the UI**: If needed, change visualization in `_prepare_visualization()`
5. **Document the Feature**: Update README.md and other documentation

### 4.4 Working with Counting Line Coordinates

The system offers two ways to specify the counting line:

#### Raw Pixel Coordinates (Default)

This method uses exact pixel positions on the screen. This gives more precise control and is more intuitive when setting up counting lines for specific videos.

```python
# In .env file or YAML config:
USE_RAW_COORDINATES=True
COUNTING_LINE_START=320,360  # X=320, Y=360 pixels
COUNTING_LINE_END=960,360    # X=960, Y=360 pixels

# In code:
counting_service = CountingService(
    counting_line=[[320, 360], [960, 360]], 
    use_raw_coordinates=True
)
```

#### Normalized Coordinates

This alternative method uses values between 0 and 1, representing the percentage of the screen's width and height. The advantage is that it works with any video resolution, but it can be less intuitive to set up.

```python
# In .env file or YAML config:
USE_RAW_COORDINATES=False
COUNTING_LINE_START=0.25,0.6  # 25% from left, 60% from top
COUNTING_LINE_END=0.75,0.6    # 75% from left, 60% from top

# In code:
counting_service = CountingService(
    counting_line=[[0.25, 0.6], [0.75, 0.6]], 
    use_raw_coordinates=False
)
```

#### How it Works

The `CountingService` handles both coordinate types internally:

```python
# Inside the CountingService update method:
if not self.use_raw_coordinates:
    # Convert normalized (0-1) coordinates to actual pixels
    line = [
        [int(self.counting_line[0][0] * width), int(self.counting_line[0][1] * height)],
        [int(self.counting_line[1][0] * width), int(self.counting_line[1][1] * height)]
    ]
else:
    # Use raw coordinates directly
    line = [
        [int(self.counting_line[0][0]), int(self.counting_line[0][1])],
        [int(self.counting_line[1][0]), int(self.counting_line[1][1])]
    ]
```

This gives you flexibility to choose the approach that works best for your use case.

### 4.5 Updating the Visual Interface

To modify the visualization elements (bounding boxes, text, etc.), you need to update the `_prepare_visualization` method in `main.py`. The system now handles original resolution rendering with coordinate scaling.

Here's a guide to updating visualization elements:

1. **Access the correct frame**:
   ```python
   # Use original frame for visualization if available
   if 'original_frame' in frame_data:
       # For high resolution output
       orig_frame = frame_data['original_frame'].copy()
       frame = frame_data['frame'].copy()
       
       # Calculate scaling factors
       scale_x = orig_frame.shape[1] / frame.shape[1]
       scale_y = orig_frame.shape[0] / frame.shape[0]
       
       # Function to scale coordinates
       def scale_coords(coords):
           if isinstance(coords, tuple) or isinstance(coords, list):
               if len(coords) == 2:  # Single point (x,y)
                   return (int(coords[0] * scale_x), int(coords[1] * scale_y))
               elif len(coords) == 4:  # Box (x1,y1,x2,y2)
                   return (int(coords[0] * scale_x), int(coords[1] * scale_y), 
                           int(coords[2] * scale_x), int(coords[3] * scale_y))
           return coords
   else:
       # Fallback to processed frame if original not available
       frame = frame_data['frame'].copy()
   ```

2. **Draw object boxes with scaling**:
   ```python
   # Example for drawing a box on original resolution
   x1, y1, x2, y2 = scale_coords(track['box'])
   cv2.rectangle(orig_frame, (x1, y1), (x2, y2), color, 2)
   ```

3. **Add text with background for readability**:
   ```python
   # Example for adding text with background
   text = f"ID:{track_id}"
   text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
   
   # Draw black background
   cv2.rectangle(orig_frame, 
                (x1, y1 - text_size[1] - 5),
                (x1 + text_size[0] + 5, y1), 
                (0, 0, 0), -1)
                
   # Draw white text
   cv2.putText(orig_frame, text, (x1 + 2, y1 - 5),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
   ```

4. **Add transparent overlays**:
   ```python
   # Create a copy for overlay
   overlay = orig_frame.copy()
   
   # Draw filled rectangle
   cv2.rectangle(overlay, (10, 10), (300, 70), (0, 0, 0), -1)
   
   # Blend with original with 60% transparency
   cv2.addWeighted(overlay, 0.6, orig_frame, 0.4, 0, orig_frame)
   
   # Draw text on top of overlay
   cv2.putText(orig_frame, f"TOTAL COUNT: {counts['total']}", (20, 50),
              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
   ```

## 5. Working with AI Models

### 5.1 Understanding the ONNX Format

The system uses ONNX (Open Neural Network Exchange) format for its AI models because it offers:

- **Cross-platform support**: Works on different operating systems and hardware
- **Hardware acceleration**: Can use GPUs and specialized processors
- **Optimized speed**: Faster inference than other formats

When working with ONNX models, remember:

1. Input shapes must match what the `ONNXDetector` class expects
2. Output format should align with YOLO's detection format
3. Models need to be trained for the specific objects you want to detect

### 5.2 Working with Different YOLO Model Versions

The system supports multiple YOLO model versions (YOLOv5, YOLOv8, YOLO11) through a flexible configuration system.

#### Configuring Model Versions

You can specify which YOLO model version you're using in two ways:

1. **Via YAML configuration**:
   ```yaml
   # In config/settings/config.yaml
   detection:
     model_versions:
       vehicle_model: "yolo11"  # Options: "yolov5", "yolov8", "yolo11"
       plate_model: "yolov8"    # Options: "yolov5", "yolov8", "yolo11"
   ```

2. **Via environment variables**:
   ```bash
   # In .env file or command line
   VEHICLE_MODEL_VERSION=yolo11
   PLATE_MODEL_VERSION=yolov8
   ```

#### How Model Version Selection Works

The detection service passes the model version to the `ONNXDetector` class, which then handles the specific output format processing:

```python
# In DetectionService.__init__:
self.vehicle_detector = ONNXDetector(
    model_path=vehicle_model_path or config.VEHICLE_DETECTION_MODEL,
    confidence_threshold=config.DETECTION_CONFIDENCE,
    iou_threshold=config.DETECTION_IOU_THRESHOLD,
    enabled_classes=config.ENABLED_VEHICLE_CLASSES,
    model_version=config.VEHICLE_MODEL_VERSION  # Pass model version
)
```

#### Adding Support for New YOLO Versions

If you need to add support for a new YOLO version:

1. **Add the version to the configuration options**:
   ```yaml
   model_versions:
     vehicle_model: "yolov9"  # Add new version option
   ```

2. **Create a dedicated processing method** in `ONNXDetector`:
   ```python
   def _process_yolov9_output(self, output, original_width, original_height):
       """Process YOLOv9 model output format"""
       # Implement processing logic for the new format
       # ...
       return detections
   ```

3. **Update the model version handling** in `_process_output`:
   ```python
   if self.model_version.lower() == 'yolov9':
       return self._process_yolov9_output(output, original_width, original_height)
   ```

#### Automatic Format Detection

Even without specifying a model version, the system attempts to auto-detect the format based on output shape and dimensions. However, explicit configuration is recommended for reliability.

When auto-detecting:
- 3D outputs (shape [1, boxes, features]) are identified as YOLO11-like
- Outputs with more than 6 features per detection are processed as YOLOv8-like
- Other formats are processed with a general approach

### 5.3 Creating Your Own Models

To train custom models for different objects:

1. **Use Ultralytics YOLOv5/v8**: This is a popular and easy-to-use framework
2. **Prepare a dataset**: Collect and label images of the objects you want to detect
3. **Train the model**: Use the YOLOv5/v8 training process
4. **Export to PyTorch**: Save the model in .pt format
5. **Convert to ONNX**: Use our conversion utility

Example workflow:
```bash
# Install Ultralytics
pip install ultralytics

# Train a model (simplified example)
yolo task=detect mode=train data=path/to/data.yaml model=yolov8n.pt epochs=100

# Convert to ONNX format
python utils/model_converter.py --model runs/train/exp/weights/best.pt
```

### 5.4 Making Models Run Faster

For better performance, especially on less powerful devices:

1. **Simplify the model**:
   ```bash
   pip install onnx-simplifier onnxruntime-tools
   python -m onnxsim input_model.onnx optimized_model.onnx
   ```

2. **Use INT8 quantization** (reduces precision but increases speed):
   ```python
   # When creating the ONNX session:
   sess_options = ort.SessionOptions()
   sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   
   session = ort.InferenceSession(model_path, sess_options=sess_options)
   ```

## 6. Testing Your Changes

### 6.1 Using the Testing Framework

The system uses pytest for automated testing:

1. **Test files** go in the `tests/` directory
2. **Name test files** with the pattern `test_*.py`
3. **Name test functions** with the pattern `test_*`

### 6.2 Writing Good Tests

Here's an example test for a service:

```python
# tests/test_detection_service.py
import pytest
import numpy as np
import cv2
from services.detection.service import DetectionService

# This creates a detection service for testing
@pytest.fixture
def detection_service():
    return DetectionService(vehicle_model_path="tests/fixtures/dummy_model.onnx")

# This creates sample data for testing
@pytest.fixture
def sample_frame_data():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        'frame': frame,
        'timestamp': 1234567890.0,
        'frame_id': 42
    }

# This tests the detect method
def test_detect(detection_service, sample_frame_data, monkeypatch):
    # Replace the real detector with a mock version
    def mock_detect(self, image):
        return [[100, 100, 200, 200, 0.8, 0]]
    
    monkeypatch.setattr(
        "services.detection.service.ONNXDetector.detect", 
        mock_detect
    )
    
    # Run the detection
    results = detection_service.detect(sample_frame_data)
    
    # Check if results are as expected
    assert 'vehicles' in results
    assert len(results['vehicles']) == 1
    assert results['frame_id'] == 42
```

### 6.3 Running the Tests

To run all tests:

```bash
pytest
```

To check code coverage (how much of your code is tested):

```bash
pytest --cov=traffic_monitoring
```

## 7. Performance Optimization

### 7.1 Tools for Finding Bottlenecks

Use these tools to identify slow parts of your code:

1. **cProfile**: Built-in Python profiler
   ```bash
   python -m cProfile -o profile.pstats main.py --source test_video.mp4
   ```

2. **snakeviz**: Visual profiling results viewer
   ```bash
   pip install snakeviz
   snakeviz profile.pstats
   ```

3. **line_profiler**: Line-by-line profiling
   ```python
   # Add @profile decorator to functions you want to profile
   @profile
   def function_to_profile():
       pass
   ```

### 7.2 Common Performance Issues and Solutions

Focus on these areas for the biggest improvements:

1. **Video Processing**: 
   - Use hardware acceleration for video decoding
   - Reduce frame resolution or skip frames

2. **AI Model Performance**: 
   - Use smaller models or quantized models
   - Adjust batch size for inference

3. **Image Operations**: 
   - Minimize resizing and format conversions
   - Use NumPy vectorized operations instead of loops

4. **OCR Performance**: 
   - Only run OCR on high-confidence license plate regions
   - Reduce OCR processing frequency

5. **UI and Visualization**: 
   - Simplify the visualization for faster rendering
   - Use headless mode when UI isn't needed

### 7.3 Hardware Acceleration Implementation

The system implements hardware acceleration using ONNX Runtime's provider system. Here's how it works:

#### Provider Selection Logic

```python
# In ONNXDetector.__init__, we configure hardware acceleration:
if config.USE_GPU:
    # Auto-detection of providers
    if config.HARDWARE_PROVIDER == "auto":
        available_providers = ort.get_available_providers()
        
        # Prioritize providers in this order
        if 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        elif 'DmlExecutionProvider' in available_providers:
            providers.append('DmlExecutionProvider')
        elif 'ROCMExecutionProvider' in available_providers:
            providers.append('ROCMExecutionProvider')
        elif 'OpenVINOExecutionProvider' in available_providers:
            providers.append('OpenVINOExecutionProvider')
    else:
        # Manual provider selection
        providers.append(f'{config.HARDWARE_PROVIDER.capitalize()}ExecutionProvider')
        
    # Always add CPU as fallback
    providers.append('CPUExecutionProvider')
```

#### Precision Control

```python
# FP16 precision for faster inference with CUDA
if config.HARDWARE_PRECISION == "fp16" and config.HARDWARE_PROVIDER == "cuda":
    provider_options = [{
        'device_id': 0,
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True
    }]
```

#### Error Handling

```python
# Graceful fallback to CPU if GPU acceleration fails
try:
    session = ort.InferenceSession(
        model_path, 
        sess_options=session_options,
        providers=providers,
        provider_options=provider_options
    )
except Exception as e:
    print(f"Error loading model with hardware acceleration: {e}")
    print("Falling back to CPU execution")
    session = ort.InferenceSession(model_path)
```

#### Adding Support for New Providers

To add support for a new hardware acceleration platform:

1. First check if ONNX Runtime supports it by running:
   ```python
   import onnxruntime as ort
   print(ort.get_all_providers())  # Shows all providers compiled in
   ```

2. Add the new provider to the auto-detection logic:
   ```python
   if 'NewProviderExecutionProvider' in available_providers:
       providers.append('NewProviderExecutionProvider')
       provider_options.append({})  # Add specific options if needed
   ```

3. Add it to the config.yaml documentation:
   ```yaml
   hardware:
     provider: "auto"  # Options: "auto", "cuda", "tensorrt", "openvino", "directml", "rocm", "newprovider"
   ```

#### Platform-Specific Optimizations

**NVIDIA (CUDA) Specific Settings:**
```python
# TensorRT for maximum NVIDIA performance
if config.HARDWARE_PROVIDER == "tensorrt":
    provider_options = [{
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
        'trt_fp16_enable': config.HARDWARE_PRECISION == "fp16"
    }]
```

**Intel (OpenVINO) Specific Settings:**
```python
# OpenVINO for Intel CPUs, GPUs, and accelerators
if config.HARDWARE_PROVIDER == "openvino":
    provider_options = [{
        'device_type': 'GPU'  # Can be 'CPU', 'GPU', 'VPU', etc.
    }]
```

**AMD (ROCm) Specific Settings:**
```python
# ROCm for AMD GPUs
if config.HARDWARE_PROVIDER == "rocm":
    provider_options = [{
        'device_id': 0
    }]
```

#### Testing Acceleration Performance

To benchmark different hardware acceleration options:

```python
def benchmark_providers(model_path, input_size=(1, 3, 640, 640), iterations=100):
    """Benchmark different ONNX Runtime providers with a model"""
    providers = ort.get_available_providers()
    results = {}
    
    # Create dummy input
    dummy_input = np.random.rand(*input_size).astype(np.float32)
    
    # Test each provider
    for provider in providers:
        try:
            # Create session with just this provider
            session = ort.InferenceSession(
                model_path, 
                providers=[provider, 'CPUExecutionProvider']
            )
            
            # Get input name
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(5):
                session.run(None, {input_name: dummy_input})
            
            # Benchmark
            start = time.time()
            for _ in range(iterations):
                session.run(None, {input_name: dummy_input})
            end = time.time()
            
            # Calculate performance
            avg_time = (end - start) / iterations
            results[provider] = {
                'avg_time': avg_time,
                'fps': 1.0 / avg_time
            }
            
            print(f"{provider}: {avg_time:.4f}s per inference, {results[provider]['fps']:.2f} FPS")
            
        except Exception as e:
            print(f"Error testing {provider}: {e}")
    
    return results
```

Add this utility to your project and use it to determine the most effective hardware acceleration for your specific deployment.

## 8. Deployment Options

### 8.1 Using Docker

Docker containers make deployment easier:

1. **Create a Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       libglib2.0-0 \
       && rm -rf /var/lib/apt/lists/*
   
   # Install Python dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Run the application
   CMD ["python", "main.py", "--no-ui"]
   ```

2. **Build the container**:
   ```bash
   docker build -t traffic-monitoring .
   ```

3. **Run the container**:
   ```bash
   docker run -it --name traffic-monitor \
       -v /path/to/models:/app/models \
       -v /path/to/data:/app/data \
       -v /path/to/videos:/app/videos \
       -e VIDEO_SOURCE=/app/videos/traffic.mp4 \
       traffic-monitoring
   ```

### 8.2 Deploying on Edge Devices

For devices like Raspberry Pi or NVIDIA Jetson:

1. **Optimize for limited resources**:
   - Use smaller models
   - Lower resolution and framerate
   - Disable features you don't need

2. **Enable hardware acceleration**:
   - Use OpenCV with GPU support
   - Enable OpenVINO for Intel devices
   - Use TensorRT for NVIDIA devices

3. **Run headless mode**:
   - Use `--no-ui` flag to avoid GUI
   - Publish results through MQTT or API

4. **Set up automatic startup**:
   ```bash
   # Create a systemd service file
   sudo nano /etc/systemd/system/traffic-monitor.service
   
   # With content like:
   [Unit]
   Description=Traffic Monitoring System
   After=network.target
   
   [Service]
   User=your_username
   WorkingDirectory=/path/to/traffic_monitoring
   ExecStart=/path/to/python /path/to/traffic_monitoring/main.py --no-ui
   Restart=on-failure
   
   [Install]
   WantedBy=multi-user.target
   
   # Enable and start the service
   sudo systemctl enable traffic-monitor
   sudo systemctl start traffic-monitor
   ```

## 9. Working with MQTT Messaging

### 9.1 Message Format

Services exchange information using JSON messages with this structure:

```json
{
  "frame_id": 42,
  "timestamp": 1234567890.0,
  "data_specific_field": "value"
}
```

Common topics include:
- `traffic_monitoring/detections`: Vehicle detection results
- `traffic_monitoring/tracks`: Vehicle tracking information
- `traffic_monitoring/counts`: Vehicle counting statistics
- `traffic_monitoring/plates`: License plate recognition results

### 9.2 Creating Custom Subscribers

You can create external applications that subscribe to these messages:

```python
import paho.mqtt.client as mqtt
import json

# Called when connecting to the broker
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("traffic_monitoring/detections")
    client.subscribe("traffic_monitoring/plates")

# Called when a message is received
def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    print(f"Received message on {msg.topic}: {data}")

# Set up the client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Connect and start listening
client.connect("localhost", 1883, 60)
client.loop_forever()
```

This allows you to build dashboards, alerts, or integrations with other systems.

## 10. Coding Standards and Best Practices

### 10.1 Code Style

Follow these guidelines for consistent code:

1. **PEP 8**: Standard Python style guide
2. **Type Hints**: Add type annotations to make code clearer
   ```python
   def process_image(image: np.ndarray) -> dict:
       """Process an image and return results"""
   ```
3. **Formatting**: Use black to automatically format code
4. **Documentation**: Add docstrings to all classes and methods
5. **Function Size**: Keep functions small and focused on one task

### 10.2 Git Workflow

Follow these practices for version control:

1. **Use Feature Branches**: Create a branch for each new feature or fix
   ```bash
   git checkout -b feature/new-detection-algorithm
   ```
2. **Write Clear Commit Messages**: Explain what and why, not how
3. **Make Focused Commits**: Each commit should have a single purpose
4. **Review Code**: Use pull requests for code review
5. **Merge Strategy**: Squash commits before merging to keep history clean

### 10.3 Error Handling

Handle errors properly:

1. **Use Specific Exceptions**: Catch only the exceptions you expect
   ```python
   try:
       file = open(filename, 'r')
   except FileNotFoundError:
       # Handle missing file
   except PermissionError:
       # Handle permission issues
   ```
2. **Add Context to Errors**: Include helpful information in error messages
3. **Recover When Possible**: Try to continue operation after errors
4. **Don't Hide Errors**: Log errors properly instead of silently ignoring them
5. **Clean Up Resources**: Use try-finally or context managers to ensure cleanup

## 11. Contributing to the Project

We welcome contributions. Here's how to contribute effectively:

1. **Fork the Repository**: Create your own copy on GitHub
2. **Create a Feature Branch**: Make changes in a new branch
3. **Follow the Guidelines**: Adhere to the coding standards
4. **Write Tests**: Add tests for new features or fixes
5. **Submit a Pull Request**: Request to merge your changes
6. **Respond to Feedback**: Address review comments

Please ensure your code follows our style guidelines and includes appropriate tests.

## 12. Troubleshooting Common Issues

### Model Loading Problems
- Check file paths are correct
- Verify ONNX model format is compatible
- Ensure dependencies are installed

### Detection Issues
- Adjust confidence thresholds
- Check input video quality
- Verify model is appropriate for your objects

### Performance Problems
- Profile to find bottlenecks
- Reduce resolution or framerate
- Use hardware acceleration if available
- Consider lighter models or algorithms

### Visual Output Issues
- Video appears resized: Check the video ingestion service's process_resolution setting
- Speed issues: Ensure output_fps matches the source video's FPS
- Visualization scaling: Make sure the coordinate scaling is correct when rendering
- Missing elements: Verify all visual elements are being added in _prepare_visualization
- Text readability: Add contrasting backgrounds to text elements for better visibility

### Counting Issues
- Vehicle not counted: Check the counting line placement and detection confidence
- Multiple counts: Ensure each tracked object is only counted once by checking the counted_tracks
- Incorrect counts: Reset the counter with the 'r' key or modify the counting logic as needed