# Developer Guide for Traffic Monitoring System

This guide provides technical details and implementation guidelines for developers who want to extend, modify, or customize the Traffic Monitoring System.

## 1. Development Environment Setup

### 1.1 Prerequisites

- Python 3.8+ with pip
- Git
- OpenCV with GPU support (optional but recommended)
- CUDA Toolkit for GPU acceleration (optional)
- MQTT broker (e.g., Mosquitto) for messaging

### 1.2 Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic_monitoring.git
   cd traffic_monitoring
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install development dependencies:
   ```bash
   pip install pytest pytest-cov black flake8
   ```

5. Set up pre-commit hooks (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## 2. Code Structure and Organization

The codebase follows a modular architecture with the following organization:

```
traffic_monitoring/
├── main.py                 # Main application entry point
├── config/
│   └── config.py           # Configuration loading and management
├── services/
│   ├── video_ingestion/    # Video capture service
│   │   └── service.py
│   ├── detection/          # Object detection service
│   │   └── service.py
│   ├── tracking/           # Object tracking service
│   │   └── service.py
│   ├── counting/           # Vehicle counting service
│   │   └── service.py
│   ├── ocr/                # License plate OCR service
│   │   └── service.py
│   └── storage/            # Database storage service
│       └── service.py
├── utils/                  # Utility functions
│   └── model_converter.py  # Converts YOLO models to ONNX
├── models/                 # Model storage directory
├── data/                   # Data storage directory
└── tests/                  # Test cases
```

Each service follows a similar pattern:

1. A main service class that handles the core functionality
2. Initialization with configuration parameters
3. Start/stop methods for lifecycle management
4. A primary processing method that performs the service's main function
5. Helper methods for internal implementation details

## 3. Service Interface Guidelines

To maintain consistency across services, follow these interface guidelines:

### 3.1 Service Class Template

```python
class NewService:
    """
    Service for [describe purpose]
    """
    def __init__(self, param1=None, param2=None):
        """
        Initialize the service
        
        Args:
            param1: Description of parameter
            param2: Description of parameter
        """
        self.param1 = param1 or config.PARAM1
        self.param2 = param2 or config.PARAM2
        
        # Initialize resources
        self.resource = None
        
        # Initialize MQTT client (if needed)
        self.client = mqtt.Client()
        self.mqtt_topic = f"{config.MQTT_TOPIC_PREFIX}/topic_name"
        
        # Try to connect to MQTT broker
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            self.client = None
    
    def start(self):
        """Start the service"""
        # Initialize resources that require explicit startup
        pass
    
    def process_data(self, input_data):
        """
        Process input data
        
        Args:
            input_data: Input data to process
            
        Returns:
            dict: Processing results
        """
        # Process data
        results = {}
        
        # Publish results to MQTT if available
        if self.client:
            self.client.publish(self.mqtt_topic, json.dumps(results))
        
        return results
    
    def stop(self):
        """Stop the service"""
        # Clean up resources
        
        # Disconnect MQTT client
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        
        print("Service stopped")
```

### 3.2 Key Interface Requirements

1. **Configuration**: Use config.py for default values, but allow overriding in constructor
2. **MQTT Integration**: Include MQTT client initialization when appropriate
3. **Error Handling**: Use try/except to handle failures gracefully
4. **Resource Management**: Properly initialize and clean up resources
5. **Documentation**: Include docstrings for all classes and methods

## 4. Extending the System

### 4.1 Adding a New Service

To add a new service to the system:

1. Create a new directory under `services/` with an appropriate name
2. Create a `service.py` file with your service implementation
3. Follow the service interface guidelines
4. Add the service to the main application in `main.py`:
   ```python
   # Import the new service
   from services.new_service.service import NewService
   
   # In TrafficMonitoringApp.__init__
   self.new_service = NewService()
   
   # In TrafficMonitoringApp.start
   # Start the service when appropriate
   
   # In TrafficMonitoringApp._process_frames
   # Call the service in the processing pipeline
   new_service_results = self.new_service.process_data(frame_data)
   
   # In TrafficMonitoringApp.stop
   # Stop the service
   self.new_service.stop()
   ```

### 4.2 Modifying Existing Services

When modifying existing services:

1. Preserve the existing interface to maintain compatibility
2. Use inheritance if you need to extend functionality:
   ```python
   class EnhancedDetectionService(DetectionService):
       """Enhanced detection service with additional capabilities"""
       
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           # Add new properties
       
       def detect(self, frame_data):
           # Call the original method
           results = super().detect(frame_data)
           
           # Enhance the results
           results['enhanced_data'] = self._enhance_detection(results)
           
           return results
       
       def _enhance_detection(self, results):
           # Add enhancement logic
           return enhanced_data
   ```

3. Add unit tests for the modified functionality

### 4.3 Implementing New Features

To implement a new feature:

1. Identify which service(s) need to be modified
2. Make changes with minimal impact on existing functionality
3. Add configuration parameters to `config.py` if needed
4. Update the user interface in `_prepare_visualization()` if applicable
5. Document the feature in the README.md and other documentation

## 5. Working with Models

### 5.1 ONNX Model Format

The system uses ONNX format for models, which provides:

- Cross-platform compatibility
- Hardware acceleration support
- Optimized inference

ONNX models must follow these requirements:

1. Input shape should be compatible with the `ONNXDetector` class
2. Output format should match YOLO's output (can be customized in `_process_output()`)
3. Trained for detecting vehicles and license plates

### 5.2 Training Custom Models

For best results with custom models:

1. Use Ultralytics YOLOv5/v8 for training
2. Train on a dataset with vehicle and license plate annotations
3. Export to PyTorch format (.pt)
4. Convert to ONNX using the provided utility

Example training workflow:
```bash
# Train model with Ultralytics
pip install ultralytics
yolo task=detect mode=train data=path/to/data.yaml model=yolov8n.pt epochs=100

# Export to ONNX
python utils/model_converter.py --model runs/train/exp/weights/best.pt
```

### 5.3 Model Optimization

To optimize models for edge devices:

1. Use model quantization:
   ```bash
   pip install onnx-simplifier onnxruntime-tools
   python -m onnxsim input_model.onnx optimized_model.onnx
   ```

2. Consider INT8 quantization for faster inference:
   ```python
   # In detection/service.py
   sess_options = ort.SessionOptions()
   sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   
   # Use this when creating the session
   self.session = ort.InferenceSession(self.model_path, sess_options=sess_options)
   ```

## 6. Unit Testing

### 6.1 Testing Framework

The system uses pytest for unit testing:

1. Test files should be placed in the `tests/` directory
2. Test file names should follow the pattern `test_*.py`
3. Test function names should follow the pattern `test_*`

### 6.2 Writing Tests

Example test for a service:

```python
# tests/test_detection_service.py
import pytest
import numpy as np
import cv2
from services.detection.service import DetectionService

@pytest.fixture
def detection_service():
    # Create a service instance for testing
    return DetectionService(vehicle_model_path="tests/fixtures/dummy_model.onnx")

@pytest.fixture
def sample_frame_data():
    # Create sample frame data
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        'frame': frame,
        'timestamp': 1234567890.0,
        'frame_id': 42
    }

def test_detect(detection_service, sample_frame_data, monkeypatch):
    # Mock the detector's detect method
    def mock_detect(self, image):
        return [[100, 100, 200, 200, 0.8, 0]]
    
    monkeypatch.setattr(
        "services.detection.service.ONNXDetector.detect", 
        mock_detect
    )
    
    # Test detection
    results = detection_service.detect(sample_frame_data)
    
    # Check results
    assert 'vehicles' in results
    assert len(results['vehicles']) == 1
    assert results['frame_id'] == 42
```

### 6.3 Running Tests

Run tests with:

```bash
pytest
```

For code coverage:

```bash
pytest --cov=traffic_monitoring
```

## 7. Performance Profiling

### 7.1 Profiling Tools

Use these tools to identify performance bottlenecks:

1. **cProfile**: Standard Python profiler
   ```bash
   python -m cProfile -o profile.pstats main.py --source test_video.mp4
   ```

2. **snakeviz**: Visualize profiling results
   ```bash
   pip install snakeviz
   snakeviz profile.pstats
   ```

3. **line_profiler**: Line-by-line profiling
   ```python
   # Add decorators to functions to profile
   @profile
   def function_to_profile():
       pass
   ```

### 7.2 Performance Hotspots

Common performance hotspots to optimize:

1. **Video Decoding**: Use hardware acceleration when available
2. **Model Inference**: Optimize batch size and model complexity
3. **OpenCV Operations**: Minimize image resizing and format conversions
4. **OCR Processing**: Limit OCR to regions with high confidence plate detections
5. **Visualization**: Reduce drawing operations for headless operation

## 8. Deployment

### 8.1 Docker Deployment

The system can be deployed using Docker:

1. Create a Dockerfile:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   RUN apt-get update && apt-get install -y \
       libgl1-mesa-glx \
       libglib2.0-0 \
       && rm -rf /var/lib/apt/lists/*
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   CMD ["python", "main.py", "--no-ui"]
   ```

2. Build the Docker image:
   ```bash
   docker build -t traffic-monitoring .
   ```

3. Run the container:
   ```bash
   docker run -it --name traffic-monitor \
       -v /path/to/models:/app/models \
       -v /path/to/data:/app/data \
       -v /path/to/videos:/app/videos \
       -e VIDEO_SOURCE=/app/videos/traffic.mp4 \
       traffic-monitoring
   ```

### 8.2 Edge Device Deployment

For edge devices like Raspberry Pi or NVIDIA Jetson:

1. Optimize models for the target hardware
2. Use GPU acceleration if available
3. Reduce resolution and frame rate
4. Consider headless operation with `--no-ui`
5. Set up auto-start using systemd services

## 9. Working with MQTT

### 9.1 MQTT Message Structure

Each service publishes structured JSON messages with the following common format:

```json
{
  "frame_id": 42,
  "timestamp": 1234567890.0,
  "data_specific_field": "value"
}
```

### 9.2 Custom Subscribers

To implement custom MQTT subscribers:

```python
import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("traffic_monitoring/detections")
    client.subscribe("traffic_monitoring/plates")

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    print(f"Received message on {msg.topic}: {data}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("localhost", 1883, 60)
client.loop_forever()
```

## 10. Guidelines and Best Practices

### 10.1 Code Style

Follow these guidelines for code consistency:

1. Use PEP 8 style guide
2. Use type hints for clarity
3. Use black for code formatting
4. Document all classes and methods with docstrings
5. Keep functions focused on a single responsibility

### 10.2 Git Workflow

Follow these Git practices:

1. Use feature branches for new features or fixes
2. Write descriptive commit messages
3. Keep commits focused on single changes
4. Create pull requests for code review
5. Squash commits before merging

### 10.3 Error Handling

Follow these error handling practices:

1. Use specific exception types
2. Log exceptions with context
3. Recover gracefully from failures
4. Don't swallow exceptions silently
5. Provide meaningful error messages

## 11. Contributing

We welcome contributions to the Traffic Monitoring System. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

Please ensure your code follows our style guidelines and includes appropriate tests.