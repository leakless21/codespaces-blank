import cv2
import numpy as np
import onnxruntime as ort
import time
import paho.mqtt.client as mqtt
import json
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class ONNXDetector:
    """
    ONNX-based object detector for vehicles and license plates
    """
    def __init__(self, model_path=None, confidence_threshold=None, iou_threshold=None, enabled_classes=None, model_version=None):
        """
        Initialize the ONNX detector
        
        Args:
            model_path (str): Path to the ONNX model
            confidence_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for non-maximum suppression
            enabled_classes (list): List of class IDs to detect, None for all classes
            model_version (str): YOLO model version ("yolov5", "yolov8", "yolo11")
        """
        self.model_path = model_path or config.VEHICLE_DETECTION_MODEL
        self.confidence_threshold = confidence_threshold or config.DETECTION_CONFIDENCE
        self.iou_threshold = iou_threshold or config.DETECTION_IOU_THRESHOLD
        self.enabled_classes = enabled_classes
        self.model_version = model_version  # Store model version for processing
        
        # Check if model file exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Configure session options with hardware acceleration settings
        session_options = ort.SessionOptions()
        
        # Set up hardware acceleration based on configuration
        providers = []
        provider_options = []
        
        if config.USE_GPU:
            # Set provider based on configuration or auto-detect
            if config.HARDWARE_PROVIDER == "auto":
                # Auto-detect available providers
                available_providers = ort.get_available_providers()
                print(f"Available ONNX Runtime providers: {available_providers}")
                
                # Prioritize providers (CUDA > DirectML > ROCm > OpenVINO > CPU)
                if 'CUDAExecutionProvider' in available_providers:
                    providers.append('CUDAExecutionProvider')
                    provider_options.append({})
                elif 'DmlExecutionProvider' in available_providers:
                    providers.append('DmlExecutionProvider')
                    provider_options.append({})
                elif 'ROCMExecutionProvider' in available_providers:
                    providers.append('ROCMExecutionProvider')
                    provider_options.append({})
                elif 'OpenVINOExecutionProvider' in available_providers:
                    providers.append('OpenVINOExecutionProvider')
                    provider_options.append({})
            else:
                # Use specific provider from configuration
                if config.HARDWARE_PROVIDER == "cuda":
                    providers.append('CUDAExecutionProvider')
                    # Configure CUDA options for half precision if requested
                    if config.HARDWARE_PRECISION == "fp16":
                        provider_options.append({'device_id': 0, 'gpu_mem_limit': 2 * 1024 * 1024 * 1024, 'arena_extend_strategy': 'kNextPowerOfTwo', 'cudnn_conv_algo_search': 'EXHAUSTIVE', 'do_copy_in_default_stream': True})
                    else:
                        provider_options.append({})
                elif config.HARDWARE_PROVIDER == "tensorrt":
                    providers.append('TensorrtExecutionProvider')
                    provider_options.append({})
                elif config.HARDWARE_PROVIDER == "directml":
                    providers.append('DmlExecutionProvider')
                    provider_options.append({})
                elif config.HARDWARE_PROVIDER == "openvino":
                    providers.append('OpenVINOExecutionProvider')
                    provider_options.append({})
                elif config.HARDWARE_PROVIDER == "rocm":
                    providers.append('ROCMExecutionProvider')
                    provider_options.append({})
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        provider_options.append({})
        
        # Print selected providers
        print(f"Using ONNX Runtime providers: {providers}")
        
        # Load the ONNX model with configured providers
        print(f"Loading ONNX model from {self.model_path}")
        try:
            self.session = ort.InferenceSession(
                self.model_path, 
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options
            )
            
            # Check which provider was actually used
            used_provider = self.session.get_providers()[0]
            print(f"Using provider: {used_provider}")
            
        except Exception as e:
            print(f"Error loading model with hardware acceleration, falling back to CPU: {e}")
            self.session = ort.InferenceSession(self.model_path)
        
        # Get model metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Get input dimensions
        self.input_width = self.input_shape[2]
        self.input_height = self.input_shape[3]
        
        print(f"Model loaded: input shape={self.input_shape}")
        if self.enabled_classes:
            print(f"Detecting only class IDs: {self.enabled_classes}")
    
    def preprocess(self, image):
        """
        Preprocess image for inference
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Resize
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0,1] and convert to NCHW format
        input_tensor = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def detect(self, image):
        """
        Detect objects in an image
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            list: List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        # Get original image dimensions
        original_height, original_width = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Process output (assuming YOLO output format)
        detections = self._process_output(outputs[0], original_width, original_height)
        
        return detections
    
    def _process_output(self, output, original_width, original_height):
        """
        Process YOLO model output
        
        Args:
            output (numpy.ndarray): Raw model output
            original_width (int): Original image width
            original_height (int): Original image height
            
        Returns:
            list: List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        # Process based on model version if specified
        if self.model_version:
            # Handle model-specific output format based on version
            if self.model_version.lower() == 'yolo11':
                return self._process_yolo11_output(output, original_width, original_height)
            elif self.model_version.lower() == 'yolov8':
                return self._process_yolov8_output(output, original_width, original_height)
            elif self.model_version.lower() == 'yolov5':
                return self._process_yolov5_output(output, original_width, original_height)
        
        # Auto-detect model format based on output shape
        if output.ndim == 3:  # YOLO11 format (1, num_boxes, 85)
            # Flatten first dimension for YOLO11 output
            output = output.reshape(-1, output.shape[-1])
            
        # Filter by confidence
        try:
            valid_detections = output[output[:, 4] > self.confidence_threshold]
        except IndexError:
            # If we get an index error, try the YOLO11 specific format
            # Check output shape to understand model format
            print(f"Model output shape: {output.shape}")
            
            # For YOLO11, reformat the output
            num_classes = output.shape[1] - 5  # Subtracting x,y,w,h,conf
            try:
                # Extract boxes, objectness scores, and classification scores
                boxes = output[:, :4]  # x, y, w, h
                objectness = output[:, 4]  # objectness score
                class_scores = output[:, 5:]  # classification scores
                
                # Filter by objectness score
                mask = objectness > self.confidence_threshold
                filtered_boxes = boxes[mask]
                filtered_scores = class_scores[mask]
                
                if len(filtered_boxes) == 0:
                    return []
                
                # Get class with highest score and its confidence
                class_ids = np.argmax(filtered_scores, axis=1)
                confidences = filtered_scores[np.arange(len(filtered_scores)), class_ids]
                
                # Create final detections format
                valid_detections = np.column_stack((
                    filtered_boxes,  # x, y, w, h
                    objectness[mask],  # objectness score
                    class_ids  # class id
                ))
            except Exception as e:
                print(f"Error processing YOLO output: {e}")
                return []
        
        if len(valid_detections) == 0:
            return []
        
        # Get bounding box coordinates
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in valid_detections:
            # Get confidence and class id
            if detection.shape[0] > 6:  # YOLOv8 format with class scores
                # YOLOv8 output format
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id] * detection[4]  # obj_conf * cls_conf
            else:  # YOLO11 or simple format
                # Already processed above for YOLO11
                confidence = detection[4]
                class_id = int(detection[5])
            
            if confidence > self.confidence_threshold:
                if self.enabled_classes and class_id not in self.enabled_classes:
                    continue
                
                # Get bounding box coordinates
                x, y, w, h = detection[0:4]
                
                # Convert to corner format (x1, y1, x2, y2)
                x1 = int((x - w/2) * original_width)
                y1 = int((y - h/2) * original_height)
                x2 = int((x + w/2) * original_width)
                y2 = int((y + h/2) * original_height)
                
                # Ensure box is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_width, x2)
                y2 = min(original_height, y2)
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.iou_threshold)
        
        detections = []
        for i in indices:
            # OpenCV 4.x has different output format for NMSBoxes
            if isinstance(i, (list, tuple)):
                i = i[0]
            
            box = boxes[i]
            detections.append(box + [confidences[i], class_ids[i]])
        
        return detections

    def _process_yolov8_output(self, output, original_width, original_height):
        """Process YOLOv8 model output format"""
        # YOLOv8 has a different output format than v5 and v11
        if output.ndim == 3:
            output = output.squeeze(0)  # Remove batch dimension
            
        # Filter by confidence
        valid_detections = output[output[:, 4] > self.confidence_threshold]
        
        if len(valid_detections) == 0:
            return []
        
        # YOLOv8 outputs format: [x, y, w, h, conf, cls1, cls2, ...]
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in valid_detections:
            # Get class predictions (starting at index 5)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] * detection[4]  # class_conf * obj_conf
            
            if confidence > self.confidence_threshold:
                if self.enabled_classes and class_id not in self.enabled_classes:
                    continue
                
                # Get bounding box coordinates (center_x, center_y, width, height)
                x, y, w, h = detection[0:4]
                
                # Convert to corner format (x1, y1, x2, y2)
                x1 = int((x - w/2) * original_width)
                y1 = int((y - h/2) * original_height)
                x2 = int((x + w/2) * original_width)
                y2 = int((y + h/2) * original_height)
                
                # Ensure box is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_width, x2)
                y2 = min(original_height, y2)
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.iou_threshold)
        
        detections = []
        for i in indices:
            # OpenCV 4.x has different output format for NMSBoxes
            if isinstance(i, (list, tuple)):
                i = i[0]
            
            box = boxes[i]
            detections.append(box + [confidences[i], class_ids[i]])
        
        return detections
    
    def _process_yolo11_output(self, output, original_width, original_height):
        """Process YOLO11 model output format"""
        # YOLO11 format is typically (1, num_boxes, 85)
        if output.ndim == 3:
            output = output.reshape(-1, output.shape[-1])
        
        # Extract boxes, objectness scores, and classification scores
        boxes = output[:, :4]  # x, y, w, h
        objectness = output[:, 4]  # objectness score
        class_scores = output[:, 5:]  # classification scores
        
        # Filter by objectness score
        mask = objectness > self.confidence_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = class_scores[mask]
        filtered_objectness = objectness[mask]
        
        if len(filtered_boxes) == 0:
            return []
        
        # Get class with highest score and its confidence
        class_ids = np.argmax(filtered_scores, axis=1)
        class_confidences = filtered_scores[np.arange(len(filtered_scores)), class_ids]
        confidences = filtered_objectness * class_confidences  # In YOLO11, confidence is obj_conf * cls_conf
        
        # Process detections - convert to corner format and apply class filtering
        boxes = []
        final_confidences = []
        final_class_ids = []
        
        for i in range(len(filtered_boxes)):
            class_id = class_ids[i]
            
            if self.enabled_classes and class_id not in self.enabled_classes:
                continue
                
            confidence = confidences[i]
            if confidence < self.confidence_threshold:
                continue
                
            # Get box coordinates
            x, y, w, h = filtered_boxes[i]
            
            # Convert to corner format (x1, y1, x2, y2)
            x1 = int((x - w/2) * original_width)
            y1 = int((y - h/2) * original_height)
            x2 = int((x + w/2) * original_width)
            y2 = int((y + h/2) * original_height)
            
            # Ensure box is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_width, x2)
            y2 = min(original_height, y2)
            
            boxes.append([x1, y1, x2, y2])
            final_confidences.append(float(confidence))
            final_class_ids.append(int(class_id))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, final_confidences, self.confidence_threshold, self.iou_threshold)
        
        detections = []
        for i in indices:
            # OpenCV 4.x has different output format for NMSBoxes
            if isinstance(i, (list, tuple)):
                i = i[0]
            
            box = boxes[i]
            detections.append(box + [final_confidences[i], final_class_ids[i]])
        
        return detections
    
    def _process_yolov5_output(self, output, original_width, original_height):
        """Process YOLOv5 model output format"""
        # YOLOv5 format is similar to YOLO11 but with some differences
        if output.ndim == 3:
            output = output.squeeze(0)  # Remove batch dimension
        
        # YOLOv5 outputs format: [x, y, w, h, conf, cls1, cls2, ...]
        # YOLOv5 older versions: [x, y, w, h, cls1, cls2, ..., conf]
        # We need to detect the format
        
        # Check if confidence is at index 4 (newer format) or last column (older format)
        if output.shape[1] > 6:
            # Newer YOLOv5 format (similar to YOLOv8)
            valid_detections = output[output[:, 4] > self.confidence_threshold]
            conf_idx = 4
            cls_start_idx = 5
        else:
            # Older YOLOv5 format
            valid_detections = output[output[:, -1] > self.confidence_threshold]
            conf_idx = -1
            cls_start_idx = 5
        
        if len(valid_detections) == 0:
            return []
        
        boxes = []
        confidences = []
        class_ids = []
        
        for detection in valid_detections:
            # Determine format and extract confidence and class
            if conf_idx == 4:
                # Newer format
                scores = detection[cls_start_idx:]
                class_id = np.argmax(scores)
                confidence = detection[conf_idx]
            else:
                # Older format
                class_id = np.argmax(detection[5:-1])
                confidence = detection[conf_idx]
            
            if confidence > self.confidence_threshold:
                if self.enabled_classes and class_id not in self.enabled_classes:
                    continue
                
                # Get bounding box coordinates (center_x, center_y, width, height)
                x, y, w, h = detection[0:4]
                
                # Convert to corner format (x1, y1, x2, y2)
                x1 = int((x - w/2) * original_width)
                y1 = int((y - h/2) * original_height)
                x2 = int((x + w/2) * original_width)
                y2 = int((y + h/2) * original_height)
                
                # Ensure box is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(original_width, x2)
                y2 = min(original_height, y2)
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.iou_threshold)
        
        detections = []
        for i in indices:
            # OpenCV 4.x has different output format for NMSBoxes
            if isinstance(i, (list, tuple)):
                i = i[0]
            
            box = boxes[i]
            detections.append(box + [confidences[i], class_ids[i]])
        
        return detections


class DetectionService:
    """
    Service for detecting vehicles and license plates in video frames
    """
    def __init__(self, vehicle_model_path=None, plate_model_path=None):
        """
        Initialize the detection service
        
        Args:
            vehicle_model_path (str): Path to the vehicle detection model
            plate_model_path (str): Path to the license plate detection model
        """
        # Initialize vehicle detector with enabled classes from config
        self.vehicle_detector = ONNXDetector(
            model_path=vehicle_model_path or config.VEHICLE_DETECTION_MODEL,
            confidence_threshold=config.DETECTION_CONFIDENCE,
            iou_threshold=config.DETECTION_IOU_THRESHOLD,
            enabled_classes=config.ENABLED_VEHICLE_CLASSES,
            model_version=config.VEHICLE_MODEL_VERSION
        )
        
        # Initialize plate detector
        self.plate_detector = ONNXDetector(
            model_path=plate_model_path or config.PLATE_DETECTION_MODEL,
            confidence_threshold=config.DETECTION_CONFIDENCE,
            iou_threshold=config.DETECTION_IOU_THRESHOLD,
            model_version=config.PLATE_MODEL_VERSION
        )
        
        # Store class names for visualization
        self.vehicle_class_names = config.VEHICLE_CLASS_NAMES
        
        # Initialize MQTT client
        self.client = mqtt.Client()
        self.mqtt_topic = f"{config.MQTT_TOPIC_PREFIX}/detections"
        
        # Connect to MQTT broker
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f"Connected to MQTT broker at {config.MQTT_BROKER}:{config.MQTT_PORT}")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            print("Running in offline mode")
            self.client = None
    
    def detect(self, frame_data):
        """
        Detect vehicles and license plates in a frame
        
        Args:
            frame_data (dict): Frame data with keys 'frame', 'timestamp', 'frame_id'
            
        Returns:
            dict: Detection results
        """
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        
        # Detect vehicles
        start_time = time.time()
        vehicle_detections = self.vehicle_detector.detect(frame)
        vehicle_time = time.time() - start_time
        
        # Detect license plates
        # For each vehicle, crop the region and detect plates
        plate_detections = []
        for vehicle in vehicle_detections:
            x1, y1, x2, y2, conf, class_id = vehicle
            
            # Add padding to the vehicle box
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            crop_x1 = max(0, x1 - padding_x)
            crop_y1 = max(0, y1 - padding_y)
            crop_x2 = min(frame.shape[1], x2 + padding_x)
            crop_y2 = min(frame.shape[0], y2 + padding_y)
            
            # Crop vehicle region
            vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if vehicle_crop.size > 0:
                # Detect plates in the vehicle crop
                plate_dets = self.plate_detector.detect(vehicle_crop)
                
                # Adjust coordinates to original frame
                for plate in plate_dets:
                    plate_x1, plate_y1, plate_x2, plate_y2, plate_conf, plate_class = plate
                    
                    # Adjust plate coordinates
                    plate_x1 += crop_x1
                    plate_y1 += crop_y1
                    plate_x2 += crop_x1
                    plate_y2 += crop_y1
                    
                    plate_detections.append([
                        plate_x1, plate_y1, plate_x2, plate_y2, 
                        plate_conf, plate_class
                    ])
        
        plate_time = time.time() - start_time - vehicle_time
        total_time = time.time() - start_time
        
        # Create detection results
        results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'vehicles': vehicle_detections,
            'plates': plate_detections,
            'processing_time': {
                'vehicle_detection': vehicle_time,
                'plate_detection': plate_time,
                'total': total_time
            }
        }
        
        # Publish results to MQTT
        if self.client:
            # Convert numpy values to Python native types for JSON serialization
            mqtt_results = {
                'frame_id': int(frame_id),
                'timestamp': float(timestamp),
                'vehicles': [[float(v) for v in vehicle] for vehicle in vehicle_detections],
                'plates': [[float(p) for p in plate] for plate in plate_detections],
                'processing_time': {
                    'vehicle_detection': float(vehicle_time),
                    'plate_detection': float(plate_time),
                    'total': float(total_time)
                }
            }
            self.client.publish(self.mqtt_topic, json.dumps(mqtt_results))
        
        return results
    
    def stop(self):
        """Stop the detection service"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        print("Detection service stopped")

# Simple usage example
if __name__ == "__main__":
    # Import VideoIngestionService for testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.video_ingestion.service import VideoIngestionService
    
    # Initialize services
    video_service = VideoIngestionService()
    detection_service = DetectionService()
    
    # Start video service
    video_service.start()
    
    try:
        while True:
            # Get frame
            frame_data = video_service.get_frame()
            if frame_data:
                # Detect objects
                results = detection_service.detect(frame_data)
                
                # Draw detections
                frame = frame_data['frame'].copy()
                
                # Draw vehicles
                for vehicle in results['vehicles']:
                    x1, y1, x2, y2, conf, class_id = vehicle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Vehicle: {conf:.2f}", (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw plates
                for plate in results['plates']:
                    x1, y1, x2, y2, conf, class_id = plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, f"Plate: {conf:.2f}", (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Show performance info
                cv2.putText(frame, f"FPS: {1/results['processing_time']['total']:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Detections', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # No frame available, wait a bit
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        video_service.stop()
        detection_service.stop()
        cv2.destroyAllWindows()