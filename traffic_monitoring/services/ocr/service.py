import cv2
import numpy as np
import time
import json
import paho.mqtt.client as mqtt
import sys
import os
from pathlib import Path
import easyocr

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class OCRService:
    """
    Service for reading license plates using OCR
    """
    def __init__(self, languages=None, gpu=None):
        """
        Initialize the OCR service
        
        Args:
            languages (list): List of language codes
            gpu (bool): Whether to use GPU for OCR
        """
        self.languages = languages or config.OCR_LANGUAGES
        self.gpu = gpu if gpu is not None else config.OCR_GPU
        
        # Initialize EasyOCR reader
        print(f"Initializing EasyOCR reader with languages: {self.languages}, GPU: {self.gpu}")
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
        
        # Plate recognition history
        self.plate_history = {}  # {track_id: [plate_texts]}
        
        # Initialize MQTT client
        self.client = mqtt.Client()
        self.mqtt_topic = f"{config.MQTT_TOPIC_PREFIX}/plates"
        
        # Connect to MQTT broker
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f"Connected to MQTT broker at {config.MQTT_BROKER}:{config.MQTT_PORT}")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            print("Running in offline mode")
            self.client = None
    
    def read_plates(self, frame_data, detection_results, tracking_results):
        """
        Read license plates in the frame
        
        Args:
            frame_data (dict): Frame data with keys 'frame', 'timestamp', 'frame_id'
            detection_results (dict): Detection results from detection service
            tracking_results (dict): Tracking results from tracking service
            
        Returns:
            dict: OCR results
        """
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        plates = detection_results['plates']
        tracks = tracking_results['tracks']
        
        # Create a mapping from box coordinates to track_id
        track_boxes = {}
        for track in tracks:
            x1, y1, x2, y2 = track['box']
            box_key = f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}"
            track_boxes[box_key] = track['track_id']
        
        # OCR results
        ocr_results = []
        
        # Process each plate
        start_time = time.time()
        for plate in plates:
            x1, y1, x2, y2, conf, class_id = plate
            
            # Skip if confidence is too low
            if conf < 0.4:  # Higher threshold for OCR to reduce false positives
                continue
            
            # Create plate key for track lookup
            plate_key = f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}"
            
            # Try to find the track_id for this plate
            track_id = None
            for track_box_key, tid in track_boxes.items():
                # Simple IoU check
                tb_x1, tb_y1, tb_x2, tb_y2 = map(float, track_box_key.split(','))
                if self._calculate_iou(
                    [x1, y1, x2, y2], [tb_x1, tb_y1, tb_x2, tb_y2]) > 0.5:
                    track_id = tid
                    break
            
            # If no track found, generate temporary ID
            if track_id is None:
                track_id = f"plate_{len(ocr_results)}"
            
            # Crop plate region with padding
            padding_x = int((x2 - x1) * 0.05)
            padding_y = int((y2 - y1) * 0.05)
            crop_x1 = max(0, int(x1) - padding_x)
            crop_y1 = max(0, int(y1) - padding_y)
            crop_x2 = min(frame.shape[1], int(x2) + padding_x)
            crop_y2 = min(frame.shape[0], int(y2) + padding_y)
            
            plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if plate_crop.size == 0:
                continue
            
            # Apply preprocessing to improve OCR
            plate_crop = self._preprocess_plate(plate_crop)
            
            # Perform OCR on the plate crop
            ocr_start = time.time()
            ocr_result = self.reader.readtext(plate_crop)
            ocr_time = time.time() - ocr_start
            
            # Extract text from OCR results
            plate_text = ""
            for detection in ocr_result:
                text = detection[1]
                conf = detection[2]
                
                # Only include text with sufficient confidence
                if conf > 0.4:
                    if plate_text:
                        plate_text += " "
                    plate_text += text
            
            # Clean up the plate text (filter out non-alphanumeric chars)
            plate_text = self._clean_plate_text(plate_text)
            
            # If text was found, add to results
            if plate_text:
                # Add to plate history for this track
                if track_id not in self.plate_history:
                    self.plate_history[track_id] = []
                
                self.plate_history[track_id].append(plate_text)
                
                # Get the most common plate text from history
                final_text = self._get_most_common_plate(track_id)
                
                result = {
                    'track_id': track_id,
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'raw_text': plate_text,
                    'text': final_text,
                    'confidence': float(conf),
                    'processing_time': float(ocr_time)
                }
                
                ocr_results.append(result)
        
        processing_time = time.time() - start_time
        
        # Create the final results
        results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'plates': ocr_results,
            'processing_time': processing_time
        }
        
        # Publish results to MQTT
        if self.client and ocr_results:
            mqtt_results = {
                'frame_id': int(frame_id),
                'timestamp': float(timestamp),
                'plates': [
                    {
                        'track_id': r['track_id'],
                        'text': r['text'],
                        'confidence': float(r['confidence']),
                    } for r in ocr_results
                ]
            }
            self.client.publish(self.mqtt_topic, json.dumps(mqtt_results))
        
        return results
    
    def _preprocess_plate(self, img):
        """
        Preprocess license plate image for better OCR results
        
        Args:
            img (numpy.ndarray): Input license plate image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Resize plate image to improve OCR
        h, w = img.shape[:2]
        
        # Enlarge small plates
        min_width = 200
        if w < min_width:
            scale = min_width / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Invert back to normal
        binary = cv2.bitwise_not(binary)
        
        # Return both processed and original images
        # (EasyOCR will choose the best one)
        return img
    
    def _clean_plate_text(self, text):
        """
        Clean up the plate text
        
        Args:
            text (str): Raw plate text
            
        Returns:
            str: Cleaned plate text
        """
        # Convert to uppercase
        text = text.upper()
        
        # Remove spaces and common misread characters
        text = text.replace(' ', '')
        
        # Keep only alphanumeric characters
        cleaned_text = ''.join(c for c in text if c.isalnum())
        
        return cleaned_text
    
    def _get_most_common_plate(self, track_id, min_occurrences=2):
        """
        Get the most common plate text from history
        
        Args:
            track_id: Track ID
            min_occurrences (int): Minimum number of occurrences required
            
        Returns:
            str: Most common plate text
        """
        if track_id not in self.plate_history:
            return ""
        
        # Count occurrences of each plate text
        from collections import Counter
        counter = Counter(self.plate_history[track_id])
        
        # Get the most common plate text
        if counter:
            most_common = counter.most_common(1)[0]
            if most_common[1] >= min_occurrences:
                return most_common[0]
        
        # If no consensus yet, return the latest one
        if self.plate_history[track_id]:
            return self.plate_history[track_id][-1]
        
        return ""
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes
        
        Args:
            box1 (list): First box [x1, y1, x2, y2]
            box2 (list): Second box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def stop(self):
        """Stop the OCR service"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        print("OCR service stopped")

# Simple usage example
if __name__ == "__main__":
    # Import necessary services for testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.video_ingestion.service import VideoIngestionService
    from services.detection.service import DetectionService
    from services.tracking.service import TrackingService
    
    # Initialize services
    video_service = VideoIngestionService()
    detection_service = DetectionService()
    tracking_service = TrackingService()
    ocr_service = OCRService()
    
    # Start video service
    video_service.start()
    
    try:
        while True:
            # Get frame
            frame_data = video_service.get_frame()
            if frame_data:
                # Detect objects
                detection_results = detection_service.detect(frame_data)
                
                # Track objects
                tracking_results = tracking_service.update(frame_data, detection_results)
                
                # Read plates
                ocr_results = ocr_service.read_plates(frame_data, detection_results, tracking_results)
                
                # Draw results
                frame = frame_data['frame'].copy()
                
                # Draw vehicle detections
                for vehicle in detection_results['vehicles']:
                    x1, y1, x2, y2, conf, class_id = vehicle
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw plate detections and OCR results
                for plate in ocr_results['plates']:
                    x1, y1, x2, y2 = plate['box']
                    text = plate['text']
                    
                    # Draw plate box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
                    # Draw text background
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, 
                                 (int(x1), int(y1) - text_size[1] - 10),
                                 (int(x1) + text_size[0] + 10, int(y1)), 
                                 (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(frame, text, (int(x1) + 5, int(y1) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show performance info
                fps = 1 / (detection_results['processing_time']['total'] + 
                          tracking_results['processing_time'] +
                          ocr_results['processing_time'])
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('OCR', frame)
                
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
        tracking_service.stop()
        ocr_service.stop()
        cv2.destroyAllWindows()