import cv2
import numpy as np
import time
import json
import paho.mqtt.client as mqtt
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class CountingService:
    """
    Service for counting vehicles crossing a line
    """
    def __init__(self, counting_line=None, use_raw_coordinates=None):
        """
        Initialize the counting service
        
        Args:
            counting_line (list): List of two points defining the counting line
                [[x1, y1], [x2, y2]] in normalized coordinates (0-1) or raw pixel coordinates
            use_raw_coordinates (bool): If True, counting_line contains raw pixel coordinates
                instead of normalized coordinates. If None, uses config.USE_RAW_COORDINATES.
        """
        # Get configuration
        self.use_raw_coordinates = use_raw_coordinates if use_raw_coordinates is not None else getattr(config, 'USE_RAW_COORDINATES', False)
        
        # Get counting line from parameters or configuration
        if counting_line is not None:
            self.counting_line = counting_line
        else:
            # Try to get appropriate counting line from config
            if self.use_raw_coordinates and hasattr(config, 'RAW_COUNTING_LINE'):
                self.counting_line = config.RAW_COUNTING_LINE
            elif hasattr(config, 'COUNTING_LINE'):
                self.counting_line = config.COUNTING_LINE
            else:
                # Fallback to default (horizontal line in the middle)
                print("Warning: No counting line configured. Using default (horizontal line in the middle).")
                self.counting_line = [[0.25, 0.5], [0.75, 0.5]]
                self.use_raw_coordinates = False
        
        # Validate counting line format
        if not (isinstance(self.counting_line, list) and 
                len(self.counting_line) == 2 and 
                all(isinstance(p, list) and len(p) == 2 for p in self.counting_line)):
            print(f"Invalid counting line format: {self.counting_line}. Using default.")
            self.counting_line = [[0.25, 0.5], [0.75, 0.5]]
            self.use_raw_coordinates = False
            
        # Initialize counters
        self.counted_tracks = {}  # {track_id: {'timestamp': ts}}
        self.counts = {'total': 0}  # Only track total count
        self.reset_time = time.time()
        
        # Initialize MQTT client
        self.client = mqtt.Client()
        self.mqtt_topic = f"{config.MQTT_TOPIC_PREFIX}/counts"
        
        # Connect to MQTT broker
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f"Connected to MQTT broker at {config.MQTT_BROKER}:{config.MQTT_PORT}")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            print("Running in offline mode")
            self.client = None
            
        # Log counting line setup
        if self.use_raw_coordinates:
            print(f"Using raw pixel coordinates for counting line: {self.counting_line}")
        else:
            print(f"Using normalized coordinates for counting line: {self.counting_line}")
    
    def update(self, frame_data, tracking_results):
        """
        Update counts with new tracking results
        
        Args:
            frame_data (dict): Frame data with keys 'frame', 'timestamp', 'frame_id'
            tracking_results (dict): Tracking results from tracking service
            
        Returns:
            dict: Counting results
        """
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        tracks = tracking_results['tracks']
        
        # Get frame dimensions to convert normalized line coordinates to pixels
        height, width = frame.shape[:2]
        
        # Convert normalized line coordinates to pixels or use raw coordinates
        if not self.use_raw_coordinates:
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
        
        # Store new counts for this frame
        new_counts = []
        
        # Process each track
        start_time = time.time()
        for track in tracks:
            track_id = track['track_id']
            path = track['path']
            
            # Need at least two points to check line crossing
            if len(path) < 2:
                continue
            
            # Get last two path points
            p1 = path[-2]
            p2 = path[-1]
            
            # Check if the track has already been counted
            if track_id in self.counted_tracks:
                continue
            
            # Check if line is crossed
            if self._check_line_crossed(line, p1, p2):
                # Update total count only
                self.counts['total'] += 1
                
                # Record this track as counted
                self.counted_tracks[track_id] = {
                    'timestamp': timestamp
                }
                
                # Record new count event
                new_counts.append({
                    'track_id': track_id,
                    'timestamp': timestamp,
                    'location': p2  # Location of crossing
                })
        
        processing_time = time.time() - start_time
        
        # Prepare counting results
        results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'counts': self.counts,
            'new_counts': new_counts,
            'counting_line': line,
            'processing_time': processing_time
        }
        
        # Publish counts to MQTT if available
        if self.client and new_counts:
            mqtt_results = {
                'frame_id': int(frame_id),
                'timestamp': float(timestamp),
                'counts': self.counts,
                'new_counts': [
                    {
                        'track_id': c['track_id'],
                        'timestamp': c['timestamp']
                    } for c in new_counts
                ]
            }
            self.client.publish(self.mqtt_topic, json.dumps(mqtt_results))
        
        return results
    
    def _check_line_crossed(self, line, p1, p2):
        """
        Check if a line segment (p1, p2) crosses the counting line
        
        Args:
            line (list): Counting line [[x1, y1], [x2, y2]]
            p1 (tuple): First point (x, y)
            p2 (tuple): Second point (x, y)
            
        Returns:
            bool: True if the line is crossed, False otherwise
        """
        # Line segments A: line[0] -> line[1] and B: p1 -> p2
        # Check if they intersect
        
        def ccw(A, B, C):
            """Check if three points are counterclockwise"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        A, B = line
        C, D = p1, p2
        
        # Check if line segments intersect
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    def reset_counts(self):
        """Reset all counters"""
        self.counts = {'total': 0}  # Only total count
        self.counted_tracks = {}
        self.reset_time = time.time()
        
        print("Counting statistics reset")
        
        # Publish reset event
        if self.client:
            reset_msg = {
                'timestamp': self.reset_time,
                'event': 'reset',
                'counts': self.counts
            }
            self.client.publish(self.mqtt_topic, json.dumps(reset_msg))
    
    def stop(self):
        """Stop the counting service"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        print("Counting service stopped")

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
    
    # Example of using raw coordinates for the counting line
    # counting_service = CountingService(
    #     counting_line=[[320, 360], [960, 360]], 
    #     use_raw_coordinates=True
    # )
    
    # Using default configuration (can be either raw or normalized based on config)
    counting_service = CountingService()
    
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
                
                # Count objects
                counting_results = counting_service.update(frame_data, tracking_results)
                
                # Draw results
                frame = frame_data['frame'].copy()
                
                # Draw counting line
                line = counting_results['counting_line']
                cv2.line(frame, tuple(line[0]), tuple(line[1]), (0, 0, 255), 2)
                
                # Draw counts
                counts = counting_results['counts']
                cv2.putText(frame, f"TOTAL: {counts['total']}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw tracks
                for track in tracking_results['tracks']:
                    x1, y1, x2, y2 = track['box']
                    track_id = track['track_id']
                    
                    # Change color if counted
                    color = (0, 165, 255)  # Orange for normal tracks
                    if track_id in counting_service.counted_tracks:
                        color = (0, 255, 0)  # Green for counted tracks
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw track ID
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw path
                    path = track['path']
                    if len(path) >= 2:
                        for i in range(1, len(path)):
                            cv2.line(frame, path[i-1], path[i], color, 2)
                
                # Show frame
                cv2.imshow('Counting', frame)
                
                # Press 'q' to quit, 'r' to reset counts
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    counting_service.reset_counts()
            else:
                # No frame available, wait a bit
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        video_service.stop()
        detection_service.stop()
        tracking_service.stop()
        counting_service.stop()
        cv2.destroyAllWindows()