import cv2
import numpy as np
import time
import json
import paho.mqtt.client as mqtt
import sys
import os
from pathlib import Path
from boxmot import BoxTracker

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class TrackingService:
    """
    Service for tracking vehicles across video frames
    """
    def __init__(self, tracker_type=None, tracking_confidence=None):
        """
        Initialize the tracking service
        
        Args:
            tracker_type (str): Type of tracker to use (bytetrack, botsort, etc.)
            tracking_confidence (float): Confidence threshold for tracking
        """
        self.tracker_type = tracker_type or config.TRACKER_TYPE
        self.tracking_confidence = tracking_confidence or config.TRACKING_CONFIDENCE
        
        # Initialize BoxMOT tracker
        self.tracker = BoxTracker(
            tracker_type=self.tracker_type,
            fps=30,  # Will be updated with actual FPS
            track_thresh=self.tracking_confidence
        )
        
        # To keep track of active tracks and their paths
        self.active_tracks = {}  # {track_id: {frames: count, path: [(x, y), ...], first_seen: timestamp}}
        
        # Initialize MQTT client
        self.client = mqtt.Client()
        self.mqtt_topic = f"{config.MQTT_TOPIC_PREFIX}/tracks"
        
        # Connect to MQTT broker
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f"Connected to MQTT broker at {config.MQTT_BROKER}:{config.MQTT_PORT}")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            print("Running in offline mode")
            self.client = None
    
    def update(self, frame_data, detection_results):
        """
        Update trackers with new detections
        
        Args:
            frame_data (dict): Frame data with keys 'frame', 'timestamp', 'frame_id'
            detection_results (dict): Detection results from detection service
            
        Returns:
            dict: Tracking results with vehicle tracks
        """
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        vehicles = detection_results['vehicles']
        
        if not vehicles:
            # No vehicles detected, return empty tracking results
            return {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'tracks': [],
                'processing_time': 0
            }
        
        # Convert vehicle detections to format expected by BoxMOT
        # [x1, y1, x2, y2, conf, class_id]
        vehicle_detections = np.array(vehicles)
        
        # Update tracker
        start_time = time.time()
        tracks = self.tracker.update(vehicle_detections, frame)
        processing_time = time.time() - start_time
        
        # Process and format tracking results
        tracking_results = []
        
        # BoxMOT tracks format: [x1, y1, x2, y2, track_id, class_id, conf]
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id, conf = track
            
            # Calculate center point of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Initialize track if new, or update existing
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    'frames': 1,
                    'path': [(center_x, center_y)],
                    'first_seen': timestamp,
                    'class_id': int(class_id)
                }
            else:
                self.active_tracks[track_id]['frames'] += 1
                self.active_tracks[track_id]['path'].append((center_x, center_y))
            
            # Calculate speed (in pixels per frame) if enough points exist
            speed = 0
            if len(self.active_tracks[track_id]['path']) >= 2:
                # Get last two points and calculate displacement
                p1 = self.active_tracks[track_id]['path'][-2]
                p2 = self.active_tracks[track_id]['path'][-1]
                dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                speed = dist  # Speed in pixels per frame
            
            tracking_results.append({
                'track_id': int(track_id),
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'center': [center_x, center_y],
                'class_id': int(class_id),
                'confidence': float(conf),
                'frames_tracked': self.active_tracks[track_id]['frames'],
                'speed': float(speed),
                'path': self.active_tracks[track_id]['path'][-10:]  # Keep last 10 points only to limit size
            })
        
        # Clean up old tracks that haven't been updated in a while
        self._clean_inactive_tracks()
        
        results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'tracks': tracking_results,
            'processing_time': processing_time
        }
        
        # Publish results to MQTT
        if self.client:
            # Create simplified version with smaller data size
            mqtt_results = {
                'frame_id': int(frame_id),
                'timestamp': float(timestamp),
                'tracks': [
                    {
                        'track_id': t['track_id'],
                        'box': t['box'],
                        'center': t['center'],
                        'class_id': t['class_id'],
                        'confidence': t['confidence'],
                        'frames_tracked': t['frames_tracked'],
                        'speed': t['speed']
                    } for t in tracking_results
                ],
                'processing_time': float(processing_time)
            }
            self.client.publish(self.mqtt_topic, json.dumps(mqtt_results))
        
        return results
    
    def _clean_inactive_tracks(self, max_frames_missing=30):
        """
        Clean up tracks that haven't been updated in a while
        
        Args:
            max_frames_missing (int): Maximum number of frames a track can be missing
        """
        # Get list of active track IDs from the tracker
        active_ids = set(track[4] for track in self.tracker.tracked_objects)
        
        # Remove tracks that are no longer active
        to_remove = []
        for track_id in self.active_tracks:
            if track_id not in active_ids:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.active_tracks[track_id]
    
    def stop(self):
        """Stop the tracking service"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        print("Tracking service stopped")

# Simple usage example
if __name__ == "__main__":
    # Import necessary services for testing
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.video_ingestion.service import VideoIngestionService
    from services.detection.service import DetectionService
    
    # Initialize services
    video_service = VideoIngestionService()
    detection_service = DetectionService()
    tracking_service = TrackingService()
    
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
                
                # Draw results
                frame = frame_data['frame'].copy()
                
                # Draw tracks
                for track in tracking_results['tracks']:
                    x1, y1, x2, y2 = track['box']
                    track_id = track['track_id']
                    confidence = track['confidence']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                    
                    # Draw track ID
                    cv2.putText(frame, f"ID: {track_id}", (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
                    # Draw path
                    path = track['path']
                    if len(path) >= 2:
                        for i in range(1, len(path)):
                            cv2.line(frame, path[i-1], path[i], (0, 165, 255), 2)
                
                # Show performance info
                cv2.putText(frame, f"Tracks: {len(tracking_results['tracks'])}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Tracking', frame)
                
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
        cv2.destroyAllWindows()