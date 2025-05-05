import cv2
import time
import threading
import queue
from pathlib import Path
import paho.mqtt.client as mqtt
import json
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class VideoIngestionService:
    """
    Service for ingesting video streams and publishing frames to a message queue
    """
    def __init__(self, source=None, max_queue_size=5, frame_skip=None):
        """
        Initialize the video ingestion service
        
        Args:
            source (str): Path to video file, RTSP URL, or device ID (int)
            max_queue_size (int): Maximum size of the frame queue
            frame_skip (int): Number of frames to skip between processing
        """
        self.source = source or config.VIDEO_SOURCE
        self.frame_skip = frame_skip or config.FRAME_SKIP
        self.process_resolution = config.PROCESS_RESOLUTION
        self.frames_queue = queue.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.capture = None
        self.capture_thread = None
        self.frame_count = 0
        self.last_position = 0  # Track the last frame position
        
        # Initialize MQTT client
        self.client = mqtt.Client()
        self.mqtt_topic = f"{config.MQTT_TOPIC_PREFIX}/frames"
        
        # Connect to MQTT broker
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            self.client.loop_start()
            print(f"Connected to MQTT broker at {config.MQTT_BROKER}:{config.MQTT_PORT}")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            print("Running in offline mode")
            self.client = None
    
    def start(self):
        """Start the video ingestion service"""
        if self.is_running:
            print("Video ingestion service is already running")
            return
        
        # Convert source to int if it's a number (for webcam)
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)
        
        try:
            self.capture = cv2.VideoCapture(self.source)
            if not self.capture.isOpened():
                raise ValueError(f"Failed to open video source: {self.source}")
            
            # Get video properties
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video source opened: {self.source}")
            print(f"Original video size: {self.width}x{self.height}, FPS: {self.fps}")
            
            # Start capture thread
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting video ingestion: {e}")
            if self.capture:
                self.capture.release()
            return False
    
    def stop(self):
        """Stop the video ingestion service"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.capture:
            self.capture.release()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        print("Video ingestion service stopped")
    
    def get_total_frames(self):
        """
        Get the total number of frames in the video source
        
        Returns:
            int: Total frames or 0 if unavailable (e.g., for livestreams/webcams)
        """
        if self.capture is None:
            return 0
            
        # This only works for video files, not for cameras/streams
        total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Return 0 for live sources (webcams, RTSP streams)
        if total <= 0 or isinstance(self.source, int) or (
                isinstance(self.source, str) and 
                (self.source.startswith('rtsp://') or 
                 self.source.startswith('http://') or 
                 self.source.startswith('https://'))):
            return 0
        
        return total
    
    def rewind_one_frame(self):
        """
        Rewind the video by one frame
        
        This is useful when we need to re-read the first frame,
        for example after initialization.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.capture is None:
            return False
            
        # Only attempt to rewind for file sources, not cameras
        if isinstance(self.source, str) and Path(self.source).exists():
            current_pos = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Store for rewind if this is first frame
            if current_pos <= 1:
                self.last_position = 0
            else:
                self.last_position = current_pos - 1
                
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.last_position)
            return True
        
        return False

    def _capture_frames(self):
        """Thread function for capturing frames"""
        while self.is_running:
            # Save the current position before reading for rewind capability
            if isinstance(self.source, str) and Path(self.source).exists():
                self.last_position = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
                
            ret, frame = self.capture.read()
            if not ret:
                # If video file ends, loop back to beginning
                if isinstance(self.source, str) and Path(self.source).exists():
                    self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("Failed to capture frame, stopping")
                    self.is_running = False
                    break
            
            self.frame_count += 1
            
            # Skip frames if needed
            if self.frame_count % self.frame_skip != 0:
                continue
            
            # Resize frame to processing resolution
            if self.process_resolution:
                frame = cv2.resize(frame, self.process_resolution)
            
            # Add frame to queue, drop oldest if full
            if self.frames_queue.full():
                try:
                    self.frames_queue.get_nowait()
                except queue.Empty:
                    pass
            
            timestamp = time.time()
            self.frames_queue.put({
                'frame': frame,
                'timestamp': timestamp,
                'frame_id': self.frame_count
            })
            
            # Publish frame metadata to MQTT if available
            if self.client:
                # Create a JSON message with frame metadata
                # (we don't send the actual frame via MQTT as it would be too large)
                msg = {
                    'timestamp': timestamp,
                    'frame_id': self.frame_count,
                    'height': frame.shape[0],
                    'width': frame.shape[1]
                }
                self.client.publish(self.mqtt_topic, json.dumps(msg))
    
    def get_frame(self, timeout=1.0):
        """
        Get the next frame from the queue
        
        Args:
            timeout (float): Timeout in seconds
            
        Returns:
            dict: Frame data with keys 'frame', 'timestamp', 'frame_id'
        """
        try:
            return self.frames_queue.get(timeout=timeout)
        except queue.Empty:
            return None

# Simple usage example
if __name__ == "__main__":
    service = VideoIngestionService()
    service.start()
    
    try:
        while True:
            frame_data = service.get_frame()
            if frame_data:
                frame = frame_data['frame']
                cv2.imshow('Frame', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # No frame available, wait a bit
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        service.stop()
        cv2.destroyAllWindows()