import cv2
import numpy as np
import time
import json
import paho.mqtt.client as mqtt
import sys
import os
from pathlib import Path
import torch
import logging

# Import trackers directly from boxmot package
from boxmot import ByteTrack, BotSort, StrongSort, OcSort, DeepOcSort

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

# Set up logging
logger = logging.getLogger(__name__)

class TrackingService:
    """
    Service for tracking vehicles across video frames.
    
    This service integrates with BoxMOT trackers to maintain the identity
    of vehicles as they move through the video. It supports multiple tracker
    types including ByteTrack, BoTSORT, StrongSORT, OCSort, and DeepOCSORT.
    """
    def __init__(self, tracker_type=None, tracking_confidence=None):
        """
        Initialize the tracking service
        
        Args:
            tracker_type (str): Type of tracker to use (bytetrack, botsort, strongsort, ocsort, deepocsort)
            tracking_confidence (float): Confidence threshold for tracking
        """
        self.tracker_type = tracker_type or config.TRACKER_TYPE
        self.tracking_confidence = tracking_confidence or config.TRACKING_CONFIDENCE
        
        logger.info(f"Initializing tracking service with {self.tracker_type} tracker")
        
        # Create device for trackers that support torch
        self.device = 'cpu'
        if config.USE_GPU and torch.cuda.is_available():
            self.device = 'cuda'
            logger.info(f"Using {self.device} for tracking")
        
        # Half precision for GPU if specified
        self.half = False
        if self.device == 'cuda' and config.HARDWARE_PRECISION == 'fp16':
            self.half = True
            logger.info("Using half precision (FP16)")
        
        # Initialize tracker based on type
        self._initialize_tracker()
        
        # Dictionary to keep track of active tracks and their paths
        self.active_tracks = {}  # {track_id: {frames: count, path: [(x, y), ...], first_seen: timestamp}}
        
        # Initialize MQTT client
        self.client = mqtt.Client()
        self.mqtt_topic = f"{config.MQTT_TOPIC_PREFIX}/tracks"
        
        # Connect to MQTT broker
        try:
            self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            self.client.loop_start()
            logger.info(f"Connected to MQTT broker at {config.MQTT_BROKER}:{config.MQTT_PORT}")
        except Exception as e:
            logger.warning(f"Failed to connect to MQTT broker: {e}")
            logger.info("Running in offline mode")
            self.client = None
    
    def _initialize_tracker(self):
        """Initialize the appropriate tracker based on tracker_type"""
        
        if self.tracker_type.lower() == 'bytetrack':
            # Get ByteTrack specific parameters from config
            track_thresh = config.get_config('tracking.bytetrack.track_thresh', 0.6)
            track_buffer = config.get_config('tracking.bytetrack.track_buffer', 30)
            match_thresh = config.get_config('tracking.bytetrack.match_thresh', 0.9)
            frame_rate = config.TRACKER_FRAME_RATE
            
            # Initialize ByteTrack with correct parameters
            logger.info(f"Initializing ByteTrack with min_conf={self.tracking_confidence}, track_buffer={track_buffer}")
            self.tracker = ByteTrack(
                min_conf=self.tracking_confidence,
                track_thresh=track_thresh,
                match_thresh=match_thresh,
                track_buffer=track_buffer,
                frame_rate=frame_rate
            )
            
        elif self.tracker_type.lower() == 'botsort':
            # Get BoTSORT specific parameters from config
            track_high_thresh = config.get_config('tracking.botsort.track_high_thresh', 0.6)
            track_low_thresh = config.get_config('tracking.botsort.track_low_thresh', 0.1)
            new_track_thresh = config.get_config('tracking.botsort.new_track_thresh', 0.7)
            track_buffer = config.get_config('tracking.botsort.track_buffer', 30)
            match_thresh = config.get_config('tracking.botsort.match_thresh', 0.7)
            cmc_method = config.get_config('tracking.botsort.cmc_method', 'sparseOptFlow')
            proximity_thresh = config.get_config('tracking.botsort.proximity_thresh', 0.5)
            appearance_thresh = config.get_config('tracking.botsort.appearance_thresh', 0.25)
            with_reid = config.get_config('tracking.botsort.with_reid', True)
            
            # Check for ReID weights file
            reid_weights = Path(config.REID_WEIGHTS_PATH)
            if with_reid and not reid_weights.exists():
                logger.warning(f"ReID weights file not found at {reid_weights}")
                logger.info("ReID will be disabled. Re-identification accuracy may be reduced.")
                with_reid = False
                reid_weights = None
            
            # Initialize BoTSORT
            logger.info(f"Initializing BoTSORT with track_high_thresh={track_high_thresh}, ReID={with_reid}")
            self.tracker = BotSort(
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                cmc_method=cmc_method,
                proximity_thresh=proximity_thresh,
                appearance_thresh=appearance_thresh,
                with_reid=with_reid,
                device=self.device,
                fp16=self.half,
                reid_weights=str(reid_weights) if reid_weights else None
            )
            
        elif self.tracker_type.lower() == 'strongsort':
            # Get StrongSORT specific parameters from config
            max_dist = config.get_config('tracking.strongsort.max_dist', 0.2)
            max_iou_distance = config.get_config('tracking.strongsort.max_iou_distance', 0.7)
            max_age = config.TRACKER_MAX_AGE
            n_init = config.get_config('tracking.strongsort.n_init', 3)
            nn_budget = config.get_config('tracking.strongsort.nn_budget', 100)
            mc_lambda = config.get_config('tracking.strongsort.mc_lambda', 0.995)
            ema_alpha = config.get_config('tracking.strongsort.ema_alpha', 0.9)
            
            # Check for ReID weights file
            reid_weights = Path(config.REID_WEIGHTS_PATH)
            if not reid_weights.exists():
                logger.warning(f"ReID weights file not found at {reid_weights}")
                logger.info("Using default weights. Re-identification accuracy may be reduced.")
                reid_weights = None
            
            # Initialize StrongSORT
            logger.info(f"Initializing StrongSORT with max_dist={max_dist}, max_age={max_age}")
            self.tracker = StrongSort(
                reid_weights=str(reid_weights) if reid_weights else None,
                device=self.device,
                fp16=self.half,
                max_dist=max_dist,
                max_iou_distance=max_iou_distance,
                max_age=max_age,
                n_init=n_init,
                nn_budget=nn_budget,
                mc_lambda=mc_lambda,
                ema_alpha=ema_alpha
            )
            
        elif self.tracker_type.lower() == 'ocsort':
            # Get OCSort specific parameters from config
            det_thresh = config.get_config('tracking.ocsort.det_thresh', 0.6)
            max_age = config.get_config('tracking.ocsort.max_age', 30)
            min_hits = config.get_config('tracking.ocsort.min_hits', 3)
            asso_threshold = config.get_config('tracking.ocsort.iou_threshold', 0.3)
            delta_t = config.get_config('tracking.ocsort.delta_t', 3)
            asso_func = config.get_config('tracking.ocsort.asso_func', 'iou')
            inertia = config.get_config('tracking.ocsort.inertia', 0.2)
            use_byte = config.get_config('tracking.ocsort.use_byte', False)
            
            # Initialize OCSort with correct parameters
            logger.info(f"Initializing OCSort with min_conf={self.tracking_confidence}, max_age={max_age}")
            self.tracker = OcSort(
                min_conf=self.tracking_confidence,
                max_age=max_age,
                min_hits=min_hits,
                asso_threshold=asso_threshold,
                delta_t=delta_t,
                asso_func=asso_func,
                inertia=inertia,
                use_byte=use_byte
            )
            
        elif self.tracker_type.lower() == 'deepocsort':
            # Get DeepOCSort specific parameters from config
            min_confidence = config.get_config('tracking.deepocsort.min_confidence', 0.5)
            max_age = config.get_config('tracking.deepocsort.max_age', 30)
            min_hits = config.get_config('tracking.deepocsort.min_hits', 3)
            iou_threshold = config.get_config('tracking.deepocsort.iou_threshold', 0.3)
            delta_t = config.get_config('tracking.deepocsort.delta_t', 3)
            asso_func = config.get_config('tracking.deepocsort.asso_func', 'iou')
            inertia = config.get_config('tracking.deepocsort.inertia', 0.2)
            w_association_emb = config.get_config('tracking.deepocsort.w_association_emb', 0.75)
            
            # Check for ReID weights file
            reid_weights = Path(config.REID_WEIGHTS_PATH)
            if not reid_weights.exists():
                logger.warning(f"ReID weights file not found at {reid_weights}")
                logger.info("Using default weights. Re-identification accuracy may be reduced.")
                reid_weights = None
            
            # Initialize DeepOCSORT with correct parameters
            logger.info(f"Initializing DeepOCSORT with min_confidence={min_confidence}, max_age={max_age}")
            self.tracker = DeepOcSort(
                model_weights=str(reid_weights) if reid_weights else None,
                device=self.device,
                fp16=self.half,
                min_confidence=min_confidence,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                delta_t=delta_t,
                asso_func=asso_func,
                inertia=inertia,
                w_association_emb=w_association_emb
            )
        else:
            logger.warning(f"Tracker type '{self.tracker_type}' not recognized. Falling back to ByteTrack.")
            self.tracker = ByteTrack(
                min_conf=self.tracking_confidence,
                track_thresh=0.6,
                match_thresh=0.9,
                track_buffer=30,
                frame_rate=30
            )
    
    def update(self, frame_data, detection_results):
        """
        Update trackers with new detections
        
        Args:
            frame_data (dict): Frame data with keys 'frame', 'timestamp', 'frame_id'
            detection_results (dict): Detection results from detection service
            
        Returns:
            dict: Tracking results with vehicle tracks
        """
        # Extract frame data
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        vehicles = detection_results['vehicles']
        
        # Return empty results if no vehicles detected
        if not vehicles:
            return {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'tracks': [],
                'processing_time': 0
            }
        
        # Convert vehicle detections to numpy array
        # Format: [x1, y1, x2, y2, conf, class_id]
        vehicle_detections = np.array(vehicles)
        
        # Measure processing time
        start_time = time.time()
        
        # Update tracker based on tracker type
        if self.tracker_type.lower() == 'bytetrack':
            # ByteTrack has a unique update signature that differs from other trackers
            # It requires image dimensions and returns STrack objects
            im_shape = [frame.shape[0], frame.shape[1]]
            
            # Update tracker
            online_targets = self.tracker.update(
                vehicle_detections,
                im_shape,
                (frame.shape[0], frame.shape[1])
            )
            
            # Convert STrack objects to standard format [x1, y1, x2, y2, track_id, class_id, conf]
            tracks = []
            for t in online_targets:
                if t.is_activated:
                    tlwh = t.tlwh
                    tid = t.track_id
                    cls = t.cls if hasattr(t, 'cls') else 0  # Default to class 0 if not available
                    score = t.score if hasattr(t, 'score') else 0.0
                    
                    # Optional filtering for vertical bounding boxes
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        x1, y1, w, h = tlwh
                        tracks.append([
                            x1, y1, x1 + w, y1 + h,  # Convert to xyxy
                            tid,  # Track ID
                            cls,  # Class ID
                            score  # Confidence score
                        ])
        
        elif self.tracker_type.lower() in ['botsort', 'strongsort', 'ocsort', 'deepocsort']:
            # These trackers follow a more standard pattern that takes detections and frame,
            # and return a numpy array of [x1, y1, x2, y2, track_id, class_id, conf]
            
            # Some trackers might require detections in a specific format
            # or need to be updated differently
            if self.tracker_type.lower() == 'ocsort':
                # OCSort doesn't need the frame, just the detections
                try:
                    tracks = self.tracker.update(vehicle_detections)
                except Exception as e:
                    logger.error(f"Error updating OCSort tracker: {e}")
                    tracks = np.array([])
            else:
                # Most other trackers need both detections and the frame
                try:
                    tracks = self.tracker.update(vehicle_detections, frame)
                except Exception as e:
                    logger.error(f"Error updating {self.tracker_type} tracker: {e}")
                    tracks = np.array([])
        
        else:
            # Fallback if tracker type is not recognized
            logger.warning(f"Unknown tracker type: {self.tracker_type}, returning empty tracks")
            tracks = np.array([])
            
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Ensure tracks is a numpy array
        if not isinstance(tracks, np.ndarray) and len(tracks) > 0:
            tracks = np.array(tracks)
        
        # Process tracks to update active_tracks and prepare results
        tracking_results = []
        
        if len(tracks) > 0:
            for track in tracks:
                try:
                    # Extract data from track
                    x1, y1, x2, y2, track_id, class_id, conf = track
                    
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Cast track_id to int (some trackers might return float)
                    track_id = int(track_id)
                    
                    # Initialize new track or update existing
                    if track_id not in self.active_tracks:
                        self.active_tracks[track_id] = {
                            'frames': 1,
                            'path': [(center_x, center_y)],
                            'first_seen': timestamp,
                            'class_id': int(class_id),
                            'last_update_frame': frame_id
                        }
                    else:
                        self.active_tracks[track_id]['frames'] += 1
                        self.active_tracks[track_id]['path'].append((center_x, center_y))
                        self.active_tracks[track_id]['last_update_frame'] = frame_id
                    
                    # Calculate speed based on path
                    speed = 0
                    if len(self.active_tracks[track_id]['path']) >= 2:
                        # Get last two points and calculate displacement
                        p1 = self.active_tracks[track_id]['path'][-2]
                        p2 = self.active_tracks[track_id]['path'][-1]
                        dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                        speed = dist  # pixels per frame
                        
                    # Add to tracking results
                    tracking_results.append({
                        'track_id': track_id,
                        'box': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [center_x, center_y],
                        'class_id': int(class_id),
                        'confidence': float(conf),
                        'frames_tracked': self.active_tracks[track_id]['frames'],
                        'speed': float(speed),
                        'path': self.active_tracks[track_id]['path'][-10:]  # Keep last 10 points to limit size
                    })
                except Exception as e:
                    logger.error(f"Error processing track {track}: {e}")
                    continue
        
        # Clean up old tracks
        self._clean_inactive_tracks(frame_id)
        
        # Prepare results
        results = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'tracks': tracking_results,
            'processing_time': processing_time
        }
        
        # Publish results to MQTT if connected
        if self.client:
            # Create simplified version for MQTT (smaller payload)
            mqtt_results = {
                'frame_id': int(frame_id),
                'timestamp': float(timestamp),
                'tracks': [
                    {
                        'track_id': t['track_id'],
                        'box': t['box'],
                        'center': t['center'],
                        'class_id': t['class_id'],
                        'confidence': float(t['confidence']),
                        'frames_tracked': t['frames_tracked'],
                        'speed': t['speed']
                    } for t in tracking_results
                ],
                'processing_time': float(processing_time)
            }
            self.client.publish(self.mqtt_topic, json.dumps(mqtt_results))
        
        return results
    
    def _clean_inactive_tracks(self, current_frame_id, max_frames_missing=30):
        """
        Clean up tracks that haven't been updated in a while
        
        Args:
            current_frame_id (int): Current frame ID
            max_frames_missing (int): Max frames a track can be missing before removal
        """
        # Identify tracks to remove based on last update frame
        to_remove = []
        for track_id, track_info in self.active_tracks.items():
            frames_since_update = current_frame_id - track_info.get('last_update_frame', 0)
            if frames_since_update > max_frames_missing:
                to_remove.append(track_id)
        
        # Remove inactive tracks
        for track_id in to_remove:
            del self.active_tracks[track_id]
            logger.debug(f"Removed inactive track {track_id}")
    
    def stop(self):
        """Stop the tracking service and clean up resources"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        logger.info("Tracking service stopped")

# Simple test if run directly
if __name__ == "__main__":
    # Set up logging for testing
    logging.basicConfig(level=logging.INFO)
    
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
        logger.info("Interrupted")
    finally:
        video_service.stop()
        detection_service.stop()
        tracking_service.stop()
        cv2.destroyAllWindows()