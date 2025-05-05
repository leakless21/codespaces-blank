import os
import sys
import time
import cv2
import argparse
import threading
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import services
from services.video_ingestion.service import VideoIngestionService
from services.detection.service import DetectionService
from services.tracking.service import TrackingService
from services.counting.service import CountingService
from services.ocr.service import OCRService
from services.storage.service import StorageService
from config import config

class TrafficMonitoringApp:
    """
    Main application for traffic monitoring
    """
    def __init__(self, video_source=None, show_ui=True, record_output=False):
        """
        Initialize the traffic monitoring application
        
        Args:
            video_source (str): Path to video file, RTSP URL, or device ID
            show_ui (bool): Whether to show the user interface
            record_output (bool): Whether to record the output video
        """
        self.video_source = video_source or config.VIDEO_SOURCE
        self.show_ui = show_ui
        self.record_output = record_output
        self.running = False
        
        # Initialize services
        print("Initializing services...")
        
        self.video_service = VideoIngestionService(source=self.video_source)
        self.detection_service = DetectionService()
        self.tracking_service = TrackingService()
        self.counting_service = CountingService()
        self.ocr_service = OCRService()
        self.storage_service = StorageService()
        
        # Video writer for recording output
        self.video_writer = None
        
        print("Services initialized")
    
    def start(self):
        """Start the traffic monitoring application"""
        if self.running:
            print("Application is already running")
            return
        
        print("Starting traffic monitoring application...")
        
        # Start services
        self.storage_service.start()
        self.video_service.start()
        
        # Start main loop
        self.running = True
        
        # Initialize video writer if recording output
        if self.record_output:
            self._init_video_writer()
        
        # Process frames until stopped
        self._process_frames()
    
    def stop(self):
        """Stop the traffic monitoring application"""
        print("Stopping traffic monitoring application...")
        
        # Stop services
        self.running = False
        self.video_service.stop()
        self.detection_service.stop()
        self.tracking_service.stop()
        self.counting_service.stop()
        self.ocr_service.stop()
        self.storage_service.stop()
        
        # Release video writer
        if self.video_writer is not None:
            self.video_writer.release()
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        print("Traffic monitoring application stopped")
    
    def _init_video_writer(self):
        """Initialize the video writer for recording output"""
        # Get video properties from the first frame
        frame_data = self.video_service.get_frame()
        if frame_data is None:
            print("Failed to get a frame for initializing video writer")
            return
        
        frame = frame_data['frame']
        height, width = frame.shape[:2]
        
        # Create output directory if it doesn't exist
        output_dir = Path(config.DATA_DIR) / "recordings"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output file name based on current time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"traffic_monitoring_{timestamp}.mp4")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, config.OUTPUT_FPS, (width, height))
        
        print(f"Recording output to {output_path}")
    
    def _process_frames(self):
        """Process frames from the video service"""
        try:
            frame_count = 0
            while self.running:
                # Get frame from video service
                frame_data = self.video_service.get_frame()
                if frame_data is None:
                    time.sleep(0.01)  # Short sleep if no frame available
                    continue
                
                frame_count += 1
                
                # Process the frame through the pipeline
                start_time = time.time()
                
                # 1. Detect vehicles and license plates
                detection_results = self.detection_service.detect(frame_data)
                
                # 2. Track vehicles
                tracking_results = self.tracking_service.update(frame_data, detection_results)
                
                # 3. Count vehicles crossing line
                counting_results = self.counting_service.update(frame_data, tracking_results)
                
                # 4. Read license plates
                ocr_results = self.ocr_service.read_plates(frame_data, detection_results, tracking_results)
                
                # Calculate FPS
                processing_time = time.time() - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # Show UI if enabled
                if self.show_ui:
                    self._display_results(frame_data, detection_results, tracking_results, 
                                        counting_results, ocr_results, fps)
                
                # Record output if enabled
                if self.record_output and self.video_writer is not None:
                    # Generate the same visualized frame as in display_results
                    vis_frame = self._prepare_visualization(frame_data, detection_results, 
                                                          tracking_results, counting_results, 
                                                          ocr_results, fps)
                    self.video_writer.write(vis_frame)
                
                # Check for key press
                if self.show_ui:
                    key = cv2.waitKey(1) & 0xFF
                    
                    # 'q' to quit
                    if key == ord('q'):
                        self.running = False
                        break
                    
                    # 'r' to reset counting
                    elif key == ord('r'):
                        self.counting_service.reset_counts()
                        print("Counting statistics reset")
                
        except KeyboardInterrupt:
            print("Interrupted")
        except Exception as e:
            print(f"Error in processing loop: {e}")
        finally:
            self.stop()
    
    def _prepare_visualization(self, frame_data, detection_results, tracking_results, 
                              counting_results, ocr_results, fps):
        """
        Prepare visualization frame
        
        Returns:
            numpy.ndarray: Visualization frame
        """
        frame = frame_data['frame'].copy()
        
        # Draw counting line
        line = counting_results['counting_line']
        cv2.line(frame, tuple(line[0]), tuple(line[1]), (0, 0, 255), 2)
        
        # Draw tracks with different colors based on counting status
        for track in tracking_results['tracks']:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['box']
            
            # Determine color based on counting status
            color = (0, 165, 255)  # Orange for normal tracks
            if track_id in self.counting_service.counted_tracks:
                if self.counting_service.counted_tracks[track_id]['direction'] > 0:
                    color = (0, 255, 0)  # Green for up/right
                else:
                    color = (0, 0, 255)  # Red for down/left
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw track ID
            cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw path
            path = track['path']
            if len(path) >= 2:
                for i in range(1, len(path)):
                    cv2.line(frame, path[i-1], path[i], color, 2)
        
        # Draw plate detections and OCR results
        for plate in ocr_results['plates']:
            x1, y1, x2, y2 = plate['box']
            text = plate['text']
            
            # Draw plate box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # Draw text background
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, 
                         (int(x1), int(y1) - text_size[1] - 10),
                         (int(x1) + text_size[0] + 10, int(y1)), 
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (int(x1) + 5, int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw counts as a semi-transparent overlay
        counts = counting_results['counts']
        overlay = frame.copy()
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw counts text
        cv2.putText(frame, f"UP: {counts['up']}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"DOWN: {counts['down']}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"TOTAL: {counts['total']}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def _display_results(self, frame_data, detection_results, tracking_results, 
                        counting_results, ocr_results, fps):
        """Display the results on screen"""
        # Prepare visualization
        vis_frame = self._prepare_visualization(frame_data, detection_results, 
                                              tracking_results, counting_results, 
                                              ocr_results, fps)
        
        # Show frame
        cv2.imshow('Traffic Monitoring', vis_frame)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Traffic Monitoring Application')
    parser.add_argument('--source', help='Video source (file path, RTSP URL, or device ID)')
    parser.add_argument('--no-ui', action='store_true', help='Disable UI')
    parser.add_argument('--record', action='store_true', help='Record output video')
    
    args = parser.parse_args()
    
    # Create application
    app = TrafficMonitoringApp(
        video_source=args.source,
        show_ui=not args.no_ui,
        record_output=args.record
    )
    
    # Start application
    try:
        app.start()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        app.stop()

if __name__ == "__main__":
    main()