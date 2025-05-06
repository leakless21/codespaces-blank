import os
import sys
import time
import cv2
import argparse
import threading
import queue
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
    def __init__(self, video_source=None, show_ui=True, record_output=False, output_path=None, render_video_only=False):
        """
        Initialize the traffic monitoring application
        
        Args:
            video_source (str): Path to video file, RTSP URL, or device ID
            show_ui (bool): Whether to show the user interface
            record_output (bool): Whether to record the output video
            output_path (str): Path to save the output video (if None, a default path will be used)
            render_video_only (bool): Whether to only render a video without live display
        """
        self.video_source = video_source or config.VIDEO_SOURCE
        self.show_ui = show_ui and not render_video_only  # Disable UI if render_video_only is True
        self.record_output = record_output or render_video_only  # Always record if render_video_only is True
        self.output_path = output_path
        self.render_video_only = render_video_only
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
        self.video_writer_lock = threading.Lock()
        
        # Create a dedicated queue and thread for video writing to prevent multithreading issues
        self.video_queue = queue.Queue(maxsize=30)  # Limit queue size to prevent memory issues
        self.video_writer_thread = None
        self.video_writer_running = False
        
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
            
            # Start video writer thread if initialized successfully
            if self.video_writer is not None:
                self.video_writer_running = True
                self.video_writer_thread = threading.Thread(target=self._video_writer_thread)
                self.video_writer_thread.daemon = True
                self.video_writer_thread.start()
                print("Video writer thread started")
        
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
        
        # Stop video writer thread
        if self.video_writer_running:
            self.video_writer_running = False
            # Put None in the queue to signal the thread to exit
            try:
                self.video_queue.put(None, block=False)
            except queue.Full:
                # Clear one item if queue is full
                try:
                    self.video_queue.get_nowait()
                    self.video_queue.put(None, block=False)
                except:
                    pass
                    
            # Wait for video thread to finish
            if self.video_writer_thread:
                self.video_writer_thread.join(timeout=2.0)
                print("Video writer thread stopped")
        
        # Release video writer with lock to prevent concurrent access
        if self.video_writer is not None:
            with self.video_writer_lock:
                self.video_writer.release()
                self.video_writer = None
        
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
        
        # Get the source video's FPS
        source_fps = self.video_service.get_fps()
        print(f"Using source video FPS: {source_fps}")
        
        # Create output directory if it doesn't exist
        if self.output_path:
            output_path = self.output_path
            # Ensure the parent directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        else:
            output_dir = Path(config.DATA_DIR) / "recordings"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output file name based on current time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"traffic_monitoring_{timestamp}.mp4")
        
        # Default to .mp4 format with modern codecs
        if not (output_path.lower().endswith('.mp4') or output_path.lower().endswith('.mkv')):
            output_path = output_path.rsplit('.', 1)[0] + '.mp4'
            
        print(f"Recording output to {output_path}")
        
        # Try different codecs in order of preference for modern formats
        try:
            # Try H.264 codec first (excellent compression and wide compatibility)
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, source_fps, (width, height))
            
            if not self.video_writer.isOpened():
                print("Failed with H264 codec, trying X264")
                
                # Try X264 codec next (another H.264 implementation)
                fourcc = cv2.VideoWriter_fourcc(*'X264')
                self.video_writer = cv2.VideoWriter(
                    output_path, fourcc, source_fps, (width, height))
                
                if not self.video_writer.isOpened():
                    print("Failed with X264 codec, trying XVID")
                    
                    # Try XVID codec (good compatibility)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(
                        output_path, fourcc, source_fps, (width, height))
                    
                    if not self.video_writer.isOpened():
                        print("Failed with XVID codec, trying MP4V")
                        
                        # Try MP4V codec (another option for MP4 containers)
                        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                        self.video_writer = cv2.VideoWriter(
                            output_path, fourcc, source_fps, (width, height))
                        
                        if not self.video_writer.isOpened():
                            print("Failed with MP4V codec, falling back to AVI format with MJPG")
                            
                            # Fallback to reliable AVI format with MJPG if all else fails
                            output_path = output_path.rsplit('.', 1)[0] + '.avi'
                            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                            self.video_writer = cv2.VideoWriter(
                                output_path, fourcc, source_fps, (width, height))
                            
                            if not self.video_writer.isOpened():
                                print("Video writer could not be initialized. Turning off recording.")
                                self.video_writer = None
                                self.record_output = False
            
        except Exception as e:
            print(f"Error initializing video writer: {e}")
            print("Video recording will be disabled")
            self.video_writer = None
            self.record_output = False
            
        # Return frame to pipeline
        self.video_service.rewind_one_frame()
    
    def _process_frames(self):
        """Process frames from the video service"""
        try:
            frame_count = 0
            
            # Try to get a more accurate frame count estimate
            estimated_frames = self.video_service.estimate_total_frames()
            reported_frames = self.video_service.get_total_frames()
            
            # Use the most reliable frame count, with some validation
            if estimated_frames > 0 and estimated_frames > reported_frames * 0.8:
                # The estimated count seems reasonable (not too low compared to reported)
                total_frames = estimated_frames
                print(f"Using estimated frame count: {total_frames}")
            elif reported_frames > 0:
                # Fall back to the reported count
                total_frames = reported_frames
                print(f"Using reported frame count: {total_frames}")
            else:
                # Can't determine frame count
                total_frames = 0
                print("Unable to determine total frame count, progress reporting will be disabled")
            
            # Add a safety margin to prevent exceeding 100%
            if total_frames > 0:
                # Add 5% safety margin to total_frames to avoid exceeding 100%
                total_frames = int(total_frames * 1.05)
                print(f"Total frames with safety margin: {total_frames}")
            
            # For render-only mode, show a progress bar
            if self.render_video_only and total_frames > 0:
                print(f"Rendering video: 0/{total_frames} frames (0%)")
                
            # If render-only mode is enabled but recording fails, we'll need to show the UI
            if self.render_video_only and not self.video_writer:
                print("WARNING: Recording failed but render-only mode was requested.")
                print("Falling back to interactive mode with UI display.")
                self.show_ui = True
            
            last_update_time = time.time()
            max_accurate_progress = 99.0  # Never show 100% until truly complete
            
            while self.running:
                # Get frame from video service
                frame_data = self.video_service.get_frame()
                if frame_data is None:
                    print("Finished processing all frames")
                    self.running = False
                    break
                
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
                if self.record_output and self.video_writer is not None and self.video_writer_running:
                    try:
                        # Generate the same visualized frame as in display_results
                        vis_frame = self._prepare_visualization(frame_data, detection_results, 
                                                             tracking_results, counting_results, 
                                                             ocr_results, fps)
                        
                        # Add frame to queue for the dedicated writer thread instead of writing directly
                        try:
                            if not self.video_queue.full():
                                self.video_queue.put(vis_frame, block=False)
                            else:
                                # Skip this frame if queue is full
                                if frame_count % 30 == 0:  # Only log occasionally to avoid spamming
                                    print("Warning: Video writing queue is full, skipping frame")
                        except Exception as e:
                            print(f"Error queuing video frame: {e}")
                    except Exception as e:
                        print(f"Error preparing video frame: {e}")
                        print("Disabling video recording due to errors")
                        self.record_output = False
                        self.video_writer_running = False
                        if self.video_writer:
                            try:
                                with self.video_writer_lock:
                                    self.video_writer.release()
                                    self.video_writer = None
                            except:
                                pass
                        # If we were in render-only mode, enable UI since recording failed
                        if self.render_video_only:
                            self.show_ui = True
                            print("Falling back to interactive mode with UI")
                
                # Show progress for render-only mode (limit updates to once per second)
                current_time = time.time()
                if self.render_video_only and (current_time - last_update_time) >= 1.0:
                    last_update_time = current_time
                    if total_frames > 0:
                        # Calculate progress with a cap to prevent exceeding max_accurate_progress
                        progress = min((frame_count / total_frames) * 100, max_accurate_progress)
                        print(f"\rProcessing: {frame_count} frames ({progress:.1f}%)", end="")
                    else:
                        # If total_frames is unknown, just show the current frame count
                        print(f"\rProcessing frame: {frame_count}", end="")
        finally:
            # Show final progress message
            if self.render_video_only:
                if total_frames > 0:
                    # Use 100% for final message only
                    print(f"\nCompleted processing {frame_count} frames (100%)")
                else:
                    print(f"\nProcessed {frame_count} frames total")
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
                color = (0, 255, 0)  # Green for counted tracks
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw track ID with better visibility
            text = f"ID:{track_id}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, 
                         (int(x1), int(y1) - text_size[1] - 5),
                         (int(x1) + text_size[0] + 5, int(y1)), 
                         (0, 0, 0), -1)
            cv2.putText(frame, text, (int(x1) + 2, int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
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
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(frame, timestamp, (20, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw counts as a semi-transparent overlay
        counts = counting_results['counts']
        overlay = frame.copy()
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (10, 10), (300, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw total count text - larger and more prominent
        cv2.putText(frame, f"TOTAL COUNT: {counts['total']}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
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
    
    def _video_writer_thread(self):
        """Thread function for writing video frames"""
        while self.video_writer_running:
            try:
                # Get frame from queue with timeout to check for exit condition
                frame = self.video_queue.get(timeout=0.5)
                
                # None is a signal to exit the thread
                if frame is None:
                    break
                
                # Write frame with lock to prevent concurrent access
                with self.video_writer_lock:
                    if self.video_writer is not None:
                        self.video_writer.write(frame)
                
                # Mark task as done
                self.video_queue.task_done()
                
            except queue.Empty:
                # Timeout, just continue
                continue
            except Exception as e:
                print(f"Error in video writer thread: {e}")
        
        print("Video writer thread exited")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Traffic Monitoring Application')
    parser.add_argument('--source', help='Video source (file path, RTSP URL, or device ID)')
    parser.add_argument('--no-ui', action='store_true', help='Disable UI')
    parser.add_argument('--record', action='store_true', help='Record output video')
    parser.add_argument('--output', help='Output video file path')
    parser.add_argument('--render-video', action='store_true', 
                        help='Only render output video without live display')
    
    args = parser.parse_args()
    
    # Create application
    app = TrafficMonitoringApp(
        video_source=args.source,
        show_ui=not args.no_ui,
        record_output=args.record,
        output_path=args.output,
        render_video_only=args.render_video
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