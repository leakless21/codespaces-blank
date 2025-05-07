import subprocess
import sys
import time
import threading
import os
import socket
import logging
import signal
import shutil
import platform

# Set up logging
logger = logging.getLogger(__name__)

class MQTTBroker:
    """
    A utility to automatically manage an MQTT broker instance.
    This will attempt to connect to an existing broker first,
    and if that fails, it will start an embedded broker.
    """
    def __init__(self, host="localhost", port=1883):
        self.host = host
        self.port = port
        self.process = None
        self.running = False
        self.thread = None
        self.is_embedded = False
        self.lock = threading.Lock()
        self.config_path = None

    def _is_port_in_use(self):
        """Check if the MQTT broker port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0

    def _find_mosquitto(self):
        """Find the Mosquitto executable path"""
        mosquitto_names = ['mosquitto', 'mosquitto.exe']
        
        # Check PATH first
        for name in mosquitto_names:
            path = shutil.which(name)
            if path:
                return path
                
        # Check common installation paths
        common_paths = []
        
        if platform.system() == 'Windows':
            program_files = os.environ.get('ProgramFiles', 'C:\\Program Files')
            program_files_x86 = os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)')
            common_paths.extend([
                os.path.join(program_files, 'Mosquitto', 'mosquitto.exe'),
                os.path.join(program_files_x86, 'Mosquitto', 'mosquitto.exe'),
            ])
        elif platform.system() == 'Linux':
            common_paths.extend([
                '/usr/sbin/mosquitto',
                '/usr/local/sbin/mosquitto',
                '/usr/bin/mosquitto',
                '/usr/local/bin/mosquitto',
            ])
        elif platform.system() == 'Darwin':  # macOS
            common_paths.extend([
                '/usr/local/sbin/mosquitto',
                '/usr/local/bin/mosquitto',
                '/opt/homebrew/bin/mosquitto',
            ])
            
        for path in common_paths:
            if os.path.isfile(path):
                return path
                
        return None

    def _create_config(self):
        """Create a minimal Mosquitto configuration file"""
        import tempfile
        
        config_dir = tempfile.mkdtemp(prefix="mqtt_broker_")
        config_path = os.path.join(config_dir, "mosquitto.conf")
        
        with open(config_path, 'w') as f:
            f.write("# Auto-generated Mosquitto configuration for embedded broker\n")
            f.write(f"port {self.port}\n")
            f.write("allow_anonymous true\n")
            f.write("log_type error\n")
            f.write("log_type warning\n")
            
            # Create a persistence directory
            persistence_dir = os.path.join(config_dir, "data")
            os.makedirs(persistence_dir, exist_ok=True)
            f.write(f"persistence true\n")
            f.write(f"persistence_location {persistence_dir}\n")
            
        self.config_path = config_path
        return config_path

    def _start_broker_process(self):
        """Start the Mosquitto broker process"""
        mosquitto_path = self._find_mosquitto()
        if not mosquitto_path:
            logger.error("Mosquitto executable not found. Cannot start embedded broker.")
            return False
            
        logger.info(f"Found Mosquitto executable at {mosquitto_path}")
        
        # Create config file
        config_path = self._create_config()
        
        # Start the broker
        try:
            self.process = subprocess.Popen(
                [mosquitto_path, "-c", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Wait a moment for it to start
            time.sleep(1)
            
            # Check if process is still running
            if self.process.poll() is not None:
                # Process terminated already, get error
                _, stderr = self.process.communicate()
                logger.error(f"Mosquitto broker failed to start: {stderr}")
                return False
                
            logger.info(f"Started embedded MQTT broker on port {self.port}")
            self.is_embedded = True
            return True
            
        except Exception as e:
            logger.error(f"Error starting Mosquitto broker: {e}")
            return False

    def _cleanup_broker(self):
        """Clean up the broker process and config files"""
        if self.process:
            try:
                if platform.system() == 'Windows':
                    self.process.terminate()
                else:
                    # SIGTERM is more graceful than SIGKILL
                    os.kill(self.process.pid, signal.SIGTERM)
                    
                # Wait for process to terminate
                self.process.wait(timeout=5)
                logger.info("Embedded MQTT broker terminated")
                
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                if platform.system() == 'Windows':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)])
                else:
                    os.kill(self.process.pid, signal.SIGKILL)
                logger.warning("Had to forcibly terminate embedded MQTT broker")
                
            except Exception as e:
                logger.error(f"Error terminating MQTT broker: {e}")
                
            self.process = None
        
        # Clean up config directory
        if self.config_path:
            try:
                config_dir = os.path.dirname(self.config_path)
                shutil.rmtree(config_dir, ignore_errors=True)
                logger.info(f"Removed temporary config directory {config_dir}")
            except Exception as e:
                logger.error(f"Error removing config directory: {e}")
                
            self.config_path = None

    def start(self):
        """
        Start the MQTT broker if needed.
        First tries to connect to an existing broker,
        and if that fails, starts an embedded broker.
        
        Returns:
            bool: True if broker is available (either existing or newly started)
        """
        with self.lock:
            if self.running:
                return True
                
            # Check if there's already an MQTT broker running
            if self._is_port_in_use():
                logger.info(f"Found existing MQTT broker at {self.host}:{self.port}")
                self.running = True
                self.is_embedded = False
                return True
                
            # No broker found, try to start one
            logger.info(f"No MQTT broker found on {self.host}:{self.port}, starting embedded broker...")
            success = self._start_broker_process()
            self.running = success
            
            return success

    def stop(self):
        """Stop the embedded MQTT broker if it was started by this utility"""
        with self.lock:
            if not self.running or not self.is_embedded:
                return
                
            logger.info("Stopping embedded MQTT broker...")
            self._cleanup_broker()
            self.running = False
            self.is_embedded = False

    def __enter__(self):
        """Support for 'with' statement"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        """Support for 'with' statement"""
        self.stop()

# Simple usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    broker = MQTTBroker()
    if broker.start():
        print(f"MQTT broker is {'pre-existing' if not broker.is_embedded else 'started'}")
        print("Press Ctrl+C to exit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            broker.stop()
    else:
        print("Failed to ensure MQTT broker is running")