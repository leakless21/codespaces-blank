import sqlite3
import json
import datetime
import time
import paho.mqtt.client as mqtt
import sys
import os
import threading
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

class StorageService:
    """
    Service for storing traffic monitoring data in a database
    """
    def __init__(self, db_url=None):
        """
        Initialize the storage service
        
        Args:
            db_url (str): Database URL
        """
        self.db_url = db_url or config.DB_URL
        
        # Extract path from SQLite URL
        if self.db_url.startswith('sqlite:///'):
            self.db_path = self.db_url[10:]
        else:
            self.db_path = self.db_url
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.conn = None
        self.init_db()
        
        # Initialize MQTT client for subscribing to data
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        # Flag to indicate if service is running
        self.is_running = False
        self.mqtt_thread = None
        
        # Connect to MQTT broker
        try:
            self.mqtt_client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
            print(f"Connected to MQTT broker at {config.MQTT_BROKER}:{config.MQTT_PORT}")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
            print("Running in offline mode")
    
    def init_db(self):
        """Initialize the database with required tables"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            
            # Vehicles table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                timestamp REAL,
                frame_id INTEGER,
                x1 REAL,
                y1 REAL,
                x2 REAL,
                y2 REAL,
                class_id INTEGER,
                confidence REAL
            )
            ''')
            
            # Plates table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                timestamp REAL,
                frame_id INTEGER,
                plate_text TEXT,
                confidence REAL,
                x1 REAL,
                y1 REAL,
                x2 REAL,
                y2 REAL
            )
            ''')
            
            # Counts table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                track_id INTEGER,
                direction TEXT,
                count_type TEXT
            )
            ''')
            
            # Stats table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                total_count INTEGER,
                up_count INTEGER,
                down_count INTEGER
            )
            ''')
            
            self.conn.commit()
            print(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def start(self):
        """Start the storage service"""
        if self.is_running:
            print("Storage service is already running")
            return
        
        self.is_running = True
        
        # Start MQTT client in a separate thread
        self.mqtt_thread = threading.Thread(target=self.mqtt_client.loop_forever)
        self.mqtt_thread.daemon = True
        self.mqtt_thread.start()
        
        print("Storage service started")
    
    def stop(self):
        """Stop the storage service"""
        self.is_running = False
        
        if self.mqtt_client:
            self.mqtt_client.disconnect()
        
        if self.conn:
            self.conn.close()
        
        print("Storage service stopped")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the MQTT broker"""
        # Subscribe to relevant topics
        client.subscribe(f"{config.MQTT_TOPIC_PREFIX}/detections")
        client.subscribe(f"{config.MQTT_TOPIC_PREFIX}/tracks")
        client.subscribe(f"{config.MQTT_TOPIC_PREFIX}/plates")
        client.subscribe(f"{config.MQTT_TOPIC_PREFIX}/counts")
        print("Subscribed to MQTT topics")
    
    def on_mqtt_message(self, client, userdata, msg):
        """Callback for when a message is received from the MQTT broker"""
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode('utf-8'))
            
            # Process message based on topic
            if topic == f"{config.MQTT_TOPIC_PREFIX}/detections":
                self.process_detections(payload)
            elif topic == f"{config.MQTT_TOPIC_PREFIX}/tracks":
                self.process_tracks(payload)
            elif topic == f"{config.MQTT_TOPIC_PREFIX}/plates":
                self.process_plates(payload)
            elif topic == f"{config.MQTT_TOPIC_PREFIX}/counts":
                self.process_counts(payload)
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def process_detections(self, data):
        """Process detection data"""
        # We don't store all detections in the database
        # Only store vehicle tracks and plate readings
        pass
    
    def process_tracks(self, data):
        """Process tracking data"""
        timestamp = data.get('timestamp')
        frame_id = data.get('frame_id')
        tracks = data.get('tracks', [])
        
        # Store each track in the database
        if self.conn:
            cursor = self.conn.cursor()
            for track in tracks:
                try:
                    # Insert vehicle track
                    cursor.execute(
                        '''
                        INSERT INTO vehicles (
                            track_id, timestamp, frame_id, x1, y1, x2, y2, class_id, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            track['track_id'],
                            timestamp,
                            frame_id,
                            track['box'][0],
                            track['box'][1],
                            track['box'][2],
                            track['box'][3],
                            track['class_id'],
                            track['confidence']
                        )
                    )
                except Exception as e:
                    print(f"Error inserting track: {e}")
            
            self.conn.commit()
    
    def process_plates(self, data):
        """Process license plate data"""
        timestamp = data.get('timestamp')
        frame_id = data.get('frame_id')
        plates = data.get('plates', [])
        
        # Store each plate in the database
        if self.conn:
            cursor = self.conn.cursor()
            for plate in plates:
                try:
                    # Get box coordinates if available, otherwise use defaults
                    box = plate.get('box', [0, 0, 0, 0])
                    
                    # Insert plate detection
                    cursor.execute(
                        '''
                        INSERT INTO plates (
                            track_id, timestamp, frame_id, plate_text, confidence, x1, y1, x2, y2
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''',
                        (
                            plate['track_id'],
                            timestamp,
                            frame_id,
                            plate['text'],
                            plate['confidence'],
                            box[0] if len(box) > 0 else 0,
                            box[1] if len(box) > 1 else 0,
                            box[2] if len(box) > 2 else 0,
                            box[3] if len(box) > 3 else 0
                        )
                    )
                except Exception as e:
                    print(f"Error inserting plate: {e}")
            
            self.conn.commit()
    
    def process_counts(self, data):
        """Process counting data"""
        timestamp = data.get('timestamp')
        counts = data.get('counts', {})
        new_counts = data.get('new_counts', [])
        
        # Store count statistics
        if self.conn and counts:
            cursor = self.conn.cursor()
            try:
                # Insert count statistics
                cursor.execute(
                    '''
                    INSERT INTO stats (
                        timestamp, total_count, up_count, down_count
                    ) VALUES (?, ?, ?, ?)
                    ''',
                    (
                        timestamp,
                        counts.get('total', 0),
                        counts.get('up', 0),
                        counts.get('down', 0)
                    )
                )
            except Exception as e:
                print(f"Error inserting stats: {e}")
        
        # Store individual count events
        if self.conn and new_counts:
            cursor = self.conn.cursor()
            for count in new_counts:
                try:
                    # Insert count event
                    cursor.execute(
                        '''
                        INSERT INTO counts (
                            timestamp, track_id, direction, count_type
                        ) VALUES (?, ?, ?, ?)
                        ''',
                        (
                            count.get('timestamp', timestamp),
                            count.get('track_id', 0),
                            str(count.get('direction', 0)),
                            count.get('type', 'unknown')
                        )
                    )
                except Exception as e:
                    print(f"Error inserting count: {e}")
            
            self.conn.commit()
    
    def add_vehicle(self, track_id, frame_id, timestamp, box, class_id, confidence):
        """
        Add a vehicle to the database
        
        Args:
            track_id (int): Track ID
            frame_id (int): Frame ID
            timestamp (float): Timestamp
            box (list): Bounding box [x1, y1, x2, y2]
            class_id (int): Class ID
            confidence (float): Detection confidence
        """
        if self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    '''
                    INSERT INTO vehicles (
                        track_id, timestamp, frame_id, x1, y1, x2, y2, class_id, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        track_id,
                        timestamp,
                        frame_id,
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        class_id,
                        confidence
                    )
                )
                self.conn.commit()
            except Exception as e:
                print(f"Error adding vehicle: {e}")
    
    def add_plate(self, track_id, frame_id, timestamp, plate_text, confidence, box):
        """
        Add a license plate to the database
        
        Args:
            track_id (int): Track ID
            frame_id (int): Frame ID
            timestamp (float): Timestamp
            plate_text (str): License plate text
            confidence (float): OCR confidence
            box (list): Bounding box [x1, y1, x2, y2]
        """
        if self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    '''
                    INSERT INTO plates (
                        track_id, timestamp, frame_id, plate_text, confidence, x1, y1, x2, y2
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        track_id,
                        timestamp,
                        frame_id,
                        plate_text,
                        confidence,
                        box[0],
                        box[1],
                        box[2],
                        box[3]
                    )
                )
                self.conn.commit()
            except Exception as e:
                print(f"Error adding plate: {e}")
    
    def add_count(self, track_id, timestamp, direction, count_type):
        """
        Add a count event to the database
        
        Args:
            track_id (int): Track ID
            timestamp (float): Timestamp
            direction (int): Direction of movement (1=up, -1=down)
            count_type (str): Type of count (up, down)
        """
        if self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    '''
                    INSERT INTO counts (
                        timestamp, track_id, direction, count_type
                    ) VALUES (?, ?, ?, ?)
                    ''',
                    (
                        timestamp,
                        track_id,
                        str(direction),
                        count_type
                    )
                )
                self.conn.commit()
            except Exception as e:
                print(f"Error adding count: {e}")
    
    def update_stats(self, timestamp, total_count, up_count, down_count):
        """
        Update statistics in the database
        
        Args:
            timestamp (float): Timestamp
            total_count (int): Total count
            up_count (int): Up count
            down_count (int): Down count
        """
        if self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    '''
                    INSERT INTO stats (
                        timestamp, total_count, up_count, down_count
                    ) VALUES (?, ?, ?, ?)
                    ''',
                    (
                        timestamp,
                        total_count,
                        up_count,
                        down_count
                    )
                )
                self.conn.commit()
            except Exception as e:
                print(f"Error updating stats: {e}")
    
    def get_latest_stats(self):
        """
        Get the latest statistics from the database
        
        Returns:
            dict: Latest statistics
        """
        if self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    '''
                    SELECT * FROM stats
                    ORDER BY timestamp DESC
                    LIMIT 1
                    '''
                )
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'timestamp': row[1],
                        'total_count': row[2],
                        'up_count': row[3],
                        'down_count': row[4]
                    }
            except Exception as e:
                print(f"Error getting latest stats: {e}")
        
        return {}
    
    def get_plate_counts(self):
        """
        Get the count of detected plates
        
        Returns:
            int: Number of detected plates
        """
        if self.conn:
            cursor = self.conn.cursor()
            try:
                cursor.execute('SELECT COUNT(DISTINCT plate_text) FROM plates')
                row = cursor.fetchone()
                return row[0] if row else 0
            except Exception as e:
                print(f"Error getting plate counts: {e}")
        
        return 0
    
    def get_vehicle_counts_by_time(self, interval='hour'):
        """
        Get vehicle counts aggregated by time interval
        
        Args:
            interval (str): Time interval ('hour', 'day', 'week')
            
        Returns:
            list: Vehicle counts by time interval
        """
        if self.conn:
            cursor = self.conn.cursor()
            try:
                if interval == 'hour':
                    # Group by hour
                    cursor.execute(
                        '''
                        SELECT 
                            strftime('%Y-%m-%d %H:00:00', datetime(timestamp, 'unixepoch')) as hour,
                            COUNT(*) as count
                        FROM counts
                        GROUP BY hour
                        ORDER BY hour DESC
                        LIMIT 24
                        '''
                    )
                elif interval == 'day':
                    # Group by day
                    cursor.execute(
                        '''
                        SELECT 
                            strftime('%Y-%m-%d', datetime(timestamp, 'unixepoch')) as day,
                            COUNT(*) as count
                        FROM counts
                        GROUP BY day
                        ORDER BY day DESC
                        LIMIT 30
                        '''
                    )
                elif interval == 'week':
                    # Group by week
                    cursor.execute(
                        '''
                        SELECT 
                            strftime('%Y-%W', datetime(timestamp, 'unixepoch')) as week,
                            COUNT(*) as count
                        FROM counts
                        GROUP BY week
                        ORDER BY week DESC
                        LIMIT 12
                        '''
                    )
                
                return cursor.fetchall()
            except Exception as e:
                print(f"Error getting vehicle counts by time: {e}")
        
        return []

# Simple usage example
if __name__ == "__main__":
    # Create storage service
    storage_service = StorageService()
    
    # Start service
    storage_service.start()
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Periodically check stats
            stats = storage_service.get_latest_stats()
            if stats:
                print(f"Latest stats: UP={stats['up_count']}, DOWN={stats['down_count']}, TOTAL={stats['total_count']}")
            
            # Get plate count
            plate_count = storage_service.get_plate_counts()
            print(f"Unique plates: {plate_count}")
            
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        storage_service.stop()