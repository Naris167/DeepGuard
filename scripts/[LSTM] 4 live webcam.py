import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
import threading
import time
import queue
import sys
from send_message import LineNotifier

# ======================== CONFIGURATION ========================

# Model paths
LSTM_MODEL_PATH = "./models/fall_detection_20251017_213109.h5"  # CHANGE THIS to your trained model
YOLO_MODEL_PATH = "./models/yolo11l-pose.pt"

# Webcam settings
WEBCAM_INDEX = 0  # 0 for default webcam, change if you have multiple cameras
TARGET_FPS = 24
WEBCAM_WIDTH = 1280   # Actual capture resolution (affects YOLO accuracy)
WEBCAM_HEIGHT = 720   # Actual capture resolution (affects YOLO accuracy)
DISPLAY_WIDTH = 640   # Display window width
DISPLAY_HEIGHT = 360  # Display window height

# Parameters
WINDOW_SIZE = 48  # frames (2 seconds at 24fps)
PREDICTION_STRIDE = 24  # frames (1 second at 24fps)
MAX_FORWARD_FILL = 48  # frames
SMOOTHING_WINDOW = 4  # moving average of last 4 predictions
CONFIDENCE_THRESHOLD = 0.5  # threshold for fall detection

# Buffer management
MAX_BUFFER_SIZE_GB = 10  # Reset buffer when exceeds 10GB

# Visualization
SHOW_BBOX = True  # Set to False to hide bounding boxes on left window


notifier = LineNotifier(env_path="./scripts/.env")

# ================================================================


class PersonTracker:
    """Track individual person's frame buffer and predictions"""
    
    def __init__(self, track_id):
        self.track_id = track_id
        self.buffer = []  # Will hold up to 48 frames
        self.last_valid_frame = None
        self.frames_since_valid = 0
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.frames_since_prediction = 0
        self.saved_frames = set()
    
    def add_frame(self, normalized_keypoints, frame_num):
        """Add a frame to the buffer with forward-fill logic"""
        
        if normalized_keypoints is not None:
            self.buffer.append(normalized_keypoints)
            self.last_valid_frame = normalized_keypoints.copy()
            self.frames_since_valid = 0
        else:
            self.frames_since_valid += 1
            
            if self.last_valid_frame is not None and self.frames_since_valid <= MAX_FORWARD_FILL:
                filled_frame = self.last_valid_frame.copy()
                filled_frame[:, 2] = 0.0
                self.buffer.append(filled_frame)
            else:
                self.buffer.append(np.full((17, 3), -1.0))
        
        if len(self.buffer) > WINDOW_SIZE:
            self.buffer.pop(0)
        
        self.frames_since_prediction += 1
    
    def can_predict(self):
        """Check if buffer is ready for prediction"""
        return len(self.buffer) == WINDOW_SIZE and self.frames_since_prediction >= PREDICTION_STRIDE
    
    def get_window(self):
        """Get current 48-frame window"""
        return np.array(self.buffer)
    
    def add_prediction(self, prediction):
        """Add prediction and return smoothed result"""
        self.prediction_history.append(prediction)
        self.frames_since_prediction = 0
        
        if len(self.prediction_history) > 0:
            return np.mean(self.prediction_history)
        return prediction
    
    def is_fall_detected(self, smoothed_prediction):
        """Check if fall is detected based on smoothed prediction"""
        return smoothed_prediction > CONFIDENCE_THRESHOLD
    
    def get_frames_to_save(self, current_frame_num):
        """Get list of frame numbers to save (12 frames from current window)"""
        window_start = current_frame_num - WINDOW_SIZE + 1
        
        frames_to_save = []
        for idx in range(0, WINDOW_SIZE, 4):
            absolute_frame_num = window_start + idx
            if absolute_frame_num not in self.saved_frames:
                frames_to_save.append(absolute_frame_num)
                self.saved_frames.add(absolute_frame_num)
        
        return frames_to_save


class FrameBuffer:
    """Thread-safe frame buffer with size monitoring"""
    
    def __init__(self):
        self.frames = {}  # frame_num -> frame_data
        self.lock = threading.Lock()
        self.current_size_bytes = 0
        self.max_size_bytes = MAX_BUFFER_SIZE_GB * 1024 * 1024 * 1024
    
    def add_frame(self, frame_num, frame):
        """Add frame to buffer"""
        with self.lock:
            frame_size = frame.nbytes
            self.frames[frame_num] = frame.copy()
            self.current_size_bytes += frame_size
    
    def get_frame(self, frame_num):
        """Get frame by frame number"""
        with self.lock:
            return self.frames.get(frame_num)
    
    def should_reset(self):
        """Check if buffer should be reset"""
        with self.lock:
            return self.current_size_bytes > self.max_size_bytes
    
    def reset(self):
        """Clear buffer"""
        with self.lock:
            self.frames.clear()
            self.current_size_bytes = 0
            print("\n⚠️  Buffer exceeded 10GB - RESETTING")
    
    def get_size_mb(self):
        """Get current buffer size in MB"""
        with self.lock:
            return self.current_size_bytes / (1024 * 1024)


def normalize_by_bbox(keypoints, bbox):
    """Normalize keypoints by bounding box"""
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width <= 0 or bbox_height <= 0:
        return None
    
    normalized = keypoints.copy()
    normalized[:, 0] = (keypoints[:, 0] - x1) / bbox_width
    normalized[:, 1] = (keypoints[:, 1] - y1) / bbox_height
    
    return normalized


def setup_output_folder():
    """Create output folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"./demos/{timestamp}_webcam"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def save_fall_frames(frame_buffer, frame_numbers, output_folder, track_id, confidence):
    """Save specific frames from buffer"""
    for frame_num in frame_numbers:
        frame = frame_buffer.get_frame(frame_num)
        
        if frame is not None:
            filename = f"{frame_num}_{confidence:.3f}_person_{track_id}.png"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"  Saved frame {frame_num} (person {track_id})")


def draw_bbox(frame, bbox, is_fall, track_id):
    """Draw bounding box on frame"""
    if not SHOW_BBOX:
        return
    
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 0, 255) if is_fall else (0, 255, 0)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID:{track_id}", (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def capture_thread(frame_queue, frame_buffer, stop_event, reset_event):
    """Thread 1: Capture from webcam and display raw feed"""
    
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        stop_event.set()
        return
    
    print(f"✓ Webcam opened at {WEBCAM_WIDTH}x{WEBCAM_HEIGHT} @ {TARGET_FPS} fps")
    
    frame_num = 0
    frame_time = 1.0 / TARGET_FPS
    
    while not stop_event.is_set():
        start_time = time.time()
        
        ret, frame = cap.read()
        
        if not ret:
            print("ERROR: Cannot read from webcam")
            break
        
        # Check if buffer needs reset
        if frame_buffer.should_reset():
            frame_buffer.reset()
            reset_event.set()  # Signal processing thread to reset trackers
        
        # Add to buffer and queue
        frame_buffer.add_frame(frame_num, frame)
        
        try:
            frame_queue.put((frame_num, frame.copy()), timeout=0.1)
        except queue.Full:
            pass  # Skip if queue is full
        
        # Display raw feed (right window)
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow('Right - Live Webcam', display_frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break
        
        frame_num += 1
        
        # Maintain target FPS
        elapsed = time.time() - start_time
        sleep_time = frame_time - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    cap.release()
    print("\nCapture thread stopped")


def processing_thread(frame_queue, frame_buffer, stop_event, reset_event, 
                     lstm_model, yolo_model, output_folder):
    """Thread 2: Process frames with YOLO + LSTM and display"""
    
    person_trackers = {}
    
    while not stop_event.is_set():
        try:
            # Get frame from queue
            frame_num, frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        # Check if reset requested
        if reset_event.is_set():
            person_trackers.clear()
            print("  Trackers cleared after buffer reset")
            reset_event.clear()
        
        # Create copy for processing
        processed_frame = frame.copy()
        
        # Run YOLO pose detection
        results = yolo_model.track(processed_frame, persist=True, verbose=False)
        
        current_track_ids = set()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id
            
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)
                keypoints = results[0].keypoints.data.cpu().numpy()
                
                for idx, track_id in enumerate(track_ids):
                    current_track_ids.add(track_id)
                    
                    person_keypoints = keypoints[idx]
                    bbox = boxes[idx]
                    
                    normalized = normalize_by_bbox(person_keypoints, bbox)
                    
                    if track_id not in person_trackers:
                        person_trackers[track_id] = PersonTracker(track_id)
                        print(f"  New person detected: ID {track_id}")
                    
                    tracker = person_trackers[track_id]
                    tracker.add_frame(normalized, frame_num)
                    
                    is_fall = False
                    smoothed_prediction = 0.0
                    
                    if tracker.can_predict():
                        window = tracker.get_window()
                        X = np.expand_dims(window, axis=0)
                        
                        prediction = lstm_model.predict(X, verbose=0)[0][0]
                        smoothed_prediction = tracker.add_prediction(prediction)
                        is_fall = tracker.is_fall_detected(smoothed_prediction)
                        
                        status = "FALL DETECTED!" if is_fall else "No fall"
                        print(f"  Frame {frame_num} | Person {track_id} | "
                              f"Pred: {prediction:.3f} | Smoothed: {smoothed_prediction:.3f} | {status}")
                        
                        if is_fall:
                            frames_to_save = tracker.get_frames_to_save(frame_num)
                            if frames_to_save:
                                print(f"  Saving {len(frames_to_save)} frames for person {track_id}...")
                                save_fall_frames(frame_buffer, frames_to_save, output_folder, 
                                               track_id, smoothed_prediction)
                                result = notifier.send_fall_detection_alert(
                                            camera_id="web cam demo",
                                            location="กล้อง webcam",
                                            timestamp=datetime.now(),
                                            message="⚠️ กรุณารีบตรวจสอบทันที!",
                                            group_id="Cf7291686ec355f681be9fc4df8b23f4f"  # Specific group
                                        )
                                print(result)
                    
                    draw_bbox(processed_frame, bbox, is_fall, track_id)
        
        # Clean up trackers
        trackers_to_remove = []
        for track_id, tracker in person_trackers.items():
            if track_id not in current_track_ids:
                tracker.frames_since_valid += 1
                if tracker.frames_since_valid > MAX_FORWARD_FILL:
                    trackers_to_remove.append(track_id)
        
        for track_id in trackers_to_remove:
            print(f"  Removing tracker for person {track_id}")
            del person_trackers[track_id]
        
        # Add buffer size info
        buffer_mb = frame_buffer.get_size_mb()
        cv2.putText(processed_frame, f"Buffer: {buffer_mb:.1f}MB", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display processed frame (left window)
        display_frame = cv2.resize(processed_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        cv2.imshow('Left - Fall Detection', display_frame)
        cv2.waitKey(1)
    
    print("Processing thread stopped")


def main():
    print("="*70)
    print("FALL DETECTION - LIVE WEBCAM")
    print("="*70)
    
    # Check model files
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"ERROR: LSTM model not found at {LSTM_MODEL_PATH}")
        return
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"ERROR: YOLO model not found at {YOLO_MODEL_PATH}")
        return
    
    # Load models
    print("\n[1/3] Loading models...")
    lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
    print(f"  ✓ LSTM model loaded")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"  ✓ YOLO model loaded")
    
    # Setup output folder
    output_folder = setup_output_folder()
    print(f"  ✓ Output folder: {output_folder}")
    
    # Create shared objects
    print("\n[2/3] Initializing threads...")
    frame_queue = queue.Queue(maxsize=10)
    frame_buffer = FrameBuffer()
    stop_event = threading.Event()
    reset_event = threading.Event()
    
    # Create threads
    capture = threading.Thread(
        target=capture_thread,
        args=(frame_queue, frame_buffer, stop_event, reset_event)
    )
    
    processing = threading.Thread(
        target=processing_thread,
        args=(frame_queue, frame_buffer, stop_event, reset_event, 
              lstm_model, yolo_model, output_folder)
    )
    
    # Start threads
    print("\n[3/3] Starting live detection...")
    print("  Press 'q' to quit")
    print("-"*70)
    
    capture.start()
    processing.start()
    
    # Wait for threads to finish
    capture.join()
    processing.join()
    
    # Cleanup
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("WEBCAM SESSION COMPLETE")
    print("="*70)
    print(f"Output folder: {output_folder}")
    print(f"Final buffer size: {frame_buffer.get_size_mb():.1f}MB")
    print("="*70)


if __name__ == "__main__":
    main()