import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
import tensorflow as tf
from tensorflow import keras
from ultralytics import YOLO
from send_message import LineNotifier

# ======================== CONFIGURATION ========================

# Model paths
LSTM_MODEL_PATH = "./models/fall_detection_20251017_213109.h5"  # CHANGE THIS to your trained model
YOLO_MODEL_PATH = "./models/yolo11l-pose.pt"

# Video path
VIDEO_PATH = "./demos/scenario/[MEDIUM] - cam3-2.avi"  # CHANGE THIS

# Parameters
WINDOW_SIZE = 48  # frames
PREDICTION_STRIDE = 12  # frames (1 second at 24fps)
MAX_FORWARD_FILL = 48  # frames
SMOOTHING_WINDOW = 3  # moving average of last 4 predictions
CONFIDENCE_THRESHOLD = 0.5  # threshold for fall detection

# Visualization
SHOW_BBOX = True  # Set to False to hide bounding boxes

notifier = LineNotifier(env_path="./scripts/.env")

# ================================================================


class PersonTracker:
    """Track individual person's frame buffer and predictions"""
    
    def __init__(self, track_id):
        self.track_id = track_id
        self.buffer = []  # Will hold up to 48 frames
        self.last_valid_frame = None
        self.frames_since_valid = 0
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)  # Last 4 predictions
        self.frames_since_prediction = 0
        self.saved_frames = set()  # Track which absolute frame numbers were saved
    
    def add_frame(self, normalized_keypoints, frame_num):
        """Add a frame to the buffer with forward-fill logic"""
        
        if normalized_keypoints is not None:
            # Valid frame
            self.buffer.append(normalized_keypoints)
            self.last_valid_frame = normalized_keypoints.copy()
            self.frames_since_valid = 0
        else:
            # Null frame - apply forward fill
            self.frames_since_valid += 1
            
            if self.last_valid_frame is not None and self.frames_since_valid <= MAX_FORWARD_FILL:
                # Forward fill with confidence=0
                filled_frame = self.last_valid_frame.copy()
                filled_frame[:, 2] = 0.0  # Set all confidence to 0
                self.buffer.append(filled_frame)
            else:
                # No previous data or gap too long
                self.buffer.append(np.full((17, 3), -1.0))
        
        # Keep only last 48 frames
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
        
        # Calculate moving average
        if len(self.prediction_history) > 0:
            return np.mean(self.prediction_history)
        return prediction
    
    def is_fall_detected(self, smoothed_prediction):
        """Check if fall is detected based on smoothed prediction"""
        return smoothed_prediction > CONFIDENCE_THRESHOLD
    
    def get_frames_to_save(self, current_frame_num):
        """
        Get list of frame numbers to save (12 frames from current window)
        Returns only frames that haven't been saved yet
        """
        # Calculate absolute frame numbers for the current window
        # Window starts at (current_frame_num - WINDOW_SIZE + 1)
        window_start = current_frame_num - WINDOW_SIZE + 1
        
        # Save every 4th frame: indices 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44
        frames_to_save = []
        for idx in range(0, WINDOW_SIZE, 4):
            absolute_frame_num = window_start + idx
            if absolute_frame_num not in self.saved_frames:
                frames_to_save.append(absolute_frame_num)
                self.saved_frames.add(absolute_frame_num)
        
        return frames_to_save


def normalize_by_bbox(keypoints, bbox):
    """
    Normalize keypoints by bounding box
    
    Args:
        keypoints: (17, 3) array - [x, y, confidence]
        bbox: [x1, y1, x2, y2] - YOLO format (xyxy)
    
    Returns:
        normalized: (17, 3) array - [x_norm, y_norm, confidence]
    """
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Avoid division by zero
    if bbox_width <= 0 or bbox_height <= 0:
        return None
    
    normalized = keypoints.copy()
    normalized[:, 0] = (keypoints[:, 0] - x1) / bbox_width   # x normalized
    normalized[:, 1] = (keypoints[:, 1] - y1) / bbox_height  # y normalized
    # confidence stays the same
    
    return normalized


def setup_output_folder(video_path):
    """Create output folder with timestamp and video name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = Path(video_path).stem
    output_folder = f"./demos/{timestamp}_{video_name}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def save_fall_frames(video_capture, frame_numbers, output_folder, track_id, confidence):
    """
    Save specific frames from video
    
    Args:
        video_capture: cv2.VideoCapture object
        frame_numbers: list of frame numbers to save
        output_folder: where to save images
        track_id: person's track_id
        confidence: prediction confidence
    """
    current_pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
    
    for frame_num in frame_numbers:
        # Seek to specific frame
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = video_capture.read()
        
        if ret:
            filename = f"{frame_num}_{confidence:.3f}_person_{track_id}.png"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"  Saved frame {frame_num} (person {track_id})")
    
    # Restore position
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos)


def draw_bbox(frame, bbox, is_fall, track_id):
    """Draw bounding box on frame"""
    if not SHOW_BBOX:
        return
    
    x1, y1, x2, y2 = map(int, bbox)
    
    # Color: Red if fall, Green if no fall
    color = (0, 0, 255) if is_fall else (0, 255, 0)
    
    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Optional: Draw track_id (small label)
    cv2.putText(frame, f"ID:{track_id}", (x1, y1-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    print("="*70)
    print("FALL DETECTION INFERENCE - LIVE VIDEO PROCESSING")
    print("="*70)
    
    # Check if model files exist
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"ERROR: LSTM model not found at {LSTM_MODEL_PATH}")
        print("Please update LSTM_MODEL_PATH in the code")
        return
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"ERROR: YOLO model not found at {YOLO_MODEL_PATH}")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the code")
        return
    
    # Load models
    print("\n[1/4] Loading models...")
    lstm_model = keras.models.load_model(LSTM_MODEL_PATH)
    print(f"  ✓ LSTM model loaded: {LSTM_MODEL_PATH}")
    
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"  ✓ YOLO model loaded: {YOLO_MODEL_PATH}")
    
    # Open video
    print("\n[2/4] Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  ✓ Video: {VIDEO_PATH}")
    print(f"    FPS: {fps}")
    print(f"    Total frames: {total_frames}")
    print(f"    Resolution: {width}x{height}")
    
    # Setup output folder
    output_folder = setup_output_folder(VIDEO_PATH)
    print(f"  ✓ Output folder: {output_folder}")
    
    # Initialize trackers
    print("\n[3/4] Initializing tracking...")
    person_trackers = {}  # track_id -> PersonTracker
    frame_num = 0
    
    print("\n[4/4] Processing video...")
    print("  Press 'q' to quit")
    print("-"*70)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("\nEnd of video")
            break
        
        # Run YOLO pose detection
        results = yolo_model.track(frame, persist=True, verbose=False)
        
        # Get current frame's detections
        current_track_ids = set()
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
            track_ids = results[0].boxes.id
            
            if track_ids is not None:
                track_ids = track_ids.cpu().numpy().astype(int)
                keypoints = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
                
                for idx, track_id in enumerate(track_ids):
                    current_track_ids.add(track_id)
                    
                    # Get keypoints and bbox for this person
                    person_keypoints = keypoints[idx]  # (17, 3)
                    bbox = boxes[idx]  # [x1, y1, x2, y2]
                    
                    # Normalize keypoints
                    normalized = normalize_by_bbox(person_keypoints, bbox)
                    
                    # Create tracker if new person
                    if track_id not in person_trackers:
                        person_trackers[track_id] = PersonTracker(track_id)
                        print(f"  New person detected: ID {track_id}")
                    
                    tracker = person_trackers[track_id]
                    
                    # Add frame to buffer
                    tracker.add_frame(normalized, frame_num)
                    
                    # Check if ready to predict
                    is_fall = False
                    smoothed_prediction = 0.0
                    
                    if tracker.can_predict():
                        # Get window and predict
                        window = tracker.get_window()
                        X = np.expand_dims(window, axis=0)  # (1, 48, 17, 3)
                        
                        prediction = lstm_model.predict(X, verbose=0)[0][0]
                        smoothed_prediction = tracker.add_prediction(prediction)
                        is_fall = tracker.is_fall_detected(smoothed_prediction)
                        
                        # Log prediction
                        status = "FALL DETECTED!" if is_fall else "No fall"
                        print(f"  Frame {frame_num} | Person {track_id} | "
                              f"Pred: {prediction:.3f} | Smoothed: {smoothed_prediction:.3f} | {status}")
                        
                        # Save frames if fall detected
                        if is_fall:
                            frames_to_save = tracker.get_frames_to_save(frame_num)
                            if frames_to_save:
                                print(f"  Saving {len(frames_to_save)} frames for person {track_id}...")
                                save_fall_frames(cap, frames_to_save, output_folder, 
                                               track_id, smoothed_prediction)
                                result = notifier.send_fall_detection_alert(
                                            camera_id="CAM-03-stairs",
                                            location="บันได",
                                            timestamp=datetime.now(),
                                            message="⚠️ กรุณารีบตรวจสอบทันที!",
                                            group_id="Cf7291686ec355f681be9fc4df8b23f4f"  # Specific group
                                        )
                                print(result)
                    
                    # Draw bbox
                    draw_bbox(frame, bbox, is_fall, track_id)
        
        # Clean up trackers for persons not seen in >48 frames
        trackers_to_remove = []
        for track_id, tracker in person_trackers.items():
            if track_id not in current_track_ids:
                tracker.frames_since_valid += 1
                if tracker.frames_since_valid > MAX_FORWARD_FILL:
                    trackers_to_remove.append(track_id)
        
        for track_id in trackers_to_remove:
            print(f"  Removing tracker for person {track_id} (not seen for >{MAX_FORWARD_FILL} frames)")
            del person_trackers[track_id]
        
        # Display frame
        cv2.imshow('Fall Detection', frame)
        
        # Wait for key press (1ms) - video will wait for processing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped by user")
            break
        
        frame_num += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total frames processed: {frame_num}")
    print(f"Output folder: {output_folder}")
    print("="*70)


if __name__ == "__main__":
    main()