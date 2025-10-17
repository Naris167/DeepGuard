import os
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FallDetectionPreprocessor:
    def __init__(self, json_folder: str, output_folder: str):
        """
        Initialize preprocessor
        
        Args:
            json_folder: Root folder containing JSON files (will scan recursively)
            output_folder: Folder to save pkl files
        """
        self.json_folder = json_folder
        self.output_folder = output_folder
        
        # Create output folder if not exists
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_videos': 0,
            'videos_with_no_persons': 0,
            'videos_with_no_labels': 0,  # NEW
            'total_persons': 0,
            'persons_removed_no_keypoints': 0,
            'persons_removed_too_short': 0,
            'persons_removed_missing_bbox': 0,
            'persons_removed_invalid_bbox': 0,
            'persons_saved': 0,
            'frames_with_missing_bbox': 0,
            'frames_with_invalid_bbox': 0
        }

    def print_statistics(self):
        """Print processing statistics"""
        logger.info("\n" + "="*50)
        logger.info("PREPROCESSING STATISTICS")
        logger.info("="*50)
        logger.info(f"Total videos processed: {self.stats['total_videos']}")
        logger.info(f"Videos with no persons: {self.stats['videos_with_no_persons']}")
        logger.info(f"Videos with no labels (incomplete): {self.stats['videos_with_no_labels']}")  # NEW
        logger.info(f"Total persons found: {self.stats['total_persons']}")
        logger.info(f"Persons removed (no keypoints): {self.stats['persons_removed_no_keypoints']}")
        logger.info(f"Persons removed (too short): {self.stats['persons_removed_too_short']}")
        logger.info(f"Frames with missing bbox: {self.stats['frames_with_missing_bbox']}")
        logger.info(f"Frames with invalid bbox: {self.stats['frames_with_invalid_bbox']}")
        logger.info(f"Persons successfully saved: {self.stats['persons_saved']}")
        logger.info("="*50)
    
    def process_all_videos(self):
        """Process all JSON files in folder and subfolders"""
        logger.info(f"Scanning for JSON files in: {self.json_folder}")
        
        json_files = []
        for root, dirs, files in os.walk(self.json_folder):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    json_files.append(json_path)
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        for json_path in json_files:
            try:
                self.process_video(json_path)
            except Exception as e:
                logger.error(f"Error processing {json_path}: {str(e)}")
                continue
        
        self.print_statistics()
    
    def process_video(self, json_path: str):
        """Process a single video JSON file"""
        logger.info(f"Processing: {json_path}")
        
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.stats['total_videos'] += 1
        video_hash = data['video_hash']
        video_metadata = data['video_metadata']
        persons = data.get('persons', [])
        
        if len(persons) == 0:
            logger.warning(f"Video {video_hash} has no persons - skipping")
            self.stats['videos_with_no_persons'] += 1
            return
        
        # NEW: Check if all persons have empty action_labels
        has_any_labels = False
        for person in persons:
            action_labels = person.get('action_labels', [])
            if action_labels and len(action_labels) > 0:
                has_any_labels = True
                break
        
        if not has_any_labels:
            logger.warning(f"Video {video_hash} has no action labels for any person (incomplete labeling) - skipping")
            self.stats['videos_with_no_labels'] += 1
            return
        
        # Group persons by person_id
        persons_by_id = defaultdict(list)
        for person in persons:
            person_id = person['person_id']
            persons_by_id[person_id].append(person)
        
        # Process each unique person
        for person_id, person_entries in persons_by_id.items():
            try:
                self.stats['total_persons'] += 1
                
                # Merge entries with same person_id
                merged_person = self.merge_person_entries(person_entries, video_metadata)
                
                if merged_person is None:
                    continue
                
                # Quality checks
                if not self.quality_check(merged_person, video_hash, person_id):
                    continue
                
                # Process person data
                processed_data = self.process_person(merged_person, video_metadata)
                
                if processed_data is None:
                    continue
                
                # Save to pkl
                self.save_person_data(processed_data, video_hash, person_id)
                self.stats['persons_saved'] += 1
                
            except Exception as e:
                logger.error(f"Error processing person {person_id} in video {video_hash}: {str(e)}")
                continue
    
    def merge_person_entries(self, person_entries: List[Dict], video_metadata: Dict) -> Optional[Dict]:
        """Merge multiple entries with same person_id"""
        if len(person_entries) == 1:
            return person_entries[0]
        
        logger.info(f"Merging {len(person_entries)} entries for person_id {person_entries[0]['person_id']}")
        
        total_frames = video_metadata['total_frames']
        person_id = person_entries[0]['person_id']
        
        # Initialize merged keypoints sequence
        merged_sequence = [None] * total_frames
        
        # Merge keypoints frame by frame
        for frame_num in range(total_frames):
            valid_frames = []
            
            # Collect all non-null data for this frame
            for entry in person_entries:
                if frame_num < len(entry['keypoints_sequence']):
                    frame_data = entry['keypoints_sequence'][frame_num]
                    if frame_data['keypoints'] is not None:
                        valid_frames.append(frame_data)
            
            if len(valid_frames) == 0:
                # All null
                merged_sequence[frame_num] = {
                    'frame_num': frame_num,
                    'keypoints': None,
                    'bbox': None,
                    'bbox_confidence': None
                }
            elif len(valid_frames) == 1:
                # Only one valid
                merged_sequence[frame_num] = valid_frames[0]
            else:
                # Multiple valid - average them
                merged_sequence[frame_num] = self.average_frames(valid_frames, frame_num)
        
        # Merge action labels
        merged_labels = []
        for entry in person_entries:
            if 'action_labels' in entry and entry['action_labels']:
                merged_labels.extend(entry['action_labels'])
        
        # Sort and handle overlaps
        merged_labels = self.merge_action_labels(merged_labels)
        
        return {
            'person_id': person_id,
            'keypoints_sequence': merged_sequence,
            'action_labels': merged_labels,
            'main_subject_age_group': person_entries[0].get('main_subject_age_group'),
            'person_mobility': person_entries[0].get('person_mobility')
        }
    
    def average_frames(self, frames: List[Dict], frame_num: int) -> Dict:
        """Average multiple frame data"""
        n = len(frames)
        
        # Average keypoints
        avg_keypoints = []
        for kp_idx in range(17):
            x_sum = sum(f['keypoints'][kp_idx][0] for f in frames)
            y_sum = sum(f['keypoints'][kp_idx][1] for f in frames)
            c_sum = sum(f['keypoints'][kp_idx][2] for f in frames)
            avg_keypoints.append([x_sum/n, y_sum/n, c_sum/n])
        
        # Average bbox
        bbox_sum = [sum(f['bbox'][i] for f in frames) for i in range(4)]
        avg_bbox = [b/n for b in bbox_sum]
        
        # Average bbox confidence
        avg_bbox_conf = sum(f['bbox_confidence'] for f in frames) / n
        
        return {
            'frame_num': frame_num,
            'keypoints': avg_keypoints,
            'bbox': avg_bbox,
            'bbox_confidence': avg_bbox_conf
        }
    
    def merge_action_labels(self, labels: List[Dict]) -> List[Dict]:
        """Merge and resolve overlapping action labels"""
        if not labels:
            return []
        
        # Sort by start_frame
        labels.sort(key=lambda x: x['start_frame'])
        
        # Create frame-level label map
        frame_labels = {}
        for label in labels:
            for frame in range(label['start_frame'], label['end_frame'] + 1):
                current_label = label['label']
                
                # If overlap, prioritize "fall"
                if frame in frame_labels:
                    if current_label == 'fall' or frame_labels[frame] == 'fall':
                        frame_labels[frame] = 'fall'
                else:
                    frame_labels[frame] = current_label
        
        # Convert back to ranges
        if not frame_labels:
            return []
        
        merged = []
        sorted_frames = sorted(frame_labels.keys())
        start_frame = sorted_frames[0]
        current_label = frame_labels[start_frame]
        
        for i in range(1, len(sorted_frames)):
            frame = sorted_frames[i]
            if frame_labels[frame] != current_label or frame != sorted_frames[i-1] + 1:
                # End current segment
                merged.append({
                    'start_frame': start_frame,
                    'end_frame': sorted_frames[i-1],
                    'label': current_label
                })
                start_frame = frame
                current_label = frame_labels[frame]
        
        # Add last segment
        merged.append({
            'start_frame': start_frame,
            'end_frame': sorted_frames[-1],
            'label': current_label
        })
        
        return merged
    
    def quality_check(self, person: Dict, video_hash: str, person_id: int) -> bool:
        """Check if person data meets quality requirements"""
        keypoints_sequence = person['keypoints_sequence']
        
        # Check if person has any valid keypoints
        valid_frames = [f for f in keypoints_sequence if f['keypoints'] is not None]
        
        if len(valid_frames) == 0:
            logger.warning(f"Person {person_id} in video {video_hash} has no valid keypoints - removing")
            self.stats['persons_removed_no_keypoints'] += 1
            return False
        
        # Check for missing bbox when keypoints exist
        for frame in valid_frames:
            if frame['bbox'] is None:
                logger.warning(f"Person {person_id} frame {frame['frame_num']} has keypoints but missing bbox - will remove this frame")
                self.stats['frames_with_missing_bbox'] += 1
                frame['keypoints'] = None  # Mark as invalid
        
        # Check for invalid bbox dimensions
        for frame in valid_frames:
            if frame['bbox'] is not None:
                bbox_width = frame['bbox'][2]
                bbox_height = frame['bbox'][3]
                if bbox_width <= 0 or bbox_height <= 0:
                    logger.warning(f"Person {person_id} frame {frame['frame_num']} has invalid bbox dimensions - will remove this frame")
                    self.stats['frames_with_invalid_bbox'] += 1
                    frame['keypoints'] = None  # Mark as invalid
        
        # Recount valid frames after removing bad ones
        valid_frames = [f for f in keypoints_sequence if f['keypoints'] is not None]
        
        if len(valid_frames) == 0:
            logger.warning(f"Person {person_id} in video {video_hash} has no valid frames after quality checks - removing")
            self.stats['persons_removed_no_keypoints'] += 1
            return False
        
        # Check sequence length
        if len(valid_frames) < 48:
            logger.warning(f"Person {person_id} in video {video_hash} has only {len(valid_frames)} valid frames (<48) - removing")
            self.stats['persons_removed_too_short'] += 1
            return False
        
        return True
    
    def process_person(self, person: Dict, video_metadata: Dict) -> Optional[Dict]:
        """Process person data: extract sequence, pad, normalize"""
        keypoints_sequence = person['keypoints_sequence']
        action_labels = person.get('action_labels', [])
        
        # Find first and last valid frame
        valid_frame_nums = [f['frame_num'] for f in keypoints_sequence if f['keypoints'] is not None]
        
        if not valid_frame_nums:
            return None
        
        first_frame = min(valid_frame_nums)
        last_frame = max(valid_frame_nums)
        
        # Extract sequence (only frames where person exists)
        person_sequence = keypoints_sequence[first_frame:last_frame + 1]
        
        # Create frame-level labels for this sequence
        frame_labels = self.create_frame_labels(
            person_sequence, 
            action_labels, 
            first_frame,
            video_metadata['total_frames']
        )
        
        # Normalize keypoints by bbox
        normalized_keypoints = []
        bboxes = []
        valid_frames = []
        
        for frame in person_sequence:
            if frame['keypoints'] is not None and frame['bbox'] is not None:
                # Normalize
                normalized_kp = self.normalize_by_bbox(frame['keypoints'], frame['bbox'])
                normalized_keypoints.append(normalized_kp)
                bboxes.append(frame['bbox'])
                valid_frames.append(True)
            else:
                # Keep null as [-1, -1, -1]
                normalized_keypoints.append([[-1, -1, -1] for _ in range(17)])
                bboxes.append([0, 0, 0, 0])  # Placeholder
                valid_frames.append(False)
        
        # Add padding (24 frames before and after)
        padding = [[[-1, -1, -1] for _ in range(17)] for _ in range(24)]
        padding_bbox = [[0, 0, 0, 0] for _ in range(24)]
        padding_label = ['no_fall'] * 24
        padding_valid = [False] * 24
        
        normalized_keypoints = padding + normalized_keypoints + padding
        bboxes = padding_bbox + bboxes + padding_bbox
        frame_labels = padding_label + frame_labels + padding_label
        valid_frames = padding_valid + valid_frames + padding_valid
        
        # Convert to numpy arrays
        keypoints_array = np.array(normalized_keypoints, dtype=np.float32)
        bboxes_array = np.array(bboxes, dtype=np.float32)
        frame_labels_array = np.array(frame_labels, dtype=object)
        valid_frames_array = np.array(valid_frames, dtype=bool)
        
        return {
            'person_id': person['person_id'],
            'keypoints': keypoints_array,
            'bboxes': bboxes_array,
            'frame_labels': frame_labels_array,
            'valid_frames': valid_frames_array,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'total_frames': len(keypoints_array)
        }
    
    def create_frame_labels(self, sequence: List[Dict], action_labels: List[Dict], 
                           first_frame: int, total_video_frames: int) -> List[str]:
        """Create frame-level labels for the person sequence"""
        sequence_length = len(sequence)
        frame_labels = ['no_fall'] * sequence_length
        
        for action in action_labels:
            start = action['start_frame']
            end = action['end_frame']
            label = action['label']
            
            # Convert to sequence-relative indices
            seq_start = max(0, start - first_frame)
            seq_end = min(sequence_length - 1, end - first_frame)
            
            if seq_start <= seq_end and seq_start < sequence_length and seq_end >= 0:
                for i in range(seq_start, seq_end + 1):
                    if label == 'fall':
                        frame_labels[i] = 'fall'
                    elif frame_labels[i] != 'fall':  # Don't override fall labels
                        frame_labels[i] = 'no_fall'
        
        return frame_labels
    
    def normalize_by_bbox(self, keypoints: List[List[float]], bbox: List[float]) -> List[List[float]]:
        """Normalize keypoints by bounding box"""
        bbox_left, bbox_top, bbox_width, bbox_height = bbox
        
        normalized = []
        for kp in keypoints:
            x, y, conf = kp
            x_norm = (x - bbox_left) / bbox_width
            y_norm = (y - bbox_top) / bbox_height
            normalized.append([x_norm, y_norm, conf])
        
        return normalized
    
    def save_person_data(self, data: Dict, video_hash: str, person_id: int):
        """Save processed person data to pkl file"""
        filename = f"{video_hash}_person_{person_id}.pkl"
        filepath = os.path.join(self.output_folder, filename)
        
        save_data = {
            'video_hash': video_hash,
            'person_id': data['person_id'],
            'keypoints': data['keypoints'],
            'bboxes': data['bboxes'],
            'frame_labels': data['frame_labels'],
            'valid_frames': data['valid_frames'],
            'first_frame': data['first_frame'],
            'last_frame': data['last_frame'],
            'total_frames': data['total_frames']
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved: {filename} ({data['total_frames']} frames, "
                   f"{np.sum(data['frame_labels'] == 'fall')} fall frames)")
    
    def print_statistics(self):
        """Print processing statistics"""
        logger.info("\n" + "="*50)
        logger.info("PREPROCESSING STATISTICS")
        logger.info("="*50)
        logger.info(f"Total videos processed: {self.stats['total_videos']}")
        logger.info(f"Videos with no persons: {self.stats['videos_with_no_persons']}")
        logger.info(f"Total persons found: {self.stats['total_persons']}")
        logger.info(f"Persons removed (no keypoints): {self.stats['persons_removed_no_keypoints']}")
        logger.info(f"Persons removed (too short): {self.stats['persons_removed_too_short']}")
        logger.info(f"Frames with missing bbox: {self.stats['frames_with_missing_bbox']}")
        logger.info(f"Frames with invalid bbox: {self.stats['frames_with_invalid_bbox']}")
        logger.info(f"Persons successfully saved: {self.stats['persons_saved']}")
        logger.info("="*50)


def main():
    """Main function"""
    # Configure paths
    json_folder = "./data/Label"
    output_folder = "./data/preprocessed"
    
    # Create preprocessor and run
    preprocessor = FallDetectionPreprocessor(json_folder, output_folder)
    preprocessor.process_all_videos()
    
    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()