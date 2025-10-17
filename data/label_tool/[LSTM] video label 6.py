import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import hashlib
import threading
import time
from collections import OrderedDict
import json
# For YOLO (will be used in Phase 3)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO features will be disabled.")

# Add with other imports at the top
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. GPU detection disabled.")

def check_gpu_availability():
    """Check if CUDA/GPU is available for PyTorch"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            # Also check if GPU memory is accessible
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                return True, gpu_count, f"{gpu_name} ({total_memory:.1f}GB)"
            except:
                return True, gpu_count, gpu_name
        else:
            return False, 0, "No GPU"
    except ImportError:
        return False, 0, "PyTorch not installed"
    except Exception as e:
        return False, 0, f"Error: {str(e)}"

class MarkerItem:
    """Represents a single visual marker on canvas"""
    def __init__(self, pair_id, marker_type, frame_pos, canvas_item_id):
        self.pair_id = pair_id
        self.marker_type = marker_type  # 'start' or 'end'
        self.frame_pos = frame_pos      # Data: frame number
        self.canvas_item_id = canvas_item_id  # Visual: canvas item ID

class MarkerPair:
    """Represents a pair of start/end markers"""
    def __init__(self, pair_id, start_frame, end_frame, color):
        self.pair_id = pair_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.color = color
        self.start_marker_item = None  # MarkerItem reference
        self.end_marker_item = None    # MarkerItem reference

class ToggleCheckButtonGroup:
    """Checkbutton group that behaves like radio buttons with unselect capability"""
    
    def __init__(self, parent, options, command=None):
        self.parent = parent
        self.options = options  # List of (value, text) tuples
        self.command = command
        self.checkbox_vars = {}
        self.checkboxes = {}
        self.enabled = True
        
        self.create_checkbuttons()
    
    def create_checkbuttons(self):
        """Create checkbuttons with radio button behavior"""
        for value, text in self.options:
            var = tk.BooleanVar()
            self.checkbox_vars[value] = var
            
            cb = tk.Checkbutton(
                self.parent,
                text=text,
                variable=var,
                command=lambda v=value: self.on_checkbox_click(v),
                font=("Arial", 9),
                indicatoron=1,
                padx=5,
                pady=2
            )
            self.checkboxes[value] = cb
    
    def on_checkbox_click(self, clicked_option):
        """Handle checkbox click with mutual exclusion"""
        if not self.enabled:
            # If disabled, revert the click
            self.checkbox_vars[clicked_option].set(False)
            return
            
        # If this checkbox was just checked
        if self.checkbox_vars[clicked_option].get():
            # Uncheck all other checkboxes (radio button behavior)
            for option, var in self.checkbox_vars.items():
                if option != clicked_option:
                    var.set(False)
        
        if self.command:
            self.command()
    
    def get_selection(self):
        """Get currently selected option or None"""
        for option, var in self.checkbox_vars.items():
            if var.get():
                return option
        return None
    
    def set_selection(self, value):
        """Programmatically set selection"""
        # Clear all first
        for var in self.checkbox_vars.values():
            var.set(False)
        
        # Set the specified value if it exists
        if value and value in self.checkbox_vars:
            self.checkbox_vars[value].set(True)
    
    def enable_group(self):
        """Enable all checkbuttons in group"""
        self.enabled = True
        for cb in self.checkboxes.values():
            cb.configure(state='normal')
    
    def disable_group(self):
        """Disable all checkbuttons in group and clear selection"""
        self.enabled = False
        # Clear selection first
        for var in self.checkbox_vars.values():
            var.set(False)
        # Then disable
        for cb in self.checkboxes.values():
            cb.configure(state='disabled')
    
    def arrange_responsive(self, available_width):
        """Arrange checkbuttons responsively - prefer horizontal layout"""
        # Clear current grid
        for cb in self.checkboxes.values():
            cb.grid_forget()
        
        if not self.checkboxes:
            return 0
        
        # Estimate button width more generously
        total_buttons = len(self.options)
        avg_text_length = sum(len(option[1]) for option in self.options) / total_buttons
        estimated_button_width = max(80, avg_text_length * 7 + 40)
        
        # Calculate how many buttons can fit - prefer horizontal
        # Use 90% of available width to account for padding
        usable_width = int(available_width * 0.9)
        buttons_per_row = min(total_buttons, max(1, usable_width // estimated_button_width))
        
        # For small groups (2-4 buttons), try to keep them on one row
        if total_buttons <= 4 and usable_width >= (estimated_button_width * total_buttons):
            buttons_per_row = total_buttons
        
        row = 0
        col = 0
        for value, text in self.options:
            cb = self.checkboxes[value]
            cb.grid(row=row, column=col, sticky='w', padx=5, pady=2)
            col += 1
            if col >= buttons_per_row:
                col = 0
                row += 1
        
        return row + 1

class AutoSaveManager:
    """Handles debounced auto-saving with 2-second delay"""
    
    def __init__(self, save_callback):
        self.save_callback = save_callback
        self.timer = None
        self.loading_in_progress = False
        self.switching_markers = False  # NEW: Flag for marker switching
    
    def trigger_save(self):
        """Trigger save after 2-second delay, canceling any previous timer"""
        if self.loading_in_progress or self.switching_markers:  # MODIFIED: Check switching flag
            return
            
        if self.timer:
            self.timer.cancel()
        
        self.timer = threading.Timer(2.0, self.save_callback)
        self.timer.daemon = True
        self.timer.start()
    
    def set_loading_flag(self, loading):
        """Set loading flag to prevent auto-save during data loading"""
        self.loading_in_progress = loading
    
    def set_switching_flag(self, switching):  # NEW: Method to set switching flag
        """Set switching flag to prevent auto-save during marker switching"""
        self.switching_markers = switching
    
    def cancel_pending(self):
        """Cancel any pending save operations"""
        if self.timer:
            self.timer.cancel()
            self.timer = None

class MarkerManager:
    """Manages video markers using proper tkinter patterns"""
    
    def __init__(self, canvas, colors, on_change_callback):
        self.canvas = canvas
        self.colors = colors
        self.on_change_callback = on_change_callback
        
        # NEW: Per-person marker storage
        self.current_person_id = None  # Currently selected person
        self.all_persons_markers = {}  # {person_id: {'marker_pairs': {}, 'selected_pair_index': None}}
        
        # Data storage - now using sequential indexing
        self.marker_pairs = {}          # sequential_index -> MarkerPair
        self.marker_items = {}          # canvas_item_id -> MarkerItem
        self.selected_pair_index = None  # Changed from selected_pair_id
        
        # Drag state
        self.dragging_item = None
        self.drag_start_x = 0
        
        # Video properties
        self.total_frames = 0
        self.scrub_width = 0
        self.canvas_width = 0
    
    def set_current_person(self, person_id):
        """Switch to a different person's markers"""
        # Save current person's state if exists
        if self.current_person_id and self.current_person_id in self.all_persons_markers:
            self.all_persons_markers[self.current_person_id]['marker_pairs'] = self.marker_pairs
            self.all_persons_markers[self.current_person_id]['selected_pair_index'] = self.selected_pair_index
        
        # Switch to new person
        self.current_person_id = person_id
        
        # Initialize person's markers if not exists
        if person_id not in self.all_persons_markers:
            self.all_persons_markers[person_id] = {
                'marker_pairs': {},
                'selected_pair_index': None
            }
        
        # Load this person's markers
        self.marker_pairs = self.all_persons_markers[person_id]['marker_pairs']
        self.selected_pair_index = self.all_persons_markers[person_id]['selected_pair_index']
        
        # Clear canvas items (will be redrawn)
        self.marker_items = {}
        
        self.on_change_callback()

    def get_current_person_markers(self):
        """Get current person's marker data"""
        if self.current_person_id and self.current_person_id in self.all_persons_markers:
            return self.all_persons_markers[self.current_person_id]
        return None

    def clear_all_persons_markers(self):
        """Clear markers for all persons"""
        self.all_persons_markers = {}
        self.current_person_id = None
        self.marker_pairs = {}
        self.marker_items = {}
        self.selected_pair_index = None

    def load_person_markers_from_data(self, person_id, action_labels):
        """Load markers for a person from action_labels data"""
        if person_id not in self.all_persons_markers:
            self.all_persons_markers[person_id] = {
                'marker_pairs': {},
                'selected_pair_index': None
            }
        
        # Create marker pairs from action labels
        marker_pairs = {}
        for idx, action in enumerate(action_labels, 1):
            start_frame = action['start_frame']
            end_frame = action['end_frame']
            color = self.get_pair_color(idx)
            
            pair = MarkerPair(idx, start_frame, end_frame, color)
            marker_pairs[idx] = pair
        
        self.all_persons_markers[person_id]['marker_pairs'] = marker_pairs
        self.all_persons_markers[person_id]['selected_pair_index'] = None
        
    def check_marker_overlap(self, start_frame, end_frame, exclude_pair_id=None):
        """
        Check if new marker would overlap with existing markers for current person only.
        Frames are inclusive ranges. Adjacent frames are OK (e.g., [100-147] and [148-200] don't overlap)
        """
        if not self.marker_pairs:
            return False
        
        for pair_id, pair in self.marker_pairs.items():
            if exclude_pair_id and pair_id == exclude_pair_id:
                continue
            
            # Check overlap: Two ranges [a1, a2] and [b1, b2] overlap if:
            # a1 <= b2 AND b1 <= a2
            # 
            # For adjacent frames (e.g., [100, 147] and [148, 200]):
            # 100 <= 200 (TRUE) AND 148 <= 147 (FALSE) = FALSE (no overlap) ✓
            #
            # For overlapping (e.g., [100, 150] and [148, 200]):
            # 100 <= 200 (TRUE) AND 148 <= 150 (TRUE) = TRUE (overlap) ✓
            
            if start_frame <= pair.end_frame and pair.start_frame <= end_frame:
                return True  # Overlap detected
        
        return False
        
    def set_video_info(self, total_frames):
        """Set video information"""
        self.total_frames = total_frames
        
    def frame_to_screen_x(self, frame_num):
        """Convert frame number to screen X coordinate"""
        if self.total_frames <= 1:
            return 10
        return 10 + (frame_num / (self.total_frames - 1)) * self.scrub_width
    
    def screen_x_to_frame(self, screen_x):
        """Convert screen X coordinate to frame number"""
        if self.scrub_width <= 0:
            return 0
        position = max(0, min(1, (screen_x - 10) / self.scrub_width))
        return int(position * (self.total_frames - 1))
    
    def get_pair_color(self, pair_id):
        """Get color for pair ID"""
        return self.colors[(pair_id - 1) % len(self.colors)]
    
    def find_non_overlapping_position(self, current_pos):
        """Find position with 20-frame spacing from existing markers"""
        def has_overlap(pos):
            for pair in self.marker_pairs.values():
                if (abs(pair.start_frame - pos) < 20 or 
                    abs(pair.end_frame - pos) < 20 or
                    abs(pair.start_frame - (pos + 40)) < 20 or
                    abs(pair.end_frame - (pos + 40)) < 20):
                    return True
            return False
        
        if not has_overlap(current_pos):
            return current_pos
        
        # Search for non-overlapping position
        search_range = 100
        for offset in range(20, search_range, 5):
            # Try positive offset
            new_pos = min(current_pos + offset, self.total_frames - 41)
            if new_pos >= 0 and not has_overlap(new_pos):
                return new_pos
            
            # Try negative offset
            new_pos = max(current_pos - offset, 0)
            if new_pos + 40 < self.total_frames and not has_overlap(new_pos):
                return new_pos
        
        return current_pos
    
    def create_marker_visual(self, pair_id, marker_type, frame_pos, color):
        """Create visual marker using tkinter best practices"""
        screen_x = self.frame_to_screen_x(frame_pos)
        
        if marker_type == 'start':
            # Upward triangle at y=5-15
            item_id = self.canvas.create_polygon(
                screen_x, 5, screen_x-6, 15, screen_x+6, 15,
                fill=color, outline='black', width=1,
                tags=(f"marker", f"pair_{pair_id}", f"start_{pair_id}")
            )
        else:
            # Downward triangle at y=25-15  
            item_id = self.canvas.create_polygon(
                screen_x, 25, screen_x-6, 15, screen_x+6, 15,
                fill=color, outline='black', width=1,
                tags=(f"marker", f"pair_{pair_id}", f"end_{pair_id}")
            )
        
        # KEY: Bind events directly to this specific item
        self.canvas.tag_bind(item_id, '<Button-1>', self.on_marker_click)
        self.canvas.tag_bind(item_id, '<B1-Motion>', self.on_marker_drag)
        self.canvas.tag_bind(item_id, '<ButtonRelease-1>', self.on_marker_release)
        
        return item_id
    
    def get_next_sequential_index(self):
        """Get the next available sequential index"""
        if not self.marker_pairs:
            return 1
        return max(self.marker_pairs.keys()) + 1

    def renumber_markers_after_deletion(self, deleted_index):
        """Renumber all markers after deletion to maintain sequential order"""
        new_pairs = {}
        
        for idx, pair in sorted(self.marker_pairs.items()):
            if idx < deleted_index:
                # Keep markers before deleted index as is
                new_pairs[idx] = pair
                pair.pair_id = idx  # Ensure pair_id matches sequential index
            elif idx > deleted_index:
                # Shift markers after deleted index down by 1
                new_idx = idx - 1
                pair.pair_id = new_idx
                pair.color = self.get_pair_color(new_idx)  # Update color
                new_pairs[new_idx] = pair
        
        self.marker_pairs = new_pairs
        
        # Update selected index if needed
        if self.selected_pair_index:
            if self.selected_pair_index == deleted_index:
                self.selected_pair_index = None
            elif self.selected_pair_index > deleted_index:
                self.selected_pair_index -= 1

    def get_pair_color(self, sequential_index):
        """Get color for sequential index (not pair_id)"""
        return self.colors[(sequential_index - 1) % len(self.colors)]

    def create_marker_pair(self, current_frame_idx):
        """Create marker pair with proper sequential indexing"""
        if self.total_frames == 0:
            return None
        
        if not self.current_person_id:
            return "no_person"  # No person selected
        
        # Start with current position
        start_pos = current_frame_idx
        end_pos = min(start_pos + 40, self.total_frames - 1)
        
        # Check for overlap
        if self.check_marker_overlap(start_pos, end_pos):
            # Try to find non-overlapping position
            found_position = False
            for offset in range(0, self.total_frames, 10):
                test_start = min(current_frame_idx + offset, self.total_frames - 41)
                test_end = min(test_start + 40, self.total_frames - 1)
                
                if not self.check_marker_overlap(test_start, test_end):
                    start_pos = test_start
                    end_pos = test_end
                    found_position = True
                    break
            
            if not found_position:
                return "overlap"  # Cannot find non-overlapping position
        
        # Get next sequential index
        seq_index = self.get_next_sequential_index()
        color = self.get_pair_color(seq_index)
        
        # Create the pair data with sequential index as pair_id
        pair = MarkerPair(seq_index, start_pos, end_pos, color)
        
        # Create visual markers
        start_item_id = self.create_marker_visual(seq_index, 'start', start_pos, color)
        end_item_id = self.create_marker_visual(seq_index, 'end', end_pos, color)
        
        # Create marker items
        start_marker = MarkerItem(seq_index, 'start', start_pos, start_item_id)
        end_marker = MarkerItem(seq_index, 'end', end_pos, end_item_id)
        
        # Link everything together
        pair.start_marker_item = start_marker
        pair.end_marker_item = end_marker
        
        # Store in mappings using sequential index
        self.marker_pairs[seq_index] = pair
        self.marker_items[start_item_id] = start_marker
        self.marker_items[end_item_id] = end_marker
        
        # Auto-select new pair
        self.selected_pair_index = seq_index
        self.highlight_selected_pair()
        
        self.on_change_callback()
        return pair

    def delete_selected_pair(self):
        """Delete currently selected marker pair and return the deleted index"""
        if self.selected_pair_index is None:
            return None
        
        pair = self.marker_pairs.get(self.selected_pair_index)
        if not pair:
            return None
        
        deleted_index = self.selected_pair_index
        
        # Remove visual items from canvas
        if pair.start_marker_item:
            self.canvas.delete(pair.start_marker_item.canvas_item_id)
            del self.marker_items[pair.start_marker_item.canvas_item_id]
        
        if pair.end_marker_item:
            self.canvas.delete(pair.end_marker_item.canvas_item_id)
            del self.marker_items[pair.end_marker_item.canvas_item_id]
        
        # Remove from data storage
        del self.marker_pairs[self.selected_pair_index]
        
        # Renumber remaining markers
        self.renumber_markers_after_deletion(deleted_index)
        
        # Clear selection
        self.selected_pair_index = None
        self.canvas.delete("selection_highlight")
        
        self.on_change_callback()
        return deleted_index

    def on_marker_click(self, event):
        """Handle marker click - uses tkinter best practices"""
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        clicked_item = self.canvas.find_closest(x, y)[0]
        
        if clicked_item not in self.marker_items:
            return
        
        marker_item = self.marker_items[clicked_item]
        
        # Use sequential index
        self.selected_pair_index = marker_item.pair_id
        self.highlight_selected_pair()
        
        self.dragging_item = clicked_item
        self.drag_start_x = x
        
        self.on_change_callback()
    
    def on_marker_drag(self, event):
        """Handle marker dragging with overlap prevention - fixed for fast dragging"""
        if not self.dragging_item:
            return
            
        # Convert coordinates properly
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Constrain to scrub bar area
        constrained_x = max(10, min(x, 10 + self.scrub_width))
        
        # Calculate proposed frame position FIRST (before moving visual)
        new_frame = self.screen_x_to_frame(constrained_x)
        marker_item = self.marker_items[self.dragging_item]
        
        # Get the pair data
        pair = self.marker_pairs[marker_item.pair_id]
        
        # Calculate proposed new positions based on which marker is being dragged
        if marker_item.marker_type == 'start':
            proposed_start = max(0, min(new_frame, self.total_frames - 1))
            proposed_end = pair.end_frame
            
            # Validate: start cannot be after end
            if proposed_start > proposed_end:
                return  # Don't allow invalid range
        else:
            proposed_start = pair.start_frame
            proposed_end = max(0, min(new_frame, self.total_frames - 1))
            
            # Validate: end cannot be before start
            if proposed_end < proposed_start:
                return  # Don't allow invalid range
        
        # Check overlap with proposed position (excluding current pair)
        if self.check_marker_overlap(proposed_start, proposed_end, exclude_pair_id=marker_item.pair_id):
            # OVERLAP DETECTED - Don't update anything, keep at current valid position
            return
        
        # VALID POSITION - Now calculate the visual movement needed
        # Calculate where the marker SHOULD be based on the new valid frame
        target_screen_x = self.frame_to_screen_x(new_frame)
        
        # Get current visual position
        current_coords = self.canvas.coords(self.dragging_item)
        if marker_item.marker_type == 'start':
            current_x = current_coords[0]  # Triangle point (top)
        else:
            current_x = current_coords[0]  # Triangle point (bottom)
        
        # Calculate actual delta to move
        dx = target_screen_x - current_x
        
        # NOW move the visual marker to the correct position
        self.canvas.move(self.dragging_item, dx, 0)
        
        # Update the pair data
        if marker_item.marker_type == 'start':
            pair.start_frame = proposed_start
        else:
            pair.end_frame = proposed_end
        
        # Update marker item frame position
        marker_item.frame_pos = new_frame
        
        # Update highlight to match new positions
        self.update_selection_highlight()
        
        # Update drag start position for next move
        self.drag_start_x = target_screen_x
        
        self.on_change_callback()
    
    def on_marker_release(self, event):
        """Handle drag release"""
        self.dragging_item = None
        self.on_change_callback()
    
    def highlight_selected_pair(self):
        """Proper selection highlighting"""
        self.canvas.delete("selection_highlight")
        
        if self.selected_pair_index and self.selected_pair_index in self.marker_pairs:
            self.update_selection_highlight()

    def update_selection_highlight(self):
        """Update selection highlight position"""
        if not self.selected_pair_index or self.selected_pair_index not in self.marker_pairs:
            return
        
        pair = self.marker_pairs[self.selected_pair_index]
        start_x = self.frame_to_screen_x(pair.start_frame)
        end_x = self.frame_to_screen_x(pair.end_frame)
        
        self.canvas.delete("selection_highlight")
        
        self.canvas.create_rectangle(
            start_x - 8, 3, end_x + 8, 27,
            outline='orange', width=2, fill='',
            tags="selection_highlight"
        )
        
        self.canvas.tag_lower("selection_highlight")
    
    def draw_markers(self, canvas_width):
        """Draw all markers and ensure event bindings are preserved"""
        self.canvas_width = canvas_width
        self.scrub_width = canvas_width - 20
        
        # Clear canvas
        self.canvas.delete("all")
        
        if canvas_width <= 1:
            return
        
        # Draw scrub bar background
        self.canvas.create_rectangle(10, 5, canvas_width - 10, 25, 
                                fill='lightgray', outline='gray')
        
        # Clear old marker items mapping
        self.marker_items = {}
        
        # Recreate all markers at correct positions with proper colors
        for seq_index in sorted(self.marker_pairs.keys()):
            pair = self.marker_pairs[seq_index]
            
            # Update color based on sequential position
            pair.color = self.get_pair_color(seq_index)
            
            # Create new visual markers with event bindings
            start_item_id = self.create_marker_visual(
                seq_index, 'start', pair.start_frame, pair.color
            )
            end_item_id = self.create_marker_visual(
                seq_index, 'end', pair.end_frame, pair.color
            )
            
            # Create new marker items
            start_marker = MarkerItem(seq_index, 'start', pair.start_frame, start_item_id)
            end_marker = MarkerItem(seq_index, 'end', pair.end_frame, end_item_id)
            
            # Update references
            pair.start_marker_item = start_marker
            pair.end_marker_item = end_marker
            
            # Update marker items mapping
            self.marker_items[start_item_id] = start_marker
            self.marker_items[end_item_id] = end_marker
        
        # Restore selection highlight
        self.highlight_selected_pair()
    
    def get_selected_pair(self):
        """Get currently selected pair"""
        return self.marker_pairs.get(self.selected_pair_index)
    
    def update_pair(self, seq_index, start_frame, end_frame):
        """Update pair frames manually with overlap checking"""
        pair = self.marker_pairs.get(seq_index)
        if not pair:
            return False
        
        # Validate frame range
        start_frame = max(0, min(start_frame, self.total_frames - 1))
        end_frame = max(0, min(end_frame, self.total_frames - 1))
        
        # Check overlap (excluding current pair)
        if self.check_marker_overlap(start_frame, end_frame, exclude_pair_id=seq_index):
            return False  # Would cause overlap
        
        pair.start_frame = start_frame
        pair.end_frame = end_frame

        if pair.start_marker_item:
            pair.start_marker_item.frame_pos = pair.start_frame
        if pair.end_marker_item:
            pair.end_marker_item.frame_pos = pair.end_frame

        self.on_change_callback()
        return True
    
    def get_marker_pairs_list(self):
        """Get list of marker pairs for external use"""
        return [
            {
                'id': seq_index,  # Use sequential index
                'start': pair.start_frame,
                'end': pair.end_frame,
                'color': pair.color
            }
            for seq_index, pair in sorted(self.marker_pairs.items())
        ]
    
    def update_pair(self, seq_index, start_frame, end_frame):
        """Update pair frames manually"""
        pair = self.marker_pairs.get(seq_index)
        if not pair:
            return False

        pair.start_frame = max(0, min(start_frame, self.total_frames - 1))
        pair.end_frame = max(0, min(end_frame, self.total_frames - 1))

        if pair.start_marker_item:
            pair.start_marker_item.frame_pos = pair.start_frame
        if pair.end_marker_item:
            pair.end_marker_item.frame_pos = pair.end_frame

        self.on_change_callback()
        return True

class ActivityTypeManager:
    """Manages interactions between label section and activity type sections"""
    
    def __init__(self, falling_groups, general_group, general_other_widgets, on_change_callback):
        self.falling_groups = falling_groups  # [front_back_group, left_right_group]
        self.general_group = general_group
        self.general_other_checkbox = general_other_widgets[0]  # Other checkbox
        self.general_other_entry = general_other_widgets[1]      # Other entry field
        self.on_change_callback = on_change_callback
    
    def update_activity_sections_state(self, label_selection):
        """Update activity sections based on label selection"""
        if label_selection == "normal":
            # Disable falling direction, enable general action
            for group in self.falling_groups:
                group.disable_group()
            self.general_group.enable_group()
            # Enable "Other" widgets for general action
            self.general_other_checkbox.configure(state='normal')
            self.general_other_entry.configure(state='normal')
            
        elif label_selection == "fall":
            # Enable falling direction, disable general action  
            for group in self.falling_groups:
                group.enable_group()
            self.general_group.disable_group()
            # Disable "Other" widgets for general action
            self.general_other_checkbox.configure(state='disabled')
            self.general_other_entry.configure(state='disabled')
            # Clear the "Other" selection if it was checked
            self.general_other_checkbox.var.set(False)
            
        else:
            # "other" or empty - enable both
            for group in self.falling_groups:
                group.enable_group()
            self.general_group.enable_group()
            # Enable "Other" widgets for general action
            self.general_other_checkbox.configure(state='normal')
            self.general_other_entry.configure(state='normal')
    
    def handle_falling_direction_change(self):
        """Handle falling direction change - clear general action"""
        has_falling_selection = any(group.get_selection() for group in self.falling_groups)
        if has_falling_selection:
            self.general_group.set_selection(None)
            # Also clear "Other" if selected
            self.general_other_checkbox.var.set(False)
        self.on_change_callback()
    
    def handle_general_action_change(self):
        """Handle general action change - clear falling direction"""
        if self.general_group.get_selection():
            for group in self.falling_groups:
                group.set_selection(None)
        self.on_change_callback()
    
    def get_combined_fall_direction(self):
        """Get combined fall direction in format like 'fall_left_front'"""
        directions = []
        for group in self.falling_groups:
            selection = group.get_selection()
            if selection:
                directions.append(selection)
        
        if directions:
            return "fall_" + "_".join(directions)
        return ""

class JSONDataManager:
    """Manages JSON data storage for video labels"""
    
    def __init__(self, json_folder_path):
        self.json_folder_path = json_folder_path
        self.video_hashes_in_storage = set()  # Store existing hashes
        self.hash_to_json_path = {}  # NEW: Maps hash -> full JSON file path
        self.load_existing_hashes()
    
    def load_existing_hashes(self):
        """Load all video hashes from existing JSON files - RECURSIVE"""
        if not os.path.exists(self.json_folder_path):
            os.makedirs(self.json_folder_path)
            return
        
        self.video_hashes_in_storage.clear()
        self.hash_to_json_path.clear()  # NEW: Clear path tracking
        
        # NEW: Use os.walk for recursive search
        for root, dirs, files in os.walk(self.json_folder_path):
            for filename in files:
                if filename.endswith('.json'):
                    # Extract hash from filename (remove .json extension)
                    video_hash = filename[:-5]
                    full_path = os.path.join(root, filename)
                    
                    self.video_hashes_in_storage.add(video_hash)
                    self.hash_to_json_path[video_hash] = full_path  # NEW: Track location
    
    def load_video_data(self, video_hash):
        """Load JSON data for specific video"""
        if not video_hash:
            return None
        
        # NEW: Use tracked path if available, otherwise fallback to root
        if video_hash in self.hash_to_json_path:
            json_path = self.hash_to_json_path[video_hash]
        else:
            json_path = os.path.join(self.json_folder_path, f"{video_hash}.json")
        
        if not os.path.exists(json_path):
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading JSON for {video_hash}: {e}")
            return None
    
    def save_video_data(self, video_hash, data):
        """Save JSON data for specific video"""
        if not video_hash:
            return False
        
        # NEW: Determine save path
        if video_hash in self.hash_to_json_path:
            # Existing file - save to same location
            json_path = self.hash_to_json_path[video_hash]
        else:
            # New file - save to root folder
            json_path = os.path.join(self.json_folder_path, f"{video_hash}.json")
            # Track this new path
            self.hash_to_json_path[video_hash] = json_path
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Add to cache
            self.video_hashes_in_storage.add(video_hash)
            return True
        except Exception as e:
            print(f"Error saving JSON for {video_hash}: {e}")
            return False
    
    def video_exists(self, video_hash):
        """Check if video data exists"""
        return video_hash in self.video_hashes_in_storage
    
    def delete_video_data(self, video_hash):
        """Delete JSON file for video"""
        if not video_hash:
            return False
        
        # NEW: Use tracked path
        if video_hash in self.hash_to_json_path:
            json_path = self.hash_to_json_path[video_hash]
        else:
            json_path = os.path.join(self.json_folder_path, f"{video_hash}.json")
        
        try:
            if os.path.exists(json_path):
                os.remove(json_path)
                self.video_hashes_in_storage.discard(video_hash)
                self.hash_to_json_path.pop(video_hash, None)  # NEW: Remove tracking
            return True
        except Exception as e:
            print(f"Error deleting JSON for {video_hash}: {e}")
            return False
    
    def get_all_video_hashes(self):
        """Get list of all video hashes in storage"""
        return list(self.video_hashes_in_storage)

class FallDetectionLabeler:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Fall Detection Video Labeling Tool - Redesigned Markers")
        self.root.geometry("1600x1000")
        
        # Video data
        self.video_files = []
        self.video_hashes = {}
        self.current_video_path = None
        self.current_video = None
        self.current_frame = None
        self.is_playing = False
        self.current_frame_idx = 0
        self.total_frames = 0
        self.fps = 30
        self.video_width = 0
        self.video_height = 0
        self.color_mode = ""
        
        # UI Components
        self.pair_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        self.is_scrubbing = False
        self.loop_start = None
        self.loop_end = None
        
        # Managers
        self.auto_save_manager = AutoSaveManager(self.perform_auto_save)
        self.marker_manager = None
        self.activity_manager = None
        
        # Checkbutton groups
        self.checkbutton_groups = {}
        
        # Custom text variables for "other" options
        self.label_custom_var = tk.StringVar()
        self.cam_custom_var = tk.StringVar()
        self.env_custom_var = tk.StringVar()
        self.general_custom_var = tk.StringVar()
        self.quality_custom_var = tk.StringVar()
        
        # Frame editing variables
        self.start_frame_var = tk.StringVar()
        self.end_frame_var = tk.StringVar()
        self.num_persons_var = tk.StringVar(value="1")
        
        # JSON data management
        self.json_folder_path = None
        self.json_manager = None
        self.current_video_data = None  # Currently loaded JSON data
        self.label_cache = {}  # Cache labels per person per marker: {person_id: {marker_id: {label, fall_direction, general_action}}}

        # YOLO management
        self.yolo_model = None
        self.yolo_model_path = None
        self.yolo_results = None  # Cached YOLO results for current video
        self.selected_person_id = None  # User-assigned ID (can be duplicate)
        self.selected_yolo_track_id = None  # YOLO's unique ID (primary key)
        self.yolo_device = tk.StringVar(value="auto")

        # Visualization toggle (persistent across videos)
        self.show_skeleton = tk.BooleanVar(value=True)
        
        # ADD THIS LINE:
        self.previous_selected_pair_index = None  # Track previous pair for proper caching

        # Initialize JSON folder
        self.initialize_json_folder()
        
        # Setup GUI
        self.setup_gui()
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind frame editing variables
        self.start_frame_var.trace('w', self.on_frame_manual_edit)
        self.end_frame_var.trace('w', self.on_frame_manual_edit)
        
        # Setup text traces
        self.setup_text_traces()
    
    # COCO 17 keypoint names for reference
    KEYPOINT_NAMES = [
        'nose',           # 0
        'left_eye',       # 1
        'right_eye',      # 2
        'left_ear',       # 3
        'right_ear',      # 4
        'left_shoulder',  # 5
        'right_shoulder', # 6
        'left_elbow',     # 7
        'right_elbow',    # 8
        'left_wrist',     # 9
        'right_wrist',    # 10
        'left_hip',       # 11
        'right_hip',      # 12
        'left_knee',      # 13
        'right_knee',     # 14
        'left_ankle',     # 15
        'right_ankle'     # 16
    ]
    
    def setup_text_traces(self):
        """Setup text variable traces for auto-save"""
        text_vars = [
            self.label_custom_var, self.cam_custom_var, self.env_custom_var,
            self.general_custom_var, self.quality_custom_var, self.num_persons_var
            # Removed start_frame_var and end_frame_var as they're handled separately
        ]
        
        for var in text_vars:
            var.trace('w', lambda *args: self.trigger_auto_save())
    
    def on_window_resize(self, event):
        """Handle window resize event"""
        if event.widget == self.root:
            if hasattr(self, '_resize_after_id'):
                self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = self.root.after(100, self.update_responsive_layout)
    
    def update_responsive_layout(self):
        """Update responsive layout for all checkbutton groups"""
        try:
            if hasattr(self, 'scrollable_frame'):
                canvas_width = self.scrollable_frame.winfo_width()
                if canvas_width > 1:
                    available_width = canvas_width - 40
                    for group in self.checkbutton_groups.values():
                        if hasattr(group, 'arrange_responsive'):
                            group.arrange_responsive(available_width)
        except:
            pass
    
    def initialize_json_folder(self):
        """Ask user to select JSON folder on startup"""
        while True:
            response = messagebox.askyesno(
                "JSON Folder Required", 
                "A folder is required to store video labels.\n\n"
                "YES: Select existing folder\n"
                "NO: Create/Select new folder\n\n"
                "All labels will be saved as JSON files in this folder."
            )
            
            if response is None:  # User closed dialog
                messagebox.showinfo("Exiting", "Cannot proceed without a JSON folder. Exiting...")
                self.root.quit()
                self.root.destroy()
                import sys
                sys.exit()
            
            # Both YES and NO open folder dialog
            folder_path = filedialog.askdirectory(
                title="Select folder for JSON labels" if response else "Create/Select folder for JSON labels"
            )
            
            if folder_path:
                self.json_folder_path = folder_path
                self.json_manager = JSONDataManager(folder_path)
                
                # Show success message with info
                num_existing = len(self.json_manager.video_hashes_in_storage)
                messagebox.showinfo(
                    "Success", 
                    f"Using folder: {folder_path}\n\n"
                    f"Found {num_existing} existing video label(s)"
                )
                return
            else:
                # User cancelled folder dialog
                response = messagebox.askretrycancel(
                    "No Folder Selected", 
                    "You must select or create a folder to continue."
                )
                if not response:  # User clicked Cancel
                    messagebox.showinfo("Exiting", "Cannot proceed without a JSON folder. Exiting...")
                    self.root.quit()
                    self.root.destroy()
                    import sys
                    sys.exit()
                # If Retry, loop continues
    
    def setup_gui(self):
        """Setup the main GUI with resizable panes"""
        # Main container frame
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create horizontal PanedWindow for resizable sections (using tk instead of ttk)
        main_paned = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, 
                                    sashwidth=5, sashrelief=tk.RAISED)
        main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Setup panels - wrap in frames for ttk styling
        left_wrapper = ttk.Frame(main_paned)
        left_frame = self.setup_left_panel(left_wrapper)
        left_frame.pack(fill=tk.BOTH, expand=True)
        
        middle_wrapper = ttk.Frame(main_paned)
        middle_frame = self.setup_middle_panel(middle_wrapper)
        middle_frame.pack(fill=tk.BOTH, expand=True)
        
        right_wrapper = ttk.Frame(main_paned)
        right_frame = self.setup_right_panel(right_wrapper)
        right_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add frames to PanedWindow with minsize
        main_paned.add(left_wrapper, minsize=200, width=250)
        main_paned.add(middle_wrapper, minsize=400, width=800)
        main_paned.add(right_wrapper, minsize=350, width=450)
        
        # Initialize marker manager with new system
        self.marker_manager = MarkerManager(
            self.scrub_canvas, 
            self.pair_colors,
            self.on_marker_change
        )
        
        # Bind keyboard shortcuts
        self.root.bind('<space>', lambda e: self.toggle_play_pause())
        self.root.bind('<Left>', lambda e: self.previous_frame(1))
        self.root.bind('<Right>', lambda e: self.next_frame(1))
        
        # Person management shortcuts
        self.root.bind('<Delete>', lambda e: self.delete_selected_person())  # Delete key
        self.root.bind('<F2>', lambda e: self.edit_person_id())  # F2 to edit
        
        self.root.focus_set()

    def setup_left_panel(self, parent):
        """Setup left panel - SCROLLABLE"""
        # Create outer container frame (like right panel)
        left_frame = ttk.Frame(parent)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(left_frame)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_container = ttk.Frame(canvas)
        
        # Configure scroll region
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        scrollable_container.bind("<Configure>", configure_scroll_region)
        canvas_frame = canvas.create_window((0, 0), window=scrollable_container, anchor="nw")
        
        def configure_canvas(event):
            canvas.itemconfig(canvas_frame, width=event.width)
        
        canvas.bind('<Configure>', configure_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create PanedWindow inside scrollable container
        left_paned = ttk.PanedWindow(scrollable_container, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)
        
        # Pass left_paned as parent to subsections
        video_files_frame = self.setup_video_files_section(left_paned)
        pose_estimator_frame = self.setup_pose_estimator_section(left_paned)
        info_container = self.setup_info_section(left_paned)
        
        # Add to PanedWindow
        left_paned.add(video_files_frame, weight=2)
        left_paned.add(pose_estimator_frame, weight=0)
        left_paned.add(info_container, weight=0)
        
        # Mousewheel binding
        def bind_mousewheel(widget):
            """Bind mousewheel to a widget and all its children - SKIP scrollable widgets"""
            
            # Skip these widget types - they have their own scrolling
            skip_types = (tk.Listbox, ttk.Treeview, tk.Text, tk.Canvas)
            
            if not isinstance(widget, skip_types):
                # Only bind if not a scrollable widget
                def _on_mousewheel(event):
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                    return "break"
                
                widget.bind("<MouseWheel>", _on_mousewheel)
                widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
                widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))
            
            # Always recurse to children
            for child in widget.winfo_children():
                bind_mousewheel(child)
        
        self.root.after(100, lambda: bind_mousewheel(scrollable_container))
        
        return left_frame

    def setup_video_files_section(self, parent):
        video_files_frame = ttk.LabelFrame(parent, text="Video Files")
        
        # Folder selection
        ttk.Button(video_files_frame, text="Select Video Folder", 
                command=self.select_folder).pack(pady=10, padx=10, fill=tk.X)
        
        # Status summary label
        self.status_label = ttk.Label(video_files_frame, 
                                    text="Complete: 0 | Incomplete: 0 | Unprocessed: 0",
                                    font=('Arial', 9))
        self.status_label.pack(pady=5, padx=10)
        
        # Video tree with scrollbar
        tree_frame = ttk.Frame(video_files_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Create Treeview with columns
        self.video_tree = ttk.Treeview(tree_frame, columns=('status',), show='tree')
        self.video_tree.heading('#0', text='Video Files')
        self.video_tree.column('#0', width=250)
        self.video_tree.column('status', width=0, stretch=False)  # Hidden column for status
        
        # Configure tags for different states
        self.video_tree.tag_configure('complete', background='#90EE90')  # Light green
        self.video_tree.tag_configure('incomplete', background='#FFD700')  # Gold/yellow
        self.video_tree.tag_configure('unprocessed', foreground='#808080')  # Gray text
        self.video_tree.tag_configure('folder', font=('Arial', 9, 'bold'))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical', command=self.video_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal', command=self.video_tree.xview)
        self.video_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack elements
        self.video_tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event
        self.video_tree.bind('<<TreeviewSelect>>', self.on_video_select)
        
        # Initialize dictionaries for storing paths and hashes
        self.tree_items_to_paths = {}
        self.video_status_cache = {}
        
        return video_files_frame

    def setup_pose_estimator_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Pose Estimator")
        
        # Model selection
        model_frame = ttk.Frame(frame)
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(model_frame, text="Select YOLO Model", 
                command=self.select_yolo_model).pack(side=tk.LEFT, padx=5)
        
        self.model_label = ttk.Label(model_frame, text="No model selected", 
                                    font=('Arial', 8, 'italic'))
        self.model_label.pack(side=tk.LEFT, padx=5)
        
        # ADD THIS SECTION - Device selection
        device_frame = ttk.LabelFrame(frame, text="Processing Device")
        device_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Check GPU availability
        gpu_available, gpu_count, gpu_name = check_gpu_availability()
        
        device_info_frame = ttk.Frame(device_frame)
        device_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        if gpu_available:
            ttk.Label(device_info_frame, 
                    text=f"✓ GPU Available: {gpu_name}", 
                    foreground="green", 
                    font=('Arial', 9, 'bold')).pack(anchor=tk.W)
            ttk.Label(device_info_frame, 
                    text=f"  {gpu_count} GPU(s) detected", 
                    font=('Arial', 8)).pack(anchor=tk.W)
        else:
            ttk.Label(device_info_frame, 
                    text="⚠ No GPU detected - will use CPU", 
                    foreground="orange", 
                    font=('Arial', 9, 'bold')).pack(anchor=tk.W)
            ttk.Label(device_info_frame, 
                    text="  Processing will be slower", 
                    font=('Arial', 8), 
                    foreground="gray").pack(anchor=tk.W)
        
        # Device selection radio buttons
        device_select_frame = ttk.Frame(device_frame)
        device_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(device_select_frame, 
                        text="Auto (Use GPU if available)", 
                        variable=self.yolo_device, 
                        value="auto").pack(anchor=tk.W)
        
        if gpu_available:
            ttk.Radiobutton(device_select_frame, 
                            text="Force GPU", 
                            variable=self.yolo_device, 
                            value="cuda").pack(anchor=tk.W)
        
        ttk.Radiobutton(device_select_frame, 
                        text="Force CPU (Slower)", 
                        variable=self.yolo_device, 
                        value="cpu").pack(anchor=tk.W)
        # END OF ADDED SECTION
        
        # Optional: GPU status check button
        ttk.Button(device_select_frame, 
                text="Check GPU Status", 
                command=self.verify_gpu_usage).pack(anchor=tk.W, pady=5)
        
        # Threshold settings
        threshold_frame = ttk.LabelFrame(frame, text="Detection Thresholds")
        threshold_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Bbox confidence
        bbox_frame = ttk.Frame(threshold_frame)
        bbox_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(bbox_frame, text="Bbox Confidence:", font=('Arial', 9)).pack(side=tk.LEFT)
        self.bbox_conf_var = tk.StringVar(value="0.3")
        ttk.Entry(bbox_frame, textvariable=self.bbox_conf_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Temporal filtering
        temporal_frame = ttk.Frame(threshold_frame)
        temporal_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(temporal_frame, text="Temporal Filter (frames):", font=('Arial', 9)).pack(side=tk.LEFT)
        self.temporal_filter_var = tk.StringVar(value="25")
        ttk.Entry(temporal_frame, textvariable=self.temporal_filter_var, width=8).pack(side=tk.LEFT, padx=5)
        
        # Run YOLO button
        self.run_yolo_button = ttk.Button(frame, text="Run YOLO on Current Video",
                                        command=self.run_yolo_on_video,
                                        state='disabled')
        self.run_yolo_button.pack(pady=10, padx=10, fill=tk.X)
        
        # Person list
        person_list_frame = ttk.LabelFrame(frame, text="Detected Persons")
        person_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.person_listbox = tk.Listbox(person_list_frame, height=4)
        self.person_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.person_listbox.bind('<<ListboxSelect>>', self.on_person_select)
        self.person_listbox.bind('<Button-3>', self.show_person_context_menu)
        
        # Person controls
        person_controls = ttk.Frame(frame)
        person_controls.pack(fill=tk.X, padx=10, pady=5)
        
        self.delete_person_button = ttk.Button(person_controls, text="Delete Person",
                                                command=self.delete_selected_person,
                                                state='disabled')
        self.delete_person_button.pack(side=tk.LEFT, padx=2)
        
        self.edit_person_id_button = ttk.Button(person_controls, text="Edit Person ID",
                                                command=self.edit_person_id,
                                                state='disabled')
        self.edit_person_id_button.pack(side=tk.LEFT, padx=2)
        
        # Visualization toggle
        viz_frame = ttk.Frame(frame)
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Checkbutton(viz_frame, text="Show Skeleton & Keypoints",
                        variable=self.show_skeleton,
                        command=self.on_skeleton_toggle).pack(anchor=tk.W)
        
        return frame
    
    def setup_info_section(self, parent):
        info_container = ttk.Frame(parent)
        
        # Video info section
        info_frame = ttk.LabelFrame(info_container, text="Video Information")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.video_filename_label = ttk.Label(info_frame, text="No video selected", 
                                            wraplength=280, font=('Arial', 9, 'bold'))
        self.video_filename_label.pack(pady=5, padx=10, anchor=tk.W)
        
        self.video_size_label = ttk.Label(info_frame, text="Size: - x -", font=('Arial', 9))
        self.video_size_label.pack(pady=2, padx=10, anchor=tk.W)
        
        self.video_color_label = ttk.Label(info_frame, text="Color: -", font=('Arial', 9))
        self.video_color_label.pack(pady=2, padx=10, anchor=tk.W)
        
        # Marker pair selection
        pair_frame = ttk.LabelFrame(info_container, text="Marker Pair Selection")
        pair_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.pair_listbox = tk.Listbox(pair_frame, height=4)
        self.pair_listbox.pack(fill=tk.X, padx=10, pady=5)
        self.pair_listbox.bind('<<ListboxSelect>>', self.on_pair_select)
        
        # Pair details
        details_frame = ttk.LabelFrame(info_container, text="Pair Details")
        details_frame.pack(fill=tk.X, padx=5, pady=5)
        
        frame_input_frame = ttk.Frame(details_frame)
        frame_input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        frame_input_frame.grid_columnconfigure(1, weight=1)
        frame_input_frame.grid_columnconfigure(3, weight=1)
        
        ttk.Label(frame_input_frame, text="Start:", font=('Arial', 9)).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.start_frame_entry = ttk.Entry(frame_input_frame, textvariable=self.start_frame_var, width=8)
        self.start_frame_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        
        ttk.Label(frame_input_frame, text="End:", font=('Arial', 9)).grid(row=0, column=2, sticky=tk.W, padx=(10, 5))
        self.end_frame_entry = ttk.Entry(frame_input_frame, textvariable=self.end_frame_var, width=8)
        self.end_frame_entry.grid(row=0, column=3, sticky=tk.EW, padx=5)
        
        self.duration_label = ttk.Label(details_frame, text="Duration: 0.0 seconds", font=('Arial', 9))
        self.duration_label.pack(pady=5, padx=10, anchor=tk.W)
        
        return info_container

    def setup_middle_panel(self, parent):
        """Setup middle panel with video player and controls - return the frame"""
        middle_frame = ttk.LabelFrame(parent, text="Video Player")
        
        # Video display
        self.video_label = tk.Label(middle_frame, bg='black', text="No video loaded",
                                fg='white', font=('Arial', 16))
        self.video_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        # Scrub bar with redesigned markers
        self.setup_scrub_bar(middle_frame)
        
        # Action markers section
        markers_frame = ttk.LabelFrame(middle_frame, text="Action Markers")
        markers_frame.pack(pady=10, padx=20, fill=tk.X)
        
        markers_control = ttk.Frame(markers_frame)
        markers_control.pack(pady=10)
        
        ttk.Button(markers_control, text="Create Marker Pair", 
                command=self.create_marker_pair).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(markers_control, text="Delete Selected Pair", 
                command=self.delete_selected_pair).pack(side=tk.LEFT, padx=5)
        
        # Control buttons
        controls_frame = ttk.Frame(middle_frame)
        controls_frame.pack(pady=10)
        
        ttk.Button(controls_frame, text="<<5", command=lambda: self.previous_frame(5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="<1", command=lambda: self.previous_frame(1)).pack(side=tk.LEFT, padx=2)
        
        self.play_button = ttk.Button(controls_frame, text="▶ Play", command=self.toggle_play_pause)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="1>", command=lambda: self.next_frame(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="5>>", command=lambda: self.next_frame(5)).pack(side=tk.LEFT, padx=2)
        
        # Frame info
        self.frame_info_label = ttk.Label(middle_frame, text="Frame: 0 / 0")
        self.frame_info_label.pack(pady=5)
        
        return middle_frame

    def setup_scrub_bar(self, parent):
        """Setup custom scrub bar with redesigned markers"""
        scrub_frame = ttk.Frame(parent)
        scrub_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Create canvas for custom scrub bar
        self.scrub_canvas = tk.Canvas(scrub_frame, height=30, bg='lightgray')
        self.scrub_canvas.pack(fill=tk.X)
        
        # Bind events for scrub bar interaction (for timeline scrubbing, not marker dragging)
        self.scrub_canvas.bind('<Button-1>', self.on_scrub_click)
        self.scrub_canvas.bind('<B1-Motion>', self.on_scrub_drag)
        self.scrub_canvas.bind('<ButtonRelease-1>', self.on_scrub_release)

    def setup_right_panel(self, parent):
        """Setup right panel with labeling tools - with proper scrolling"""
        right_frame = ttk.LabelFrame(parent, text="Labeling Tools")
        
        # Create scrollable frame
        canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.scrollable_frame.bind("<Configure>", configure_scroll_region)
        
        canvas_frame = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Update canvas frame width when canvas resizes
        def configure_canvas(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_frame, width=canvas_width)
            # Trigger responsive layout update
            self.update_responsive_layout()
        
        canvas.bind('<Configure>', configure_canvas)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Setup mouse wheel scrolling
        def bind_mousewheel(widget):
            """Bind mousewheel to a widget and all its children"""
            def _on_mousewheel(event):
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                return "break"
            
            # Bind to the widget
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
            widget.bind("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux
            
            # Recursively bind to all children
            for child in widget.winfo_children():
                bind_mousewheel(child)
        
        # Store the bind function for later use after widgets are created
        self.bind_mousewheel_func = lambda: bind_mousewheel(self.scrollable_frame)
        
        self.setup_labeling_content()
        
        # Apply mousewheel binding after all widgets are created
        self.root.after(100, self.bind_mousewheel_func)
        
        return right_frame

    def setup_labeling_content(self):
        """Setup labeling content with checkbutton groups"""
        # Note: Video Information, Marker Pair Selection, and Pair Details 
        # have been moved to the left panel
        
        # Label section with checkbuttons
        self.setup_label_section()
        
        # Activity type section
        self.setup_activity_type_section()
        
        # Environment section
        self.setup_environment_section()
        
        # Save/Refresh buttons
        button_frame = ttk.Frame(self.scrollable_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="💾 Save Labels", command=self.manual_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Refresh", command=self.refresh_labels).pack(side=tk.LEFT, padx=5)
        
        # Initialize responsive layout
        self.root.after(100, self.update_responsive_layout)

    # def setup_pair_details(self):
    #     """Setup pair details section with better expansion"""
    #     details_frame = ttk.LabelFrame(self.scrollable_frame, text="Pair Details")
    #     details_frame.pack(fill=tk.X, padx=10, pady=5)
        
    #     # Frame inputs with grid weight configuration
    #     frame_input_frame = ttk.Frame(details_frame)
    #     frame_input_frame.pack(fill=tk.X, padx=10, pady=5)
        
    #     # Configure grid columns for expansion
    #     frame_input_frame.grid_columnconfigure(1, weight=1)
    #     frame_input_frame.grid_columnconfigure(3, weight=1)
        
    #     ttk.Label(frame_input_frame, text="Start:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
    #     self.start_frame_entry = ttk.Entry(frame_input_frame, textvariable=self.start_frame_var, width=10)
    #     self.start_frame_entry.grid(row=0, column=1, sticky=tk.EW, padx=5)
        
    #     ttk.Label(frame_input_frame, text="End:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
    #     self.end_frame_entry = ttk.Entry(frame_input_frame, textvariable=self.end_frame_var, width=10)
    #     self.end_frame_entry.grid(row=0, column=3, sticky=tk.EW, padx=5)
        
    #     self.duration_label = ttk.Label(details_frame, text="Duration: 0.0 seconds")
    #     self.duration_label.pack(pady=5, padx=10, anchor=tk.W)

    def setup_label_section(self):
        """Setup label section with better horizontal expansion and fixed width for Other"""
        label_frame = ttk.LabelFrame(self.scrollable_frame, text="Label")
        label_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Standard labels
        label_grid_frame = ttk.Frame(label_frame)
        label_grid_frame.pack(fill=tk.X, padx=10, pady=5)
        
        label_options = [("fall", "Fall"), ("normal", "Normal")]
        self.checkbutton_groups['label'] = ToggleCheckButtonGroup(
            label_grid_frame, label_options, self.on_label_change
        )
        
        # Custom label with controlled width
        custom_frame = ttk.Frame(label_frame)
        custom_frame.pack(fill=tk.X, padx=10, pady=2)
        
        # Use grid for better alignment
        self.label_other_var = tk.BooleanVar()
        self.label_other_cb = tk.Checkbutton(
            custom_frame, text="Other:", variable=self.label_other_var,
            command=self.on_label_other_click, font=("Arial", 9),
            indicatoron=1, padx=5, pady=2
        )
        self.label_other_cb.grid(row=0, column=0, sticky=tk.W)
        
        self.label_custom_entry = ttk.Entry(custom_frame, textvariable=self.label_custom_var, width=20)
        self.label_custom_entry.grid(row=0, column=1, padx=5, sticky=tk.W)

    def setup_activity_type_section(self):
        """Setup activity type section with better layout and fixed width for Other"""
        activity_frame = ttk.LabelFrame(self.scrollable_frame, text="Activity Type")
        activity_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Falling direction
        fall_dir_frame = ttk.LabelFrame(activity_frame, text="Falling Direction")
        fall_dir_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Container for both direction groups
        directions_container = ttk.Frame(fall_dir_frame)
        directions_container.pack(fill=tk.X, padx=10, pady=2)
        
        # Front/Back
        fb_frame = ttk.Frame(directions_container)
        fb_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        fb_options = [("front", "Front"), ("back", "Back")]
        self.checkbutton_groups['front_back'] = ToggleCheckButtonGroup(
            fb_frame, fb_options, self.on_falling_direction_change
        )
        
        # Left/Right
        lr_frame = ttk.Frame(directions_container)
        lr_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        lr_options = [("left", "Left"), ("right", "Right")]
        self.checkbutton_groups['left_right'] = ToggleCheckButtonGroup(
            lr_frame, lr_options, self.on_falling_direction_change
        )
        
        # General action with expanded options
        general_frame = ttk.LabelFrame(activity_frame, text="General Action")
        general_frame.pack(fill=tk.X, padx=5, pady=5)
        
        general_grid_frame = ttk.Frame(general_frame)
        general_grid_frame.pack(fill=tk.X, padx=10, pady=2)
        
        # Expanded general options
        general_options = [
            ("sitting", "Sitting"),
            ("standing", "Standing"), 
            ("walking", "Walking"),
            ("hopping", "Hopping"),
            ("bending", "Bending"),
            ("lying", "Lying"),
            ("crawling", "Crawling")
        ]
        self.checkbutton_groups['general'] = ToggleCheckButtonGroup(
            general_grid_frame, general_options, self.on_general_action_change
        )
        
        # General other with controlled width
        general_other_frame = ttk.Frame(general_frame)
        general_other_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.general_other_var = tk.BooleanVar()
        self.general_other_cb = tk.Checkbutton(
            general_other_frame, text="Other:", variable=self.general_other_var,
            command=self.on_general_other_click, font=("Arial", 9),
            indicatoron=1, padx=5, pady=2
        )
        self.general_other_cb.grid(row=0, column=0, sticky=tk.W)
        # Store reference to variable for ActivityTypeManager
        self.general_other_cb.var = self.general_other_var
        
        self.general_custom_entry = ttk.Entry(general_other_frame, textvariable=self.general_custom_var, width=20)
        self.general_custom_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # Initialize activity manager with "Other" widgets
        falling_groups = [
            self.checkbutton_groups['front_back'],
            self.checkbutton_groups['left_right']
        ]
        general_group = self.checkbutton_groups['general']
        general_other_widgets = (self.general_other_cb, self.general_custom_entry)
        
        self.activity_manager = ActivityTypeManager(
            falling_groups, general_group, general_other_widgets, self.trigger_auto_save
        )
    
    def setup_environment_section(self):
        """Setup environment section with better horizontal expansion and fixed widths"""
        env_frame = ttk.LabelFrame(self.scrollable_frame, text="Camera & Environment")
        env_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Camera angle
        cam_frame = ttk.LabelFrame(env_frame, text="Camera Angle")
        cam_frame.pack(fill=tk.X, padx=5, pady=5)
        
        cam_grid_frame = ttk.Frame(cam_frame)
        cam_grid_frame.pack(fill=tk.X, padx=10, pady=2)
        
        cam_options = [("body_level", "Body Level"), ("overhead", "Overhead")]
        self.checkbutton_groups['camera'] = ToggleCheckButtonGroup(
            cam_grid_frame, cam_options, self.trigger_auto_save
        )
        
        # Camera other with controlled width
        cam_other_frame = ttk.Frame(cam_frame)
        cam_other_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.cam_other_var = tk.BooleanVar()
        self.cam_other_cb = tk.Checkbutton(
            cam_other_frame, text="Other:", variable=self.cam_other_var,
            command=self.on_cam_other_click, font=("Arial", 9),
            indicatoron=1, padx=5, pady=2
        )
        self.cam_other_cb.grid(row=0, column=0, sticky=tk.W)
        
        self.cam_custom_entry = ttk.Entry(cam_other_frame, textvariable=self.cam_custom_var, width=20)
        self.cam_custom_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # Environment
        room_frame = ttk.LabelFrame(env_frame, text="Environment")
        room_frame.pack(fill=tk.X, padx=5, pady=5)
        
        room_grid_frame = ttk.Frame(room_frame)
        room_grid_frame.pack(fill=tk.X, padx=10, pady=2)
        
        room_options = [("living_room", "Living Room"), ("bed_room", "Bed Room"),
                    ("bathroom", "Bathroom"), ("kitchen", "Kitchen")]
        self.checkbutton_groups['environment'] = ToggleCheckButtonGroup(
            room_grid_frame, room_options, self.trigger_auto_save
        )
        
        # Environment other with controlled width
        env_other_frame = ttk.Frame(room_frame)
        env_other_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.env_other_var = tk.BooleanVar()
        self.env_other_cb = tk.Checkbutton(
            env_other_frame, text="Other:", variable=self.env_other_var,
            command=self.on_env_other_click, font=("Arial", 9),
            indicatoron=1, padx=5, pady=2
        )
        self.env_other_cb.grid(row=0, column=0, sticky=tk.W)
        
        self.env_custom_entry = ttk.Entry(env_other_frame, textvariable=self.env_custom_var, width=20)
        self.env_custom_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        # Person details
        person_frame = ttk.LabelFrame(env_frame, text="Person Details")
        person_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Number of persons
        num_frame = ttk.Frame(person_frame)
        num_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(num_frame, text="Number of persons:").pack(side=tk.LEFT)
        
        self.num_persons_entry = ttk.Entry(num_frame, textvariable=self.num_persons_var, width=5)
        self.num_persons_entry.pack(side=tk.LEFT, padx=5)
        
        # Age group
        age_frame = ttk.LabelFrame(person_frame, text="Main Subject Age Group")
        age_frame.pack(fill=tk.X, padx=5, pady=2)
        
        age_grid_frame = ttk.Frame(age_frame)
        age_grid_frame.pack(fill=tk.X, padx=10, pady=2)
        
        age_options = [("young", "Young"), ("middle_aged", "Middle Aged"), ("elderly", "Elderly")]
        self.checkbutton_groups['age'] = ToggleCheckButtonGroup(
            age_grid_frame, age_options, self.trigger_auto_save
        )
        
        # Person mobility
        mobility_frame = ttk.LabelFrame(person_frame, text="Person Mobility")
        mobility_frame.pack(fill=tk.X, padx=5, pady=2)
        
        mobility_grid_frame = ttk.Frame(mobility_frame)
        mobility_grid_frame.pack(fill=tk.X, padx=10, pady=2)
        
        mobility_options = [("normal", "Normal"), ("assisted_walking", "Assisted Walking"),
                        ("wheelchair", "Wheelchair")]
        self.checkbutton_groups['mobility'] = ToggleCheckButtonGroup(
            mobility_grid_frame, mobility_options, self.trigger_auto_save
        )
        
        # Video quality
        quality_frame = ttk.LabelFrame(env_frame, text="Video Quality")
        quality_frame.pack(fill=tk.X, padx=5, pady=5)
        
        quality_grid_frame = ttk.Frame(quality_frame)
        quality_grid_frame.pack(fill=tk.X, padx=10, pady=2)
        
        quality_options = [("good", "Good"), ("blurry", "Blurry"), ("poor_lighting", "Poor Lighting")]
        self.checkbutton_groups['quality'] = ToggleCheckButtonGroup(
            quality_grid_frame, quality_options, self.trigger_auto_save
        )
        
        # Quality other with controlled width
        quality_other_frame = ttk.Frame(quality_frame)
        quality_other_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.quality_other_var = tk.BooleanVar()
        self.quality_other_cb = tk.Checkbutton(
            quality_other_frame, text="Other:", variable=self.quality_other_var,
            command=self.on_quality_other_click, font=("Arial", 9),
            indicatoron=1, padx=5, pady=2
        )
        self.quality_other_cb.grid(row=0, column=0, sticky=tk.W)
        
        self.quality_custom_entry = ttk.Entry(quality_other_frame, textvariable=self.quality_custom_var, width=20)
        self.quality_custom_entry.grid(row=0, column=1, padx=5, sticky=tk.W)
    
    # Event handlers for "other" checkbuttons
    def on_label_other_click(self):
        """Handle label other checkbox"""
        if self.label_other_var.get():
            # Clear standard selections
            self.checkbutton_groups['label'].set_selection(None)
        else:
            # Clear custom text when unchecking
            self.label_custom_var.set("")
        self.on_label_change()
    
    def on_general_other_click(self):
        """Handle general action other checkbox"""
        if self.general_other_var.get():
            self.checkbutton_groups['general'].set_selection(None)
            # Clear falling directions when selecting general "Other"
            self.checkbutton_groups['front_back'].set_selection(None)
            self.checkbutton_groups['left_right'].set_selection(None)
        else:
            self.general_custom_var.set("")
        self.on_general_action_change()
    
    def on_cam_other_click(self):
        """Handle camera other checkbox"""
        if self.cam_other_var.get():
            self.checkbutton_groups['camera'].set_selection(None)
        else:
            self.cam_custom_var.set("")
        self.trigger_auto_save()
    
    def on_env_other_click(self):
        """Handle environment other checkbox"""
        if self.env_other_var.get():
            self.checkbutton_groups['environment'].set_selection(None)
        else:
            self.env_custom_var.set("")
        self.trigger_auto_save()
    
    def on_quality_other_click(self):
        """Handle quality other checkbox"""
        if self.quality_other_var.get():
            self.checkbutton_groups['quality'].set_selection(None)
        else:
            self.quality_custom_var.set("")
        self.trigger_auto_save()
    
    # Main event handlers
    def on_label_change(self):
        """Handle label change and update activity sections"""
        label_selection = self.get_current_label_selection()
        if self.activity_manager:
            self.activity_manager.update_activity_sections_state(label_selection)
        self.trigger_auto_save()
    
    def on_falling_direction_change(self):
        """Handle falling direction change"""
        if self.activity_manager:
            self.activity_manager.handle_falling_direction_change()
        self.trigger_auto_save()
    
    def on_general_action_change(self):
        """Handle general action change"""
        if self.activity_manager:
            self.activity_manager.handle_general_action_change()
        self.trigger_auto_save()
    
    def get_current_label_selection(self):
        """Get current label selection"""
        standard_selection = self.checkbutton_groups['label'].get_selection()
        if standard_selection:
            return standard_selection
        elif self.label_other_var.get():
            return "other"
        return ""
    
    def trigger_auto_save(self):
        """Trigger auto-save through manager and cache current labels"""
        # CRITICAL: Check switching flag BEFORE caching to prevent overwriting during load
        if self.auto_save_manager.switching_markers or self.auto_save_manager.loading_in_progress:
            # Don't cache during switching or loading
            print(f"Skipping cache during switching/loading")
            return
        
        # Cache current pair labels BEFORE triggering save
        if self.selected_yolo_track_id and self.marker_manager.selected_pair_index:
            self.cache_current_pair_labels()
            print(f"Cached labels for YOLO:{self.selected_yolo_track_id}, Pair:{self.marker_manager.selected_pair_index}")
            
        self.update_person_listbox()
        
        self.auto_save_manager.trigger_save()
    
    # Video management methods
    def select_folder(self):
        """Select folder and calculate hashes"""
        folder_path = filedialog.askdirectory(title="Select Video Folder")
        if folder_path:
            self.load_video_files_with_hash(folder_path)
    
    def load_video_files_with_hash(self, folder_path):
        """Load video files recursively and calculate MD5 hashes"""
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
        
        # Clear existing tree and data
        self.video_tree.delete(*self.video_tree.get_children())
        self.tree_items_to_paths = {}
        self.video_files = []
        self.video_hashes = {}
        self.video_status_cache = {}  # Clear status cache
        
        # Find all video files using os.walk
        video_files_found = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, folder_path)
                    video_files_found.append((full_path, relative_path))
        
        if not video_files_found:
            messagebox.showwarning("Warning", "No video files found in selected folder")
            return
        
        # Set video_files list immediately
        self.video_files = [vf[0] for vf in video_files_found]
        
        # Build tree structure first (on main thread)
        self.build_tree_structure(folder_path, video_files_found)
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Video Files")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="Calculating hashes and checking status...").pack(pady=20)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=100)
        progress_bar.pack(pady=10, padx=20, fill=tk.X)
        
        self.progress_label = ttk.Label(progress_window, text="0 / 0")
        self.progress_label.pack(pady=10)
        
        # Process videos in separate thread
        def process_videos():
            processed_count = 0
            total_count = len(video_files_found)
            
            for i, (video_path, _) in enumerate(video_files_found):
                # Update progress
                progress = ((i + 1) / total_count) * 100
                self.root.after(0, lambda p=progress, c=i+1, t=total_count: 
                            (progress_var.set(p), self.progress_label.config(text=f"{c} / {t}")))
                
                # Calculate hash
                hash_md5 = hashlib.md5()
                try:
                    with open(video_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)
                    video_hash = hash_md5.hexdigest()
                    self.video_hashes[video_path] = video_hash
                    
                    # Check completeness
                    status = self.check_video_completeness(video_hash)
                    
                    # Update tree item status on main thread
                    self.root.after(0, lambda vp=video_path, s=status: self.update_video_tree_status(vp, s))
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")
                    self.video_hashes[video_path] = "error"
            
            # Final update after all processing is done
            self.root.after(100, self.update_status_summary)  # Small delay to ensure all tree updates complete
            self.root.after(150, lambda: messagebox.showinfo("Success", f"Loaded {total_count} video files"))
            self.root.after(200, progress_window.destroy)
        
        thread = threading.Thread(target=process_videos)
        thread.daemon = True
        thread.start()
    
    def finish_video_loading(self):
        """Finish loading videos after hash calculation"""
        self.video_listbox.delete(0, tk.END)
        for video_path in self.video_files:
            filename = os.path.basename(video_path)
            self.video_listbox.insert(tk.END, filename)
        
        messagebox.showinfo("Success", f"Loaded {len(self.video_files)} video files")
    
    def on_video_select(self, event):
        """Handle video selection from tree"""
        # Save current labels before switching
        if self.current_video_path:
            self.perform_auto_save()
        
        # Get selected item
        selection = self.video_tree.selection()
        if not selection:
            return
        
        item_id = selection[0]
        
        # Check if it's a video file (not a folder)
        if item_id in self.tree_items_to_paths:
            video_path = self.tree_items_to_paths[item_id]
            self.load_video(video_path)
            
            # Update status after loading
            if video_path in self.video_hashes:
                video_hash = self.video_hashes[video_path]
                status = self.check_video_completeness(video_hash)
                self.update_video_tree_status(video_path, status)
    
    def load_video(self, video_path):
        """Load selected video and its JSON data"""
        if self.current_video:
            self.current_video.release()
        
        # Clear pair details BEFORE loading new video data
        self.start_frame_var.set("")
        self.end_frame_var.set("")
        self.duration_label.config(text="Duration: 0.0 seconds")
        
        # CLEAR video-level fields when switching videos
        self.checkbutton_groups['camera'].set_selection(None)
        self.cam_other_var.set(False)
        self.cam_custom_var.set("")

        self.checkbutton_groups['environment'].set_selection(None)
        self.env_other_var.set(False)
        self.env_custom_var.set("")

        self.checkbutton_groups['age'].set_selection(None)
        self.checkbutton_groups['mobility'].set_selection(None)

        self.checkbutton_groups['quality'].set_selection(None)
        self.quality_other_var.set(False)
        self.quality_custom_var.set("")

        # Clear pair-specific fields
        self.clear_pair_specific_ui()
        
        # CLEAR all marker and person state when switching videos
        self.selected_person_id = None
        self.label_cache = {}
        if self.marker_manager:
            self.marker_manager.clear_all_persons_markers()
            self.marker_manager.current_person_id = None
        
        self.current_video_path = video_path
        self.current_video = cv2.VideoCapture(video_path)
        
        if not self.current_video.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            return
        
        # Get video properties
        self.total_frames = int(self.current_video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.current_video.get(cv2.CAP_PROP_FPS)) or 30
        self.video_width = int(self.current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine color mode
        ret, frame = self.current_video.read()
        if ret:
            if len(frame.shape) == 3:
                self.color_mode = "BGR" if frame.shape[2] == 3 else "BGRA"
            else:
                self.color_mode = "Grayscale"
        
        self.current_frame_idx = 0
        
        # Update marker manager
        self.marker_manager.set_video_info(self.total_frames)
        
        # Update GUI
        filename = os.path.basename(video_path)
        self.video_filename_label.config(text=f"File: {filename}")
        self.video_size_label.config(text=f"Size: {self.video_width} x {self.video_height}")
        self.video_color_label.config(text=f"Color: {self.color_mode}")
        
        # NEW: Load JSON data if exists
        video_hash = self.video_hashes.get(video_path, "")
        if video_hash and self.json_manager:
            self.current_video_data = self.json_manager.load_video_data(video_hash)
            
            if self.current_video_data:
                # Video has existing data
                self.load_video_data_from_json()
            else:
                # New video - no data yet
                self.current_video_data = None
                self.yolo_results = None
                self.selected_person_id = None
                
                # Disable person controls
                self.delete_person_button.config(state='disabled')
                self.edit_person_id_button.config(state='disabled')
                
                # Clear person list
                self.person_listbox.delete(0, tk.END)
                
                # Enable YOLO run button if model is selected
                if self.yolo_model_path:
                    self.run_yolo_button.config(state='normal')
        
        # Show first frame and update UI
        self.show_frame()
        self.update_scrub_bar()
        self.update_pair_listbox()
        
        # Stop playing when new video loads
        self.is_playing = False
        self.play_button.config(text="▶ Play")
    
    def load_video_data_from_json(self):
        """Load video data from JSON"""
        if not self.current_video_data:
            return
        
        print(f"=== LOADING VIDEO DATA FROM JSON ===")
        
        # Mark that we have pose data (for visualization)
        self.yolo_results = "loaded_from_json"
        
        # Clear all person markers and cache
        self.marker_manager.clear_all_persons_markers()
        self.label_cache = {}
        
        # Load markers and labels for all persons
        for person in self.current_video_data.get('persons', []):
            yolo_track_id = person['yolo_track_id']
            person_id = person['person_id']
            action_labels = person.get('action_labels', [])
            
            print(f"Loading Person {person_id} (YOLO:{yolo_track_id}): {len(action_labels)} action labels")
            
            if action_labels:
                # Load markers
                self.marker_manager.load_person_markers_from_data(yolo_track_id, action_labels)
                
                # Initialize cache for this YOLO track
                if yolo_track_id not in self.label_cache:
                    self.label_cache[yolo_track_id] = {}
                
                # Cache labels for each marker
                for idx, action in enumerate(action_labels, 1):
                    self.label_cache[yolo_track_id][idx] = {
                        'label': action.get('label', ''),
                        'fall_direction': action.get('fall_direction', None),
                        'general_action': action.get('general_action', None)
                    }
                    print(f"  Cached marker {idx}: label='{action.get('label', '')}', "
                        f"fall_dir='{action.get('fall_direction', '')}', "
                        f"general='{action.get('general_action', '')}'")
        
        # Update person listbox
        self.update_person_listbox()
        
        # Enable person controls
        self.delete_person_button.config(state='normal')
        self.edit_person_id_button.config(state='normal')
        
        # Enable YOLO run button for re-running
        if self.yolo_model_path:
            self.run_yolo_button.config(state='normal')
        
        # Load video-level metadata to UI
        self.load_video_metadata_to_ui()
        
        print(f"✓ Loaded data for {len(self.current_video_data.get('persons', []))} person(s)")
    
    def check_person_completeness(self, person):
        """
        Check person's marker completeness
        Returns:
            0 = No markers (gray)
            1 = Has markers but incomplete (yellow)
            2 = All markers complete (green)
        """
        action_labels = person.get('action_labels', [])
        
        if not action_labels:
            return 0  # No markers - gray
        
        # Check if all markers are complete
        all_complete = True
        
        for action in action_labels:
            label = action.get('label', '')
            
            # Must have label
            if not label or label == '':
                all_complete = False
                break
            
            # Check if required activity fields are filled
            if label == 'fall':
                # For falls, must have fall direction
                if not action.get('fall_direction'):
                    all_complete = False
                    break
            elif label == 'normal':
                # For normal, must have general action
                if not action.get('general_action'):
                    all_complete = False
                    break
            # For other labels, either fall_dir or general_action is acceptable
            else:
                has_activity = (action.get('fall_direction') or 
                            action.get('general_action'))
                if not has_activity:
                    all_complete = False
                    break
        
        if all_complete:
            return 2  # Green - complete
        else:
            return 1  # Yellow - incomplete
    
    def update_person_listbox(self):
        """Update person listbox with color coding"""
        self.person_listbox.delete(0, tk.END)
        
        if not self.current_video_data or 'persons' not in self.current_video_data:
            return
        
        for person in self.current_video_data['persons']:
            person_id = person['person_id']
            yolo_id = person.get('yolo_track_id', '?')
            
            # Format: "Person 1 [YOLO: 7]"
            display_text = f"Person {person_id} [YOLO: {yolo_id}]"
            self.person_listbox.insert(tk.END, display_text)
            
            # Determine color based on completeness
            status = self.check_person_completeness(person)
            
            # Apply color to the last inserted item
            idx = self.person_listbox.size() - 1
            
            if status == 0:
                # Gray - no markers
                self.person_listbox.itemconfig(idx, fg='gray')
            elif status == 1:
                # Yellow/Orange - has markers but incomplete
                self.person_listbox.itemconfig(idx, fg='#FF8C00')  # Dark orange
            elif status == 2:
                # Green - complete
                self.person_listbox.itemconfig(idx, fg='green')
    
    def load_video_metadata_to_ui(self):
        """Load video-level metadata to UI fields"""
        if not self.current_video_data or 'video_metadata' not in self.current_video_data:
            return
        
        metadata = self.current_video_data['video_metadata']
        
        # Load camera angle
        camera = metadata.get('camera_angle', '')
        if camera:
            if camera in ['body_level', 'overhead']:
                self.checkbutton_groups['camera'].set_selection(camera)
            else:
                self.cam_other_var.set(True)
                self.cam_custom_var.set(camera)
        
        # Load environment
        env = metadata.get('environment', '')
        if env:
            if env in ['living_room', 'bed_room', 'bathroom', 'kitchen']:
                self.checkbutton_groups['environment'].set_selection(env)
            else:
                self.env_other_var.set(True)
                self.env_custom_var.set(env)
        
        # Load video quality
        quality = metadata.get('video_quality', '')
        if quality:
            if quality in ['good', 'blurry', 'poor_lighting']:
                self.checkbutton_groups['quality'].set_selection(quality)
            else:
                self.quality_other_var.set(True)
                self.quality_custom_var.set(quality)
    
    def show_frame(self):
        """Display current frame with proper fit mode and pose visualization"""
        if not self.current_video:
            return
        
        self.current_video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.current_video.read()
        
        if not ret:
            return
        
        # Draw bboxes and keypoints if YOLO results exist
        if self.current_video_data and 'persons' in self.current_video_data:
            frame = self.draw_pose_on_frame(frame)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get label dimensions
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        if label_width <= 1:
            label_width = 800
        if label_height <= 1:
            label_height = 600
        
        # Calculate scaling with padding
        max_width = max(400, label_width - 40)
        max_height = max(300, label_height - 40)
        
        height, width = frame_rgb.shape[:2]
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image)
        
        # Update label
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo
        
        # Update frame info
        self.frame_info_label.config(text=f"Frame: {self.current_frame_idx + 1} / {self.total_frames}")
    
    def draw_pose_on_frame(self, frame):
        """Draw bboxes and keypoints on frame for all persons"""
        if not self.current_video_data:
            return frame
        
        persons = self.current_video_data.get('persons', [])
        
        # Quick exit if no persons
        if not persons:
            return frame
        
        for person in persons:
            person_id = person['person_id']
            
            # Get frame data for current frame
            keypoints_seq = person.get('keypoints_sequence', [])
            if self.current_frame_idx >= len(keypoints_seq):
                continue
            
            frame_data = keypoints_seq[self.current_frame_idx]
            
            # Check if person is detected in this frame
            if frame_data.get('bbox') is None or frame_data.get('keypoints') is None:
                continue  # Person not detected in this frame
            
            bbox = frame_data['bbox']
            keypoints = frame_data['keypoints']
            
            # Determine color based on selection
            yolo_track_id = person.get('yolo_track_id')

            # Highlight all persons with the same person_id as selected
            if self.selected_person_id and person_id == self.selected_person_id:
                color = (0, 0, 255)  # Bright RED for all with same person_id
                thickness = 3
            else:
                color = self.get_person_color(person_id)
                thickness = 2
            
            # Draw bbox
            x, y, w, h = bbox
            cv2.rectangle(frame, 
                        (int(x), int(y)), 
                        (int(x + w), int(y + h)),
                        color, thickness)
            
            # Draw person ID label with background
            label_text = f"Person {person_id}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame,
                        (int(x), int(y) - label_size[1] - 10),
                        (int(x) + label_size[0] + 10, int(y)),
                        color, -1)
            
            # Draw label text
            cv2.putText(frame, label_text,
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw keypoints and skeleton if toggle is on
            if self.show_skeleton.get():
                self.draw_keypoints_and_skeleton(frame, keypoints, color)
        
        return frame
    
    def get_person_color(self, person_id):
        """Get BGR color for person (not red, which is reserved for selection)"""
        # Colors in BGR format (avoiding red)
        colors = [
            (255, 140, 0),    # Blue
            (0, 255, 0),      # Green
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 165, 255),    # Orange
            (203, 192, 255),  # Pink
            (128, 128, 128),  # Gray
            (0, 255, 255),    # Yellow
            (255, 128, 0),    # Deep blue
            (128, 0, 128),    # Purple
        ]
        
        # Use person_id to select color (wrap around if more persons than colors)
        color_idx = (person_id - 1) % len(colors)
        return colors[color_idx]
    
    def get_bgr_color(self, color_name):
        """Convert color name to BGR tuple for OpenCV"""
        color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'orange': (0, 165, 255),
            'purple': (255, 0, 255),
            'brown': (42, 42, 165),
            'pink': (203, 192, 255),
            'gray': (128, 128, 128),
            'cyan': (255, 255, 0),
            'yellow': (0, 255, 255)
        }
        return color_map.get(color_name, (255, 255, 255))
    
    def validate_keypoint(self, kpt):
        """Validate keypoint has correct format and is within frame bounds"""
        if not kpt or len(kpt) < 3:
            return False
        
        x, y, conf = kpt[0], kpt[1], kpt[2]
        
        # Check if within frame bounds
        if x < 0 or x >= self.video_width or y < 0 or y >= self.video_height:
            return False
        
        # Check confidence
        if conf <= 0:
            return False
        
        return True
    
    def draw_keypoints_and_skeleton(self, frame, keypoints, color):
        """Draw 17 COCO keypoints and skeleton connections"""
        if not keypoints or len(keypoints) != 17:
            return
        
        # COCO 17 keypoint skeleton connections
        skeleton = [
            (0, 1), (0, 2),        # Nose to eyes
            (1, 3), (2, 4),        # Eyes to ears
            (0, 5), (0, 6),        # Nose to shoulders
            (5, 6),                # Shoulders
            (5, 7), (7, 9),        # Left arm
            (6, 8), (8, 10),       # Right arm
            (5, 11), (6, 12),      # Shoulders to hips
            (11, 12),              # Hips
            (11, 13), (13, 15),    # Left leg
            (12, 14), (14, 16)     # Right leg
        ]
        
        # Confidence threshold for drawing
        conf_threshold = 0.3
        
        # Draw skeleton lines first (so they appear behind keypoints)
        for conn in skeleton:
            pt1_idx, pt2_idx = conn
            
            if pt1_idx >= len(keypoints) or pt2_idx >= len(keypoints):
                continue
            
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            # Check if both keypoints have sufficient confidence
            if len(pt1) >= 3 and len(pt2) >= 3:
                if pt1[2] > conf_threshold and pt2[2] > conf_threshold:
                    cv2.line(frame,
                            (int(pt1[0]), int(pt1[1])),
                            (int(pt2[0]), int(pt2[1])),
                            color, 2)
        
        # Draw keypoint dots on top
        for i, kpt in enumerate(keypoints):
            if self.validate_keypoint(kpt) and kpt[2] > conf_threshold:
                # Draw outer circle (outline)
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 5, (0, 0, 0), -1)
                # Draw inner circle (colored)
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 4, color, -1)
    
    # Scrub bar interaction (for timeline, not markers)
    def update_scrub_bar(self):
        """Update scrub bar with markers and slider thumb"""
        if not self.current_video:
            return
        
        canvas = self.scrub_canvas
        canvas_width = canvas.winfo_width()
        
        if canvas_width <= 1:
            self.root.after(100, self.update_scrub_bar)
            return
        
        # Draw markers using new system
        self.marker_manager.draw_markers(canvas_width)
        
        # Draw slider thumb
        if self.total_frames > 0:
            scrub_width = canvas_width - 20
            thumb_pos = 10 + (self.current_frame_idx / (self.total_frames - 1)) * scrub_width
            canvas.create_oval(thumb_pos - 6, 12, thumb_pos + 6, 18, 
                             fill='darkblue', outline='black', tags="thumb")
    
    def on_scrub_click(self, event):
        """Handle scrub bar click for timeline seeking (not marker interaction)"""
        if not self.current_video:
            return
        
        # Convert to canvas coordinates
        x, y = self.scrub_canvas.canvasx(event.x), self.scrub_canvas.canvasy(event.y)
        
        # Check if clicking on thumb or timeline
        canvas_width = self.scrub_canvas.winfo_width()
        scrub_width = canvas_width - 20
        
        # Only handle timeline seeking if not clicking on markers
        clicked_item = self.scrub_canvas.find_closest(x, y)[0]
        item_tags = self.scrub_canvas.gettags(clicked_item)
        
        # If clicked on marker, let marker system handle it
        if "marker" in item_tags:
            return
        
        # Otherwise, seek video
        if x >= 10 and x <= 10 + scrub_width:
            position = (x - 10) / scrub_width
            self.current_frame_idx = int(position * (self.total_frames - 1))
            self.show_frame()
            self.update_scrub_bar()
            self.is_scrubbing = True
    
    def on_scrub_drag(self, event):
        """Handle scrub bar drag for timeline seeking"""
        if not self.current_video or not self.is_scrubbing:
            return
        
        canvas_width = self.scrub_canvas.winfo_width()
        scrub_width = canvas_width - 20
        
        x = max(10, min(event.x, 10 + scrub_width))
        position = (x - 10) / scrub_width
        self.current_frame_idx = int(position * (self.total_frames - 1))
        self.show_frame()
        self.update_scrub_bar()
    
    def on_scrub_release(self, event):
        """Handle scrub bar release"""
        self.is_scrubbing = False
    
    # FIXED: Marker management using new system
    def on_marker_change(self):
        """Handle marker changes from timeline clicks"""
        # Set flag to prevent auto-save during switching
        self.auto_save_manager.set_switching_flag(True)
        
        # Cache PREVIOUS pair labels before switching (using previous index, not current!)
        # This is critical because marker manager has already changed selected_pair_index
        # So the UI still has the PREVIOUS pair's data, which we need to cache
        if self.selected_yolo_track_id and self.previous_selected_pair_index:
            print(f"Caching previous pair {self.previous_selected_pair_index} before switching to {self.marker_manager.selected_pair_index}")
            self.cache_current_pair_labels(pair_id=self.previous_selected_pair_index)
        
        # Update previous pair tracking for next switch
        self.previous_selected_pair_index = self.marker_manager.selected_pair_index
        
        # Update listbox and sync selection
        self.update_pair_listbox()
        self.update_pair_details()
        self.setup_loop_playback()
        
        # Clear and load pair-specific data
        if self.marker_manager.selected_pair_index and self.selected_yolo_track_id:
            self.clear_pair_specific_ui()
            self.load_cached_labels_for_pair(self.selected_yolo_track_id, 
                                            self.marker_manager.selected_pair_index)
        elif not self.marker_manager.selected_pair_index:
            # No pair selected - clear UI
            self.clear_pair_specific_ui()
        
        # Re-enable auto-save after switching is complete
        self.auto_save_manager.set_switching_flag(False)
    
    def create_marker_pair(self):
        """Create new marker pair for selected person"""
        if not self.current_video:
            return
        
        if not self.selected_yolo_track_id:
            messagebox.showwarning(
                "No Person Selected",
                "Please select a person from the 'Detected Persons' list first."
            )
            return
        
        # Call MarkerManager to create the marker
        result = self.marker_manager.create_marker_pair(self.current_frame_idx)
        
        # Handle result
        if result == "no_person":
            messagebox.showwarning(
                "No Person Selected",
                "Please select a person from the 'Detected Persons' list first."
            )
        elif result == "overlap":
            messagebox.showwarning(
                "Cannot Create Marker",
                "The new marker would overlap with an existing marker.\n\n"
                "Please move to a different frame or adjust existing markers."
            )
        elif result is None:
            messagebox.showerror(
                "Error",
                "Failed to create marker pair."
            )
        else:
            # Success - result is a MarkerPair object (truthy)
            self.update_scrub_bar()
            self.update_pair_listbox()
            self.update_person_listbox()
            self.trigger_auto_save()
    
    def delete_selected_pair(self):
        """Delete selected marker pair with confirmation"""
        if not self.marker_manager.selected_pair_index:
            messagebox.showwarning("Warning", "No pair selected")
            return
        
        # Get selected pair info
        selected_index = self.marker_manager.selected_pair_index
        selected_pair = self.marker_manager.get_selected_pair()
        
        if not selected_pair:
            return
        
        # Show confirmation dialog
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete Marker Pair {selected_index}?\n\n"
            f"Frames: {selected_pair.start_frame} - {selected_pair.end_frame}\n"
            f"This action cannot be undone."
        )
        
        if not result:
            return
        
        # Delete the marker and get the deleted index
        deleted_index = self.marker_manager.delete_selected_pair()
        
        if deleted_index:
            
            # Update UI
            self.update_scrub_bar()
            self.clear_loop_playback()
            self.update_pair_listbox()
            self.update_person_listbox()
            self.trigger_auto_save()
    
    def update_pair_listbox(self):
        """Update marker pair listbox using new system"""
        self.pair_listbox.delete(0, tk.END)
        
        marker_pairs = self.marker_manager.get_marker_pairs_list()
        for pair in marker_pairs:
            duration = (pair['end'] - pair['start']) / self.fps
            self.pair_listbox.insert(tk.END, 
                f"Pair {pair['id']}: Frame {pair['start']}-{pair['end']} ({duration:.1f}s)")
        
        # Maintain selection
        if self.marker_manager.selected_pair_index:
            for i, pair in enumerate(marker_pairs):
                if pair['id'] == self.marker_manager.selected_pair_index:
                    self.pair_listbox.selection_set(i)
                    break
    
    def on_pair_select(self, event):
        """Handle pair selection with label caching"""
        selection = self.pair_listbox.curselection()
        if selection:
            # Set flag to prevent auto-save during switching
            self.auto_save_manager.set_switching_flag(True)
            
            # Save current pair's labels before switching
            if self.marker_manager.selected_pair_index:
                self.cache_current_pair_labels()
            
            idx = selection[0]
            marker_pairs = self.marker_manager.get_marker_pairs_list()
            
            if idx < len(marker_pairs):
                selected_id = marker_pairs[idx]['id']
                
                # Check if clicking same pair - unselect
                if selected_id == self.marker_manager.selected_pair_index:
                    # self.previous_selected_pair_index = None
                    self.marker_manager.selected_pair_index = None
                    self.pair_listbox.selection_clear(0, tk.END)
                    self.clear_loop_playback()
                    self.clear_pair_specific_ui()
                    self.previous_selected_pair_index = None
                else:
                    # Cache previous pair before switching
                    if self.marker_manager.selected_pair_index and self.selected_yolo_track_id:
                        self.cache_current_pair_labels()  # Cache current before switching
                    
                    # Update tracking BEFORE changing selection
                    self.previous_selected_pair_index = self.marker_manager.selected_pair_index
                    
                    self.marker_manager.selected_pair_index = selected_id
                    self.setup_loop_playback()
                    
                    # Load cached labels for this pair
                    if self.selected_yolo_track_id:
                        self.clear_pair_specific_ui()
                        self.load_cached_labels_for_pair(self.selected_yolo_track_id, selected_id)
                
                self.marker_manager.highlight_selected_pair()
                self.update_pair_details()
            
            # Re-enable auto-save after switching is complete
            self.auto_save_manager.set_switching_flag(False)
            
            # UPDATE: Sync previous tracking to current selection after loading is complete
            # This ensures the next switch (whether from listbox or scrub bar) knows the current pair
            self.previous_selected_pair_index = self.marker_manager.selected_pair_index
    
    def update_pair_details(self):
        """Update pair details display using new system"""
        selected_pair = self.marker_manager.get_selected_pair()
        
        if selected_pair is None:
            self.start_frame_var.set("")
            self.end_frame_var.set("")
            self.duration_label.config(text="Duration: 0.0 seconds")
            return
        
        # Temporarily disable auto-save while updating frame values
        self.auto_save_manager.set_switching_flag(True)
        
        self.start_frame_var.set(str(selected_pair.start_frame))
        self.end_frame_var.set(str(selected_pair.end_frame))
        duration = (selected_pair.end_frame - selected_pair.start_frame) / self.fps
        self.duration_label.config(text=f"Duration: {duration:.1f} seconds")
        
        # Re-enable auto-save
        self.auto_save_manager.set_switching_flag(False)
    
    def on_frame_manual_edit(self, *args):
        """Handle manual frame editing with overlap checking"""
        if self.marker_manager.selected_pair_index is None:
            return
        
        # Skip if we're in the middle of switching markers
        if self.auto_save_manager.switching_markers:
            return
        
        try:
            start_frame = int(self.start_frame_var.get() or 0)
            end_frame = int(self.end_frame_var.get() or 0)
            
            # Validate
            if start_frame > end_frame:
                return  # Invalid range
            
            success = self.marker_manager.update_pair(
                self.marker_manager.selected_pair_index, start_frame, end_frame
            )
            
            if not success:
                # Show warning about overlap
                messagebox.showwarning(
                    "Cannot Update",
                    "The specified frame range overlaps with another marker pair.\n"
                    "Please choose a different range."
                )
                # Revert to original values
                selected_pair = self.marker_manager.get_selected_pair()
                if selected_pair:
                    self.start_frame_var.set(str(selected_pair.start_frame))
                    self.end_frame_var.set(str(selected_pair.end_frame))
                return
            
            # Update successful
            self.update_scrub_bar()
            selected_pair = self.marker_manager.get_selected_pair()
            if selected_pair:
                duration = (selected_pair.end_frame - selected_pair.start_frame) / self.fps
                self.duration_label.config(text=f"Duration: {duration:.1f} seconds")
            
            # Trigger auto-save for manual edits
            self.trigger_auto_save()
        except ValueError:
            pass
    
    # Video playback methods
    def setup_loop_playback(self):
        """Setup looping playback for selected pair"""
        selected_pair = self.marker_manager.get_selected_pair()
        if selected_pair:
            self.loop_start = selected_pair.start_frame
            self.loop_end = selected_pair.end_frame
        else:
            self.clear_loop_playback()
    
    def clear_loop_playback(self):
        """Clear looping playback"""
        self.loop_start = None
        self.loop_end = None
    
    def toggle_play_pause(self):
        """Toggle video playback"""
        if not self.current_video:
            return
        
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="⏸ Pause")
            self.play_video()
        else:
            self.play_button.config(text="▶ Play")
    
    def play_video(self):
        """Play video with looping support"""
        def play_loop():
            while self.is_playing:
                if self.loop_start is not None and self.loop_end is not None:
                    # Loop within selected pair
                    if self.current_frame_idx >= self.loop_end:
                        self.current_frame_idx = self.loop_start
                    else:
                        self.current_frame_idx += 1
                else:
                    # Normal playback
                    if self.current_frame_idx >= self.total_frames - 1:
                        self.is_playing = False
                        self.root.after(0, lambda: self.play_button.config(text="▶ Play"))
                        break
                    else:
                        self.current_frame_idx += 1
                
                self.root.after(0, self.show_frame)
                self.root.after(0, self.update_scrub_bar)
                time.sleep(1.0 / self.fps)
        
        if self.is_playing:
            thread = threading.Thread(target=play_loop)
            thread.daemon = True
            thread.start()
    
    def previous_frame(self, count):
        """Go to previous frame(s)"""
        if self.current_video:
            self.current_frame_idx = max(0, self.current_frame_idx - count)
            self.show_frame()
            self.update_scrub_bar()
    
    def next_frame(self, count):
        """Go to next frame(s)"""
        if self.current_video:
            self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + count)
            self.show_frame()
            self.update_scrub_bar()
    
    def perform_auto_save(self):
        """Perform actual saving operation to JSON"""
        if not self.current_video_data:
            print("No video data to save")
            return
        
        print("=== AUTO-SAVE TRIGGERED ===")
        
        success = self.save_current_video_data()
        
        if success:
            # Update tree status
            video_hash = self.video_hashes.get(self.current_video_path, "")
            if video_hash:
                status = self.check_video_completeness(video_hash)
                self.update_video_tree_status(self.current_video_path, status)
                self.update_status_summary()  # ADD THIS LINE
            
            # Update person listbox colors
            self.update_person_listbox()  # ADD THIS LINE
            
            print("✓ Auto-save completed")
        else:
            print("✗ Auto-save failed")
    
    def get_final_label_value(self):
        """Get final label value"""
        standard_selection = self.checkbutton_groups['label'].get_selection()
        if standard_selection:
            return standard_selection
        elif self.label_other_var.get():
            return self.label_custom_var.get()
        return ""
    
    def get_final_value(self, group_name, other_var, custom_var):
        """Get final value considering 'other' option"""
        if other_var.get():
            return custom_var.get()
        return self.checkbutton_groups[group_name].get_selection() or ""
    
    def get_fall_direction_for_pair(self):
        """Get fall direction for current pair"""
        if self.activity_manager:
            return self.activity_manager.get_combined_fall_direction()
        return ""
    
    def get_general_action_for_pair(self):
        """Get general action for current pair"""
        standard_selection = self.checkbutton_groups['general'].get_selection()
        if standard_selection:
            return standard_selection
        elif self.general_other_var.get():
            return self.general_custom_var.get()
        return ""
    
    def manual_save(self):
        """Manual save operation"""
        if not self.current_video_data:
            messagebox.showwarning("Nothing to Save", "No video data to save")
            return
        
        print("\n=== MANUAL SAVE TRIGGERED ===")
        
        # Cache current labels before saving
        if self.selected_yolo_track_id and self.marker_manager.selected_pair_index:
            self.cache_current_pair_labels()
        
        success = self.save_current_video_data()
        
        if success:
            # Update current video status in tree
            if self.current_video_path and self.current_video_path in self.video_hashes:
                video_hash = self.video_hashes[self.current_video_path]
                status = self.check_video_completeness(video_hash)
                self.update_video_tree_status(self.current_video_path, status)
            
            # Update summary
            self.update_status_summary()
            
            print("✓ Manual save completed\n")
            messagebox.showinfo("Success", "✓ Labels saved successfully!")
        else:
            print("✗ Manual save failed\n")
            messagebox.showerror("Error", "Failed to save labels")
    
    def refresh_labels(self):
        """Refresh labels from JSON"""
        if not self.current_video_path:
            messagebox.showwarning("No Video", "Please load a video first")
            return
        
        video_hash = self.video_hashes.get(self.current_video_path, "")
        if not video_hash:
            return
        
        # Reload from JSON
        self.current_video_data = self.json_manager.load_video_data(video_hash)
        
        if self.current_video_data:
            self.load_video_data_from_json()
            self.update_pair_listbox()
            self.update_pair_details()
            self.update_scrub_bar()
            self.show_frame()
            messagebox.showinfo("Success", "✓ Labels refreshed from JSON!")
        else:
            messagebox.showinfo("No Data", "No saved data found for this video")
    
    def clear_pair_specific_ui(self):
        """Clear both label and activity sections"""
        # Clear label section
        self.checkbutton_groups['label'].set_selection(None)
        self.label_other_var.set(False)
        self.label_custom_var.set("")
        
        # Clear activity sections
        self.checkbutton_groups['front_back'].set_selection(None)
        self.checkbutton_groups['left_right'].set_selection(None)
        self.checkbutton_groups['general'].set_selection(None)
        self.general_other_var.set(False)
        self.general_custom_var.set("")
    
    def build_tree_structure(self, base_folder, video_files):
        """Build tree structure in Treeview"""
        # Create a dictionary to store folder items
        folder_items = {}
        
        # Sort video files by path for better organization
        video_files.sort(key=lambda x: x[1])
        
        for full_path, relative_path in video_files:
            path_parts = relative_path.split(os.sep)
            
            # If file is in root folder
            if len(path_parts) == 1:
                item_id = self.video_tree.insert('', 'end', text=path_parts[0], 
                                                tags=('unprocessed',))
                self.tree_items_to_paths[item_id] = full_path
            else:
                # Create folder structure
                current_parent = ''
                for i, part in enumerate(path_parts[:-1]):
                    folder_path = os.sep.join(path_parts[:i+1])
                    
                    if folder_path not in folder_items:
                        # Create folder node
                        parent_item = folder_items.get(os.sep.join(path_parts[:i]), '')
                        folder_id = self.video_tree.insert(parent_item, 'end', 
                                                        text=part + '/', 
                                                        tags=('folder',))
                        folder_items[folder_path] = folder_id
                    
                    current_parent = folder_items[folder_path]
                
                # Add video file to its folder
                item_id = self.video_tree.insert(current_parent, 'end', 
                                                text=path_parts[-1], 
                                                tags=('unprocessed',))
                self.tree_items_to_paths[item_id] = full_path
    
    def check_video_completeness(self, video_hash):
        """
        Check video labeling completeness
        Returns: 
            0 = No JSON (gray) - YOLO never run
            1 = YOLO run but incomplete (yellow) - either no markers or incomplete labels
            2 = Complete (green) - all markers have complete labels
        """
        if not self.json_manager:
            return 0
        
        # Check if JSON exists
        if not self.json_manager.video_exists(video_hash):
            return 0  # Gray - YOLO never run
        
        # Load data
        data = self.json_manager.load_video_data(video_hash)
        
        if not data:
            return 0  # Error loading - treat as never run
        
        persons = data.get('persons', [])
        
        if not persons:
            return 1  # Yellow - YOLO run but no persons detected
        
        # Check if ANY person has markers
        has_any_markers = False
        all_markers_complete = True
        
        for person in persons:
            action_labels = person.get('action_labels', [])
            
            if action_labels:
                has_any_markers = True
                
                # Check completeness of each marker
                for action in action_labels:
                    label = action.get('label', '')
                    
                    # Must have label
                    if not label or label == '':
                        all_markers_complete = False
                        break
                    
                    # Check if required activity fields are filled
                    if label == 'fall':
                        # For falls, must have fall direction
                        if not action.get('fall_direction'):
                            all_markers_complete = False
                            break
                    elif label == 'normal':
                        # For normal, must have general action
                        if not action.get('general_action'):
                            all_markers_complete = False
                            break
                    # For other labels, either fall_dir or general_action is acceptable
                    else:
                        has_activity = (action.get('fall_direction') or 
                                    action.get('general_action'))
                        if not has_activity:
                            all_markers_complete = False
                            break
            
            if not all_markers_complete:
                break
        
        if not has_any_markers:
            return 1  # Yellow - YOLO run but no markers created yet
        
        if all_markers_complete:
            return 2  # Green - all markers complete
        else:
            return 1  # Yellow - has markers but incomplete
    
    def update_video_tree_status(self, video_path, status):
        """Update tree item status color based on completeness"""
        # Find the tree item for this video
        for item_id, path in self.tree_items_to_paths.items():
            if path == video_path:
                # Clear all status tags first
                current_tags = list(self.video_tree.item(item_id, 'tags'))
                new_tags = [tag for tag in current_tags if tag not in ['complete', 'incomplete', 'unprocessed']]
                
                # Add appropriate status tag
                if status == 0:
                    new_tags.append('unprocessed')
                elif status == 1:
                    new_tags.append('incomplete')
                elif status == 2:
                    new_tags.append('complete')
                
                # Update tree item
                self.video_tree.item(item_id, tags=tuple(new_tags))
                
                # Store in cache for quick access
                if video_path in self.video_hashes:
                    video_hash = self.video_hashes[video_path]
                    self.video_status_cache[video_hash] = status
                
                break
    
    def update_status_summary(self):
        """Update the status summary label"""
        complete = 0
        incomplete = 0
        unprocessed = 0
        
        # Count based on actual tree items and their tags
        for item_id in self.tree_items_to_paths:
            tags = self.video_tree.item(item_id, 'tags')
            if 'complete' in tags:
                complete += 1
            elif 'incomplete' in tags:
                incomplete += 1
            elif 'unprocessed' in tags:
                unprocessed += 1
        
        total = complete + incomplete + unprocessed
        
        self.status_label.config(
            text=f"Complete: {complete} | Incomplete: {incomplete} | Unprocessed: {unprocessed} | Total: {total}"
        )
    
    def select_yolo_model(self):
        """Select YOLO model file"""
        model_path = filedialog.askopenfilename(
            title="Select YOLO Pose Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")]
        )
        if model_path:
            self.yolo_model_path = model_path
            self.model_label.config(text=os.path.basename(model_path))
            
            # Enable run button if video is loaded
            if self.current_video_path:
                self.run_yolo_button.config(state='normal')
            
            messagebox.showinfo("Model Selected", f"Selected: {os.path.basename(model_path)}")

    def run_yolo_on_video(self):
        """Run YOLO tracking on current video"""
        if not YOLO_AVAILABLE:
            messagebox.showerror("Error", "YOLO is not installed. Please install: pip install ultralytics")
            return
        
        if not self.yolo_model_path or not self.current_video_path:
            messagebox.showerror("Error", "Please select both a YOLO model and a video")
            return
        
        # Check if YOLO results already exist
        if self.current_video_data and self.current_video_data.get('persons'):
            response = messagebox.askyesno(
                "⚠️ Warning",
                "Running YOLO will replace ALL existing labels for this video.\n"
                "All person IDs, marker pairs, and action labels will be LOST.\n\n"
                "This action cannot be undone.\n\n"
                "Continue?",
                icon='warning'
            )
            if not response:
                return
        
        # Validate thresholds
        try:
            bbox_conf = float(self.bbox_conf_var.get())
            if bbox_conf < 0 or bbox_conf > 1:
                raise ValueError("Bbox confidence must be between 0 and 1")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid bbox confidence: {e}")
            return
        
        try:
            temporal_filter = int(self.temporal_filter_var.get())
            if temporal_filter < 1:
                raise ValueError("Temporal filter must be at least 1")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid temporal filter: {e}")
            return
        
        # Show progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Running YOLO Pose Estimation")
        progress_window.geometry("450x180")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (450 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (180 // 2)
        progress_window.geometry(f"450x180+{x}+{y}")
        
        ttk.Label(progress_window, text="Processing video with YOLO...", 
                font=('Arial', 10, 'bold')).pack(pady=15)
        
        self.progress_status_label = ttk.Label(progress_window, text="Initializing...", 
                                            font=('Arial', 9))
        self.progress_status_label.pack(pady=5)
        
        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(pady=10, padx=30, fill=tk.X)
        progress_bar.start(10)
        
        self.progress_detail_label = ttk.Label(progress_window, text="", 
                                            font=('Arial', 8), foreground='gray')
        self.progress_detail_label.pack(pady=5)
        
        # Disable buttons during processing
        self.run_yolo_button.config(state='disabled')
        
        # Run in thread
        def process():
            try:
                # Update status
                self.root.after(0, lambda: self.progress_status_label.config(
                    text="Loading YOLO model..."))

                # Load model
                model = YOLO(self.yolo_model_path)

                # Check which device will be used
                device_setting = self.yolo_device.get()
                import torch
                if device_setting == "auto" and torch.cuda.is_available():
                    device_info = f"GPU: {torch.cuda.get_device_name(0)}"
                elif device_setting == "cuda":
                    if torch.cuda.is_available():
                        device_info = f"GPU: {torch.cuda.get_device_name(0)}"
                    else:
                        device_info = "GPU requested but not available - using CPU"
                else:
                    device_info = "CPU"

                self.root.after(0, lambda di=device_info: self.progress_detail_label.config(
                    text=f"Device: {di}"))
                
                self.root.after(0, lambda: self.progress_status_label.config(
                    text="Running pose estimation..."))
                self.root.after(0, lambda: self.progress_detail_label.config(
                    text="This may take several minutes depending on video length"))
                
                # Determine device to use
                device_setting = self.yolo_device.get()
                if device_setting == "auto":
                    # Auto-detect
                    import torch
                    if torch.cuda.is_available():
                        device = '0'  # Use first GPU
                        device_name = torch.cuda.get_device_name(0)
                    else:
                        device = 'cpu'
                        device_name = 'CPU'
                elif device_setting == "cuda":
                    device = '0'  # Use first GPU
                    import torch
                    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU (not available)'
                else:
                    device = 'cpu'
                    device_name = 'CPU'

                # Update progress with device info
                self.root.after(0, lambda: self.progress_detail_label.config(
                    text=f"Using device: {device_name}"))

                # Run tracking with parameters
                results = model.track(
                    self.current_video_path,
                    persist=True,
                    stream=True,
                    conf=bbox_conf,
                    verbose=False,
                    device=device  # Use selected device
                )
                
                self.root.after(0, lambda: self.progress_status_label.config(
                    text="Processing detections..."))
                
                # Process results
                self.yolo_results = self.process_yolo_results(
                    results,
                    temporal_threshold=temporal_filter
                )
                
                # Update UI on main thread
                self.root.after(0, lambda: self.finalize_yolo_processing(progress_window))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror(
                    "YOLO Error", 
                    f"Failed to process video:\n\n{error_msg}\n\n"
                    "Make sure you selected a valid YOLO pose model (.pt file)"
                ))
                self.root.after(0, progress_window.destroy)
                self.root.after(0, lambda: self.run_yolo_button.config(state='normal'))
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
    
    def process_yolo_results(self, results, temporal_threshold):
        """Process YOLO results with temporal filtering"""
        # Structure: {track_id: {frame_num: {bbox, keypoints, confidence}}}
        tracks = {}
        frame_count = 0
        
        for frame_idx, result in enumerate(results):
            frame_count += 1
            
            # Update progress periodically
            if frame_count % 30 == 0:
                self.root.after(0, lambda fc=frame_count: self.progress_detail_label.config(
                    text=f"Processed {fc} frames..."))
            
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            boxes = result.boxes
            
            # Check if keypoints exist
            if not hasattr(result, 'keypoints') or result.keypoints is None or len(result.keypoints) == 0:
                continue
            
            keypoints = result.keypoints
            
            for i in range(len(boxes)):
                # Get tracking ID
                if boxes.id is not None and len(boxes.id) > i:
                    track_id = int(boxes.id[i])
                else:
                    # No tracking ID assigned, skip this detection
                    continue
                
                bbox_conf = float(boxes.conf[i])
                bbox_xyxy = boxes.xyxy[i].cpu().numpy()
                
                # Get keypoints for this person
                if i < len(keypoints.data):
                    kpts = keypoints.data[i].cpu().numpy()  # Shape: (17, 3)
                else:
                    continue
                
                if track_id not in tracks:
                    tracks[track_id] = {}
                
                tracks[track_id][frame_idx] = {
                    'bbox': [float(bbox_xyxy[0]), float(bbox_xyxy[1]),
                            float(bbox_xyxy[2] - bbox_xyxy[0]),  # width
                            float(bbox_xyxy[3] - bbox_xyxy[1])],  # height
                    'bbox_confidence': bbox_conf,
                    'keypoints': kpts.tolist()
                }
        
        self.root.after(0, lambda: self.progress_status_label.config(
            text="Applying temporal filtering..."))
        
        # Temporal filtering: remove tracks with < temporal_threshold consecutive frames
        filtered_tracks = self.apply_temporal_filter(tracks, temporal_threshold)
        
        return filtered_tracks
    
    def verify_gpu_usage(self):
        """Verify GPU is being used - for testing/debugging"""
        try:
            import torch
            
            info = "GPU Information:\n\n"
            
            if torch.cuda.is_available():
                info += f"✓ CUDA Available: Yes\n"
                info += f"✓ GPU Count: {torch.cuda.device_count()}\n"
                info += f"✓ Current Device: {torch.cuda.current_device()}\n"
                info += f"✓ Device Name: {torch.cuda.get_device_name(0)}\n"
                
                # Memory info
                if torch.cuda.is_initialized():
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    info += f"✓ Memory Allocated: {allocated:.2f} GB\n"
                    info += f"✓ Memory Reserved: {reserved:.2f} GB\n"
            else:
                info += "✗ CUDA Not Available\n"
                info += "  Reasons could be:\n"
                info += "  - No NVIDIA GPU installed\n"
                info += "  - GPU drivers not installed\n"
                info += "  - PyTorch CPU-only version installed\n"
            
            messagebox.showinfo("GPU Status", info)
        except Exception as e:
            messagebox.showerror("Error", f"Could not check GPU status:\n{e}")

    def apply_temporal_filter(self, tracks, threshold):
        """Remove tracks that don't appear in enough consecutive frames"""
        filtered = {}
        
        for track_id, frames in tracks.items():
            frame_nums = sorted(frames.keys())
            
            if len(frame_nums) < threshold:
                continue
            
            # Find longest consecutive sequence
            max_consecutive = 1
            current_consecutive = 1
            
            for i in range(1, len(frame_nums)):
                if frame_nums[i] == frame_nums[i-1] + 1:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1
            
            # Only keep tracks with sufficient consecutive frames
            if max_consecutive >= threshold:
                filtered[track_id] = frames
        
        return filtered

    def finalize_yolo_processing(self, progress_window):
        """Finalize YOLO processing and update UI"""
        progress_window.destroy()
        
        # Re-enable button
        self.run_yolo_button.config(state='normal')
        
        if not self.yolo_results:
            messagebox.showwarning(
                "No Detections", 
                "No persons detected in video with current settings.\n\n"
                "Try adjusting:\n"
                "- Lower bbox confidence threshold\n"
                "- Lower temporal filter threshold"
            )
            return
        
        # Create new JSON structure from YOLO results
        self.create_new_video_data_from_yolo()
        
        # Clear any existing person selection
        self.selected_person_id = None
        
        # Clear marker manager for all persons
        if hasattr(self.marker_manager, 'all_persons_markers'):
            self.marker_manager.all_persons_markers = {}
        
        # Update person listbox
        self.update_person_listbox()
        
        # Enable person controls
        self.delete_person_button.config(state='normal')
        self.edit_person_id_button.config(state='normal')
        
        # Refresh video display
        self.show_frame()
        self.update_scrub_bar()
        
        # Save the new data
        self.save_current_video_data()
        
        # Update tree status
        video_hash = self.video_hashes.get(self.current_video_path, "")
        if video_hash:
            status = self.check_video_completeness(video_hash)
            self.update_video_tree_status(self.current_video_path, status)
            self.update_status_summary()
        
        num_persons = len(self.current_video_data['persons'])
        messagebox.showinfo(
            "Success", 
            f"✓ Detected {num_persons} person(s)\n\n"
            f"You can now select a person and create marker pairs to label their actions."
        )
        
    def create_new_video_data_from_yolo(self):
        """Create JSON structure from YOLO results"""
        video_hash = self.video_hashes.get(self.current_video_path, "")
        
        persons = []
        person_id = 1
        
        # Sort track IDs for consistent ordering
        for yolo_track_id in sorted(self.yolo_results.keys()):
            frames_data = self.yolo_results[yolo_track_id]
            
            # Build keypoints_sequence with null for missing frames
            keypoints_sequence = []
            for frame_num in range(self.total_frames):
                if frame_num in frames_data:
                    keypoints_sequence.append({
                        'frame_num': frame_num,
                        'keypoints': frames_data[frame_num]['keypoints'],
                        'bbox': frames_data[frame_num]['bbox'],
                        'bbox_confidence': frames_data[frame_num]['bbox_confidence']
                    })
                else:
                    # Person not detected in this frame
                    keypoints_sequence.append({
                        'frame_num': frame_num,
                        'keypoints': None,
                        'bbox': None,
                        'bbox_confidence': None
                    })
            
            persons.append({
                'person_id': person_id,
                'yolo_track_id': yolo_track_id,
                'main_subject_age_group': '',
                'person_mobility': '',
                'keypoints_sequence': keypoints_sequence,
                'action_labels': []
            })
            
            person_id += 1
        
        # Get video-level metadata from UI (if they were set before)
        camera_angle = self.get_final_value('camera', self.cam_other_var, self.cam_custom_var)
        environment = self.get_final_value('environment', self.env_other_var, self.env_custom_var)
        video_quality = self.get_final_value('quality', self.quality_other_var, self.quality_custom_var)
        
        self.current_video_data = {
            'video_hash': video_hash,
            'video_metadata': {
                'filename': os.path.basename(self.current_video_path),
                'width': self.video_width,
                'height': self.video_height,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'color_mode': self.color_mode,
                'total_duration_seconds': round(self.total_frames / self.fps, 2),
                'camera_angle': camera_angle,
                'environment': environment,
                'video_quality': video_quality,
                'number_of_person': len(persons)
            },
            'persons': persons
        }
        
        # Clear label cache
        self.label_cache = {}

    def save_current_video_data(self):
        """Save current video data to JSON - only update selected person"""
        if not self.current_video_data:
            return False
        
        video_hash = self.current_video_data['video_hash']
        
        # CRITICAL: Cache current pair labels BEFORE saving
        if self.selected_yolo_track_id and self.marker_manager.selected_pair_index:
            self.cache_current_pair_labels()
        
        # Update video metadata from UI
        self.update_video_metadata_from_ui()
        
        # CRITICAL FIX: Only update SELECTED person's data
        # Other persons' data should remain unchanged in JSON
        if self.selected_yolo_track_id:
            self.update_person_data_from_ui(self.selected_yolo_track_id)
            print(f"Updated data for selected YOLO track: {self.selected_yolo_track_id}")
        
        # Save to file
        success = self.json_manager.save_video_data(video_hash, self.current_video_data)
        
        if not success:
            print(f"ERROR: Failed to save video data for hash {video_hash}")
            return False
        
        print(f"✓ Saved data for {len(self.current_video_data['persons'])} person(s)")
        return success

    def update_video_metadata_from_ui(self):
        """Update video metadata from UI fields"""
        if not self.current_video_data or 'video_metadata' not in self.current_video_data:
            return
        
        metadata = self.current_video_data['video_metadata']
        
        # Update from checkbutton groups and custom fields
        metadata['camera_angle'] = self.get_final_value('camera', self.cam_other_var, self.cam_custom_var)
        metadata['environment'] = self.get_final_value('environment', self.env_other_var, self.env_custom_var)
        metadata['video_quality'] = self.get_final_value('quality', self.quality_other_var, self.quality_custom_var)
        
        # Update number of persons
        metadata['number_of_person'] = len(self.current_video_data.get('persons', []))

    def update_person_data_from_ui(self, yolo_track_id):
        """Update person's data from UI including markers - ONLY from cache"""
        person = next((p for p in self.current_video_data['persons'] 
                    if p['yolo_track_id'] == yolo_track_id), None)
        if not person:
            return
        
        # Update person-specific fields ONLY if this is the selected person
        if yolo_track_id == self.selected_yolo_track_id:
            person['main_subject_age_group'] = self.checkbutton_groups['age'].get_selection() or ''
            person['person_mobility'] = self.checkbutton_groups['mobility'].get_selection() or ''
        
        # Get markers from marker manager
        person_markers = self.marker_manager.all_persons_markers.get(yolo_track_id, {})
        marker_pairs = person_markers.get('marker_pairs', {})
        
        # CRITICAL: Only update action_labels if we have cache data for this person
        if yolo_track_id not in self.label_cache:
            print(f"  No cache for YOLO:{yolo_track_id}, keeping existing action_labels")
            return
        
        if not marker_pairs:
            # No markers - keep existing action_labels
            print(f"  No markers for YOLO:{yolo_track_id}, keeping existing action_labels")
            return
        
        # Build action_labels from markers + cache
        action_labels = []
        for pair_id in sorted(marker_pairs.keys()):
            pair = marker_pairs[pair_id]
            
            # Get cached label data for this pair
            label_data = self.get_cached_label_for_pair(yolo_track_id, pair_id)
            
            # CRITICAL: Only add if we have actual cached data
            # If cache returns empty, use existing data or empty
            action_labels.append({
                'start_frame': pair.start_frame,
                'end_frame': pair.end_frame,
                'label': label_data.get('label', ''),
                'fall_direction': label_data.get('fall_direction', None),
                'general_action': label_data.get('general_action', None)
            })
        
        person['action_labels'] = action_labels
        print(f"  Updated {len(action_labels)} action labels for YOLO track {yolo_track_id}")

    def debug_cache_state(self, location):
        """Debug helper to print cache state"""
        print(f"\n{'='*60}")
        print(f"CACHE STATE at {location}")
        print(f"{'='*60}")
        print(f"Selected YOLO: {self.selected_yolo_track_id}")
        print(f"Selected Pair: {self.marker_manager.selected_pair_index}")
        
        for yolo_id, pairs in self.label_cache.items():
            print(f"\nYOLO {yolo_id}:")
            for pair_id, labels in pairs.items():
                print(f"  Pair {pair_id}: {labels}")
        print(f"{'='*60}\n")

    def on_person_select(self, event):
        """Handle person selection from listbox - using yolo_track_id as primary key"""
        # ===== FIX 3: Disable auto-save during person switching =====
        self.auto_save_manager.set_switching_flag(True)
        
        selection = self.person_listbox.curselection()
        if not selection:
            self.auto_save_manager.set_switching_flag(False)
            return
        
        idx = selection[0]
        
        if not self.current_video_data or 'persons' not in self.current_video_data:
            self.auto_save_manager.set_switching_flag(False)
            return
        
        person = self.current_video_data['persons'][idx]
        new_yolo_track_id = person['yolo_track_id']
        new_person_id = person['person_id']
        
        # Check if clicking same YOLO track - unselect
        if new_yolo_track_id == self.selected_yolo_track_id:
            self.selected_person_id = None
            self.selected_yolo_track_id = None
            self.person_listbox.selection_clear(0, tk.END)
            
            # Clear markers view
            self.marker_manager.current_person_id = None
            self.marker_manager.marker_pairs = {}
            self.marker_manager.selected_pair_index = None
            
            # Update UI
            self.update_pair_listbox()
            self.update_pair_details()
            self.update_scrub_bar()
            self.clear_pair_specific_ui()
            self.show_frame()
            
            # Re-enable auto-save before returning
            self.auto_save_manager.set_switching_flag(False)
            return
        
        # Save current person's data if switching
        if self.selected_yolo_track_id:
            self.debug_cache_state("BEFORE switching person")
            
            # CRITICAL: Cache current labels BEFORE switching
            if self.marker_manager.selected_pair_index:
                self.cache_current_pair_labels()
            
            # Save the old person's data
            self.save_current_person_labels()
            
            # Trigger a full save to persist changes
            self.save_current_video_data()
            
            self.debug_cache_state("AFTER saving person")
        
        # Switch to new person
        self.selected_person_id = new_person_id
        self.selected_yolo_track_id = new_yolo_track_id

        # Clear previous pair tracking when switching persons
        self.previous_selected_pair_index = None  # ADD THIS LINE

        # ===== FIX 1: ALWAYS reload markers from JSON to ensure sync with cache =====
        action_labels = person.get('action_labels', [])
        if action_labels:
            # ALWAYS reload markers from JSON, even if they exist in memory
            # This ensures markers and cache IDs are synchronized
            print(f"Reloading markers from JSON for YOLO:{new_yolo_track_id}")
            self.marker_manager.load_person_markers_from_data(new_yolo_track_id, action_labels)
        else:
            # No action labels - initialize empty markers if needed
            if new_yolo_track_id not in self.marker_manager.all_persons_markers:
                self.marker_manager.all_persons_markers[new_yolo_track_id] = {
                    'marker_pairs': {},
                    'selected_pair_index': None
                }
        
        # ===== FIX 2: Always clear cache before reloading to avoid stale data =====
        if action_labels:
            # Always clear and reinitialize cache to avoid stale data
            self.label_cache[new_yolo_track_id] = {}
            
            # Reload cache from action_labels
            for idx, action in enumerate(action_labels, 1):
                self.label_cache[new_yolo_track_id][idx] = {
                    'label': action.get('label', ''),
                    'fall_direction': action.get('fall_direction', None),
                    'general_action': action.get('general_action', None)
                }
            print(f"✓ Reloaded cache for {len(action_labels)} markers from JSON")
        else:
            # No action labels - clear cache for this person
            self.label_cache[new_yolo_track_id] = {}
        
        # Switch marker manager to this YOLO track
        self.marker_manager.set_current_person(new_yolo_track_id)
        
        # Load person-specific data to UI
        self.load_person_specific_labels(new_yolo_track_id)
        
        # Update UI
        self.update_pair_listbox()
        self.update_pair_details()
        self.update_scrub_bar()
        self.show_frame()
        
        # ===== FIX 3: Re-enable auto-save after person switching is complete =====
        self.auto_save_manager.set_switching_flag(False)
        
        print(f"✓ Selected Person {new_person_id} (YOLO Track {new_yolo_track_id})")

    def delete_selected_person(self):
        """Delete selected person with confirmation"""
        if not self.validate_person_operations():  # ADD THIS LINE
            return
        
        if not self.selected_yolo_track_id:
            messagebox.showwarning("No Selection", "Please select a person to delete")
            return
        
        if not self.current_video_data or 'persons' not in self.current_video_data:
            return
        
        # Find the person by yolo_track_id
        person = next((p for p in self.current_video_data['persons'] 
                    if p['yolo_track_id'] == self.selected_yolo_track_id), None)
        
        if not person:
            return
        
        # Count how many action labels this person has
        num_labels = len(person.get('action_labels', []))
        
        # Build confirmation message
        msg = f"Are you sure you want to delete Person {self.selected_person_id}?\n\n"
        msg += f"This will permanently remove:\n"
        msg += f"  • Person {self.selected_person_id}'s pose data\n"
        msg += f"  • {num_labels} action label(s)\n"
        msg += f"  • All associated markers\n\n"
        msg += f"This action CANNOT be undone.\n\n"
        msg += f"Continue?"
        
        result = messagebox.askyesno(
            "⚠️ Confirm Deletion",
            msg,
            icon='warning'
        )
        
        if not result:
            return
        
        # Perform deletion
        deleted_yolo_track_id = self.selected_yolo_track_id
        deleted_person_id = self.selected_person_id

        # Remove person from data
        self.current_video_data['persons'] = [
            p for p in self.current_video_data['persons']
            if p['yolo_track_id'] != deleted_yolo_track_id
        ]
        
        # Update number_of_person in metadata
        self.current_video_data['video_metadata']['number_of_person'] = \
            len(self.current_video_data['persons'])
        
        # Remove from marker manager
        if deleted_yolo_track_id in self.marker_manager.all_persons_markers:
            del self.marker_manager.all_persons_markers[deleted_yolo_track_id]

        # Remove from label cache
        if deleted_yolo_track_id in self.label_cache:
            del self.label_cache[deleted_yolo_track_id]

        # Clear selection
        self.selected_person_id = None
        self.selected_yolo_track_id = None
        self.marker_manager.current_person_id = None
        self.marker_manager.marker_pairs = {}
        self.marker_manager.selected_pair_index = None
        
        # Update UI
        self.person_listbox.selection_clear(0, tk.END)
        self.update_person_listbox()
        
        # If no persons left, disable person controls
        if not self.current_video_data['persons']:
            self.delete_person_button.config(state='disabled')
            self.edit_person_id_button.config(state='disabled')
        
        self.update_pair_listbox()
        self.update_pair_details()
        self.update_scrub_bar()
        self.clear_pair_specific_ui()
        self.show_frame()
        
        # Save changes
        self.save_current_video_data()
        
        # Update tree status
        video_hash = self.video_hashes.get(self.current_video_path, "")
        if video_hash:
            status = self.check_video_completeness(video_hash)
            self.update_video_tree_status(self.current_video_path, status)
            self.update_status_summary()
        
        remaining = len(self.current_video_data['persons'])
        messagebox.showinfo(
            "Person Deleted",
            f"✓ Person {deleted_person_id} has been deleted.\n\n"
            f"Remaining persons: {remaining}"
        )

    def edit_person_id(self):
        """Allow user to edit person ID"""
        if not self.validate_person_operations():  # ADD THIS LINE
            return
    
        if not self.selected_person_id:
            messagebox.showwarning("No Selection", "Please select a person to edit")
            return
        
        if not self.current_video_data or 'persons' not in self.current_video_data:
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Edit Person ID")
        dialog.geometry("350x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (350 // 2)
        y = (dialog.winfo_screenheight() // 2) - (200 // 2)
        dialog.geometry(f"350x200+{x}+{y}")
        
        # Header
        header_frame = ttk.Frame(dialog)
        header_frame.pack(fill=tk.X, padx=20, pady=15)
        
        ttk.Label(header_frame, 
                text=f"Edit Person ID", 
                font=('Arial', 12, 'bold')).pack()
        
        # Current ID display
        info_frame = ttk.Frame(dialog)
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(info_frame, 
                text=f"Current ID:", 
                font=('Arial', 10)).grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(info_frame, 
                text=f"{self.selected_person_id}", 
                font=('Arial', 10, 'bold'),
                foreground='blue').grid(row=0, column=1, sticky=tk.W, padx=10)
        
        # Get current YOLO track ID
        person = next((p for p in self.current_video_data['persons'] 
                    if p['person_id'] == self.selected_person_id), None)
        yolo_id = person.get('yolo_track_id', '?') if person else '?'
        
        ttk.Label(info_frame, 
                text=f"YOLO Track ID:", 
                font=('Arial', 9)).grid(row=1, column=0, sticky=tk.W)
        ttk.Label(info_frame, 
                text=f"{yolo_id}", 
                font=('Arial', 9),
                foreground='gray').grid(row=1, column=1, sticky=tk.W, padx=10)
        
        # New ID input
        input_frame = ttk.Frame(dialog)
        input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(input_frame, 
                text="New ID:", 
                font=('Arial', 10)).pack(side=tk.LEFT)
        
        new_id_var = tk.StringVar(value=str(self.selected_person_id))
        entry = ttk.Entry(input_frame, textvariable=new_id_var, width=10, font=('Arial', 10))
        entry.pack(side=tk.LEFT, padx=10)
        entry.select_range(0, tk.END)
        entry.focus_set()
        
        # Error label (hidden initially)
        error_label = ttk.Label(dialog, text="", foreground="red", font=('Arial', 9))
        error_label.pack(pady=5)
        
        def validate_and_apply():
            """Validate and apply the ID change"""
            try:
                new_id = int(new_id_var.get())
                
                # Validate: must be positive
                if new_id <= 0:
                    error_label.config(text="⚠ ID must be a positive number")
                    return
                
                # Validate: cannot be same as current
                if new_id == self.selected_person_id:
                    error_label.config(text="⚠ New ID is the same as current ID")
                    return
                
                # Validate: ID must not already exist
                # existing_ids = [p['person_id'] for p in self.current_video_data['persons']]
                # if new_id in existing_ids:
                #     error_label.config(text=f"⚠ Person ID {new_id} already exists")
                #     return
                
                # Check if this will create duplicate IDs (which is OK)
                existing_ids = [p['person_id'] for p in self.current_video_data['persons'] 
                                if p['yolo_track_id'] != self.selected_yolo_track_id]
                is_duplicate = new_id in existing_ids

                # Build confirmation message
                confirm_msg = f"Change Person ID from {self.selected_person_id} to {new_id}?\n\n"
                if is_duplicate:
                    confirm_msg += "⚠ Note: This ID already exists for other YOLO track(s).\n"
                    confirm_msg += "This will merge them as the same logical person.\n\n"
                confirm_msg += "This will update:\n"
                confirm_msg += f"  • Person ID in video data\n"
                confirm_msg += f"  • All associated markers\n"
                confirm_msg += f"  • All action labels\n"
                
                if not messagebox.askyesno("Confirm Change", confirm_msg, parent=dialog):
                    return
                
                # Perform the ID change
                old_id = self.selected_person_id
                self.change_person_id(self.selected_yolo_track_id, new_id)
                
                dialog.destroy()
                
                messagebox.showinfo(
                    "Success",
                    f"✓ Person ID changed from {old_id} to {self.selected_yolo_track_id}"
                )
                
            except ValueError:
                error_label.config(text="⚠ Please enter a valid number")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=15)
        
        ttk.Button(button_frame, 
                text="Apply", 
                command=validate_and_apply,
                width=12).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, 
                text="Cancel", 
                command=dialog.destroy,
                width=12).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key to apply
        entry.bind('<Return>', lambda e: validate_and_apply())
        entry.bind('<Escape>', lambda e: dialog.destroy())

    def validate_person_operations(self):
        """Validate that person operations are allowed"""
        if not self.current_video_data:
            messagebox.showwarning(
                "No Video Data",
                "Please run YOLO on the current video first."
            )
            return False
        
        if not self.current_video_data.get('persons'):
            messagebox.showwarning(
                "No Persons",
                "No persons detected in this video.\n\n"
                "Please run YOLO to detect persons first."
            )
            return False
        
        return True

    def change_person_id(self, yolo_track_id, new_id):
        """Change person ID for specific YOLO track"""
        if not self.current_video_data:
            return False
        
        # Find person by yolo_track_id (PRIMARY KEY - never changes)
        person = next((p for p in self.current_video_data['persons'] 
                    if p['yolo_track_id'] == yolo_track_id), None)
        if not person:
            return False
        
        old_id = person['person_id']
        
        # ONLY update the person_id field (just a label now)
        person['person_id'] = new_id
        
        # Markers and cache stay under yolo_track_id (which never changes!)
        # NO need to move anything in marker_manager
        # NO need to move anything in label_cache
        
        # Update current selection display
        if self.selected_yolo_track_id == yolo_track_id:
            self.selected_person_id = new_id
            # marker_manager.current_person_id should also be yolo_track_id
        
        # Update UI
        self.update_person_listbox()
        
        # Re-select the person in listbox by yolo_track_id
        for idx, p in enumerate(self.current_video_data['persons']):
            if p['yolo_track_id'] == yolo_track_id:
                self.person_listbox.selection_set(idx)
                break
        
        # Refresh display
        self.show_frame()
        
        # Save changes
        self.save_current_video_data()
        
        # Update tree status
        video_hash = self.video_hashes.get(self.current_video_path, "")
        if video_hash:
            status = self.check_video_completeness(video_hash)
            self.update_video_tree_status(self.current_video_path, status)
        
        return True

    def on_skeleton_toggle(self):
        """Handle skeleton visualization toggle"""
        # Refresh the current frame to show/hide skeleton
        if self.current_video_path and self.current_video_data:
            self.show_frame()
    
    def save_current_person_labels(self):
        """Save current person's labels to cache and update person data"""
        if not self.selected_yolo_track_id or not self.current_video_data:
            return
        
        print(f"=== SAVING PERSON LABELS: YOLO:{self.selected_yolo_track_id} ===")
        
        # CRITICAL: Cache current pair labels first
        if self.marker_manager.selected_pair_index:
            self.cache_current_pair_labels()
        
        # Get current YOLO track's markers
        person_markers = self.marker_manager.all_persons_markers.get(self.selected_yolo_track_id)
        if not person_markers:
            print(f"  No markers for YOLO:{self.selected_yolo_track_id}")
            return
        
        # Update person data will read from cache
        self.update_person_data_from_ui(self.selected_yolo_track_id)
        
        print(f"✓ Saved labels for YOLO track {self.selected_yolo_track_id}")

    def load_person_specific_labels(self, yolo_track_id):
        """Load person-specific labels to UI"""
        if not self.current_video_data:
            return
        
        # Find person in data
        person = next((p for p in self.current_video_data['persons'] if p['yolo_track_id'] == yolo_track_id), None)
        if not person:
            return
        
        # Load person-specific fields
        age_group = person.get('main_subject_age_group', '')
        if age_group and age_group in ['young', 'middle_aged', 'elderly']:
            self.checkbutton_groups['age'].set_selection(age_group)
        else:
            self.checkbutton_groups['age'].set_selection(None)
        
        mobility = person.get('person_mobility', '')
        if mobility and mobility in ['normal', 'assisted_walking', 'wheelchair']:
            self.checkbutton_groups['mobility'].set_selection(mobility)
        else:
            self.checkbutton_groups['mobility'].set_selection(None)
        
        # Clear pair-specific UI initially
        self.clear_pair_specific_ui()
    
    def clear_pair_specific_ui(self):
        """Clear both label and activity sections WITHOUT triggering callbacks"""
        # Temporarily disable auto-save during clearing
        was_switching = self.auto_save_manager.switching_markers
        self.auto_save_manager.set_switching_flag(True)
        
        # Clear label section
        self.checkbutton_groups['label'].set_selection(None)
        self.label_other_var.set(False)
        self.label_custom_var.set("")
        
        # Clear activity sections
        self.checkbutton_groups['front_back'].set_selection(None)
        self.checkbutton_groups['left_right'].set_selection(None)
        self.checkbutton_groups['general'].set_selection(None)
        self.general_other_var.set(False)
        self.general_custom_var.set("")
        
        # Restore switching flag
        self.auto_save_manager.set_switching_flag(was_switching)
    
    def cache_current_pair_labels(self, pair_id=None):
        """Cache labels for specified pair (or currently selected pair) - using yolo_track_id"""
        if not self.selected_yolo_track_id:
            print("WARNING: Cannot cache - no person selected")
            return
        
        # Use provided pair_id or fall back to current selection
        if pair_id is None:
            pair_id = self.marker_manager.selected_pair_index
        
        if not pair_id:
            print("WARNING: Cannot cache - no pair specified")
            return
        
        # Initialize cache for YOLO track if needed
        if self.selected_yolo_track_id not in self.label_cache:
            self.label_cache[self.selected_yolo_track_id] = {}
        
        # Get current label values
        label = self.get_final_label_value()
        fall_direction = self.get_fall_direction_for_pair()
        general_action = self.get_general_action_for_pair()
        
        # Cache them
        self.label_cache[self.selected_yolo_track_id][pair_id] = {
            'label': label,
            'fall_direction': fall_direction if fall_direction else None,
            'general_action': general_action if general_action else None
        }
        
        print(f"✓ Cached: YOLO:{self.selected_yolo_track_id}, Pair:{pair_id}, "
            f"Label:'{label}', FallDir:'{fall_direction}', General:'{general_action}'")

    def get_cached_label_for_pair(self, yolo_track_id, pair_id):
        """Get cached label for specific pair - using yolo_track_id"""
        if yolo_track_id in self.label_cache and pair_id in self.label_cache[yolo_track_id]:
            return self.label_cache[yolo_track_id][pair_id]
        
        return {
            'label': '',
            'fall_direction': None,
            'general_action': None
        }

    def load_cached_labels_for_pair(self, yolo_track_id, pair_id):
        """Load cached labels to UI - using yolo_track_id"""
        print(f"Loading cached labels: YOLO:{yolo_track_id}, Pair:{pair_id}")
        
        cached = self.get_cached_label_for_pair(yolo_track_id, pair_id)
        
        print(f"  Cached data: {cached}")
        
        # Load label
        label = cached.get('label', '')
        if label:
            if label in ['fall', 'normal']:
                self.checkbutton_groups['label'].set_selection(label)
                self.label_other_var.set(False)
                self.label_custom_var.set("")
            else:
                self.checkbutton_groups['label'].set_selection(None)
                self.label_other_var.set(True)
                self.label_custom_var.set(label)
        else:
            # Clear if no label
            self.checkbutton_groups['label'].set_selection(None)
            self.label_other_var.set(False)
            self.label_custom_var.set("")
        
        # Load fall direction
        fall_dir = cached.get('fall_direction', '')
        if fall_dir and fall_dir.startswith('fall_'):
            directions = fall_dir.replace('fall_', '').split('_')
            for direction in directions:
                if direction in ['front', 'back']:
                    self.checkbutton_groups['front_back'].set_selection(direction)
                elif direction in ['left', 'right']:
                    self.checkbutton_groups['left_right'].set_selection(direction)
        else:
            # Clear if no fall direction
            self.checkbutton_groups['front_back'].set_selection(None)
            self.checkbutton_groups['left_right'].set_selection(None)
        
        # Load general action
        general = cached.get('general_action', '')
        if general:
            if general in ['sitting', 'standing', 'walking', 'hopping', 'bending', 'lying', 'crawling']:
                self.checkbutton_groups['general'].set_selection(general)
                self.general_other_var.set(False)
                self.general_custom_var.set("")
            else:
                self.checkbutton_groups['general'].set_selection(None)
                self.general_other_var.set(True)
                self.general_custom_var.set(general)
        else:
            # Clear if no general action
            self.checkbutton_groups['general'].set_selection(None)
            self.general_other_var.set(False)
            self.general_custom_var.set("")
        
        # CRITICAL: Trigger label change to update activity sections
        self.on_label_change()
        
        print(f"  Loaded to UI: label='{label}', fall_dir='{fall_dir}', general='{general}'")
    
    def show_person_context_menu(self, event):
        """Show context menu for person listbox"""
        # Select the item under cursor
        index = self.person_listbox.nearest(event.y)
        self.person_listbox.selection_clear(0, tk.END)
        self.person_listbox.selection_set(index)
        self.person_listbox.activate(index)
        
        # Trigger person selection
        self.person_listbox.event_generate('<<ListboxSelect>>')
        
        # Create context menu
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="Edit Person ID (F2)", 
                                command=self.edit_person_id)
        context_menu.add_command(label="Delete Person (Del)", 
                                command=self.delete_selected_person)
        context_menu.add_separator()
        context_menu.add_command(label="Cancel")
        
        # Show menu at cursor
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def on_closing(self):
        """Handle window closing with save"""
        print("\n=== APPLICATION CLOSING ===")
        
        # Cancel any pending auto-save
        self.auto_save_manager.cancel_pending()
        
        # Save current work if video is loaded
        if self.current_video_data:
            print("Saving current work before closing...")
            if self.selected_yolo_track_id and self.marker_manager.selected_pair_index:
                self.cache_current_pair_labels()
            if self.selected_yolo_track_id:
                self.save_current_person_labels()
            self.save_current_video_data()
            print("✓ Work saved")
        
        # Release video
        if self.current_video:
            self.current_video.release()
        
        print("✓ Application closed cleanly\n")
        
        # Destroy window
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

if __name__ == "__main__":    
    app = FallDetectionLabeler()
    app.run()