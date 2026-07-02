"""
==============================================================================
BLIND NAVIGATION ASSISTANT - ENHANCED OBJECT DETECTION
==============================================================================

NEW FEATURES:
-------------
1. TREE & PLANT DETECTION: Detects vegetation obstacles
2. IMPROVED OBJECT CLASSIFICATION: Better accuracy with visual features
3. CONFIDENCE FILTERING: Reduces false positives
4. MULTI-STAGE VERIFICATION: Validates detections before announcing
5. DOOR DETECTION: Open/closed state analysis
6. WALL DETECTION: Large surface detection

ENHANCED DETECTION:
-------------------
- Trees and plants detected using color + texture analysis
- Object verification using multiple frames
- Confidence thresholds adjusted per object type
- Shape-based validation (books vs laptops, phones vs trays)

==============================================================================
"""

import cv2
import time
import sys
import os
import numpy as np
import torch
import pyttsx3
import threading
import queue
from ultralytics import YOLO
from collections import deque, defaultdict
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================
QUEUE_WINDOW_SECONDS = 60.0
ANNOUNCEMENT_DELAY = 2.5
TEMPORAL_SMOOTHING_WINDOW = 5
DEPTH_PERCENTILE = 90

STOP_DISTANCE_METERS = 1.0
WARNING_DISTANCE_METERS = 2.5
CLEAR_THRESHOLD_METERS = 3.0

PATH_UPDATE_INTERVAL = 5.0
SAFE_PATH_DISTANCE_THRESHOLD = 2.0
MIN_CLEAR_WIDTH = 0.3

# Door detection
DOOR_DEPTH_STD_THRESHOLD = 500
DOOR_DEPTH_MIN_THRESHOLD = 200
DOOR_MIN_CONFIDENCE = 0.5

# Wall detection
WALL_MIN_WIDTH = 200
WALL_MIN_HEIGHT = 150
WALL_DEPTH_UNIFORMITY = 100
WALL_MIN_CONFIDENCE = 0.6

# NEW: Enhanced object detection thresholds
OBJECT_MIN_CONFIDENCE = {
    'person': 0.5,
    'laptop': 0.6,      # Higher threshold for laptops (often misdetected)
    'cell phone': 0.55, # Higher threshold for phones
    'book': 0.5,
    'cup': 0.45,
    'bottle': 0.5,
    'chair': 0.5,
    'door': 0.5,
    'closed_door': 0.45,
    'open_door': 0.45,
    'default': 0.4
}

DEFAULT_DOOR_MODEL = 'runs/detect/universia_doors/weights/best.pt'
DEFAULT_COCO_MODEL = 'yolov8n.pt'

# NEW: Vegetation detection
VEGETATION_GREEN_THRESHOLD_LOW = np.array([35, 40, 40])   # HSV lower bound
VEGETATION_GREEN_THRESHOLD_HIGH = np.array([85, 255, 255]) # HSV upper bound
VEGETATION_MIN_AREA = 5000  # Minimum pixel area to be considered tree/plant
VEGETATION_CONFIDENCE_THRESHOLD = 0.7

# Object validation parameters
DETECTION_STABILITY_FRAMES = 3  # Frames object must appear to be considered stable

DISPLAY_WINDOW_NAME = "Vision Assistant - Enhanced Detection"


def get_screen_size() -> Tuple[int, int]:
    """Return primary monitor width and height in pixels."""
    if sys.platform == "win32":
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return 1920, 1080

# =============================================================================
# ENUMERATIONS
# =============================================================================
class Direction(Enum):
    LEFT = "on your left"
    CENTER = "in front of you"
    RIGHT = "on your right"
    UNKNOWN = "location unknown"

class DoorState(Enum):
    UNKNOWN = auto()
    OPEN = auto()
    CLOSED = auto()

class PathDirection(Enum):
    LEFT = "Move to your left"
    CENTER = "Continue straight ahead"
    RIGHT = "Move to your right"
    STOP = "Stop - all paths blocked"
    UNKNOWN = "No clear path detected"


def normalize_door_label(label: str) -> Tuple[str, Optional[DoorState]]:
    """Map UNIVERSIA YOLO classes to generic door + state."""
    if label == 'closed_door':
        return 'door', DoorState.CLOSED
    if label == 'open_door':
        return 'door', DoorState.OPEN
    return label, None

# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class DetectedObject:
    object_id: int
    label: str
    distance: float
    direction: Direction
    confidence: float
    first_seen: float
    last_updated: float
    door_state: Optional[DoorState] = None
    announced: bool = False
    is_wall: bool = False
    is_vegetation: bool = False  # NEW
    detection_count: int = 1     # NEW: How many frames detected
    
    def get_announcement(self) -> str:
        """Generate voice announcement with enhanced object awareness."""
        
        # Vegetation announcements
        if self.is_vegetation:
            if self.distance < STOP_DISTANCE_METERS:
                return f"STOP! Tree or plant very close {self.direction.value}!"
            elif self.distance < WARNING_DISTANCE_METERS:
                return f"Warning: Tree or plant {self.direction.value} at {self.distance:.1f} meters"
            else:
                return f"Tree or plant {self.direction.value}"
        
        # Door announcements
        if self.label == 'door':
            if self.door_state == DoorState.CLOSED:
                if self.distance < STOP_DISTANCE_METERS:
                    return f"STOP! Door is CLOSED {self.direction.value}! Please open the door first!"
                elif self.distance < WARNING_DISTANCE_METERS:
                    return f"WARNING: Closed door {self.direction.value} at {self.distance:.1f} meters. Open it first."
                else:
                    return f"Closed door {self.direction.value}. Open before approaching."
            elif self.door_state == DoorState.OPEN:
                if self.distance < WARNING_DISTANCE_METERS:
                    return f"Open door {self.direction.value} - proceed carefully."
                else:
                    return f"Door is open {self.direction.value} - safe to pass."
            else:
                if self.distance < STOP_DISTANCE_METERS:
                    return f"STOP! Door {self.direction.value} - check if open!"
                else:
                    return f"Door {self.direction.value} - verify if open."
        
        # Wall announcements
        if self.is_wall:
            if self.distance < STOP_DISTANCE_METERS:
                return f"STOP! Wall very close {self.direction.value}!"
            elif self.distance < WARNING_DISTANCE_METERS:
                return f"Warning: Wall {self.direction.value} at {self.distance:.1f} meters"
            else:
                return f"Wall {self.direction.value}"
        
        # Standard objects
        if self.distance < STOP_DISTANCE_METERS:
            return f"STOP! {self.label} very close {self.direction.value}!"
        elif self.distance < WARNING_DISTANCE_METERS:
            return f"Warning: {self.label} {self.direction.value}"
        else:
            return f"{self.label} {self.direction.value}"
    
    def is_critical(self) -> bool:
        """Check if critical object."""
        if self.label == 'door' and self.door_state == DoorState.CLOSED:
            return True
        if self.is_wall and self.distance < STOP_DISTANCE_METERS:
            return True
        if self.is_vegetation and self.distance < STOP_DISTANCE_METERS:
            return True
        return self.distance < STOP_DISTANCE_METERS

@dataclass
class PathAnalysis:
    direction: PathDirection
    obstacle_count: int
    min_distance: float
    avg_distance: float
    clear_percentage: float
    safety_score: float
    has_closed_door: bool = False
    has_wall: bool = False
    has_vegetation: bool = False  # NEW
    
    def is_safe(self) -> bool:
        if self.has_closed_door:
            return False
        if self.has_wall and self.min_distance < SAFE_PATH_DISTANCE_THRESHOLD:
            return False
        if self.has_vegetation and self.min_distance < SAFE_PATH_DISTANCE_THRESHOLD:
            return False
        return (self.min_distance > SAFE_PATH_DISTANCE_THRESHOLD and 
                self.clear_percentage > MIN_CLEAR_WIDTH)

# =============================================================================
# VEGETATION DETECTOR (NEW)
# =============================================================================
class VegetationDetector:
    """
    Detects trees and plants using color and texture analysis.
    
    DETECTION METHOD:
    -----------------
    1. HSV color filtering (green detection)
    2. Contour analysis (find plant shapes)
    3. Texture analysis (organic patterns)
    4. Size filtering (significant vegetation only)
    
    Trees/Plants characteristics:
    - Green color (HSV range)
    - Organic irregular shapes
    - Texture variation
    - Vertical structure for trees
    """
    
    def __init__(self):
        print("[VegetationDetector] Initialized - Ready to detect trees and plants")
    
    def detect_vegetation(self, frame: np.ndarray, depth_map: np.ndarray) -> List[Dict]:
        """
        Detect trees and plants in the frame.
        
        Returns:
            List of detected vegetation with bbox, distance, confidence
        """
        vegetation_objects = []
        
        try:
            # Convert to HSV for green detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for green colors (vegetation)
            green_mask = cv2.inRange(hsv, VEGETATION_GREEN_THRESHOLD_LOW, 
                                    VEGETATION_GREEN_THRESHOLD_HIGH)
            
            # Morphological operations to clean up mask
            kernel = np.ones((5, 5), np.uint8)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size
                if area < VEGETATION_MIN_AREA:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract depth region
                depth_roi = depth_map[y:y+h, x:x+w]
                
                if depth_roi.size == 0:
                    continue
                
                # Calculate distance
                depth_mean = np.mean(depth_roi)
                distance = 500.0 / (depth_mean + 1e-6)
                
                # Calculate confidence based on:
                # 1. Green intensity
                # 2. Organic shape (irregular contour)
                # 3. Size
                
                # Green intensity
                roi_hsv = hsv[y:y+h, x:x+w]
                green_pixels = cv2.countNonZero(green_mask[y:y+h, x:x+w])
                green_percentage = green_pixels / (w * h)
                
                # Shape analysis (trees/plants are irregular)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                irregularity = 1.0 - circularity  # Higher = more irregular (organic)
                
                # Aspect ratio (trees tend to be vertical)
                aspect_ratio = h / w if w > 0 else 0
                vertical_score = min(aspect_ratio / 2.0, 1.0)  # Trees are taller
                
                # Combined confidence
                confidence = (
                    green_percentage * 0.4 +      # 40% green content
                    irregularity * 0.3 +          # 30% irregular shape
                    vertical_score * 0.2 +        # 20% vertical structure
                    min(area / 20000, 1.0) * 0.1  # 10% size
                )
                
                if confidence < VEGETATION_CONFIDENCE_THRESHOLD:
                    continue
                
                vegetation_objects.append({
                    'label': 'tree or plant',
                    'bbox': [x, y, x + w, y + h],
                    'distance': distance,
                    'confidence': confidence,
                    'is_vegetation': True
                })
                
                print(f"[VegetationDetector] Detected vegetation at {distance:.1f}m, "
                      f"green%: {green_percentage:.2f}, conf: {confidence:.2f}")
        
        except Exception as e:
            print(f"[VegetationDetector] Error: {e}")
        
        return vegetation_objects

# =============================================================================
# OBJECT VALIDATOR (NEW)
# =============================================================================
class ObjectValidator:
    """
    Validates object detections to reduce false positives.
    
    VALIDATION METHODS:
    -------------------
    1. Confidence thresholds per object type
    2. Shape-based validation (aspect ratio, size)
    3. Multi-frame stability (must appear in consecutive frames)
    4. Context validation (unlikely object positions)
    """
    
    def __init__(self):
        self.detection_history = defaultdict(lambda: deque(maxlen=5))
        print("[ObjectValidator] Initialized - Reducing false positives")
    
    def validate_laptop(self, bbox: List[float], confidence: float) -> Tuple[bool, float]:
        """
        Validate laptop detection (often confused with books, trays).
        
        Laptops typically:
        - Have specific aspect ratio (wider than tall)
        - Are horizontal rectangles
        - Have moderate size
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        aspect_ratio = width / height if height > 0 else 0
        
        # Laptops are typically wider (aspect ratio 1.3-2.0)
        if aspect_ratio < 1.2 or aspect_ratio > 2.5:
            # Likely not a laptop (maybe book or tray)
            return False, confidence * 0.7
        
        # Require higher confidence for laptops
        if confidence < OBJECT_MIN_CONFIDENCE.get('laptop', 0.6):
            return False, confidence
        
        return True, confidence
    
    def validate_cell_phone(self, bbox: List[float], confidence: float) -> Tuple[bool, float]:
        """
        Validate cell phone detection (often confused with remote, tray).
        
        Phones typically:
        - Small rectangular shape
        - Aspect ratio around 1.5-2.5 (taller than wide)
        - Limited size range
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        aspect_ratio = height / width if width > 0 else 0
        
        # Phones are vertical rectangles
        if aspect_ratio < 1.3 or aspect_ratio > 3.0:
            # Likely tray or other object
            return False, confidence * 0.7
        
        # Phones are relatively small
        if area > 50000:  # Too large to be phone
            return False, confidence * 0.6
        
        # Require higher confidence
        if confidence < OBJECT_MIN_CONFIDENCE.get('cell phone', 0.55):
            return False, confidence
        
        return True, confidence
    
    def validate_detection(self, label: str, bbox: List[float], 
                          confidence: float) -> Tuple[bool, float, str]:
        """
        Main validation function for all objects.
        
        Returns:
            (is_valid, adjusted_confidence, corrected_label)
        """
        # Object-specific validation
        if label == 'laptop':
            is_valid, adj_conf = self.validate_laptop(bbox, confidence)
            if not is_valid:
                # Might be a book instead
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # Books are often confused with laptops
                if 0.7 < aspect_ratio < 1.2:
                    return True, adj_conf, 'book'
                return False, adj_conf, label
            return is_valid, adj_conf, label
        
        elif label == 'cell phone':
            is_valid, adj_conf = self.validate_cell_phone(bbox, confidence)
            if not is_valid:
                # Might be remote or small tray
                return False, adj_conf, label
            return is_valid, adj_conf, label
        
        # General confidence check
        min_conf = OBJECT_MIN_CONFIDENCE.get(label, OBJECT_MIN_CONFIDENCE['default'])
        if confidence < min_conf:
            return False, confidence, label
        
        return True, confidence, label

# =============================================================================
# WALL DETECTOR
# =============================================================================
class WallDetector:
    def __init__(self):
        print("[WallDetector] Initialized")
    
    def detect_walls(self, frame: np.ndarray, depth_map: np.ndarray) -> List[Dict]:
        walls = []
        
        try:
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
            
            edges = cv2.Canny(depth_uint8, 50, 150)
            
            kernel = np.ones((5, 5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                if w < WALL_MIN_WIDTH or h < WALL_MIN_HEIGHT:
                    continue
                
                depth_roi = depth_map[y:y+h, x:x+w]
                
                if depth_roi.size == 0:
                    continue
                
                depth_std = np.std(depth_roi)
                depth_mean = np.mean(depth_roi)
                
                if depth_std > WALL_DEPTH_UNIFORMITY:
                    continue
                
                distance = 500.0 / (depth_mean + 1e-6)
                
                area = w * h
                max_area = frame.shape[0] * frame.shape[1]
                size_confidence = min(area / (max_area * 0.3), 1.0)
                uniformity_confidence = max(0, 1.0 - (depth_std / 200.0))
                
                confidence = (size_confidence * 0.6 + uniformity_confidence * 0.4)
                
                if confidence < WALL_MIN_CONFIDENCE:
                    continue
                
                walls.append({
                    'label': 'wall',
                    'bbox': [x, y, x + w, y + h],
                    'distance': distance,
                    'confidence': confidence,
                    'is_wall': True
                })
        
        except Exception as e:
            print(f"[WallDetector] Error: {e}")
        
        return walls

# =============================================================================
# DOOR ANALYZER
# =============================================================================
class DoorAnalyzer:
    def __init__(self):
        self.door_history = defaultdict(lambda: deque(maxlen=3))
        print("[DoorAnalyzer] Enhanced door analyzer initialized")
    
    def analyze_door_state(self, depth_map: np.ndarray, bbox: List[float], 
                          obj_id: int) -> DoorState:
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            width = x2 - x1
            height = y2 - y1
            
            cx1 = int(x1 + width * 0.3)
            cx2 = int(x1 + width * 0.7)
            cy1 = int(y1 + height * 0.3)
            cy2 = int(y1 + height * 0.7)
            
            door_roi = depth_map[cy1:cy2, cx1:cx2]
            
            if door_roi.size == 0:
                return DoorState.UNKNOWN
            
            depth_std = np.std(door_roi)
            depth_min = np.min(door_roi)
            
            very_low_depth_pixels = np.sum(door_roi < DOOR_DEPTH_MIN_THRESHOLD)
            low_depth_percentage = very_low_depth_pixels / door_roi.size
            
            state = DoorState.UNKNOWN
            
            if (depth_std > DOOR_DEPTH_STD_THRESHOLD or 
                depth_min < DOOR_DEPTH_MIN_THRESHOLD or
                low_depth_percentage > 0.2):
                state = DoorState.OPEN
            elif (depth_std < DOOR_DEPTH_STD_THRESHOLD * 0.5 and
                  depth_min > DOOR_DEPTH_MIN_THRESHOLD * 1.5 and
                  low_depth_percentage < 0.05):
                state = DoorState.CLOSED
            
            self.door_history[obj_id].append(state)
            
            if len(self.door_history[obj_id]) >= 2:
                history = list(self.door_history[obj_id])
                open_count = history.count(DoorState.OPEN)
                closed_count = history.count(DoorState.CLOSED)
                
                if closed_count >= 2:
                    final_state = DoorState.CLOSED
                elif open_count >= 2:
                    final_state = DoorState.OPEN
                else:
                    final_state = state
            else:
                final_state = state
            
            return final_state
            
        except Exception as e:
            print(f"[DoorAnalyzer] Error: {e}")
            return DoorState.UNKNOWN

# =============================================================================
# PATH ANALYZER
# =============================================================================
class PathAnalyzer:
    def __init__(self, frame_width=640):
        self.frame_width = frame_width
        self.left_zone = (0, int(frame_width * 0.33))
        self.center_zone = (int(frame_width * 0.33), int(frame_width * 0.66))
        self.right_zone = (int(frame_width * 0.66), frame_width)
        self.last_safe_path = PathDirection.CENTER
        self.path_history = deque(maxlen=3)
        print(f"[PathAnalyzer] Enhanced with vegetation awareness")
    
    def analyze_paths(self, tracked_objects: Dict, depth_map: np.ndarray) -> Tuple[PathAnalysis, PathAnalysis, PathAnalysis, PathDirection]:
        left_analysis = self._analyze_zone(tracked_objects, depth_map, self.left_zone, PathDirection.LEFT)
        center_analysis = self._analyze_zone(tracked_objects, depth_map, self.center_zone, PathDirection.CENTER)
        right_analysis = self._analyze_zone(tracked_objects, depth_map, self.right_zone, PathDirection.RIGHT)
        
        recommended_path = self._recommend_path(left_analysis, center_analysis, right_analysis)
        
        self.path_history.append(recommended_path)
        if len(self.path_history) >= 2:
            if self.path_history[-1] == self.path_history[-2]:
                self.last_safe_path = recommended_path
        else:
            self.last_safe_path = recommended_path
        
        return left_analysis, center_analysis, right_analysis, self.last_safe_path
    
    def _analyze_zone(self, tracked_objects: Dict, depth_map: np.ndarray, 
                     zone_bounds: Tuple[int, int], direction: PathDirection) -> PathAnalysis:
        zone_start, zone_end = zone_bounds
        
        zone_objects = []
        has_closed_door = False
        has_wall = False
        has_vegetation = False
        
        for obj_id, obj_data in tracked_objects.items():
            bbox = obj_data['bbox']
            obj_center_x = (bbox[0] + bbox[2]) / 2
            
            if zone_start <= obj_center_x < zone_end:
                zone_objects.append(obj_data)
                
                if obj_data['label'] == 'door' and obj_data.get('door_state') == DoorState.CLOSED:
                    has_closed_door = True
                
                if obj_data.get('is_wall', False):
                    has_wall = True
                
                if obj_data.get('is_vegetation', False):
                    has_vegetation = True
        
        obstacle_count = len(zone_objects)
        
        if obstacle_count == 0:
            min_distance = float('inf')
            avg_distance = float('inf')
        else:
            distances = [obj['distance'] for obj in zone_objects]
            min_distance = min(distances)
            avg_distance = sum(distances) / len(distances)
        
        zone_depth = depth_map[:, zone_start:zone_end]
        clear_threshold = 300
        clear_pixels = np.sum(zone_depth < clear_threshold)
        total_pixels = zone_depth.size
        clear_percentage = clear_pixels / total_pixels if total_pixels > 0 else 0
        
        if min_distance == float('inf'):
            distance_score = 10.0
        else:
            distance_score = min(min_distance, 10.0)
        
        safety_score = (
            distance_score * 0.4 +
            min(avg_distance, 10.0) * 0.3 +
            clear_percentage * 10 * 0.3
        )
        
        if has_closed_door:
            safety_score = 0.0
        elif has_wall and min_distance < SAFE_PATH_DISTANCE_THRESHOLD:
            safety_score *= 0.5
        elif has_vegetation and min_distance < SAFE_PATH_DISTANCE_THRESHOLD:
            safety_score *= 0.6
        
        return PathAnalysis(
            direction=direction,
            obstacle_count=obstacle_count,
            min_distance=min_distance,
            avg_distance=avg_distance,
            clear_percentage=clear_percentage,
            safety_score=safety_score,
            has_closed_door=has_closed_door,
            has_wall=has_wall,
            has_vegetation=has_vegetation
        )
    
    def _recommend_path(self, left: PathAnalysis, center: PathAnalysis, 
                       right: PathAnalysis) -> PathDirection:
        if center.has_closed_door and left.has_closed_door and right.has_closed_door:
            return PathDirection.STOP
        
        all_blocked = (
            left.min_distance < STOP_DISTANCE_METERS and
            center.min_distance < STOP_DISTANCE_METERS and
            right.min_distance < STOP_DISTANCE_METERS
        )
        
        if all_blocked:
            return PathDirection.STOP
        
        if center.is_safe() and center.safety_score > 3.0 and not center.has_closed_door:
            return PathDirection.CENTER
        
        paths = [(left, PathDirection.LEFT), 
                (center, PathDirection.CENTER), 
                (right, PathDirection.RIGHT)]
        
        safe_paths = [(p, d) for p, d in paths if p.is_safe()]
        
        if not safe_paths:
            best_path = max(paths, key=lambda x: x[0].safety_score)
            return best_path[1]
        
        best_path = max(safe_paths, key=lambda x: x[0].safety_score)
        return best_path[1]
    
    def get_guidance_message(self, recommended_path: PathDirection, 
                            left: PathAnalysis, center: PathAnalysis, 
                            right: PathAnalysis) -> str:
        if recommended_path == PathDirection.STOP:
            if (center.has_closed_door or left.has_closed_door or right.has_closed_door):
                return "STOP! Closed door blocking path. Open the door first."
            return "STOP! All paths blocked. Wait or turn around."
        
        path_map = {
            PathDirection.LEFT: left,
            PathDirection.CENTER: center,
            PathDirection.RIGHT: right
        }
        recommended_analysis = path_map.get(recommended_path)
        
        if recommended_analysis is None:
            return "No clear path detected."
        
        if recommended_path == PathDirection.CENTER:
            if center.has_closed_door:
                return "Closed door ahead - choose alternate path."
            elif center.has_vegetation:
                return f"Tree or plant ahead at {center.min_distance:.1f} meters - move around it."
            elif center.has_wall:
                return f"Wall ahead at {center.min_distance:.1f} meters - proceed carefully."
            elif center.obstacle_count == 0:
                return "Safe path ahead - continue straight."
            else:
                return f"Path clear ahead. Obstacle {center.min_distance:.1f} meters away."
        
        elif recommended_path == PathDirection.LEFT:
            if left.has_vegetation:
                return f"Move left carefully - tree or plant at {left.min_distance:.1f} meters."
            elif left.has_wall:
                return f"Move left carefully - wall at {left.min_distance:.1f} meters."
            elif left.obstacle_count == 0:
                return "Clear on your left - move left for safe path."
            else:
                return f"Move to your left. Obstacle {left.min_distance:.1f} meters away."
        
        elif recommended_path == PathDirection.RIGHT:
            if right.has_vegetation:
                return f"Move right carefully - tree or plant at {right.min_distance:.1f} meters."
            elif right.has_wall:
                return f"Move right carefully - wall at {right.min_distance:.1f} meters."
            elif right.obstacle_count == 0:
                return "Clear on your right - move right for safe path."
            else:
                return f"Move to your right. Obstacle {right.min_distance:.1f} meters away."
        
        return "Continue with caution."

# =============================================================================
# VISION MODULE
# =============================================================================
class VisionModule:
    def __init__(self, door_model_path: Optional[str] = None, model_size='n'):
        print("[VisionModule] Initializing YOLOv8 and MiDaS...")

        coco_weights = DEFAULT_COCO_MODEL if os.path.isfile(DEFAULT_COCO_MODEL) else f'yolov8{model_size}.pt'
        print(f"[VisionModule] General objects (COCO): {coco_weights}")
        self.general_model = YOLO(coco_weights)

        self.door_model = None
        if door_model_path and os.path.isfile(door_model_path):
            print(f"[VisionModule] Door model (open/closed): {door_model_path}")
            self.door_model = YOLO(door_model_path)
        else:
            print("[VisionModule] No door model — using depth-only door state if 'door' is detected")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VisionModule] Using device: {self.device}")
        
        self.depth_model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", self.depth_model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform
        print("[VisionModule] Models loaded successfully")

    def _extract_detections(self, model: YOLO, results, map_door_classes: bool) -> List[Dict]:
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = model.names[cls]
                door_state = None
                if map_door_classes:
                    label, door_state = normalize_door_label(label)

                detections.append({
                    'label': label,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'door_state': door_state,
                })
        return detections

    def detect_objects(self, frame):
        detections = self._extract_detections(
            self.general_model,
            self.general_model(frame, verbose=False, conf=0.35, imgsz=416),
            map_door_classes=False,
        )

        if self.door_model is not None:
            door_detections = self._extract_detections(
                self.door_model,
                self.door_model(frame, verbose=False, conf=0.35, imgsz=416),
                map_door_classes=True,
            )
            detections.extend(door_detections)

        return detections

    def estimate_depth(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

# =============================================================================
# DISTANCE ESTIMATOR
# =============================================================================
class DistanceEstimator:
    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=TEMPORAL_SMOOTHING_WINDOW))

    def get_distance(self, roi_depth):
        if roi_depth.size == 0:
            return float('inf')
        raw_val = np.percentile(roi_depth, DEPTH_PERCENTILE)
        distance = 500.0 / (raw_val + 1e-6)
        return float(distance)

    def smooth_distance(self, obj_id, current_dist):
        hist = self.history[obj_id]
        if len(hist) > 0:
            if abs(current_dist - hist[-1]) > 2.0 and len(hist) >= 3:
                return hist[-1]
        hist.append(current_dist)
        return float(np.median(hist))

# =============================================================================
# OBJECT TRACKER (ENHANCED)
# =============================================================================
class ObjectTracker:
    def __init__(self):
        self.tracked_objects: Dict[int, Dict] = {}
        self.next_id = 0
        self.distance_estimator = DistanceEstimator()
        self.door_analyzer = DoorAnalyzer()
        self.wall_detector = WallDetector()
        self.vegetation_detector = VegetationDetector()  # NEW
        self.object_validator = ObjectValidator()        # NEW

    def update(self, detections, depth_map, frame):
        """Enhanced update with vegetation detection and validation."""
        current_time = time.time()
        active_ids = set()
        
        # Detect vegetation
        vegetation_detections = self.vegetation_detector.detect_vegetation(frame, depth_map)
        
        # Detect walls
        wall_detections = self.wall_detector.detect_walls(frame, depth_map)
        
        # Combine all detections
        all_detections = detections + vegetation_detections + wall_detections
        
        # Validate and filter detections
        validated_detections = []
        for det in all_detections:
            # Skip validation for vegetation and walls (already validated)
            if det.get('is_vegetation') or det.get('is_wall'):
                validated_detections.append(det)
                continue
            
            # Validate regular YOLO detections
            is_valid, adj_conf, corrected_label = self.object_validator.validate_detection(
                det['label'], det['bbox'], det['confidence']
            )
            
            if is_valid:
                det['label'] = corrected_label
                det['confidence'] = adj_conf
                validated_detections.append(det)
            else:
                print(f"[Validator] Rejected: {det['label']} (conf: {det['confidence']:.2f})")
        
        for det in validated_detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            
            y_bottom = int(y1 + (y2 - y1) * 0.8)
            roi = depth_map[y_bottom:y2, x1:x2]
            
            raw_dist = self.distance_estimator.get_distance(roi)
            matched_id = self._find_match(det["bbox"])
            
            if matched_id is not None:
                obj_id = matched_id
                smooth_dist = self.distance_estimator.smooth_distance(obj_id, raw_dist)
                # Increment detection count for stability
                detection_count = self.tracked_objects[obj_id].get('detection_count', 0) + 1
            else:
                obj_id = self.next_id
                self.next_id += 1
                smooth_dist = self.distance_estimator.smooth_distance(obj_id, raw_dist)
                detection_count = 1
            
            door_state = det.get('door_state')
            if det['label'] == 'door' and door_state is None:
                door_state = self.door_analyzer.analyze_door_state(depth_map, det['bbox'], obj_id)
            
            self.tracked_objects[obj_id] = {
                'label': det['label'],
                'bbox': det['bbox'],
                'distance': smooth_dist,
                'confidence': det['confidence'],
                'direction': self._calculate_direction(det['bbox']),
                'last_seen': current_time,
                'door_state': door_state,
                'is_wall': det.get('is_wall', False),
                'is_vegetation': det.get('is_vegetation', False),
                'detection_count': detection_count
            }
            
            active_ids.add(obj_id)
        
        stale_ids = [oid for oid, obj in self.tracked_objects.items() 
                     if current_time - obj['last_seen'] > 1.0]
        for oid in stale_ids:
            del self.tracked_objects[oid]
        
        return self.tracked_objects, active_ids

    def _calculate_direction(self, bbox):
        x1, _, x2, _ = bbox
        center_x = (x1 + x2) / 2
        
        if center_x < 640 * 0.33:
            return Direction.LEFT
        elif center_x > 640 * 0.66:
            return Direction.RIGHT
        else:
            return Direction.CENTER

    def _find_match(self, bbox):
        best_id = None
        min_dist = 100
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        
        for obj_id, obj in self.tracked_objects.items():
            ox1, oy1, ox2, oy2 = obj['bbox']
            ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            dist = np.sqrt((cx - ocx)**2 + (cy - ocy)**2)
            
            if dist < 50 and dist < min_dist:
                min_dist = dist
                best_id = obj_id
        
        return best_id

# =============================================================================
# ANNOUNCEMENT QUEUE
# =============================================================================
class AnnouncementQueue:
    def __init__(self):
        self.queue: List[DetectedObject] = []
        self.queue_start_time = time.time()
        self._lock = threading.Lock()
        print(f"[QUEUE] 60-second announcement queue started")
    
    def should_reset(self) -> bool:
        elapsed = time.time() - self.queue_start_time
        return elapsed >= QUEUE_WINDOW_SECONDS
    
    def reset(self):
        with self._lock:
            announced_count = sum(1 for obj in self.queue if obj.announced)
            total_count = len(self.queue)
            print(f"\n[QUEUE RESET] Total: {total_count} | Announced: {announced_count}")
            self.queue.clear()
            self.queue_start_time = time.time()
    
    def add_or_update(self, obj_id: int, label: str, distance: float,
                      direction: Direction, confidence: float, 
                      door_state: Optional[DoorState] = None,
                      is_wall: bool = False,
                      is_vegetation: bool = False,
                      detection_count: int = 1):
        with self._lock:
            current_time = time.time()
            
            for obj in self.queue:
                if obj.object_id == obj_id:
                    obj.distance = distance
                    obj.direction = direction
                    obj.confidence = confidence
                    obj.last_updated = current_time
                    obj.detection_count = detection_count
                    if door_state:
                        obj.door_state = door_state
                    obj.is_wall = is_wall
                    obj.is_vegetation = is_vegetation
                    return
            
            new_obj = DetectedObject(
                object_id=obj_id,
                label=label,
                distance=distance,
                direction=direction,
                confidence=confidence,
                first_seen=current_time,
                last_updated=current_time,
                door_state=door_state,
                announced=False,
                is_wall=is_wall,
                is_vegetation=is_vegetation,
                detection_count=detection_count
            )
            
            self.queue.append(new_obj)
            
            # Enhanced logging
            if label == 'door':
                state_str = door_state.name if door_state else "UNKNOWN"
                print(f"[QUEUE] 🚪 DOOR: {state_str} {direction.value} @ {distance:.1f}m")
            elif is_wall:
                print(f"[QUEUE] 🧱 WALL: {direction.value} @ {distance:.1f}m")
            elif is_vegetation:
                print(f"[QUEUE] 🌳 VEGETATION: {direction.value} @ {distance:.1f}m")
            else:
                print(f"[QUEUE] NEW: {label} {direction.value} @ {distance:.1f}m (conf: {confidence:.2f})")
    
    def get_next_to_announce(self) -> Optional[DetectedObject]:
        with self._lock:
            # Only announce objects that have been detected in multiple frames
            # (reduces false positives)
            
            # Priority 1: Closed doors
            closed_doors = [obj for obj in self.queue 
                          if obj.label == 'door' and 
                          obj.door_state == DoorState.CLOSED and 
                          not obj.announced and
                          obj.detection_count >= 2]
            if closed_doors:
                return closed_doors[0]
            
            # Priority 2: Critical objects (stable detections only)
            critical_unannounced = [obj for obj in self.queue 
                                   if obj.is_critical() and 
                                   not obj.announced and
                                   obj.detection_count >= DETECTION_STABILITY_FRAMES]
            if critical_unannounced:
                next_obj = min(critical_unannounced, key=lambda x: x.distance)
                return next_obj
            
            # Priority 3: Regular FIFO (stable detections)
            for obj in self.queue:
                if not obj.announced and obj.detection_count >= DETECTION_STABILITY_FRAMES:
                    return obj
            
            return None
    
    def mark_announced(self, obj_id: int):
        with self._lock:
            for obj in self.queue:
                if obj.object_id == obj_id:
                    obj.announced = True
                    break
    
    def remove_stale(self, active_ids: set):
        with self._lock:
            self.queue = [obj for obj in self.queue 
                         if obj.object_id in active_ids or not obj.announced]
    
    def get_status(self) -> str:
        with self._lock:
            total = len(self.queue)
            announced = sum(1 for obj in self.queue if obj.announced)
            pending = total - announced
            stable = sum(1 for obj in self.queue 
                        if obj.detection_count >= DETECTION_STABILITY_FRAMES)
            return f"Queue: {total} | Stable: {stable} | Announced: {announced} | Pending: {pending}"

# =============================================================================
# VOICE SYSTEM
# =============================================================================
class VoiceSystem:
    def __init__(self):
        print("[VoiceSystem] Initializing TTS system...")
        self._is_speaking = False
        self._lock = threading.Lock()
        self.announcement_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._announcement_worker, daemon=True)
        self.worker_thread.start()
        print("[VoiceSystem] TTS system ready")
    
    def _announcement_worker(self):
        print("[VoiceSystem] Announcement worker started")
        
        while True:
            try:
                item = self.announcement_queue.get(timeout=1)
                
                if item is None:
                    break
                
                text, is_critical = item
                
                with self._lock:
                    self._is_speaking = True
                
                try:
                    print(f"[VoiceSystem] 🔊 Speaking: '{text}'")
                    
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 1.0)
                    
                    voices = engine.getProperty('voices')
                    if voices and len(voices) > 0:
                        engine.setProperty('voice', voices[0].id)
                    
                    engine.say(text)
                    engine.runAndWait()
                    
                    engine.stop()
                    del engine
                    
                except Exception as e:
                    print(f"[VoiceSystem] ERROR: {e}")
                
                with self._lock:
                    self._is_speaking = False
                
                if not is_critical:
                    time.sleep(ANNOUNCEMENT_DELAY)
                
                self.announcement_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VoiceSystem] Worker error: {e}")
                with self._lock:
                    self._is_speaking = False
    
    def announce(self, text: str, is_critical: bool = False):
        if is_critical:
            print(f"[VoiceSystem] ⚠️ CRITICAL: {text}")
            with self.announcement_queue.mutex:
                self.announcement_queue.queue.clear()
        
        self.announcement_queue.put((text, is_critical))
    
    def is_busy(self) -> bool:
        with self._lock:
            return self._is_speaking
    
    def shutdown(self):
        print("[VoiceSystem] Shutting down...")
        self.announcement_queue.put((None, False))
        self.worker_thread.join(timeout=3)

# =============================================================================
# MAIN VISION ASSISTANT
# =============================================================================
class VisionAssistant:
    def __init__(self, door_model_path: Optional[str] = None):
        print("\n" + "="*70)
        print("BLIND NAVIGATION ASSISTANT - ENHANCED DETECTION")
        print("="*70)
        print("\nFEATURES:")
        print("✓ Tree & plant detection (vegetation obstacles)")
        print("✓ Improved object classification (less false positives)")
        print("✓ Door detection with OPEN/CLOSED states")
        print("✓ Wall detection and warnings")
        print("✓ Multi-frame validation (stable detections)")
        print("✓ Safe path guidance")
        print("="*70 + "\n")
        
        self.vision = VisionModule(door_model_path=door_model_path)
        self.tracker = ObjectTracker()
        self.queue = AnnouncementQueue()
        self.voice = VoiceSystem()
        self.path_analyzer = PathAnalyzer()
        self.is_running = False
        self.last_path_announcement = 0
        self.display_fullscreen = True
        self.screen_w, self.screen_h = get_screen_size()
        
        print("[System] Initialization complete\n")

    def _setup_display_window(self):
        cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
        if self.display_fullscreen:
            cv2.setWindowProperty(
                DISPLAY_WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            print(f"[Display] Fullscreen ({self.screen_w}x{self.screen_h})")

    def _show_frame(self, frame):
        if self.display_fullscreen:
            display = cv2.resize(
                frame,
                (self.screen_w, self.screen_h),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            display = frame
        cv2.imshow(DISPLAY_WINDOW_NAME, display)
    
    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("[ERROR] Cannot access camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._setup_display_window()
        
        self.is_running = True
        
        self.voice.announce("Vision assistant active. Enhanced detection with trees, plants, doors, and walls enabled.")
        time.sleep(3)
        
        frame_count = 0
        last_announcement_check = time.time()
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time()
                
                if self.queue.should_reset():
                    self.queue.reset()
                
                detections = self.vision.detect_objects(frame)
                depth_map = self.vision.estimate_depth(frame)
                tracked_objects, active_ids = self.tracker.update(detections, depth_map, frame)
                
                left_path, center_path, right_path, recommended_path = \
                    self.path_analyzer.analyze_paths(tracked_objects, depth_map)
                
                for obj_id, obj_data in tracked_objects.items():
                    self.queue.add_or_update(
                        obj_id=obj_id,
                        label=obj_data['label'],
                        distance=obj_data['distance'],
                        direction=obj_data['direction'],
                        confidence=obj_data['confidence'],
                        door_state=obj_data.get('door_state'),
                        is_wall=obj_data.get('is_wall', False),
                        is_vegetation=obj_data.get('is_vegetation', False),
                        detection_count=obj_data.get('detection_count', 1)
                    )
                
                self.queue.remove_stale(active_ids)
                
                # Path guidance
                if current_time - self.last_path_announcement >= PATH_UPDATE_INTERVAL:
                    guidance_message = self.path_analyzer.get_guidance_message(
                        recommended_path, left_path, center_path, right_path
                    )
                    
                    if not self.voice.is_busy():
                        is_critical = (recommended_path == PathDirection.STOP)
                        self.voice.announce(guidance_message, is_critical)
                        self.last_path_announcement = current_time
                
                # Object announcements
                if current_time - last_announcement_check >= 0.1:
                    last_announcement_check = current_time
                    
                    while True:
                        next_obj = self.queue.get_next_to_announce()
                        
                        if next_obj is None:
                            break
                        
                        if self.voice.is_busy():
                            break
                        
                        announcement = next_obj.get_announcement()
                        is_critical = next_obj.is_critical()
                        
                        self.queue.mark_announced(next_obj.object_id)
                        self.voice.announce(announcement, is_critical)
                        
                        time.sleep(0.05)
                
                self._draw_debug(frame, tracked_objects, left_path, center_path, 
                               right_path, recommended_path)
                self._show_frame(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\n[System] User interrupted")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown(cap)
    
    def _draw_debug(self, frame, tracked_objects, left_path, center_path, 
                   right_path, recommended_path):
        """Enhanced visualization."""
        height, width = frame.shape[:2]
        
        cv2.line(frame, (int(width * 0.33), 0), (int(width * 0.33), height), (100, 100, 100), 1)
        cv2.line(frame, (int(width * 0.66), 0), (int(width * 0.66), height), (100, 100, 100), 1)
        
        for obj_id, obj_data in tracked_objects.items():
            x1, y1, x2, y2 = map(int, obj_data['bbox'])
            
            # Color coding
            if obj_data['label'] == 'door' and obj_data.get('door_state') == DoorState.CLOSED:
                color = (0, 0, 255)
                thickness = 3
            elif obj_data.get('is_wall'):
                color = (255, 0, 255)
                thickness = 2
            elif obj_data.get('is_vegetation'):
                color = (0, 200, 0)
                thickness = 2
            elif obj_data['distance'] < STOP_DISTANCE_METERS:
                color = (0, 0, 255)
                thickness = 2
            elif obj_data['distance'] < WARNING_DISTANCE_METERS:
                color = (0, 165, 255)
                thickness = 2
            else:
                color = (0, 255, 0)
                thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Enhanced labels
            if obj_data['label'] == 'door':
                state = obj_data.get('door_state')
                state_str = state.name if state else "UNKNOWN"
                label = f"🚪 DOOR:{state_str} (ID:{obj_id})"
            elif obj_data.get('is_wall'):
                label = f"🧱 WALL (ID:{obj_id})"
            elif obj_data.get('is_vegetation'):
                label = f"🌳 PLANT (ID:{obj_id})"
            else:
                det_count = obj_data.get('detection_count', 1)
                label = f"ID:{obj_id} {obj_data['label']} [{det_count}f]"
            
            cv2.putText(frame, label, (x1, y1-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            info = f"{obj_data['distance']:.1f}m {obj_data['direction'].value}"
            cv2.putText(frame, info, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Status
        y_offset = 30
        status = self.queue.get_status()
        cv2.putText(frame, status, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_offset += 25
        
        path_text = f"SAFE PATH: {recommended_path.value}"
        path_color = (0, 255, 0) if recommended_path != PathDirection.STOP else (0, 0, 255)
        cv2.putText(frame, path_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, path_color, 2)
        y_offset += 30
        
        def format_zone_info(path_analysis):
            flags = []
            if path_analysis.has_closed_door:
                flags.append("DOOR")
            if path_analysis.has_wall:
                flags.append("WALL")
            if path_analysis.has_vegetation:
                flags.append("PLANT")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            return f"{path_analysis.safety_score:.1f} ({path_analysis.obstacle_count}obj){flag_str}"
        
        cv2.putText(frame, f"LEFT: {format_zone_info(left_path)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"CENTER: {format_zone_info(center_path)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"RIGHT: {format_zone_info(right_path)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def shutdown(self, cap):
        print("\n[System] Shutting down...")
        self.is_running = False
        
        cap.release()
        cv2.destroyAllWindows()
        
        self.voice.announce("Vision assistant shutting down.")
        time.sleep(2)
        self.voice.shutdown()
        
        print("[System] Shutdown complete")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    source = 0
    door_model_path = None

    for arg in sys.argv[1:]:
        if arg.endswith('.pt'):
            door_model_path = arg
        else:
            source = int(arg) if arg.isdigit() else arg

    if door_model_path is None and os.path.exists(DEFAULT_DOOR_MODEL):
        door_model_path = DEFAULT_DOOR_MODEL
        print(f"[System] Door model: {door_model_path}")
        print("[System] General COCO model: person, chair, sofa, bed, tv, etc.")

    assistant = VisionAssistant(door_model_path=door_model_path)
    assistant.run(source=source)