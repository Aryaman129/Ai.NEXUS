"""
UI Element Detector for NEXUS

This module provides UI element detection capabilities using lightweight models
optimized for real-time performance on NVIDIA GPUs. It uses YOLOv8-nano for
rapid detection of UI elements such as buttons, text fields, icons, etc.

Key features:
- GPU-accelerated object detection
- Custom model for common UI elements
- Confidence scoring and filtering
- Region-of-interest tracking for efficiency
"""
import os
import time
import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# UI element types that can be detected
UI_ELEMENT_TYPES = [
    "button", "checkbox", "radio", "dropdown", "text_field", "text_area", 
    "slider", "toggle", "icon", "menu", "tab", "scrollbar", "dialog", 
    "window_control", "link", "image"
]

class UIDetector:
    """
    UI element detector using YOLOv8-nano
    
    This class provides methods for detecting UI elements in screen captures,
    with optimizations for real-time performance on NVIDIA GPUs.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the UI element detector
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - model_path: Path to custom UI element detection model
                - confidence_threshold: Minimum confidence score (0-1)
                - use_gpu: Whether to use GPU acceleration
                - batch_size: Batch size for inference
        """
        self.config = config or {}
        
        # Set default configuration
        self.model_path = self.config.get("model_path", None)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.45)
        self.use_gpu = self.config.get("use_gpu", True)
        self.batch_size = self.config.get("batch_size", 1)
        
        # Check GPU availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu else "cpu")
        
        # Load the detection model
        self.model = self._load_model()
        
        # Performance tracking
        self.inference_times = []
        self.avg_inference_time = 0
        
        logger.info(f"UIDetector initialized with device: {self.device}, "
                   f"confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self):
        """Load the YOLOv8 model for UI element detection"""
        try:
            from ultralytics import YOLO
            
            # Check if we have a custom model path
            if self.model_path and os.path.exists(self.model_path):
                model = YOLO(self.model_path)
                logger.info(f"Loaded custom UI detection model from {self.model_path}")
            else:
                # Use the YOLOv8n model and download it if not present
                model = YOLO("yolov8n.pt")
                logger.info("Loaded YOLOv8n model for UI detection")
                
            # Move to appropriate device
            model.to(self.device)
            
            # Set model parameters for efficient inference
            model.conf = self.confidence_threshold  # Confidence threshold
            model.iou = 0.45  # IoU threshold for NMS
            model.max_det = 100  # Maximum detections per image
            
            return model
            
        except ImportError:
            logger.error("Failed to import YOLO. Please install with: pip install ultralytics")
            raise ImportError("Missing required package: ultralytics")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            raise
    
    def detect_elements(self, image: np.ndarray, region_of_interest: Optional[Tuple[int, int, int, int]] = None) -> List[Dict]:
        """
        Detect UI elements in an image
        
        Args:
            image: NumPy array containing the screen image (RGB format)
            region_of_interest: Optional region to focus on (left, top, right, bottom)
            
        Returns:
            List of dictionaries containing detected UI elements, each with:
                - type: Element type (button, checkbox, etc.)
                - confidence: Detection confidence (0-1)
                - bbox: Bounding box (x1, y1, x2, y2)
                - center: Center coordinates (x, y)
        """
        start_time = time.time()
        
        # Apply region of interest if provided
        if region_of_interest:
            left, top, right, bottom = region_of_interest
            roi_image = image[top:bottom, left:right]
            offset_x, offset_y = left, top
        else:
            roi_image = image
            offset_x, offset_y = 0, 0
        
        # Run inference
        results = self.model(roi_image, verbose=False)
        
        # Process results
        detections = []
        
        if results and len(results) > 0:
            # YOLOv8 returns a list of Results objects
            result = results[0]  # Get first result (single image)
            
            # Extract boxes, confidence scores and class IDs
            if hasattr(result, 'boxes') and result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    # Convert to numpy for easier handling
                    box_data = box.data.cpu().numpy()[0]
                    
                    # Extract information
                    x1, y1, x2, y2 = map(int, box_data[:4])
                    confidence = float(box_data[4])
                    class_id = int(box_data[5])
                    
                    # Apply offset if using ROI
                    x1 += offset_x
                    y1 += offset_y
                    x2 += offset_x
                    y2 += offset_y
                    
                    # Calculate center
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Get class name
                    class_name = result.names[class_id]
                    
                    # Map YOLO classes to UI element types if using generic model
                    element_type = self._map_class_to_ui_element(class_name)
                    
                    detections.append({
                        "type": element_type,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "center": (center_x, center_y),
                        "class_name": class_name,
                        "class_id": class_id
                    })
        
        # Update performance metrics
        inference_time = time.time() - start_time
        self._update_performance_metrics(inference_time)
        
        return detections
    
    def _map_class_to_ui_element(self, class_name: str) -> str:
        """
        Map a YOLOv8 class name to a UI element type
        
        If using a generic YOLO model, this maps common objects to UI elements.
        If using a custom UI model, this may just return the class name directly.
        
        Args:
            class_name: Class name from YOLO model
            
        Returns:
            UI element type
        """
        # If using custom UI model that already has UI element classes
        if class_name.lower() in [element.lower() for element in UI_ELEMENT_TYPES]:
            return class_name.lower()
            
        # Generic YOLO model mapping
        mapping = {
            # Common mappings from COCO objects to UI elements
            "person": "icon",
            "bicycle": "icon",
            "car": "icon",
            "motorcycle": "icon",
            "airplane": "icon",
            "bus": "icon",
            "train": "icon",
            "truck": "icon",
            "boat": "icon",
            "traffic light": "icon",
            "fire hydrant": "icon",
            "stop sign": "button",
            "parking meter": "icon",
            "bench": "icon",
            "bird": "icon",
            "cat": "icon",
            "dog": "icon",
            "horse": "icon",
            "sheep": "icon",
            "cow": "icon",
            "elephant": "icon",
            "bear": "icon",
            "zebra": "icon",
            "giraffe": "icon",
            "backpack": "icon",
            "umbrella": "icon",
            "handbag": "icon",
            "tie": "icon",
            "suitcase": "icon",
            "frisbee": "icon",
            "skis": "icon",
            "snowboard": "icon",
            "sports ball": "icon",
            "kite": "icon",
            "baseball bat": "icon",
            "baseball glove": "icon",
            "skateboard": "icon",
            "surfboard": "icon",
            "tennis racket": "icon",
            "bottle": "icon",
            "wine glass": "icon",
            "cup": "icon",
            "fork": "icon",
            "knife": "icon",
            "spoon": "icon",
            "bowl": "icon",
            "banana": "icon",
            "apple": "icon",
            "sandwich": "icon",
            "orange": "icon",
            "broccoli": "icon",
            "carrot": "icon",
            "hot dog": "icon",
            "pizza": "icon",
            "donut": "icon",
            "cake": "icon",
            "chair": "icon",
            "couch": "icon",
            "potted plant": "icon",
            "bed": "icon",
            "dining table": "icon",
            "toilet": "icon",
            "tv": "window",
            "laptop": "window",
            "mouse": "icon",
            "remote": "icon",
            "keyboard": "text_field",
            "cell phone": "icon",
            "microwave": "icon",
            "oven": "icon",
            "toaster": "icon",
            "sink": "icon",
            "refrigerator": "icon",
            "book": "icon",
            "clock": "icon",
            "vase": "icon",
            "scissors": "icon",
            "teddy bear": "icon",
            "hair drier": "icon",
            "toothbrush": "icon"
        }
        
        return mapping.get(class_name.lower(), "unknown")
    
    def _update_performance_metrics(self, inference_time: float):
        """Update performance metrics for monitoring"""
        # Keep last 30 inference times for rolling average
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 30:
            self.inference_times.pop(0)
            
        # Calculate average inference time
        if self.inference_times:
            self.avg_inference_time = sum(self.inference_times) / len(self.inference_times)
    
    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        fps = 1.0 / self.avg_inference_time if self.avg_inference_time > 0 else 0
        return {
            "fps": fps,
            "avg_inference_time": self.avg_inference_time,
            "device": str(self.device),
            "confidence_threshold": self.confidence_threshold
        }
        
    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for detections
        
        Args:
            threshold: Confidence threshold (0-1)
        """
        self.confidence_threshold = max(0.1, min(0.9, threshold))
        if hasattr(self.model, 'conf'):
            self.model.conf = self.confidence_threshold
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")
        
    def train_custom_model(self, dataset_path: str, epochs: int = 50) -> str:
        """
        Train a custom UI element detection model
        
        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
            
        Returns:
            Path to trained model
        """
        try:
            # Create output directory
            output_dir = Path("models/ui_detector")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Train model
            self.model.train(
                data=dataset_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                device=0 if self.use_gpu else 'cpu',
                project=str(output_dir),
                name="custom_ui_detector"
            )
            
            # Get path to best model
            best_model_path = output_dir / "custom_ui_detector" / "weights" / "best.pt"
            
            # Update current model
            self.model_path = str(best_model_path)
            self.model = self._load_model()
            
            return self.model_path
            
        except Exception as e:
            logger.error(f"Error training custom model: {e}")
            raise
            
    def get_supported_element_types(self) -> List[str]:
        """
        Get list of supported UI element types
        
        Returns:
            List of element type strings
        """
        return UI_ELEMENT_TYPES
