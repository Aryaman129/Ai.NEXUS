"""
YOLO Object Detector for NEXUS
Enhances vision capabilities with state-of-the-art object detection
"""
import os
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class YOLODetector:
    """
    YOLO (You Only Look Once) object detector for NEXUS.
    
    This provides advanced object detection capabilities using YOLOv8,
    allowing NEXUS to identify UI elements, objects, and screen content
    with high accuracy and speed.
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", 
                confidence_threshold: float = 0.25,
                models_dir: str = "models"):
        """
        Initialize the YOLO detector
        
        Args:
            model_name: Name or path of the YOLO model to use 
                       (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
            confidence_threshold: Minimum confidence score for detections
            models_dir: Directory to store downloaded models
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Track model loading status
        self.model = None
        self.is_model_loaded = False
        self.model_loading_error = None
        
        # For tracking performance
        self.total_inference_time = 0
        self.inference_count = 0
        
        # Try to initialize the model (but don't block)
        self._try_load_model()
        
        logger.info(f"YOLO Detector initialized with model: {model_name}")
    
    def _try_load_model(self) -> None:
        """Try to load the YOLO model in a non-blocking way"""
        try:
            # Check if ultralytics is available
            import importlib
            ultralytics_spec = importlib.util.find_spec("ultralytics")
            
            if ultralytics_spec is None:
                logger.warning("Ultralytics package not found. YOLO detection will not be available.")
                logger.warning("Install with: pip install ultralytics")
                self.model_loading_error = "Ultralytics package not installed"
                return
            
            # Use dynamic import to avoid hard dependency
            from ultralytics import YOLO
            
            # Check if model exists in models directory
            model_path = self.models_dir / self.model_name
            
            if not model_path.exists():
                # If model_name doesn't point to an existing file, try to download it
                logger.info(f"Model not found at {model_path}, will use from ultralytics hub")
                self.model = YOLO(self.model_name)
            else:
                logger.info(f"Loading model from {model_path}")
                self.model = YOLO(model_path)
            
            # Set model parameters
            self.model.conf = self.confidence_threshold
            
            # Mark as successfully loaded
            self.is_model_loaded = True
            logger.info(f"YOLO model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.model_loading_error = str(e)
    
    async def ensure_model_loaded(self) -> bool:
        """
        Ensure the model is loaded before using it
        
        Returns:
            True if model is loaded, False otherwise
        """
        # If already loaded, return True
        if self.is_model_loaded:
            return True
            
        # If there was an error, don't retry
        if self.model_loading_error:
            logger.warning(f"YOLO model not available due to previous error: {self.model_loading_error}")
            return False
            
        # Try loading again
        try:
            # Importing here to avoid hard dependency
            from ultralytics import YOLO
            
            # Check if model exists in models directory
            model_path = self.models_dir / self.model_name
            
            if not model_path.exists():
                # Try to download it
                logger.info(f"Downloading model {self.model_name}")
                self.model = YOLO(self.model_name)
            else:
                self.model = YOLO(model_path)
            
            # Set model parameters
            self.model.conf = self.confidence_threshold
            
            self.is_model_loaded = True
            logger.info(f"YOLO model loaded successfully on retry: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading YOLO model on retry: {e}")
            self.model_loading_error = str(e)
            return False
    
    async def detect_objects(self, image) -> Dict[str, Any]:
        """
        Detect objects in an image
        
        Args:
            image: PIL Image or numpy array or path to image file
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Ensure model is loaded
        if not await self.ensure_model_loaded():
            return {
                "success": False,
                "error": f"YOLO model not available: {self.model_loading_error}",
                "detections": []
            }
        
        try:
            # Handle different input types
            if isinstance(image, str):
                # Path to image
                image_path = image
            elif isinstance(image, Image.Image):
                # Save PIL image to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image_path = tmp.name
                    image.save(image_path)
            elif isinstance(image, np.ndarray):
                # Convert numpy array to PIL image and save
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    image_path = tmp.name
                    Image.fromarray(image).save(image_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported image type: {type(image)}",
                    "detections": []
                }
            
            # Run inference
            results = self.model(image_path)
            
            # Process results
            detections = []
            
            for result in results:
                # Get the boxes and metadata
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Get class name and confidence
                    class_id = int(box.cls)
                    class_name = result.names[class_id]
                    confidence = float(box.conf)
                    
                    # Create detection object
                    detection = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "width": int(x2 - x1),
                            "height": int(y2 - y1),
                            "center_x": int((x1 + x2) / 2),
                            "center_y": int((y1 + y2) / 2)
                        }
                    }
                    
                    detections.append(detection)
            
            # Update performance tracking
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.inference_count += 1
            avg_time = self.total_inference_time / self.inference_count
            
            return {
                "success": True,
                "detections": detections,
                "count": len(detections),
                "inference_time": inference_time,
                "avg_inference_time": avg_time
            }
            
        except Exception as e:
            logger.error(f"Error detecting objects: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": []
            }
    
    async def detect_ui_elements(self, screenshot) -> Dict[str, Any]:
        """
        Specialized method to detect UI elements in a screenshot
        
        Args:
            screenshot: Screenshot as PIL Image or numpy array
            
        Returns:
            Dictionary with detected UI elements
        """
        # Standard object detection
        results = await self.detect_objects(screenshot)
        
        if not results["success"]:
            return results
            
        # Filter and categorize UI elements
        ui_elements = []
        ui_categories = {
            "button": ["button", "btn", "control"],
            "text_field": ["text box", "textbox", "field", "input", "textfield"],
            "dropdown": ["dropdown", "select", "combobox", "menu"],
            "checkbox": ["checkbox", "check", "toggle"],
            "radio_button": ["radio", "option"],
            "icon": ["icon", "logo", "symbol"],
            "slider": ["slider", "scrollbar"],
            "image": ["image", "picture", "photo", "img"],
            "link": ["link", "hyperlink", "url"]
        }
        
        # Process detected objects
        for detection in results["detections"]:
            class_name = detection["class_name"].lower()
            element_type = "unknown"
            
            # Determine UI element type
            for ui_type, keywords in ui_categories.items():
                if any(keyword in class_name for keyword in keywords):
                    element_type = ui_type
                    break
                    
            # Add to UI elements with type
            ui_element = {
                "type": element_type,
                "original_class": detection["class_name"],
                "confidence": detection["confidence"],
                "bounds": detection["box"]
            }
            
            ui_elements.append(ui_element)
        
        return {
            "success": True,
            "ui_elements": ui_elements,
            "count": len(ui_elements),
            "inference_time": results.get("inference_time", 0)
        }
    
    async def detect_text_regions(self, screenshot) -> Dict[str, Any]:
        """
        Detect potential text regions in a screenshot
        
        Args:
            screenshot: Screenshot as PIL Image or numpy array
            
        Returns:
            Dictionary with detected text regions
        """
        # Use object detection to find potential text regions
        results = await self.detect_objects(screenshot)
        
        if not results["success"]:
            return results
            
        # Filter for text-like objects
        text_classes = ["text", "label", "title", "paragraph", "caption", "heading"]
        text_regions = []
        
        for detection in results["detections"]:
            class_name = detection["class_name"].lower()
            
            # If the class relates to text or has high confidence
            is_text = any(text_class in class_name for text_class in text_classes)
            if is_text or detection["confidence"] > 0.6:
                region = {
                    "region_type": "text" if is_text else "potential_text",
                    "confidence": detection["confidence"],
                    "bounds": detection["box"]
                }
                
                text_regions.append(region)
        
        return {
            "success": True,
            "text_regions": text_regions,
            "count": len(text_regions),
            "inference_time": results.get("inference_time", 0)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_model_loaded:
            return {
                "loaded": False,
                "error": self.model_loading_error
            }
            
        try:
            return {
                "loaded": True,
                "model_name": self.model_name,
                "confidence_threshold": self.confidence_threshold,
                "inference_count": self.inference_count,
                "avg_inference_time": self.total_inference_time / max(1, self.inference_count),
                "model_type": type(self.model).__name__
            }
        except Exception as e:
            return {
                "loaded": True,
                "model_name": self.model_name,
                "error": str(e)
            }
    
    async def train_custom_detector(self, 
                                  dataset_path: str, 
                                  epochs: int = 10,
                                  batch_size: int = 16) -> Dict[str, Any]:
        """
        Train a custom YOLO detector on a dataset
        
        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training results
        """
        # Ensure model is loaded
        if not await self.ensure_model_loaded():
            return {
                "success": False,
                "error": f"YOLO model not available: {self.model_loading_error}"
            }
            
        try:
            # Start training
            logger.info(f"Starting training on dataset: {dataset_path}")
            results = self.model.train(
                data=dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=640
            )
            
            # Get the path to the trained model
            trained_model_path = str(self.model.export())
            
            return {
                "success": True,
                "trained_model_path": trained_model_path,
                "epochs_completed": epochs,
                "message": f"Custom model trained and saved to {trained_model_path}"
            }
            
        except Exception as e:
            logger.error(f"Error training custom model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
