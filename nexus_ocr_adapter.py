"""
NEXUS OCR Adapter

This module automatically detects and configures Tesseract OCR for NEXUS.
It follows NEXUS's adaptive philosophy by dynamically finding and configuring
OCR capabilities without relying on hard-coded paths.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def find_tesseract_installation():
    """Dynamically find Tesseract OCR installation on the system.
    
    Returns:
        str: Path to tesseract.exe if found, None otherwise
    """
    # First, check for OCR_TESSERACT environment variable
    env_tesseract = os.environ.get("OCR_TESSERACT")
    if env_tesseract and os.path.exists(env_tesseract):
        # Direct path to tesseract.exe was provided
        if env_tesseract.lower().endswith("tesseract.exe"):
            return env_tesseract
        # Directory containing tesseract.exe was provided
        else:
            tesseract_path = os.path.join(env_tesseract, "tesseract.exe")
            if os.path.exists(tesseract_path):
                return tesseract_path
    
    # Common installation directories for Windows
    common_locations = [
        r"C:\Program Files\Tesseract-OCR",
        r"C:\Program Files (x86)\Tesseract-OCR",
        r"C:\Tesseract-OCR",
        r"D:\Tesseract-OCR",
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Tesseract-OCR"),
    ]
    
    # Check common locations
    for location in common_locations:
        tesseract_path = os.path.join(location, "tesseract.exe")
        if os.path.exists(tesseract_path):
            return tesseract_path
            
    # Check if it's in PATH
    try:
        # Use subprocess with shell=True to find tesseract in PATH
        result = subprocess.run("where tesseract", shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass
        
    return None

def configure_pytesseract():
    """Configure pytesseract with the found Tesseract installation.
    
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    try:
        import pytesseract
        tesseract_path = find_tesseract_installation()
        
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"[NEXUS OCR] Successfully configured pytesseract with: {tesseract_path}")
            return True
        else:
            print("[NEXUS OCR] Tesseract not found on this system")
            return False
    except ImportError:
        print("[NEXUS OCR] pytesseract module not installed")
        return False

def test_ocr_capabilities():
    """Test if OCR is working with current configuration.
    
    Returns:
        bool: True if OCR is working, False otherwise
    """
    try:
        import pytesseract
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image with text
        img = Image.new('RGB', (200, 50), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        # Try to use a common font or default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()
                
        test_text = "NEXUS OCR"
        d.text((10, 10), test_text, fill=(0, 0, 0), font=font)
        
        # Try OCR
        result = pytesseract.image_to_string(img).strip()
        
        if result:
            print(f"[NEXUS OCR] Test successful - detected: '{result}'")
            return True
        else:
            print("[NEXUS OCR] Test failed - no text detected")
            return False
    except Exception as e:
        print(f"[NEXUS OCR] Test failed with error: {e}")
        return False

class OCRAdapter:
    """Adaptive OCR interface for NEXUS.
    
    This class provides a consistent interface for OCR functionality,
    automatically configuring and falling back as needed.
    """
    
    def __init__(self):
        self.ocr_available = False
        self.tesseract_path = None
        self.initialize()
    
    def initialize(self):
        """Initialize OCR adapter by finding and configuring Tesseract."""
        self.tesseract_path = find_tesseract_installation()
        
        if self.tesseract_path:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                self.ocr_available = True
                print(f"[NEXUS OCR] Initialized with Tesseract at: {self.tesseract_path}")
            except ImportError:
                print("[NEXUS OCR] pytesseract module not installed")
                self.ocr_available = False
        else:
            print("[NEXUS OCR] Tesseract not found, OCR functions will be limited")
            self.ocr_available = False
    
    def extract_text(self, image):
        """Extract text from an image.
        
        Args:
            image: PIL.Image or numpy array
            
        Returns:
            str: Extracted text or empty string if OCR is unavailable
        """
        if not self.ocr_available:
            return ""
            
        try:
            import pytesseract
            result = pytesseract.image_to_string(image).strip()
            return result
        except Exception as e:
            print(f"[NEXUS OCR] Error during text extraction: {e}")
            return ""
    
    def extract_text_with_boxes(self, image):
        """Extract text with bounding box information.
        
        Args:
            image: PIL.Image or numpy array
            
        Returns:
            list: List of dicts with text and box info, or empty list if OCR is unavailable
        """
        if not self.ocr_available:
            return []
            
        try:
            import pytesseract
            boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            results = []
            for i in range(len(boxes['text'])):
                if boxes['text'][i].strip():
                    results.append({
                        'text': boxes['text'][i],
                        'x': boxes['left'][i],
                        'y': boxes['top'][i],
                        'width': boxes['width'][i],
                        'height': boxes['height'][i],
                        'conf': boxes['conf'][i]
                    })
            return results
        except Exception as e:
            print(f"[NEXUS OCR] Error during box extraction: {e}")
            return []
    
    def get_status(self):
        """Get the current status of OCR capabilities.
        
        Returns:
            dict: Status information
        """
        return {
            'available': self.ocr_available,
            'tesseract_path': self.tesseract_path
        }

# Auto-initialize when imported
ocr = OCRAdapter()

if __name__ == "__main__":
    # When run directly, perform setup and testing
    print("\n" + "=" * 80)
    print(" NEXUS OCR Adapter - Setup & Diagnostic")
    print("=" * 80)
    
    path = find_tesseract_installation()
    if path:
        print(f"✅ Found Tesseract at: {path}")
        configured = configure_pytesseract()
        if configured:
            print("✅ Successfully configured pytesseract")
            working = test_ocr_capabilities()
            if working:
                print("✅ OCR is functioning correctly")
                print("\nNEXUS OCR system is ready to use!")
            else:
                print("❌ OCR test failed - may need troubleshooting")
        else:
            print("❌ Failed to configure pytesseract")
    else:
        print("❌ Tesseract not found on this system")
        print("\nPlease install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
