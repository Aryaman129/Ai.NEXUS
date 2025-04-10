"""
Tesseract OCR Diagnostic Tool

This script helps diagnose issues with Tesseract OCR installation and PATH configuration.
It will:
1. Check if Tesseract is installed
2. Check if it's in the PATH
3. Look for Tesseract in common installation locations
4. Provide guidance on fixing PATH issues
"""

import os
import sys
import subprocess
import platform
import ctypes
from pathlib import Path
import shutil

def print_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def check_command_exists(command):
    """Check if a command exists in the system PATH."""
    return shutil.which(command) is not None

def run_command(command):
    """Run a command and return its output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=False)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), -1

def check_tesseract_version():
    """Check if Tesseract is installed and get its version."""
    print_header("Checking Tesseract Installation")
    
    if check_command_exists("tesseract"):
        print("✅ Tesseract command is available in PATH")
        stdout, stderr, return_code = run_command("tesseract --version")
        
        if return_code == 0:
            print(f"✅ Tesseract version info:\n{stdout.strip()}")
            return True
        else:
            print(f"❌ Error running tesseract: {stderr}")
            return False
    else:
        print("❌ Tesseract command is NOT available in PATH")
        return False

def show_python_environment():
    """Show Python environment details."""
    print_header("Python Environment")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    try:
        import pytesseract
        print(f"✅ pytesseract module is installed (version: {pytesseract.__version__})")
        print(f"pytesseract path: {pytesseract.__path__}")
        
        tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
        print(f"Current pytesseract.tesseract_cmd setting: {tesseract_cmd}")
        
        if os.path.isfile(tesseract_cmd):
            print(f"✅ The configured tesseract executable exists")
        else:
            print(f"❌ The configured tesseract executable does NOT exist")
            
    except ImportError:
        print("❌ pytesseract module is NOT installed")
    
    try:
        import cv2
        print(f"✅ OpenCV is installed (version: {cv2.__version__})")
    except ImportError:
        print("❌ OpenCV (cv2) is NOT installed")

def check_path_environment():
    """Check the PATH environment variable."""
    print_header("PATH Environment Variable Analysis")
    
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    print(f"Number of entries in PATH: {len(path_entries)}")
    
    for i, path in enumerate(path_entries, 1):
        path = path.strip()
        if path:
            if os.path.exists(path):
                print(f"{i}. ✅ {path}")
            else:
                print(f"{i}. ❌ {path} (does not exist)")

def find_tesseract_installation():
    """Look for Tesseract in common installation locations."""
    print_header("Searching for Tesseract Installation")
    
    # Common installation paths for Tesseract on Windows
    possible_locations = [
        r"C:\Program Files\Tesseract-OCR",
        r"C:\Program Files (x86)\Tesseract-OCR",
        r"C:\Tesseract-OCR",
        *[f"{drive}:\\Tesseract-OCR" for drive in "CDEFG"],
        os.path.expanduser("~\\AppData\\Local\\Tesseract-OCR"),
        os.path.expanduser("~\\AppData\\Local\\Programs\\Tesseract-OCR"),
    ]
    
    found = False
    for location in possible_locations:
        exe_path = os.path.join(location, "tesseract.exe")
        if os.path.isfile(exe_path):
            print(f"✅ Found Tesseract executable at: {exe_path}")
            found = True
            try:
                stdout, stderr, return_code = run_command(f'"{exe_path}" --version')
                if return_code == 0:
                    print(f"   Version: {stdout.strip()}")
                else:
                    print(f"   Error getting version: {stderr}")
            except Exception as e:
                print(f"   Error: {e}")
                
    # Search in PATH folders
    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        exe_path = os.path.join(path_dir, "tesseract.exe")
        if os.path.isfile(exe_path):
            print(f"✅ Found Tesseract executable in PATH: {exe_path}")
            found = True
            
    if not found:
        print("❌ Tesseract executable not found in common locations")
        print("   You may need to install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")

def test_ocr_functionality():
    """Test if OCR functionality works with a simple image."""
    print_header("Testing OCR Functionality")
    
    try:
        import pytesseract
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image with text
        img = Image.new('RGB', (200, 50), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        
        # Try to use a common font
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            try:
                font = ImageFont.truetype("times.ttf", 15)
            except:
                font = ImageFont.load_default()
                
        d.text((10, 10), "Testing OCR 123", fill=(0, 0, 0), font=font)
        
        # Save temporary image
        temp_image_path = os.path.join(os.getcwd(), "temp_ocr_test.png")
        img.save(temp_image_path)
        print(f"Created test image at: {temp_image_path}")
        
        # Try OCR with all available methods
        print("\nTrying OCR with pytesseract:")
        try:
            text = pytesseract.image_to_string(img)
            print(f"✅ OCR result: {text.strip()}")
        except Exception as e:
            print(f"❌ Error with pytesseract: {e}")
            
            # If there's an error, try with direct command
            exe_path = pytesseract.pytesseract.tesseract_cmd
            if os.path.isfile(exe_path):
                print("\nTrying direct command execution:")
                out_file = os.path.join(os.getcwd(), "temp_out")
                command = f'"{exe_path}" "{temp_image_path}" "{out_file}"'
                stdout, stderr, return_code = run_command(command)
                
                if return_code == 0:
                    try:
                        with open(f"{out_file}.txt", 'r') as f:
                            print(f"✅ OCR result from direct command: {f.read().strip()}")
                        os.remove(f"{out_file}.txt")
                    except Exception as read_err:
                        print(f"❌ Error reading output: {read_err}")
                else:
                    print(f"❌ Direct command failed: {stderr}")

        # Cleanup
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
                
    except ImportError as e:
        print(f"❌ Could not test OCR functionality: {e}")
        print("   Make sure pytesseract and Pillow are installed (pip install pytesseract Pillow)")

def suggest_fixes():
    """Suggest fixes for common Tesseract issues."""
    print_header("Suggested Fixes")
    
    print("1. Install Tesseract OCR (if not installed):")
    print("   - Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   - Choose the appropriate installer (32-bit or 64-bit)")
    print("   - During installation, check 'Add to PATH' option")
    print()
    
    print("2. Add Tesseract to PATH manually (if needed):")
    print("   - Press Win+S, type 'environment variables', click 'Edit the system environment variables'")
    print("   - Click 'Environment Variables'")
    print("   - Under System Variables, find and select 'Path', click 'Edit'")
    print("   - Click 'New' and add the Tesseract installation directory (e.g., C:\\Program Files\\Tesseract-OCR)")
    print("   - Click 'OK' on all dialogs")
    print()
    
    print("3. Set pytesseract.pytesseract.tesseract_cmd in your code:")
    print("   Example:")
    print("   import pytesseract")
    print("   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
    print()
    
    print("4. Install pytesseract and dependencies:")
    print("   pip install pytesseract Pillow numpy opencv-python")
    print()
    
    print("5. Restart your IDE or terminal after making PATH changes")

def is_admin():
    """Check if the script is running with admin privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def main():
    """Main function to run all diagnostic checks."""
    print_header("Tesseract OCR Diagnostic Tool")
    
    if not is_admin():
        print("NOTE: This script is NOT running with administrator privileges.")
        print("Some fixes might require admin rights.")
    
    check_tesseract_version()
    show_python_environment()
    check_path_environment()
    find_tesseract_installation()
    test_ocr_functionality()
    suggest_fixes()
    
    print("\nDiagnostic completed. See above for detailed information and suggested fixes.")

if __name__ == "__main__":
    main()
