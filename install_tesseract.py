"""
Tesseract OCR Installer and Configuration Script for NEXUS

This script handles:
1. Downloading and installing Tesseract OCR
2. Configuring the PATH environment
3. Testing the installation
4. Updating NEXUS configuration to use the correct path
"""

import os
import sys
import subprocess
import tempfile
import winreg
import shutil
import ctypes
import urllib.request
from pathlib import Path
import time

def print_step(step_number, message):
    """Print a formatted step message."""
    print(f"\n[Step {step_number}] {message}")
    print("=" * 80)

def download_file(url, target_path):
    """Download a file from a URL to a target path with progress reporting."""
    print(f"Downloading from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response, open(target_path, 'wb') as out_file:
            file_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            block_size = 8192
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                
                downloaded += len(buffer)
                out_file.write(buffer)
                
                # Report progress
                if file_size > 0:
                    percent = int(downloaded * 100 / file_size)
                    sys.stdout.write(f"\rProgress: {percent}% ({downloaded} / {file_size} bytes)")
                    sys.stdout.flush()
                    
        print("\nDownload completed successfully.")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def run_command(command, cwd=None):
    """Run a command and return its output and error."""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            cwd=cwd,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode
    except Exception as e:
        return "", str(e), -1

def is_admin():
    """Check if the script is running with admin privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def add_to_path(directory, system_wide=False):
    """Add a directory to PATH environment variable."""
    if system_wide and not is_admin():
        print("WARNING: Cannot modify system PATH without admin privileges.")
        print("Will add to user PATH instead.")
        system_wide = False
        
    try:
        if system_wide:
            # System PATH
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_ALL_ACCESS) as key:
                path = winreg.QueryValueEx(key, 'Path')[0]
                if directory not in path:
                    new_path = f"{path};{directory}"
                    winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
                    print(f"Added {directory} to system PATH")
                else:
                    print(f"{directory} is already in system PATH")
        else:
            # User PATH
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Environment', 0, winreg.KEY_ALL_ACCESS) as key:
                try:
                    path = winreg.QueryValueEx(key, 'Path')[0]
                    if directory not in path:
                        new_path = f"{path};{directory}"
                        winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
                        print(f"Added {directory} to user PATH")
                    else:
                        print(f"{directory} is already in user PATH")
                except FileNotFoundError:
                    winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, directory)
                    print(f"Created new user PATH with {directory}")
        
        # Notify other processes of the change
        subprocess.call(['setx', 'DUMMY_VAR', '1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.call(['setx', 'DUMMY_VAR', ''], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Update current process environment
        os.environ['PATH'] = f"{os.environ.get('PATH', '')};{directory}"
        
        return True
    except Exception as e:
        print(f"Error updating PATH: {e}")
        return False

def update_nexus_config(tesseract_path):
    """Update NEXUS configuration to use the installed Tesseract."""
    try:
        # Update common Python files that might reference Tesseract
        for root_dir in ['src', '.']:
            for root, dirs, files in os.walk(os.path.join(os.getcwd(), root_dir)):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # Look for pytesseract import and tesseract_cmd setting
                            if 'import pytesseract' in content and 'pytesseract.pytesseract.tesseract_cmd' not in content:
                                # Add the tesseract_cmd setting after the import
                                modified_content = content.replace(
                                    'import pytesseract', 
                                    f'import pytesseract\npytesseract.pytesseract.tesseract_cmd = r"{tesseract_path}"'
                                )
                                
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    f.write(modified_content)
                                print(f"Updated {file_path} with Tesseract path")
                                
                        except Exception as e:
                            print(f"Error updating {file_path}: {e}")
        
        return True
    except Exception as e:
        print(f"Error updating NEXUS configuration: {e}")
        return False

def test_tesseract_installation(tesseract_path):
    """Test if Tesseract is correctly installed."""
    print("\nTesting Tesseract installation...")
    
    # Test direct command
    if os.path.exists(tesseract_path):
        stdout, stderr, returncode = run_command(f'"{tesseract_path}" --version')
        if returncode == 0:
            print(f"✅ Tesseract is correctly installed: {stdout.strip()}")
            return True
        else:
            print(f"❌ Error running Tesseract: {stderr}")
    else:
        print(f"❌ Tesseract executable not found at {tesseract_path}")
    
    # Try using PATH
    stdout, stderr, returncode = run_command("tesseract --version")
    if returncode == 0:
        print(f"✅ Tesseract is available in PATH: {stdout.strip()}")
        return True
    
    print("❌ Tesseract installation test failed")
    return False

def main():
    """Main function to install and configure Tesseract."""
    print("=" * 80)
    print(" NEXUS - Tesseract OCR Installer")
    print("=" * 80)
    
    # Check admin privileges
    if not is_admin():
        print("NOTE: This script is NOT running with administrator privileges.")
        print("Installation will be done for the current user only.")
    
    # Step 1: Create installation directory
    print_step(1, "Creating installation directory")
    install_dir = os.path.expanduser("~\\AppData\\Local\\Tesseract-OCR")
    os.makedirs(install_dir, exist_ok=True)
    print(f"Installation directory: {install_dir}")
    
    # Step 2: Download Tesseract installer
    print_step(2, "Downloading Tesseract OCR")
    # Check if 64-bit system
    is_64bit = sys.maxsize > 2**32
    
    if is_64bit:
        download_url = "https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe"
    else:
        # For 32-bit systems, we'll use the older version as no 5.5.0 is available for 32-bit
        download_url = "https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w32-setup-5.3.0.20221222.exe"
    
    temp_dir = tempfile.gettempdir()
    installer_path = os.path.join(temp_dir, "tesseract_installer.exe")
    
    success = download_file(download_url, installer_path)
    if not success:
        print("Failed to download Tesseract installer. Aborting.")
        return False
    
    # Step 3: Install Tesseract silently
    print_step(3, "Installing Tesseract OCR")
    
    # Silent install parameters
    install_params = f'/S /D={install_dir}'
    
    print(f"Running installer: {installer_path} {install_params}")
    stdout, stderr, returncode = run_command(f'"{installer_path}" {install_params}')
    
    if returncode != 0:
        print(f"Installation may have failed: {stderr}")
        print("Continuing with the assumption that Tesseract might be installed elsewhere...")
    else:
        print("Tesseract installer completed.")
    
    # Wait for installation to complete
    time.sleep(3)
    
    # Step 4: Verify installation
    print_step(4, "Verifying installation")
    tesseract_exe = os.path.join(install_dir, "tesseract.exe")
    
    if os.path.exists(tesseract_exe):
        print(f"✅ Tesseract executable found at: {tesseract_exe}")
    else:
        print(f"❌ Tesseract executable not found at expected location: {tesseract_exe}")
        # Try to find it elsewhere
        common_locations = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
            os.path.expanduser("~\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"),
        ]
        
        for location in common_locations:
            if os.path.exists(location):
                tesseract_exe = location
                print(f"✅ Found Tesseract at alternative location: {tesseract_exe}")
                install_dir = os.path.dirname(tesseract_exe)
                break
        else:
            print("❌ Unable to find Tesseract installation. Manual installation may be required.")
            print("Please download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
            return False
    
    # Step 5: Add to PATH
    print_step(5, "Adding Tesseract to PATH")
    add_to_path(install_dir)
    
    # Step 6: Install pytesseract if needed
    print_step(6, "Installing Python dependencies")
    stdout, stderr, returncode = run_command("pip install pytesseract pillow")
    if returncode == 0:
        print("✅ pytesseract and pillow successfully installed")
    else:
        print(f"❌ Error installing dependencies: {stderr}")
    
    # Step 7: Update NEXUS configuration
    print_step(7, "Updating NEXUS configuration")
    update_nexus_config(tesseract_exe)
    
    # Step 8: Final test
    print_step(8, "Final verification")
    test_successful = test_tesseract_installation(tesseract_exe)
    
    if test_successful:
        print("\n✅ Tesseract OCR has been successfully installed and configured!")
        print(f"Executable path: {tesseract_exe}")
        print("The NEXUS UI Intelligence system should now have OCR capabilities available.")
    else:
        print("\n❌ There were issues with the Tesseract installation.")
        print("Please try restarting your system and running this script again.")
        print("If problems persist, consider manual installation: https://github.com/UB-Mannheim/tesseract/wiki")
    
    return test_successful

if __name__ == "__main__":
    main()
