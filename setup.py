from setuptools import setup, find_packages

setup(
    name="nexus",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pyautogui",
        "torch",
        "opencv-python",
        "pillow",
        "pytest",
        "ultralytics"  # For YOLOv8
    ],
    python_requires=">=3.8",
)
