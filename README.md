# AI Scroll

A Python application that uses computer vision to detect hand gestures for controlling page scrolling:
- Show your palm to scroll down
- Make a fist to stop scrolling

## Requirements

- Python 3.7+
- Webcam
- Good lighting conditions

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:

```bash
python ai_scroll.py
```

2. Position your hand in front of the webcam in good lighting
3. Show your palm to start scrolling
4. Make a fist to stop scrolling
5. Press 'q' to quit the application

## How It Works

The application uses:
- OpenCV for webcam capture, image processing, and hand contour detection
- Skin color detection to isolate the hand
- Contour analysis and convexity defects to count fingers
- PyAutoGUI for controlling screen scrolling

The algorithm detects a palm when multiple fingers are extended and a fist when fingers are closed. 