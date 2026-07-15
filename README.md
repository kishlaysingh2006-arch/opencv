# OpenCV and all the cool stuff I built around it, this vacay 

after grasping a little bit of python from making recreating a bunch of games and messing around with some projects; i decided to get my hands dirty with something i found really cool on Insta. I started simple with a hand tracker and a face tracker before moving on to making a bit cooler (and more difficult for me to understand, much less build) stuff. With all that being said, here's all that I've managed creat till now. also, i'll try to attach a proper README with all my other projects too...

## Key Features

* **Air Mouse (System Control):** Control your desktop cursor with (somewhat) high precision by tracking your index finger.
* **Pinch-to-Click & Draw:** Uses Euclidean distance calculations between the thumb and index finger to trigger distinct state-machine events (Clicking, Dragging, Drawing). I do recommend studying the Mathematics invloved in this. I was not able to understand everything, but it's really very interesting.
* **Algorithmic Jitter Smoothing:** Implements an Exponential Moving Average (EMA) filter to translate noisy raw webcam coordinates into a buttery-smooth desktop experience. I suggest changing this value and experimenting around and find a value that feels the best for YOU. 
* **Ergonomic Sub-Region Mapping:** Maps the entire screen resolution to a small active bounding box in the center of the camera frame, allowing for full-screen navigation with minimal wrist fatigue. Can also be changed, so make sure you mess around with that. make it microscopic for maximum chaos and fun (possibly, some system issues too).
* **Object-Oriented Canvas Engine:** * Strokes are stored as vector objects in a memory stack rather than destructive raster modifications.
  * Features an instant, O(1) time complexity **Undo** function.

## Tech Stack

* **Language:** Python 3.x
* **Computer Vision:** OpenCV (`cv2`)
* **Machine Learning:** MediaPipe (Vision Tasks API)
* **Hardware Interface:** Pynput, Screeninfo
* **Math & Matrices:** NumPy, Math

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/airbrush.git](https://github.com/yourusername/airbrush.git)
   cd airbrush

# Set up a virtual environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```
setting up a virtaul environment was a must for me as I use Homebrew (curse of macOS). I was facing issues with aligning the versions of Python and Mediapipe and after a lot of wasted time and banging my head against a pointless tech wall, this is how I finally made it work. 

# Install dependencies:

```bash
pip install -r requirements.txt
```

setting up a virtual environment was a must for me as I use Homebrew (curse of macOS). I was facing issues with aligning the versions of Python and Mediapipe and after a lot of wasted time and banging my head against a pointless tech wall, this is what finally made it work. 

# Download the MediaPipe Model:

Download the hand_landmarker.task model file from the official Google MediaPipe documentation and place it in the root directory (or the designated models/ folder).

# Usage and Controls: 

```bash
python src/air_brush.py
```

It's simple. You don't even need to do anything for the face and hand tracker. just... be in front of the camera, i guess. as for the air mouse, hover your hand around. The index finger is the main man and piching will to click (distance < 5% of screen). 

as for the air brush; here you go.

Key,Action
u - Undo the last continuous stroke
c - Clear the entire canvas
r - Switch brush color to Red
g - Switch brush color to Green
b - Switch brush color to Blue
q - Quit the application

have fun! 

# Architecture and Directory Structure 

airbrush/
├── .gitignore
├── README.md
├── requirements.txt
├── models/
│   └── hand_landmarker.task
└── src/
    ├── air_mouse.py
    ├── air_brush.py
    ├── handmesh_tracker.py
    └── facemesh_tracker.py 
    
# Author 
-> Kishlay 



