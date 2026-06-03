# 🌀 JJK OpenCV: Gesture-Controlled Domain Expansion Studio



A real-time computer vision project inspired by **Jujutsu Kaisen**, where hand gestures trigger cinematic anime-style attacks and domain expansions using OpenCV, MediaPipe, and procedural visual effects.

## 🎥 Overview

JJK OpenCV transforms your webcam into an interactive anime battle system.

Using MediaPipe hand tracking and custom gesture recognition algorithms, the application detects specific hand signs and activates fully animated effects such as:

* 🔵 Hollow Purple
* ♾️ Unlimited Void
* ⛩️ Malevolent Shrine

The project combines computer vision, gesture recognition, real-time graphics rendering, particle systems, and animation design into a single interactive experience.

---

## ✨ Features

### 🖐 Real-Time Hand Tracking

* Detects and tracks hands using MediaPipe
* Extracts all 21 hand landmarks
* Supports single-hand and dual-hand gestures
* Hand stability analysis for reliable recognition

### 🎯 Custom Gesture Recognition

Recognizes anime-inspired hand signs:

| Gesture            | Effect            |
| ------------------ | ----------------- |
| Gojo Hand Sign     | Unlimited Void    |
| Sukuna Hand Sign   | Malevolent Shrine |
| Hollow Purple Sign | Hollow Purple     |

Features:

* Confidence scoring
* Gesture smoothing
* Stability verification
* Trigger cooldown system

### 🌌 Procedural Visual Effects

All effects are generated programmatically using OpenCV and NumPy.

#### Unlimited Void

* Nebula generation
* Particle explosions
* Rotating infinity ring
* Space-like visual environment

#### Malevolent Shrine

* Ground formation effects
* Shrine gate construction
* Pillar generation
* Energy orb animation

#### Hollow Purple

* Red and Blue energy orbs
* Energy streams
* Dynamic tendrils
* Orb fusion sequence
* Shockwave propagation
* Bloom lighting effects
* Debris particle simulation

---

## 🏗 Project Architecture

```text
Webcam Feed
      │
      ▼
Hand Tracking (MediaPipe)
      │
      ▼
Landmark Processing
      │
      ▼
Gesture Recognition
      │
      ▼
Gesture Validation
      │
      ▼
Effect Trigger
      │
      ▼
Animation Renderer
      │
      ▼
Display Window
```

---

## 📂 Project Structure

```text
JJK_OpenCV/
│
├── main.py
├── hand_tracking.py
├── gesture_recognition.py
│
├── gojo.py
├── sukuna.py
├── hollow_purple.py
│
├── assets/
│   └── fonts/
│
└── README.md
```

---

## 📄 File Descriptions

### main.py

Main application controller.

Responsibilities:

* Webcam capture
* User interface rendering
* Effect management
* Gesture monitoring
* Animation updates

---

### hand_tracking.py

Handles hand detection and landmark extraction using MediaPipe.

Provides:

* 21 landmark coordinates
* Handedness detection
* Palm orientation detection
* Stability measurement

---

### gesture_recognition.py

Converts hand landmark data into meaningful gestures.

Features:

* Finger angle calculations
* Finger extension detection
* Multi-hand gesture analysis
* Confidence scoring
* Temporal smoothing
* Cooldown management

---

### gojo.py

Implements the **Unlimited Void** domain expansion.

Contains:

* Nebula generation
* Particle systems
* Rotating infinity ring
* Domain collapse animation

---

### sukuna.py

Implements the **Malevolent Shrine** domain expansion.

Contains:

* Ground particle field
* Shrine construction
* Torii gate rendering
* Energy effects

---

### hollow_purple.py

Implements the **Hollow Purple** attack sequence.

Contains:

* Procedural space background
* Red and Blue energy orbs
* Orb merging animation
* Shockwave effects
* Bloom rendering
* Debris particle system

---

## 🛠 Technologies Used

### OpenCV

Used for:

* Webcam capture
* Rendering
* Drawing effects
* Image processing

### MediaPipe

Used for:

* Hand detection
* Landmark tracking
* Real-time gesture input

### NumPy

Used for:

* Vector mathematics
* Particle simulation
* Geometry calculations

### Pillow (PIL)

Used for:

* Custom font rendering
* Japanese text support

---

## ⚙ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/JJK_OpenCV.git
cd JJK_OpenCV
```

### Install Dependencies

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install pillow
```

Or:

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Project

```bash
python main.py
```

Ensure:

* Webcam is connected
* Proper lighting is available
* Hand is visible to the camera

---

## 🧠 How Gesture Recognition Works

### Step 1

MediaPipe detects hand landmarks.

### Step 2

Finger angles are calculated.

### Step 3

The system determines:

* Extended fingers
* Curled fingers
* Relative finger positions

### Step 4

Confidence scores are generated.

### Step 5

A gesture must remain stable for a fixed duration.

### Step 6

The corresponding animation is triggered.

---

## 🚀 Technical Highlights

* Real-time hand tracking
* Custom gesture recognition engine
* Temporal gesture stabilization
* Particle-based visual effects
* Procedural animation generation
* Event-driven architecture
* Real-time rendering pipeline

---
