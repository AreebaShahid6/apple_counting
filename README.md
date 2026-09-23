# 🍎 Apple Counting — YOLO26

<p align="center">
  <b>AI-Powered Apple Detection, Tracking & Counting for Dense Conveyor-Belt Scenes</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLO26-Object%20Detection-red?style=for-the-badge" alt="YOLO">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch" alt="PyTorch">
</p>

---

## 📌 Overview

**Apple Counting** is a real-time computer vision system for detecting, tracking, and counting apples from video footage.

The system is specifically designed for **dense and overlapping apple piles on conveyor belts**, where conventional object detection can struggle to identify individual apples.

Instead of simply counting every apple visible in a frame, the system uses a **reference counting line**. An apple is added to the count only when its tracked center crosses the line, similar to an automated conveyor-belt counting sensor.

The system also supports **multi-class detection**, allowing different apple categories such as:

* 🍎 `apple`
* 🟤 `damaged_apple`

Each class is displayed using a consistent color, while a live HUD provides real-time counting and processing statistics.

---

# ✨ Key Features

* 🍎 **Individual Apple Detection**
* 🔍 **YOLO26 Object Detection**
* 🎯 **Multi-Class Apple Detection**
* 🧩 **Tiled Detection for Dense Scenes**
* 🆔 **Unique Apple Tracking IDs**
* 🚧 **Reference-Line Counting**
* 📊 **Per-Class Counting**
* 🔢 **Total Apple Count**
* 🎨 **Class-Based Bounding Box Colors**
* 📈 **Live Count-Trend Graph**
* 📺 **Real-Time HUD Dashboard**
* ⚡ **Processing Speed / FPS Display**
* 🎯 **Average Detection Confidence**
* 📐 **Custom Counting-Line Calibration**
* 🎥 **Video Input & Annotated Video Output**
* ⚙️ **Configurable Detection Parameters**

---

# 🧠 How the System Works

```text
                    Input Video
                         │
                         ▼
                ┌─────────────────┐
                │  YOLO Detection │
                └────────┬────────┘
                         │
                         ▼
                  Dense Scene?
                         │
                         ▼
                 ┌──────────────┐
                 │    Tiling    │
                 └──────┬───────┘
                        │
                        ▼
              Duplicate Detection
                    Removal
                        │
                        ▼
                 Apple Tracking
                        │
                        ▼
                Unique Object IDs
                        │
                        ▼
               Counting Line Check
                        │
                 ┌──────┴──────┐
                 │             │
             Not Crossed     Crossed
                 │             │
                 │             ▼
                 │       Update Count
                 │             │
                 └──────┬──────┘
                        ▼
                 Live HUD / Stats
                        │
                        ▼
                Annotated Output
```

---

# 📂 Project Structure

```text
apple_counting/
│
├── detection.py
├── calibrate_counting_line.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── best.pt                 # Local trained model
│
├── training/
│   └── train_custom_model.py   # Kaggle training script
│
├── utils/
│   ├── __init__.py
│   ├── tiling.py               # Dense-scene tiling & duplicate merging
│   ├── tracker.py              # Apple tracking and unique IDs
│   ├── crossing.py             # Counting-line crossing logic
│   ├── hud.py                  # Live statistics and class visualization
│   ├── drawing.py              # Counting-line visualization
│   └── config.py               # Configuration and calibration
│
└── videos/
    └── input.mp4               # Local input video
```

> **Note:** `venv/`, video files, and trained model weights are excluded from GitHub using `.gitignore`.

---

# 🔍 Detection Model

The project uses a custom-trained **YOLO26 model** for apple detection.

The trained model is expected locally at:

```text
models/best.pt
```

The model can detect multiple classes, for example:

```text
0 → apple
1 → damaged_apple
```

The exact class names depend on the dataset used during training.

---

# 🧩 Dense Apple Detection

Detecting apples in a dense pile is challenging because apples may:

* Overlap heavily
* Partially hide each other
* Appear very small
* Touch one another
* Have similar colors
* Be located close to image boundaries

To improve detection in these situations, the project uses **image tiling**.

---

# 🧩 Tiling

Implemented in:

```text
utils/tiling.py
```

Instead of sending only the entire frame to the model, the frame can be divided into overlapping smaller regions.

```text
Original Frame
┌───────────────────────────────┐
│                               │
│      🍎 🍎 🍎 🍎              │
│    🍎 🍎 🍎 🍎 🍎              │
│      🍎 🍎 🍎                 │
│                               │
└───────────────────────────────┘
              │
              ▼
       ┌──────┬──────┐
       │ Tile │ Tile │
       ├──────┼──────┤
       │ Tile │ Tile │
       └──────┴──────┘
              │
              ▼
       Merge Detections
              │
              ▼
      Final Apple Detections
```

The overlapping tiles help the detector focus on smaller apple regions.

Duplicate detections created by overlapping tiles are then merged.

---

# 🆔 Apple Tracking

Implemented in:

```text
utils/tracker.py
```

After detection, each apple is assigned a unique tracking ID.

For example:

```text
Apple #1
Apple #2
Apple #3
Apple #4
```

The tracker attempts to maintain the same ID as an apple moves between frames.

This is important because simply detecting apples independently in every frame would cause the same apple to be counted repeatedly.

---

# 🚧 Line-Crossing Counting

Implemented in:

```text
utils/crossing.py
```

The system uses a reference line similar to a conveyor-belt counting sensor.

```text
       Apple Movement
             ↓
      🍎
      🍎
      🍎
-----------------------------  ← Counting Line
             ↓
          COUNT +1
```

An apple is **not counted simply because it appears in the video**.

Instead:

1. Apple is detected.
2. Apple receives a tracking ID.
3. Its position is monitored.
4. The system checks whether its center crosses the reference line.
5. The apple is counted only once when it crosses the line.

This prevents repeated counting of the same apple.

---

# 🎨 Multi-Class Detection

The system supports multiple apple classes.

For example:

| Class              | Description   |
| ------------------ | ------------- |
| 🍎 `apple`         | Normal apple  |
| 🟤 `damaged_apple` | Damaged apple |

Bounding boxes are colored consistently according to their class.

This makes it easier to visually distinguish different categories during processing.

---

# 📊 Live HUD

Implemented in:

```text
utils/hud.py
```

The system displays a real-time information panel containing:

* Per-class count
* Overall apple count
* Apples currently visible
* Average confidence
* Processing speed
* Class legend
* Count-trend graph

Example:

```text
┌─────────────────────────────┐
│       APPLE ANALYTICS       │
├─────────────────────────────┤
│ Apple          : 125        │
│ Damaged Apple  : 18         │
│ --------------------------- │
│ Total          : 143        │
│ Visible        : 21         │
│ Confidence     : 0.91       │
│ FPS            : 24.6       │
│                             │
│ Count Trend                  │
│ ▁▂▃▃▄▅▆▇████                 │
└─────────────────────────────┘
```

---

# 📐 Counting-Line Calibration

The project provides an optional calibration script:

```text
calibrate_counting_line.py
```

Run:

```powershell
python calibrate_counting_line.py --source videos/input.mp4
```

A video frame will be displayed.

### Calibration process

1. Open the calibration tool.
2. Click the first point of the counting line.
3. Click the second point.
4. Press `s` to save.
5. Run `detection.py`.

The calibrated line will then be used instead of the default line.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AreebaShahid6/apple_counting.git
```

Move into the project:

```bash
cd apple_counting
```

---

## 2. Create Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📁 Add Required Files

Before running the project, place your files locally:

### Trained Model

```text
models/best.pt
```

### Input Video

```text
videos/input.mp4
```

These files are intentionally excluded from GitHub because they can be large.

---

# ▶️ Run the Application

From the project root:

```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt
```

The processed video will be saved as:

```text
videos/output_counted.mp4
```

---

# 🛠️ Useful Commands

### ▶️ Show Processing Live

```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt --show
```

### 📐 Change Counting-Line Position

```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt --line-y 0.4
```

`0.0` represents the top of the frame and `1.0` represents the bottom.

### 📏 Change Line Tilt

```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt --line-tilt 0.4
```

A value of `0` creates a flat horizontal line.

### 🎯 Display Confidence Scores

```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt --show-conf
```

### 🍎 Count Specific Classes

```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt --classes apple
```

### ⚡ Disable Tiling

```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt --no-tile
```

Tiling can improve detection in dense scenes, while disabling it can make processing faster.

### 🎯 Change Detection Confidence

```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt --conf 0.35
```

Lower confidence:

```text
More detections
+
Potentially more false positives
```

Higher confidence:

```text
Fewer detections
+
Potentially fewer false positives
```

---

# 📈 Improving Detection Accuracy

If the model is missing apples, the main issue is usually the **training dataset and model quality**, rather than the counting logic.

The training script is located at:

```text
training/train_custom_model.py
```

The project can use a larger YOLO model and longer training schedule, for example:

```python
model = YOLO("yolo26s.pt")

model.train(
    data=FIXED_YAML,
    epochs=250,
    imgsz=640,
    batch=16,
    patience=50
)
```

For difficult scenes, useful training images include:

* Dense apple piles
* Partially hidden apples
* Different lighting
* Different conveyor angles
* Wide camera views
* Motion blur
* Different apple sizes
* Damaged apples
* Overlapping apples

---

# 🎯 Training Strategy

A strong training dataset should represent the actual environment where the model will be used.

For example:

```text
Training Dataset
       │
       ├── Dense piles
       ├── Wide shots
       ├── Close-up shots
       ├── Different lighting
       ├── Different backgrounds
       └── Damaged apples
                │
                ▼
          YOLO26 Training
                │
                ▼
          Trained Model
                │
                ▼
       Real Video Evaluation
                │
                ▼
       Identify Weak Scenes
                │
                ▼
        Add More Examples
                │
                └──────────────► Retrain
```

If certain camera angles consistently produce weak detection, adding labeled frames from those exact scenes can improve performance more effectively than simply increasing the number of training epochs.

---

# 🧮 How Counting Works

The complete counting process is:

### Step 1 — Detection

YOLO identifies apples in every frame.

### Step 2 — Tiling

Dense frames can be divided into smaller overlapping regions.

### Step 3 — Duplicate Removal

Overlapping tile predictions are merged to avoid detecting the same apple multiple times.

### Step 4 — Tracking

Each detected apple receives a unique ID.

### Step 5 — Position Tracking

The system monitors the center position of each tracked apple.

### Step 6 — Line Crossing

The system checks whether the apple crosses the reference line.

### Step 7 — Count

If the apple crosses the line for the first time:

```text
Count = Count + 1
```

### Step 8 — HUD Update

The live statistics and count-trend graph are updated.

---

# 📊 Example Processing Flow

```text
                    VIDEO FRAME
                         │
                         ▼
                ┌────────────────┐
                │  YOLO26 Model  │
                └───────┬────────┘
                        │
                        ▼
               Apple Detections
                        │
                        ▼
                  Image Tiling
                        │
                        ▼
               Duplicate Removal
                        │
                        ▼
                  Object Tracker
                        │
                        ▼
                 Unique Apple IDs
                        │
                        ▼
                 Counting Line
                        │
              ┌─────────┴─────────┐
              │                   │
          Not Crossed          Crossed
              │                   │
              │                   ▼
              │              Count Apple
              │                   │
              └─────────┬─────────┘
                        ▼
                    Live HUD
                        │
                        ▼
               Output Video
```

---

# 📁 Important Files

| File                             | Purpose                                  |
| -------------------------------- | ---------------------------------------- |
| `detection.py`                   | Main detection and counting pipeline     |
| `calibrate_counting_line.py`     | Custom counting-line calibration         |
| `utils/tiling.py`                | Dense-scene tiling and duplicate merging |
| `utils/tracker.py`               | Object tracking and unique IDs           |
| `utils/crossing.py`              | Line-crossing counting                   |
| `utils/hud.py`                   | Live statistics and visualization        |
| `utils/drawing.py`               | Drawing utilities                        |
| `utils/config.py`                | Configuration and calibration            |
| `training/train_custom_model.py` | YOLO26 model training                    |

---

# 🔐 GitHub File Policy

Large and generated files are intentionally excluded from this repository.

The `.gitignore` excludes:

```text
venv/
videos/
*.mp4
*.avi
*.mov
*.mkv
*.pt
*.pth
*.onnx
runs/
outputs/
```

Therefore, users need to add their own:

```text
models/best.pt
videos/input.mp4
```

locally before running the system.

---

# 🚀 Future Improvements

Possible future extensions include:

* [ ] Improved apple re-identification
* [ ] Advanced multi-object tracking
* [ ] Automatic camera calibration
* [ ] Conveyor speed estimation
* [ ] Apple size estimation
* [ ] Quality grading
* [ ] Automatic damaged-apple percentage
* [ ] Real-time production statistics
* [ ] Web-based monitoring dashboard
* [ ] Database integration
* [ ] Automatic daily counting reports
* [ ] Edge-device deployment
* [ ] Real-time industrial camera integration

---

# 💡 Applications

This system can be adapted for:

* 🍎 Fruit processing plants
* 🏭 Automated conveyor systems
* 📦 Food inspection
* 🔍 Quality control
* 📊 Production monitoring
* 🤖 Agricultural automation
* 🏢 Industrial computer vision

---

# 👩‍💻 Author

## Areeba Shahid

**Computer Science Graduate | Computer Vision Engineer | Machine Learning Enthusiast**

Interested in building intelligent systems using:

* 🤖 Artificial Intelligence
* 👁️ Computer Vision
* 🧠 Deep Learning
* 📊 Machine Learning
* 🐍 Python
* ⚙️ YOLO
* 🌐 IoT

### Connect

🔗 **LinkedIn:**
https://www.linkedin.com/in/areeba-shahid-1b53b231/

📧 **Email:**
[shahidareeba922@gmail.com](mailto:shahidareeba922@gmail.com)

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

<p align="center">
  <b>🍎 Turning Computer Vision into Automated Fruit Counting</b>
</p>
