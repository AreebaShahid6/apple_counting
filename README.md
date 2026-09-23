# Apple Counting — YOLO26 (Individual Detection, Multi-Class Edition)

Real-time apple detection, tracking, and counting from video — built to
handle **dense, overlapping piles** on a conveyor belt. Every apple gets
boxed individually, colored by CLASS (all "apple" boxes one color, all
"damaged_apple" boxes another). Apples are only added to the count the
moment they **cross a reference line** — like a real conveyor counting
sensor — not just for being visible in frame. A live HUD panel shows the
per-class + total count and a count-trend graph.

## Project structure
```
apple_counting/
├── models/
│   └── best.pt                  <- PASTE YOUR TRAINED MODEL HERE
├── training/
│   └── train_custom_model.py    <- script used to train best.pt (on Kaggle)
├── utils/
│   ├── tiling.py                 <- splits dense frames into tiles + merges duplicate detections
│   ├── tracker.py                <- assigns a stable, unique ID to each apple across frames
│   ├── hud.py                    <- color-by-class boxes + the live stats/legend panel
│   ├── drawing.py                <- counting-line drawing helper
│   └── config.py                 <- save/load calibration line
├── venv/                         <- your Python virtual environment (create locally, see below)
├── videos/
│   └── input.mp4                 <- PASTE YOUR VIDEO HERE
├── calibrate_counting_line.py    <- optional: click to set a reference counting line
├── detection.py                  <- MAIN SCRIPT — run this
├── requirements.txt
└── README.md
```

## 1. Set up
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Add your files
- Your video → `videos/input.mp4`
- Your trained model from Kaggle → `models/best.pt`

## 3. Run
```powershell
python detection.py --source videos/input.mp4 --weights models/best.pt
```

Output video is saved to `videos/output_counted.mp4`, with:
- Every apple boxed individually, even in a dense pile (small, clean labels by default)
- Boxes colored **by class** — every "apple" box the same color, every
  "damaged_apple" box a different color (consistent every run)
- A visible counting line drawn across the frame — apples are only added
  to the total the moment their center crosses it
- A right-side HUD panel: a color-swatch legend + count per class,
  the overall total, apples visible right now, average confidence,
  processing speed, and a trend graph of the running total

## Counting line
By default, a diagonal line is drawn across the frame (tilted, not flat
horizontal) and used automatically — no setup needed. If you want a fully
custom line (angled exactly to match your belt, positioned somewhere
specific), calibrate one first:
```powershell
python calibrate_counting_line.py --source videos/input.mp4
```
Click two points, press `s` to save — `detection.py` will automatically
use that calibrated line instead of the default the next time you run it.

## Useful options
```powershell
# Watch it process live instead of only saving the file
python detection.py --source videos/input.mp4 --weights models/best.pt --show

# Move the default line (0.0 = top of frame, 1.0 = bottom)
python detection.py --source videos/input.mp4 --weights models/best.pt --line-y 0.4

# Make the default line more/less diagonal (0 = flat horizontal)
python detection.py --source videos/input.mp4 --weights models/best.pt --line-tilt 0.4

# Show confidence scores on box labels (off by default to reduce clutter)
python detection.py --source videos/input.mp4 --weights models/best.pt --show-conf

# Only count specific classes
python detection.py --source videos/input.mp4 --weights models/best.pt --classes apple

# Faster but worse on dense piles (skips tiling)
python detection.py --source videos/input.mp4 --weights models/best.pt --no-tile

# Lower confidence catches more real apples but risks more noise;
# raise it if you're getting false positives
python detection.py --source videos/input.mp4 --weights models/best.pt --conf 0.35
```

## If detection is still missing apples (under-detecting)
This is a training-data problem, not a script problem — fix it in
`training/train_custom_model.py` on Kaggle:
```python
model = YOLO("yolo26s.pt")   # small model — better recall than nano
model.train(data=FIXED_YAML, epochs=250, imgsz=640, batch=16, patience=50, ...)
```
250 epochs on `yolo26s` will noticeably improve recall over the initial
50-epoch nano run. If certain shots (e.g. wide overview angles) are still
weak afterward, that's a genuine gap between those frames and your
training set — pull ~30-50 frames of that exact shot type from your own
video, label them in Roboflow, and merge them into the training data.
That closes the gap faster than more epochs alone.

## How individual counting works on a dense pile
1. **Tiling** (`utils/tiling.py`): each frame is cut into overlapping
   crops so the model examines apple-sized regions instead of the whole
   pile at once. Two merge passes then remove duplicate detections of
   the same apple picked up by neighboring tiles.
2. **Tracking** (`utils/tracker.py`): a centroid tracker matches each
   detected apple to the nearest one (of similar size) from the previous
   frame, so it keeps the same ID as it moves.
3. **Line-crossing count** (`utils/crossing.py`): each tracked apple is
   only added to the total the instant it crosses the counting line -
   so the number reflects apples that actually passed the checkpoint,
   not just everything visible on screen at once.
