# Real-Time Face Emotion Detection (Python + TensorFlow + OpenCV)

Production-style, modular application for real-time face detection and facial emotion classification from webcam stream.

## Features

- Real-time webcam inference
- Multi-face support
- DNN-based face detection (OpenCV SSD)
- CNN-based emotion classifier (FER2013-compatible)
- Emotion label + confidence overlay
- FPS overlay
- Pause/Resume + Quit keyboard controls
- Emotion event logging to CSV
- Optional auto-screenshot capture on selected emotions

## Emotion Classes

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Architecture

### 1) Face detection
- **Model:** OpenCV DNN SSD face detector (`res10_300x300_ssd`)
- **Input:** 300x300 BGR blob
- **Output:** Bounding boxes and detection confidence

### 2) Emotion model (CNN)
- **Input shape:** `(48, 48, 1)` grayscale face crop
- **Output:** 7-class softmax probabilities
- **Architecture:**
  - Conv(32) → BN → Conv(32) → MaxPool → Dropout(0.25)
  - Conv(64) → BN → Conv(64) → MaxPool → Dropout(0.30)
  - Conv(128) → BN → Conv(128) → MaxPool → Dropout(0.35)
  - Flatten → Dense(256) → Dropout(0.50) → Dense(7, softmax)
- **Activations:** ReLU for hidden layers, Softmax for final classification
- **Loss:** Sparse categorical crossentropy
- **Optimizer:** Adam

### 3) Real-time pipeline
1. Read webcam frame
2. Detect all faces
3. For each face: crop → grayscale → resize to 48x48 → normalize
4. Predict emotion with CNN
5. Render box + label + confidence
6. Log prediction event
7. Optional screenshot capture for configured emotions

## Project structure

```text
.
├── app
│   ├── config.py
│   ├── data/fer2013.py
│   ├── detection/face_detector.py
│   ├── inference/realtime.py
│   ├── models/emotion_cnn.py
│   └── utils/{logger.py,preprocess.py,viz.py}
├── scripts
│   ├── download_assets.py
│   └── train.py
├── assets/
├── logs/
├── captures/
├── main.py
└── requirements.txt
```

## Setup

### 1) Create and activate virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Download face detector assets

```bash
python scripts/download_assets.py
```

### 4) Train emotion model (FER2013)

```bash
python scripts/train.py --epochs 40 --batch-size 64
```

This saves the best model at:

```text
assets/emotion_cnn.keras
```

> If you already have trained weights, place them at `assets/emotion_cnn.keras`.

## Run real-time application

```bash
python main.py
```

### Optional runtime flags

```bash
python main.py --camera 0 --min-face-conf 0.6
python main.py --no-fps
```

### Keyboard controls

- `q` → quit
- `p` → pause/resume stream processing

## Output artifacts

- Emotion logs: `logs/emotion_log.csv`
- Captures: `captures/*.jpg` (for configured target emotions)

## Testing checklist

1. **No-face scenario:** camera sees no person → "No face detected" appears.
2. **Single face:** box + emotion + confidence rendered.
3. **Multiple faces:** each face receives independent prediction.
4. **Low light:** verify detection threshold tuning with `--min-face-conf`.
5. **Performance:** monitor FPS overlay and reduce webcam resolution if needed.

## Performance tips

- Use 640x480 camera resolution for lower latency.
- Keep TensorFlow on CPU unless GPU is available.
- Lower face detector confidence threshold in difficult lighting.
- Ensure good frontal lighting and near-frontal head pose.
