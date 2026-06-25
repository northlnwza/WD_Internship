# Scratch vs Contam — YOLO Segmentation Pipeline
## User Manual

---

## Overview

This pipeline trains a **YOLOv8 segmentation model** to detect and classify two types of surface defects:
- **Contam** (class 0) — contamination
- **Scratch** (class 1) — scratch marks

Input: images labeled with X-AnyLabeling (YOLO polygon format)
Output: segmentation masks drawn on images with class label + confidence

---

## Files

```
labels/
  pipeline.py        ← single script, runs everything
  config.yaml        ← all settings (edit this to tune the model)
  requirements.txt   ← Python dependencies
  MANUAL.md          ← this file

  labels/            ← your labeled images + YOLO txt files (source data)
  yolo_data/         ← auto-generated, YOLO training structure
  models/            ← auto-generated, trained weights
  test_result/       ← put new unlabeled images here for prediction
  test_result_output/← auto-generated, annotated prediction results
```

---

## Requirements

- Python 3.8+
- Internet connection (first run downloads YOLOv8 weights ~7MB)

Install dependencies once:
```powershell
pip install -r requirements.txt
```
(`ultralytics` is now included in requirements.txt.)

---

## Label Format (X-AnyLabeling output)

Each `.txt` file is a YOLO polygon label:
```
class_id  x1 y1  x2 y2  x3 y3  ...  xn yn
```
- Coordinates are **normalized** (0.0 to 1.0)
- `class_id = 0` → contam
- `class_id = 1` → scratch
- Empty `.txt` file = image examined, no defect found. Kept as a **background image** (negative sample — reduces false positives). Set `data.keep_backgrounds: false` in config.yaml to skip them instead.

---

## Configuration — config.yaml

```yaml
data:
  labels_dir: "labels"     # folder containing .png + .txt pairs
  classes:
    0: "contam"            # must match X-AnyLabeling class order
    1: "scratch"
  train_ratio: 0.80        # 80% train, 20% val
  seed: 42                 # random seed for reproducible splits
  keep_backgrounds: true   # keep empty-label images as negatives

model:
  yolo_model: "yolov8n-seg.pt"   # n=nano(fast), s=small, m=medium
  input_size: 640                # image size for training

training:
  epochs: 100              # max training epochs
  batch_size: 4            # lower if RAM runs out
  patience: 30             # stop if no improvement for N epochs
  output_dir: "models"     # where weights are saved
  freeze: 10               # freeze backbone layers (anti-overfit on small data)
  augment: true            # enable augmentation (hsv/rotate/flip/scale/mosaic)
```

> **Small dataset?** With few labeled images the model overfits fast (recall
> drops as it trains). `freeze` + `augment` above fight this, but the real fix
> is **more labeled data** — aim for 50+ images per class.

### Model size options

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n-seg.pt` | Nano | Fastest | Lower |
| `yolov8s-seg.pt` | Small | Fast | Medium |
| `yolov8m-seg.pt` | Medium | Slower | Higher |

Start with nano. Switch to small/medium when you have more data (100+ images).

---

## How to Run

### Full pipeline (prepare → verify → train → predict)

```powershell
cd C:\Users\7364239\Downloads\labels
python pipeline.py
```

This runs all 4 steps automatically:

| Step | What it does |
|------|-------------|
| 1. Prepare | Reads labels/, splits into train/val, writes yolo_data/ |
| 2. Verify | Checks every image has a matching label file |
| 3. Train | Trains YOLOv8-seg, prints live loss per epoch |
| 4. Predict | Runs model on test_result/, saves to test_result_output/ |

---

### Predict only (after training is done)

```powershell
python pipeline.py --predict
```

Runs segmentation on `test_result/` → results go to `test_result_output/`.

```powershell
# predict on a different folder
python pipeline.py --predict --folder my_folder
```

---

## Training Output

```
models/
  train/
    weights/
      best.pt       ← best checkpoint (use this for prediction)
      last.pt       ← last epoch checkpoint
    results.png     ← loss + metric curves
    confusion_matrix.png
    results.csv     ← per-epoch numbers
```

During training you will see live output like:
```
Epoch   1/100   box_loss  seg_loss  cls_loss   mAP50
          0.8       1.2       0.9      0.12
Epoch   2/100   ...
```
- `box_loss` — how well it finds the region
- `seg_loss` — how well it draws the polygon mask
- `cls_loss` — how well it classifies scratch vs contam
- `mAP50` — overall accuracy (higher = better, max 1.0)

---

## Prediction Output

Each image in `test_result_output/` shows:
- **Colored polygon mask** drawn over the defect region
- **Class label + confidence** printed next to it
- Blue = contam, Red = scratch (default YOLO colors)

---

## Adding More Data

1. Label new images with X-AnyLabeling (export as YOLO polygon format)
2. Copy new `.png` + `.txt` files into `labels/labels/`
3. Run `python pipeline.py` again — pipeline rebuilds everything from scratch

More data = better accuracy. Minimum recommended: **50+ images per class**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'ultralytics'` | Run `pip install ultralytics` |
| `No model found` | Run `python pipeline.py` (full train) before `--predict` |
| Model predicts everything as one class | Dataset too small or imbalanced — add more images |
| `batch_size` causes memory error | Lower `batch_size` in config.yaml (try 2) |
| Training too slow | Normal on CPU. Add GPU or use `yolov8n-seg.pt` (nano) |
| Empty label files ignored | Expected — images with no annotations are skipped |

---

## Workflow Summary

```
Label in X-AnyLabeling
        |
        v
  labels/labels/       (.png + .txt)
        |
        v
  python pipeline.py   (train)
        |
        v
  models/train/weights/best.pt
        |
  Put new images in test_result/
        |
        v
  python pipeline.py --predict
        |
        v
  test_result_output/  (segmentation results)
```
