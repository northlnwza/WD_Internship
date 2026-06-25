"""
Scratch vs Contam — YOLO segmentation pipeline.

Steps:
  1. Prepare  — organise images + YOLO labels into yolo_data/train|val
  2. Verify   — confirm every image has a matching label file
  3. Train    — YOLOv8-seg fine-tune, live loss in terminal
  4. Predict  — run segmentation on test_result/ -> test_result_output/

Usage:
  python pipeline.py                         # full train pipeline
  python pipeline.py --predict               # predict on test_result/ (default)
  python pipeline.py --predict --folder path/to/images
"""

import argparse
import shutil
import random
import yaml
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path="config.yaml"):
    with open(_SCRIPT_DIR / path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve(rel):
    return _SCRIPT_DIR / rel


# ── banner ────────────────────────────────────────────────────────────────────

def banner(step, title):
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  STEP {step}: {title}", flush=True)
    print("=" * 60, flush=True)


# ── step 1: prepare ───────────────────────────────────────────────────────────

def step_prepare(cfg):
    labels_dir = resolve(cfg["data"]["labels_dir"])
    yolo_dir   = resolve("yolo_data")
    class_map  = {int(k): v for k, v in cfg["data"]["classes"].items()}
    train_ratio = cfg["data"]["train_ratio"]
    seed        = cfg["data"]["seed"]
    keep_bg     = cfg["data"].get("keep_backgrounds", True)

    print(f"Classes : {class_map}")
    print(f"Source  : {labels_dir}")

    # collect image/label pairs. empty .txt = background image (no defect).
    pairs, n_bg = [], 0
    for img in sorted(labels_dir.glob("*.png")):
        txt = img.with_suffix(".txt")
        if not txt.exists():
            continue
        is_bg = txt.stat().st_size == 0
        if is_bg:
            n_bg += 1
            if not keep_bg:
                continue          # skip empty-label images when disabled
        pairs.append((img, txt))

    print(f"Found   : {len(pairs)} images "
          f"({n_bg} background{'s' if n_bg != 1 else ''}, "
          f"{'kept' if keep_bg else 'skipped'})")

    if not pairs:
        print("No image+label pairs found. Check labels_dir in config.yaml.")
        return

    # split
    random.seed(seed)
    random.shuffle(pairs)
    n_train = max(1, int(len(pairs) * train_ratio))
    splits  = {"train": pairs[:n_train], "val": pairs[n_train:]}

    # copy into yolo_data/images/{split}/ and yolo_data/labels/{split}/
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    counts = {}
    for split, items in splits.items():
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, txt in items:
            shutil.copy(img, yolo_dir / "images" / split / img.name)
            shutil.copy(txt, yolo_dir / "labels" / split / txt.name)
        counts[split] = len(items)

    # write data.yaml for YOLO
    data_yaml = {
        "path": str(yolo_dir),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(class_map),
        "names": {k: v for k, v in class_map.items()},
    }
    data_yaml_path = yolo_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'Split':<8} {'Images':>8}")
    print("-" * 18)
    for split, n in counts.items():
        print(f"{split:<8} {n:>8}")
    print("-" * 18)
    print(f"Total    {len(pairs):>8}")
    print(f"\nYOLO data -> {yolo_dir}")
    print(f"data.yaml -> {data_yaml_path}")


# ── step 2: verify ────────────────────────────────────────────────────────────

def step_verify(cfg):
    yolo_dir = resolve("yolo_data")
    ok, fail = 0, 0

    for split in ["train", "val"]:
        img_dir = yolo_dir / "images" / split
        lbl_dir = yolo_dir / "labels" / split
        if not img_dir.exists():
            print(f"  [WARN] Missing {img_dir}")
            continue
        for img in sorted(img_dir.glob("*.png")):
            lbl = lbl_dir / img.with_suffix(".txt").name
            if lbl.exists():
                ok += 1
                print(f"  [OK]   {split}/images/{img.name}  <->  {split}/labels/{lbl.name}")
            else:
                fail += 1
                print(f"  [MISS] {split}/images/{img.name}  — no label file!")

    print(f"\n{'='*55}")
    print(f"  Total: {ok+fail}  |  OK: {ok}  |  Missing labels: {fail}")
    if fail == 0:
        print("  All images have matching label files.")
    print("="*55)

    if fail > 0:
        import sys
        sys.exit(f"\nAborting: {fail} image(s) missing label files.")


# ── step 3: train ─────────────────────────────────────────────────────────────

def step_train(cfg):
    from ultralytics import YOLO

    tcfg       = cfg["training"]
    data_yaml  = resolve("yolo_data") / "data.yaml"
    model_name = cfg["model"].get("yolo_model", "yolov8n-seg.pt")
    epochs     = tcfg["epochs"]
    imgsz      = cfg["model"]["input_size"]
    batch      = tcfg["batch_size"]
    patience   = tcfg["patience"]
    freeze     = tcfg.get("freeze", 0)
    out_dir    = resolve(tcfg["output_dir"])

    print(f"Model    : {model_name}")
    print(f"Data     : {data_yaml}")
    print(f"Epochs   : {epochs}  |  imgsz: {imgsz}  |  batch: {batch}")
    print(f"Patience : {patience}  |  freeze: {freeze}")

    # small-dataset hardening: augmentation knobs (defaults match YOLO if absent)
    aug = {}
    if tcfg.get("augment", False):
        for k in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
                  "scale", "fliplr", "flipud", "mosaic"):
            if k in tcfg:
                aug[k] = tcfg[k]
        print(f"Augment  : {aug}")
    print()

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        freeze=freeze,
        project=str(out_dir),
        name="train",
        exist_ok=True,
        verbose=True,
        **aug,
    )

    best = out_dir / "train" / "weights" / "best.pt"
    print(f"\nBest checkpoint -> {best}")


# ── step 4: predict ───────────────────────────────────────────────────────────

def step_predict(cfg, folder="test_result"):
    from ultralytics import YOLO

    out_dir    = resolve(cfg["training"]["output_dir"])
    best_ckpt  = out_dir / "train" / "weights" / "best.pt"
    input_dir  = resolve(folder)
    output_dir = resolve(folder + "_output")

    if not best_ckpt.exists():
        print(f"No trained model found at {best_ckpt}")
        print("Run  python pipeline.py  first to train.")
        return

    if not input_dir.exists():
        print(f"Folder not found: {input_dir}")
        return

    output_dir.mkdir(exist_ok=True)

    imgs = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    print(f"Model  : {best_ckpt}")
    print(f"Input  : {input_dir}  ({len(imgs)} images)")
    print(f"Output : {output_dir}\n")

    model = YOLO(best_ckpt)
    model.predict(
        source=str(input_dir),
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        conf=0.25,
        verbose=True,
    )

    print(f"\nSegmentation results -> {output_dir}/")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO segmentation pipeline")
    parser.add_argument("--predict", action="store_true",
                        help="Run segmentation on test_result/ (skip training)")
    parser.add_argument("--folder", default="test_result",
                        help="Folder to predict on (default: test_result)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.predict:
        banner("*", "PREDICT")
        step_predict(cfg, folder=args.folder)
        return

    import time
    t_start = time.time()

    banner(1, "PREPARE DATASET")
    step_prepare(cfg)

    banner(2, "VERIFY DATASET")
    step_verify(cfg)

    banner(3, "TRAIN  (YOLOv8 segmentation)")
    step_train(cfg)

    banner(4, "PREDICT  ->  test_result_output/")
    step_predict(cfg)

    elapsed = time.time() - t_start
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  PIPELINE DONE  ({elapsed:.1f}s)", flush=True)
    print("  outputs:", flush=True)
    print(f"    {resolve(cfg['training']['output_dir'])}/train/weights/best.pt", flush=True)
    print(f"    {resolve('test_result_output')}/*.png  (segmentation visuals)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
