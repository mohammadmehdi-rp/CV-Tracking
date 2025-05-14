#!/usr/bin/env python3
import os, csv, numpy as np
from ultralytics import YOLO

# ─── CONFIG ──────────────────────────────────────────────────────────────
MODEL_PATH     = "runs/detect/train/weights/best.pt"
VIDEO_SRC      = "videos_rectified/out13.mp4"
TRACKER_CONFIG = "bytetrack.yaml"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45
OUT_CSV        = "runs/detect/cam_13/tracks_cam13.csv"
# ────────────────────────────────────────────────────────────────────────

# Prepare model
model = YOLO(MODEL_PATH)

# Prepare CSV writer
os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
with open(OUT_CSV, "w", newline="") as csvf:
    writer = csv.writer(csvf)
    writer.writerow(["frame","id","x1","y1","x2","y2","score"])

    # Stream inference (no full-list in RAM)
    stream = model.track(
        source=VIDEO_SRC,
        tracker=TRACKER_CONFIG,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        stream=True,
        save=False
    )

    for frame_idx, result in enumerate(stream):
        boxes = result.boxes
        # skip if no boxes object or no detections
        if boxes is None or boxes.xyxy is None:
            continue

        # fetch tensors safely
        xyxy = boxes.xyxy.cpu().numpy()         # (N,4)
        confs = (boxes.conf.cpu().numpy()
                 if boxes.conf is not None
                 else np.ones(len(xyxy), dtype=float))
        ids   = (boxes.id.cpu().numpy().astype(int)
                 if boxes.id is not None
                 else -1 * np.ones(len(xyxy), dtype=int))

        # write each detection
        for tid, (x1,y1,x2,y2), conf in zip(ids, xyxy, confs):
            writer.writerow([frame_idx, tid, x1, y1, x2, y2, conf])

print(f"✅ Wrote streaming tracks to {OUT_CSV}")
