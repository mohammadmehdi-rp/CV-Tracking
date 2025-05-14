import os
import cv2
from ultralytics import YOLO

# === Configuration ===
MODEL_PATH     = "runs/detect/train/weights/best.pt"
VIDEO_SRCS     = ["videos/out13.mp4"]
OUTPUT_ROOT    = "result/2DTracking"
TRACKER_CONFIG = "bytetrack.yaml"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

model = YOLO(MODEL_PATH)

for vid_path in VIDEO_SRCS:
    vid_name      = os.path.splitext(os.path.basename(vid_path))[0]
    out_dir       = os.path.join(OUTPUT_ROOT, vid_name)
    out_video     = os.path.join(out_dir, f"{vid_name}_annotated.mp4")
    out_label_dir = os.path.join(out_dir, "labels")
    os.makedirs(out_label_dir, exist_ok=True)

    # Video writer setup
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Stream tracking
    stream = model.track(
        source=vid_path,
        tracker=TRACKER_CONFIG,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        stream=True,
        save=False
    )

    for frame_idx, result in enumerate(stream):
        # Annotate and write frame
        annotated = result.plot()
        writer.write(annotated)

        # Dump boxes safely
        boxes = result.boxes
        # Determine how many detections we actually have
        num_boxes = len(boxes.xyxy) if (boxes is not None and boxes.xyxy is not None) else 0

        txt_path = os.path.join(out_label_dir, f"{frame_idx:06d}.txt")
        with open(txt_path, "w") as f:
            for i in range(num_boxes):
                cls  = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                # If tracker failed to assign an ID, boxes.id may be None
                tid = int(boxes.id[i]) if (hasattr(boxes, "id") and boxes.id is not None) else -1
                f.write(f"{cls} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {tid} {conf:.3f}\n")

    writer.release()
    print(f"Finished {vid_name}:")
    print(f"  Video -> {out_video}")
    print(f"  Labels -> {out_label_dir}")
