import glob
import os
import pandas as pd

# === Configuration ===
VIDEO_DIR    = "result/2DTracking/out13"           # the video folder
LABELS_DIR   = os.path.join(VIDEO_DIR, "labels")   # labels subfolder
REPORT_DIR   = os.path.join(VIDEO_DIR, "report")    # reports folder
os.makedirs(REPORT_DIR, exist_ok=True)

# Class names by index
class_names = [
    'Ball', 'Red_0', 'Red_11', 'Red_12', 'Red_16', 'Red_2',
    'Refree_F', 'Refree_M',
    'White_13', 'White_16', 'White_25', 'White_27', 'White_34'
]

records = []

# Read all .txt files in LABELS_DIR
for txt_path in sorted(glob.glob(os.path.join(LABELS_DIR, "*.txt"))):
    frame_str = os.path.splitext(os.path.basename(txt_path))[0]
    if not frame_str.isdigit():
        continue
    frame = int(frame_str)
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                continue
            cls_id = int(parts[0])
            tid    = int(parts[5])
            conf   = float(parts[6])
            records.append({
                "frame": frame,
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "track_id": tid,
                "confidence": conf
            })

# Build DataFrame
df = pd.DataFrame(records)

if df.empty:
    print(f"No tracking records found in {LABELS_DIR}")
else:
    # 1) Video summary (no video column)
    video_summary = pd.DataFrame({
        "total_frames":     [df["frame"].nunique()],
        "total_detections": [len(df)],
        "unique_tracks":    [df["track_id"].nunique()]
    })
    video_summary.to_csv(os.path.join(REPORT_DIR, "video_summary.csv"), index=False)

    # 2) Class summary
    class_summary = df.groupby("class_name").agg(
        total_detections = ("class_name", "count"),
        avg_confidence   = ("confidence", "mean"),
        unique_tracks    = ("track_id", "nunique")
    ).reset_index()
    class_summary.to_csv(os.path.join(REPORT_DIR, "class_summary.csv"), index=False)

    # 3) Track-length summary
    track_lengths = df.groupby("track_id").size().reset_index(name="length")
    track_summary = pd.DataFrame({
        "avg_track_length": [track_lengths["length"].mean()],
        "max_track_length": [track_lengths["length"].max()]
    })
    track_summary.to_csv(os.path.join(REPORT_DIR, "track_summary.csv"), index=False)

    print(f"Reports saved to {REPORT_DIR}")
