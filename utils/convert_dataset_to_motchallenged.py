import os
import glob
import cv2
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
LBL_DIR    = "preprocessed-train/labels"
IMG_DIR    = "preprocessed-train/images"
OUT_GT     = "result/2DTracking/out13/evaluation/gt.txt"
ORIG_FPS   = 25      # original video FPS
ANNOT_FPS  = 5       # your exported annotation FPS
MAX_FRAME  = 130     # number of annotated frames
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUT_GT), exist_ok=True)
    factor = ORIG_FPS // ANNOT_FPS  # 25//5 = 5

    rows = []
    label_paths = sorted(glob.glob(os.path.join(LBL_DIR, "out13_frame_*_png.rf.*.txt")))
    print(f"Found {len(label_paths)} label files")

    for lbl in label_paths:
        fname = os.path.basename(lbl)
        parts = fname.split("_")
        try:
            frame0 = int(parts[2])         # "0001" → 1
        except ValueError:
            print("Skipping unrecognized file:", fname)
            continue

        if not (1 <= frame0 <= MAX_FRAME):
            continue

        # Map to original-frame numbering
        orig_frame = (frame0 - 1) * factor + 1

        # Find the matching image (handles the hash in the name)
        pattern = os.path.join(IMG_DIR, f"out13_frame_{parts[2]}_*.jpg")
        imgs = glob.glob(pattern)
        if not imgs:
            raise FileNotFoundError(f"No image for annotated frame {frame0} (looked for {pattern})")
        img = cv2.imread(imgs[0])
        if img is None:
            raise RuntimeError(f"Failed to load image {imgs[0]}")
        H, W = img.shape[:2]

        # Load YOLOv8‐style labels: cls, xc_n, yc_n, w_n, h_n
        df = pd.read_csv(lbl, sep=" ", header=None,
                         names=["cls","xc_n","yc_n","w_n","h_n"])
        if df.empty:
            continue

        # De-normalize to absolute pixels
        df["xc"] = df.xc_n * W
        df["yc"] = df.yc_n * H
        df["w"]  = df.w_n  * W
        df["h"]  = df.h_n  * H
        df["x1"] = df.xc - df.w/2
        df["y1"] = df.yc - df.h/2

        # Collect rows: mapped frame, (placeholder) ID, x, y, w, h
        for _, r in df.iterrows():
            rows.append([
                orig_frame,
                int(r.cls),   # replace with true track ID if you have it
                r.x1, r.y1,
                r.w,  r.h
            ])

    # Write MOTChallenge GT
    gt = pd.DataFrame(rows, columns=["frame","id","x","y","w","h"])
    gt.to_csv(OUT_GT, index=False, header=False, float_format="%.3f")
    print(f"Wrote {len(gt)} boxes over {gt.frame.nunique()} frames to {OUT_GT}")

if __name__ == "__main__":
    main()
