#!/usr/bin/env python3
import os, json, cv2, numpy as np, pandas as pd

# ─── USER CONFIG ───────────────────────────────────────────────────────────
VIDEO_PATH = "videos_rectified/out13.mp4"          # rectified cam-2 video
TRACK_DIR  = "runs/detect/cam_13/tracks_cam13.csv"       # raw ByteTrack CSV
CALIB_JSON = "calib-camera/cam_13/camera_calib_real.json"
OUT_CSV     = "runs/detect/cam_13/tracks_rect_cam13.csv" # output rectified-tracks
# ──────────────────────────────────────────────────────────────────────────

def load_calib(fp):
    J    = json.load(open(fp))
    K    = np.asarray(J["mtx"],  dtype=np.float32).reshape(3,3)
    dist = np.asarray(J["dist"], dtype=np.float32).ravel()
    return K, dist

def build_map(K, dist, w, h):
    newK,_ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)
    return cv2.initUndistortRectifyMap(K, dist, None, newK, (w,h), cv2.CV_32FC1)

if __name__=="__main__":
    # 1) grab W,H from the rectified video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video '{VIDEO_PATH}'")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Video size: {W}×{H}")

    # 2) build undistort/rectify map
    K, dist = load_calib(CALIB_JSON)
    map1, map2 = build_map(K, dist, W, H)

    # 3) load raw tracks & compute centre points
    df = pd.read_csv(TRACK_DIR)
    df["u"] = (df.x1 + df.x2) / 2
    df["v"] = (df.y1 + df.y2) / 2

    # 4) remap centres into the rectified plane
    ix = np.clip(df.u.astype(int), 0, W-1)
    iy = np.clip(df.v.astype(int), 0, H-1)
    df["u_rect"] = map1[iy, ix]
    df["v_rect"] = map2[iy, ix]

    # 5) save the rectified-tracks CSV
    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    df[["frame","id","u_rect","v_rect","score"]].to_csv(OUT_CSV, index=False)
    print("✅ wrote", OUT_CSV)
