import json, os
import numpy as np
import pandas as pd
import cv2

# ─── Configuration (inlined) ─────────────────────────────────────────────
calibs = [
    "calib-camera/cam_13/camera_calib_real.json",
    "calib-camera/cam_2/camera_calib_real.json"
]
tracks = [
    "runs/detect/cam_13/tracks_rect_cam13.csv",
    "runs/detect/cam_2/tracks_rect_cam2.csv"
]
OUT_CSV = "result/world3d.csv"
# ──────────────────────────────────────────────────────────────────────────

def load_camera(fp, dtype=np.float64):
    """Load K, rvecs, tvecs → build 3×4 P matrix."""
    J = json.load(open(fp))
    K = np.array(J["mtx"],    dtype=dtype).reshape(3,3)
    R,_ = cv2.Rodrigues(np.array(J["rvecs"], dtype=dtype).reshape(3,1))
    t = np.array(J["tvecs"],   dtype=dtype).reshape(3,1)
    P = K @ np.hstack((R, t))  # 3×4 projection matrix
    return P

# ─── Load cameras ────────────────────────────────────────────────────────
P1 = load_camera(calibs[0])
P2 = load_camera(calibs[1])

# ─── Load & rename tracks ───────────────────────────────────────────────
df1 = pd.read_csv(tracks[0]).rename(
    columns={"u_rect":"u1","v_rect":"v1","score":"score1"}
)
df2 = pd.read_csv(tracks[1]).rename(
    columns={"u_rect":"u2","v_rect":"v2","score":"score2"}
)

# ─── Merge on frame & id ────────────────────────────────────────────────
merged = pd.merge(df1, df2, on=["frame","id"], how="inner")
if merged.empty:
    raise RuntimeError("No matching detections across the two views!")

# ─── Prepare point arrays (2×N) ────────────────────────────────────────
pts1 = merged[["u1","v1"]].to_numpy().T.astype(np.float64)  # shape (2,N)
pts2 = merged[["u2","v2"]].to_numpy().T.astype(np.float64)  # shape (2,N)

# ─── Triangulate ────────────────────────────────────────────────────────
pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)           # (4,N)
pts3d = (pts4d[:3] / pts4d[3]).T                           # (N,3)

# ─── Assign & save ─────────────────────────────────────────────────────
merged["X"] = pts3d[:,0]
merged["Y"] = pts3d[:,1]
merged["Z"] = pts3d[:,2]

out_df = merged[["frame","id","X","Y","Z","score1","score2"]]
os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
out_df.to_csv(OUT_CSV, index=False)
print(f"✅ Wrote {len(out_df)} points → {OUT_CSV}")
