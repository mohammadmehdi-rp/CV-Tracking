import os, argparse
import numpy as np, pandas as pd

def per_track_metrics(df, fps):
    rows = []
    for tid, g in df.groupby("id"):
        g = g.sort_values("frame")
        # planar distance frame-to-frame
        dx = np.diff(g.X_m)
        dy = np.diff(g.Y_m)
        step = np.hypot(dx, dy)
        total_d = step.sum()
        max_step = step.max() if len(step) else 0.0
        dur_frames = g.frame.max() - g.frame.min() + 1
        dur_sec = dur_frames / fps
        e2e = np.hypot(g.X_m.iloc[-1]-g.X_m.iloc[0],
                       g.Y_m.iloc[-1]-g.Y_m.iloc[0])
        straight = e2e / total_d if total_d>0 else np.nan
        rows.append([
            tid, len(g), dur_sec, total_d, e2e, straight,
            total_d/dur_sec if dur_sec>0 else np.nan,
            max_step*fps,                     # max speed
            g.Z_m.mean()
        ])
    cols = ["id","n_frames","duration_s","total_dist_m",
            "e2e_dist_m","straightness",
            "avg_speed_mps","max_speed_mps","avg_height_m"]
    return pd.DataFrame(rows, columns=cols)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv",  default="result/world3d_court.csv",
                   help="court-aligned CSV (X_m,Y_m,Z_m,frame,id)")
    p.add_argument("--fps",     type=float, default=25,
                   help="video frame-rate (default 25)")
    p.add_argument("--out_csv", default="result/track_metrics.csv",
                   help="where to save the metrics table")
    args = p.parse_args()

    df = pd.read_csv(args.in_csv)
    needed = {"id","frame","X_m","Y_m","Z_m"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"{args.in_csv} missing columns {needed-needed}")

    metrics = per_track_metrics(df, args.fps)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    metrics.to_csv(args.out_csv, index=False)
    print("✅  wrote metrics →", args.out_csv)
    print(metrics.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
