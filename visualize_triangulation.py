import os, argparse, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# court dimensions
COURT_X, COURT_Y, COURT_Z = 28.0, 15.0, 3.0

# ─────────────────────────────────────────────────────────────────────────

def pick_xyz(df):
    for triple in (("X_m","Y_m","Z_m"), ("X","Y","Z"), ("x","y","z")):
        if all(c in df.columns for c in triple):
            return triple
    sys.exit("❌  no X/Y/Z columns found")

def maybe_mm_to_m(arr):
    return arr/1000.0 if np.median(np.abs(arr)) > 50 else arr

def floor_align(XYZ):
    pca = PCA(3).fit(XYZ)
    n   = pca.components_[2]
    c   = XYZ.mean(axis=0)
    z   = n/np.linalg.norm(n)
    x   = np.cross([0,1,0], z); x /= np.linalg.norm(x)
    y   = np.cross(z, x)
    R   = np.vstack([x, y, z])
    return (R @ (XYZ-c).T).T

def robust_span(v):
    return np.percentile(v,95) - np.percentile(v,5)

def main():
    IN_CSV="result/world3d.csv"
    OUT_DIR="result"
   
    if not os.path.exists(IN_CSV):
        sys.exit(f"{IN_CSV} not found")

    os.makedirs(OUT_DIR, exist_ok=True)

    # load
    df = pd.read_csv(IN_CSV)
    Xc,Yc,Zc = pick_xyz(df)
    for c in (Xc,Yc,Zc):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=[Xc,Yc,Zc], inplace=True)
    df[[Xc,Yc,Zc]] = df[[Xc,Yc,Zc]].apply(maybe_mm_to_m)

    # floor alignment
    XYZ_rot = floor_align(df[[Xc,Yc,Zc]].to_numpy())
    df["X_a"],df["Y_a"],df["Z_a"] = XYZ_rot.T

    # robust scale (choose axis whose span >30 % of the other)
    span_x, span_y = robust_span(df.X_a), robust_span(df.Y_a)
    if span_x > 0.3*span_y:
        scale = COURT_X / span_x
        print(f"scale from X-span {span_x:.2f} m  → ×{scale:.4f}")
    else:
        scale = COURT_Y / span_y
        print(f"scale from Y-span {span_y:.2f} m  → ×{scale:.4f}")

    # scale + translate
    df["X_m"] = (df.X_a - np.percentile(df.X_a,5)) * scale
    df["Y_m"] = (df.Y_a - np.percentile(df.Y_a,5)) * scale
    df["Z_m"] =  df.Z_a * scale

    # court filter
    court = df[
        (df.X_m.between(0,COURT_X)) &
        (df.Y_m.between(0,COURT_Y)) &
        (df.Z_m.between(0,COURT_Z))
    ].reset_index(drop=True)
    print(f"kept {len(court)} / {len(df)} points inside court")

    # write CSV
    csv_out = os.path.join(OUT_DIR, "world3d_court.csv")
    court.to_csv(csv_out, index=False)
    print("✅  wrote", csv_out)

    # plot
    png_out = os.path.join(OUT_DIR, "world3d_court.png")
    fig = plt.figure(figsize=(9,7))
    ax  = fig.add_subplot(111, projection="3d")
    ids = court.get("id", pd.Series(0))
    ax.scatter(court.X_m, court.Y_m, court.Z_m,
               c=ids, cmap="tab20", s=25, alpha=.9)
    ax.set_xlim(0,COURT_X); ax.set_ylim(0,COURT_Y); ax.set_zlim(0,COURT_Z)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    plt.title("3-D positions on real court")
    plt.tight_layout()
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅  wrote", png_out)

if __name__ == "__main__":
    main()
