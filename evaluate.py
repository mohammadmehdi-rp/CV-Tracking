import os
import numpy as np
import pandas as pd
import motmetrics as mm
from scipy.optimize import linear_sum_assignment

# ── CONFIG ─────────────────────────────────────────────────────────────────────
# Paths
PARENT_ROOT = 'result/2DTracking/out13/evaluation'
GT_PATH     = os.path.join(PARENT_ROOT, 'gt.txt')
TRK_PATH    = os.path.join(PARENT_ROOT, 'track.txt')

# Scaling factors to bring GT from 640×640 into 3840×2160
SCALE_X = 3840 / 640    # = 6.0
SCALE_Y = 2160 / 640    # = 3.375
# ────────────────────────────────────────────────────────────────────────────────

def load_motchallenge(path, scale=None):
    """
    Load a MOTChallenge-style file with columns:
      frame, id, x, y, w, h[, score]
    Keep only the first 6 columns and optionally rescale X,W by scale[0] and Y,H by scale[1].
    """
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :6]
    df.columns = ['FrameId','Id','X','Y','W','H']
    if scale is not None:
        sx, sy = scale
        df[['X','W']] *= sx
        df[['Y','H']] *= sy
    return df

def iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two arrays of boxes [x,y,w,h].
    Returns an (N1 x N2) array of IoU values.
    """
    N1, N2 = boxes1.shape[0], boxes2.shape[0]
    ious = np.zeros((N1, N2), dtype=float)
    for i in range(N1):
        x1, y1, w1, h1 = boxes1[i]
        xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
        for j in range(N2):
            x2, y2, w2, h2 = boxes2[j]
            xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
            inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
            inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
            inter   = inter_w * inter_h
            union   = w1 * h1 + w2 * h2 - inter
            if union > 0:
                ious[i, j] = inter / union
    return ious

def evaluate_tracking(gt_path, trk_path):
    # Load GT (scaled) and tracker (unscaled)
    gt  = load_motchallenge(gt_path, scale=(SCALE_X, SCALE_Y))
    trk = load_motchallenge(trk_path, scale=None)

    acc = mm.MOTAccumulator(auto_id=True)
    frames = sorted(set(gt.FrameId) | set(trk.FrameId))

    for f in frames:
        g = gt[gt.FrameId == f]
        t = trk[trk.FrameId == f]
        gt_ids  = g.Id.values
        trk_ids = t.Id.values

        if len(gt_ids) and len(trk_ids):
            # compute distance matrix = 1 - IoU
            ious = iou_matrix(
                g[['X','Y','W','H']].values,
                t[['X','Y','W','H']].values
            )
            dists = 1.0 - ious
        else:
            dists = np.empty((len(gt_ids), len(trk_ids)))

        acc.update(gt_ids, trk_ids, dists)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota','idf1'], name='eval')

    # Rename for clarity
    summary = summary.rename(
        index={'eval':'Results'},
        columns={'mota':'MOTA','idf1':'IDF1'}
    )

    print(mm.io.render_summary(
        summary, 
        formatters=mh.formatters, 
        namemap={'MOTA':'MOTA','IDF1':'IDF1'}
    ))

def compute_average_iou(gt_path, trk_path):
    """
    Independently compute average IoU over Hungarian-matched pairs,
    for cross-checking MOTP.
    """
    gt  = pd.read_csv(gt_path, header=None, names=["frame","id","X","Y","W","H"])
    trk = pd.read_csv(trk_path, header=None, names=["frame","id","X","Y","W","H","score"])

    # scale GT to full-res
    gt[['X','W']] *= SCALE_X
    gt[['Y','H']] *= SCALE_Y

    all_ious = []
    frames = sorted(set(gt.frame) & set(trk.frame))
    for f in frames:
        g = gt[gt.frame == f][['X','Y','W','H']].values
        t = trk[trk.frame == f][['X','Y','W','H']].values
        if g.size == 0 or t.size == 0:
            continue

        iou_mat = iou_matrix(g, t)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        for i,j in zip(row_ind, col_ind):
            all_ious.append(iou_mat[i,j])

    avg_iou = float(np.mean(all_ious)) if all_ious else 0.0
    print(f"Average IoU over {len(all_ious)} matches: {avg_iou:.4f}")
    return avg_iou

if __name__ == "__main__":
    evaluate_tracking(GT_PATH, TRK_PATH)
    compute_average_iou(GT_PATH, TRK_PATH)
