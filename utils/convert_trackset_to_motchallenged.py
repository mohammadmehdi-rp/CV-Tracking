import os, glob, pandas as pd

def convert_track_to_mot(
    txt_folder:str,
    out_path:str,
    orig_fps:int=25,
    tgt_fps:int=5
):
    step = orig_fps // tgt_fps
    if step < 1:
        raise ValueError("orig_fps must be >= tgt_fps")

    # make sure output dir exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    files = sorted(glob.glob(os.path.join(txt_folder, "*.txt")))
    print(f"Found {len(files)} label files in {txt_folder}")

    for path in files:
        base   = os.path.basename(path)
        frame0 = int(os.path.splitext(base)[0])
        frame  = frame0 + 1
        if (frame - 1) % step != 0:
            continue

        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                # expect exactly 7 parts: class,x1,y1,x2,y2,tid,score
                if len(parts) < 7:
                    continue

                # unpack
                _, x1, y1, x2, y2, tid, score = parts
                x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
                tid            = int(tid)
                score          = float(score)

                w, h = x2 - x1, y2 - y1
                rows.append([frame, tid, x1, y1, w, h, score])

    # build DF & dump
    cols = ["frame","id","x","y","w","h","score"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_path, index=False, header=False, float_format="%.3f")
    print(f"Wrote {len(df)} detections across {df['frame'].nunique()} frames to {out_path}")


if __name__ == "__main__":
    convert_track_to_mot(
      txt_folder="result/2DTracking/out13/labels",
      out_path  ="result/2DTracking/out13/evaluation/track.txt",
      orig_fps=25,
      tgt_fps=5
    )
