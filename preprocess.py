import cv2
import os
import glob
import shutil

def batch_clahe(input_img_dir, output_img_dir, clip_limit=2.0, tile_grid=(8,8), exts=('jpg','jpeg','png')):
    """
    Apply CLAHE to all images in input_img_dir and save results to output_img_dir,
    preserving filenames.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)

    for ext in exts:
        for img_path in glob.glob(os.path.join(input_img_dir, f'*.{ext}')):
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️  Skipping unreadable file: {img_path}")
                continue

            lab      = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b  = cv2.split(lab)
            l_eq     = clahe.apply(l)
            lab_eq   = cv2.merge([l_eq, a, b])
            img_eq   = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

            filename = os.path.basename(img_path)
            save_path = os.path.join(output_img_dir, filename)
            cv2.imwrite(save_path, img_eq)
            print(f"✔️  Saved: {save_path}")

def copy_labels(input_label_dir, output_label_dir):
    """
    Copy the entire labels folder (and its contents) to the new output directory.
    """
    if os.path.exists(output_label_dir):
        shutil.rmtree(output_label_dir)
    shutil.copytree(input_label_dir, output_label_dir)
    print(f"📋 Labels folder copied to: {output_label_dir}")


INPUT_IMG_DIR   = "train/images"
INPUT_LABEL_DIR = "train/labels"
OUTPUT_ROOT     = "preprocessed-train"
    
# output sub-folders
OUTPUT_IMG_DIR   = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_LABEL_DIR = os.path.join(OUTPUT_ROOT, "labels")

# 1) Process images
batch_clahe(INPUT_IMG_DIR, OUTPUT_IMG_DIR, clip_limit=2.0, tile_grid=(8,8))

# 2) Copy labels folder
copy_labels(INPUT_LABEL_DIR, OUTPUT_LABEL_DIR)
