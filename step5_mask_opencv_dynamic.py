# step5_masking_opencv.py
import os, json, cv2
from pathlib import Path
import numpy as np

IMG_DIR   = Path("outputs_cv2/images")   # originals from step 3
META_DIR  = Path("outputs_cv2/meta")     # per-image ROI from step 3 (dynamic)
MASK_DIR  = Path("outputs_cv2/masked")   # To store the dynamic
MASK_DIR.mkdir(parents=True, exist_ok=True)

#  mask styles
def mask_black(img, x, y, w, h):
    out = img.copy()
    cv2.rectangle(out, (x, y), (x+w, y+h), (0,0,0), thickness=-1)
    return out

def mask_blur(img, x, y, w, h, k=9):
    out = img.copy()
    roi = out[y:y+h, x:x+w]
    k = max(3, k | 1)  # make odd
    out[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k, k), 0)
    return out

def mask_pixelate(img, x, y, w, h, block=4):
    out = img.copy()
    roi = out[y:y+h, x:x+w]
    sh, sw = roi.shape[:2]
    bh, bw = max(1, sh//block), max(1, sw//block)
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    out[y:y+h, x:x+w] = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_NEAREST)
    return out

MASK_MODE = "black"  # we can also use "blur" or "pixelate" but for now just black

def apply_mask(img, x, y, w, h):
    if MASK_MODE == "black":
        return mask_black(img, x, y, w, h)
    if MASK_MODE == "blur":
        return mask_blur(img, x, y, w, h, k=9)
    if MASK_MODE == "pixelate":
        return mask_pixelate(img, x, y, w, h, block=4)
    return mask_black(img, x, y, w, h)

def main():
    metas = sorted(META_DIR.glob("img_*.meta.json"))
    if not metas:
        raise SystemExit(f"No meta files in {META_DIR}. Did you run Step 3 dynamic extractor?")

    count = 0
    for m in metas:
        meta = json.loads(m.read_text())
        # paths are saved in meta are RGB pngs; OpenCV reads BGR (its fine for masking)
        img_path = Path(meta["image_path"])
        roi = meta["roi"]  # {"x": x, "y": y0, "w": w, "h": h}

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        # This is a safety clamp in case of edge cases!
        h_img, w_img = img.shape[:2]
        x = max(0, min(x, w_img-1))
        y = max(0, min(y, h_img-1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        masked = apply_mask(img, x, y, w, h)
        out_path = MASK_DIR / img_path.name  # To keep same filename
        cv2.imwrite(str(out_path), masked)

        count += 1
        if count % 200 == 0:
            print(f"Masked {count} images...")

    print(f"Done. Saved {count} masked images to {MASK_DIR}")

if __name__ == "__main__":
    main()
