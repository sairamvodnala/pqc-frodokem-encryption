# step123_extract_patches_opencv_dynamic.py
import os, json
import cv2
import numpy as np
import torchvision
import torchvision.transforms as T

# files Config
OUT_DIR   = "outputs_cv2"
PATCH_DIR = os.path.join(OUT_DIR, "patches")
IMG_DIR   = os.path.join(OUT_DIR, "images")
META_DIR  = os.path.join(OUT_DIR, "meta")
os.makedirs(PATCH_DIR, exist_ok=True)
os.makedirs(IMG_DIR,   exist_ok=True)
os.makedirs(META_DIR,  exist_ok=True)

SENSITIVE  = {3, 5}    # here cat=3, dog=5
PATCH_SIZE = 16        # 16x16 ROI on CIFAR-10 (32x32)

# Loading the  CIFAR-10 data
transform = T.ToTensor()
testset   = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
classes   = testset.classes
print("Classes:", classes)

def tensor_to_u8_hwc(img_tensor):
    # [C,H,W] in [0,1]  to [H,W,C] uint8
    x = (img_tensor.clamp(0,1) * 255.0).byte().numpy()
    return np.transpose(x, (1,2,0))

def pick_dynamic_roi_sobel(img_hwc_u8, patch=16):
    """
    so here it Picks top-left (x,y) of patch x patch with maximum Sobel magnitude sum.
    then Returns (x, y, w, h) and the magnitude map (just for debug if needed).
    """
    h, w, _ = img_hwc_u8.shape
    gray = cv2.cvtColor(img_hwc_u8, cv2.COLOR_RGB2GRAY)

    # Sobel grads to magnitude
    gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Fast window sum via box filter (sum, not normalized)
    energy = cv2.boxFilter(mag, ddepth=-1, ksize=(patch, patch), normalize=False)

    # Valid top-lefts: [0..h-patch], [0..w-patch]
    H = h - patch + 1
    W = w - patch + 1
    sub = energy[:H, :W]
    y0, x0 = np.unravel_index(np.argmax(sub), sub.shape)

    return int(x0), int(y0), patch, patch

count = 0
for idx, (img_tensor, label) in enumerate(testset):
    if label not in SENSITIVE:
        continue

    # Converting to HWC uint8 for OpenCV
    img_np = tensor_to_u8_hwc(img_tensor)

    # so, DYNAMIC ROI (content-based)
    x, y0, w, h = pick_dynamic_roi_sobel(img_np, patch=PATCH_SIZE)

    # Cropoing the dynamic patch
    patch_np = img_np[y0:y0+h, x:x+w, :]

    # To Save the full original image (just for reference)
    cv2.imwrite(f"{IMG_DIR}/img_{idx:05d}_{classes[label]}.png",
                cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Saving the patch
    cv2.imwrite(f"{PATCH_DIR}/patch_{idx:05d}_{classes[label]}.png",
                cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR))

    #Saving a visual with the ROI drawn (useful for reports)
    vis = img_np.copy()
    cv2.rectangle(vis, (x, y0), (x+w, y0+h), (255, 0, 0), thickness=1)  # blue box
    cv2.imwrite(f"{IMG_DIR}/img_{idx:05d}_{classes[label]}_box.png",
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    #Saving the metadata for encryption/decryption steps 
    meta = {
        "index": idx,
        "label_id": int(label),
        "label_name": classes[label],
        "image_path": f"{IMG_DIR}/img_{idx:05d}_{classes[label]}.png",
        "patch_path": f"{PATCH_DIR}/patch_{idx:05d}_{classes[label]}.png",
        "roi": {"x": x, "y": y0, "w": w, "h": h},
        "image_size": {"w": int(img_np.shape[1]), "h": int(img_np.shape[0])},
        "patch_size": PATCH_SIZE,
        "policy": "sobel_max_energy"
    }
    with open(f"{META_DIR}/img_{idx:05d}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    count += 1
    if count % 200 == 0:
        print(f"Processed {count} sensitive images...")

print(f"Done. Saved {count} sensitive images + dynamic patches.")
print(f"- Full images: {IMG_DIR}")
print(f"- Patches:     {PATCH_DIR}")
print(f"- Metadata:    {META_DIR}")
