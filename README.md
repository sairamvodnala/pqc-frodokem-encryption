# Sensitivity-Based Image Region Encryption Pipeline Using FrodoKEM

**Author:** Sai Ram Vodnala  
**Date:** August 12, 2025

## Overview
This project implements a **proof-of-concept pipeline** for protecting sensitive regions in images using **post-quantum cryptography**.

It demonstrates:
1. **Dynamic sensitive region detection** in CIFAR-10 images (cats & dogs).
2. **Patch extraction** of the sensitive area (16×16 pixels).
3. **Encryption** of the patch using **FrodoKEM (LWE-based)** with AES-256-GCM.
4. **Visual masking** (black box) of the same region in the original image.

The approach allows:
- **Secure storage/transmission** of the sensitive patch.
- **Public sharing** of the masked image without revealing sensitive details.

---

## Pipeline Steps

### Step 1 — Loading CIFAR-10
Loading CIFAR-10 (TorchVision) test set and filtering for sensitive classes (`cat`, `dog`).

### Step 2 — Detecting Sensitive Regions
Using **Sobel gradient magnitude** to find the most textured 16×16 patch in the image.

### Step 3 — Extracting & Saving Patches
Save:
- Original image
- Sensitive patch
- ROI overlay visualization
- Metadata JSON (coordinates, label, file paths)

### Step 4 — Encrypting Patches (Post-Quantum)
Encrypt each patch using:
- **FrodoKEM-640-AES** for key encapsulation.
- **HKDF-SHA256** to derive a 32-byte AES key.
- **AES-256-GCM** for symmetric encryption.

Artifacts saved:
- `*.kem_ct.bin` — FrodoKEM ciphertext
- `*.patch.enc` — AES-GCM encrypted patch (nonce||ciphertext)
- `*.meta.json` — encryption metadata
- `manifest.json` — summary of all encryptions

### Step 5 — Masking Regions in Images
Using ROI coordinates from metadata, apply a **black-box mask** and save to `outputs_cv2/masked/`.

---

## Directory Structure

```plaintext
outputs_cv2/
├── images/        # Original images + ROI overlays
├── patches/       # Extracted sensitive patches
├── meta/          # Metadata JSON files
├── encrypted/     # Encrypted patches + manifest.json
└── masked/        # Masked images (black box)
```
---

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>

2. Create a Python virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate     # Linux/Mac
    venv\Scripts\activate        # Windows

3. Install dependencies:
    ```bash
    pip install -r requirements.txt

Download CIFAR-10 dataset automatically when running Step 1 script.

---

## Usage

1. Step 1–3: Detect & Extract Patches
   ```bash
   python step123_extract_patches_opencv_dynamic.py

2. Step 4: Encrypt Patches
   ```bash
   python step4_encrypt_patches_frodokem.py

3. Step 5: Mask Regions
   ```bash
   python step5_mask_replace_opencv.py

---

**License**
Open Source -- feel free to modify and use for research and educational purposes.
