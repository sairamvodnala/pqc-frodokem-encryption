# step4_encrypt_patches_frodokem.py
import os
import json
import secrets
from pathlib import Path

import cv2
import numpy as np

# PQC KEM (liboqs-python 0.14 API)
import oqs

# Symmetric AEAD + KDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

PATCH_DIR = Path("outputs_cv2/patches")   # from Step 3 (OpenCV)
ENC_DIR   = Path("outputs_cv2/encrypted")
ENC_DIR.mkdir(parents=True, exist_ok=True)

# Choosing the FrodoKEM parameter set
FRODO_KEM = "FrodoKEM-640-AES"  # its a good balance for demo as it is POC.

def read_patch_bytes(patch_path: Path):
    """Loading 16x16 RGB patch with OpenCV and return (bytes, shape)."""
    img_bgr = cv2.imread(str(patch_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to load {patch_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape  # should be 16,16,3
    return img_rgb.tobytes(), (h, w, c)

def hkdf32(secret: bytes) -> bytes:
    """Derive a 32-byte key from the KEM shared secret using HKDF-SHA256."""
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"patch-aead-v1",
    ).derive(secret)

def main():
    # To Ensure Frodo is actually enabled in your liboqs build
    enabled = set(oqs.get_enabled_kem_mechanisms())
    if FRODO_KEM not in enabled:
        raise RuntimeError(
            f"{FRODO_KEM} not enabled in liboqs. Enabled: "
            f"{sorted([m for m in enabled if 'Frodo' in m])}"
        )

    patch_paths = sorted(PATCH_DIR.glob("patch_*.png"))
    if not patch_paths:
        raise RuntimeError(f"No patches found in {PATCH_DIR}. Did you run Step 3?")

    manifest = []

    # 1) Receiver (server) keypair once (secret key is kept inside the object in 0.14)
    with oqs.KeyEncapsulation(FRODO_KEM) as kem_receiver, oqs.KeyEncapsulation(FRODO_KEM) as kem_sender:
        public_key = kem_receiver.generate_keypair()  # 0.14 returns ONLY public key

        # 2) Iterating the patches and encrypting each
        for patch_path in patch_paths:
            patch_bytes, shape = read_patch_bytes(patch_path)

            # KEM: sender encapsulates to receiver's public key (0.14 API)
            # returns (ciphertext, shared_secret)
            ciphertext_kem, shared_secret_sender = kem_sender.encap_secret(public_key)

            # DEM: derive AES-256 key via HKDF from the shared secret
            key = hkdf32(shared_secret_sender)  # 32 bytes
            aesgcm = AESGCM(key)
            nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
            ct = aesgcm.encrypt(nonce, patch_bytes, associated_data=None)

            # 3)Now, Saving the artifacts
            out_base = ENC_DIR / patch_path.stem  # e.g., outputs_cv2/encrypted/patch_00000_cat
            out_base.parent.mkdir(parents=True, exist_ok=True)

            kem_ct_path = out_base.with_suffix(".kem_ct.bin")
            aead_path   = out_base.with_suffix(".patch.enc")
            meta_path   = out_base.with_suffix(".meta.json")

            kem_ct_path.write_bytes(ciphertext_kem)
            aead_path.write_bytes(nonce + ct)  # to store nonce||ciphertext

            meta = {
                "patch_png": str(patch_path.name),
                "shape": {"h": shape[0], "w": shape[1], "c": shape[2]},
                "kem": {
                    "alg": FRODO_KEM,
                    "kem_ciphertext_len": len(ciphertext_kem),
                    "shared_secret_len": len(shared_secret_sender),
                },
                "aead": {"scheme": "AES-256-GCM", "nonce_len": 12, "ciphertext_len": len(ct)},
                "kdf": {"type": "HKDF-SHA256", "length": 32, "info": "patch-aead-v1"},
            }
            meta_path.write_text(json.dumps(meta, indent=2))

            manifest.append({
                "id": patch_path.stem,
                "kem_ct": kem_ct_path.name,
                "aead": aead_path.name,
                "meta": meta_path.name
            })

    # 4) Writing  a top-level manifest
    (ENC_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Done. Encrypted {len(manifest)} patches with {FRODO_KEM}.")
    print(f"Artifacts in: {ENC_DIR}")
    print("Files per patch: *.kem_ct.bin (FrodoKEM ct), *.patch.enc (nonce||AES‑GCM ct), *.meta.json")

if __name__ == "__main__":
    main()
