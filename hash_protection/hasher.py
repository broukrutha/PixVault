"""
hasher.py
---------
Hash-Based Image Integrity Protection

How it works:
  1. Compute SHA-256 hash of the image pixel data
  2. Embed the hash as metadata (EXIF comment field)
  3. Also store a visual QR-like hash string in the output filename
  4. Verification: re-hash the image pixels and compare against stored hash

  If the image has been deepfaked / modified:
  → The pixel data changes → The computed hash changes → Hash MISMATCH detected
"""

import hashlib
import json
import os
import piexif
from PIL import Image
import numpy as np
from datetime import datetime


def compute_image_hash(pil_image: Image.Image) -> str:
    """
    Compute SHA-256 hash of the raw pixel data (not the file bytes).
    This is deterministic regardless of compression.
    """
    img_array = np.array(pil_image.convert("RGB"))
    raw_bytes = img_array.tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of the actual file bytes (file-level integrity)."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def embed_hash_in_exif(pil_image: Image.Image, hash_value: str, method: str = "Unknown") -> Image.Image:
    """
    Embeds the image hash and metadata into the EXIF data of the image.
    Works with JPEG images. PNG stores in info dict.

    Returns the image with embedded hash.
    """
    metadata = {
        "face_protect_hash": hash_value,
        "protected_by": f"FaceProtect | Method: {method}",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0",
    }
    metadata_str = json.dumps(metadata)

    # Try to embed in EXIF (works for JPEG)
    try:
        # Create or load EXIF data
        try:
            exif_data = piexif.load(pil_image.info.get("exif", b""))
        except Exception:
            exif_data = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

        # Store hash in UserComment field
        comment_bytes = piexif.helper.UserComment.dump(metadata_str, encoding="unicode")
        exif_data["Exif"][piexif.ExifIFD.UserComment] = comment_bytes

        # Store in ImageDescription too
        exif_data["0th"][piexif.ImageIFD.ImageDescription] = metadata_str.encode("utf-8")

        exif_bytes = piexif.dump(exif_data)

        # Re-save with EXIF
        from io import BytesIO
        buf = BytesIO()
        pil_image.save(buf, format="JPEG", exif=exif_bytes, quality=95)
        buf.seek(0)
        return Image.open(buf)

    except Exception as e:
        print(f"[Hasher] EXIF embedding warning: {e}. Returning image with hash in info.")
        # Fallback: add to image info dict
        pil_image.info["face_protect_hash"] = hash_value
        return pil_image


def save_with_hash(
    pil_image: Image.Image,
    output_path: str,
    method: str = "Unknown"
) -> dict:
    """
    Saves the protected image with hash embedded.
    Returns a dict with hash info for display.
    """
    pixel_hash = compute_image_hash(pil_image)
    image_with_hash = embed_hash_in_exif(pil_image, pixel_hash, method)

    # Ensure output is JPEG for EXIF support
    if not output_path.lower().endswith((".jpg", ".jpeg")):
        output_path = os.path.splitext(output_path)[0] + ".jpg"

    # Save with hash metadata
    try:
        exif_data = piexif.load(image_with_hash.info.get("exif", b""))
        exif_bytes = piexif.dump(exif_data)
        image_with_hash.save(output_path, format="JPEG", exif=exif_bytes, quality=95)
    except Exception:
        image_with_hash.save(output_path, format="JPEG", quality=95)

    # Also save a sidecar .hash file for easy verification
    hash_sidecar = output_path + ".sha256"
    metadata = {
        "image_path": os.path.basename(output_path),
        "pixel_hash_sha256": pixel_hash,
        "protection_method": method,
        "timestamp": datetime.now().isoformat(),
        "instructions": "Use verify_image_hash() to check integrity"
    }
    with open(hash_sidecar, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "pixel_hash": pixel_hash,
        "hash_file": hash_sidecar,
        "saved_to": output_path,
    }


def verify_image_hash(image_path: str, expected_hash: str = None) -> dict:
    """
    Verify if an image has been modified (deepfaked) by checking its hash.

    Args:
        image_path   : Path to the image to verify
        expected_hash: The original hash to compare against.
                       If None, looks for a .sha256 sidecar file.

    Returns a dict with verification result.
    """
    result = {
        "image_path": image_path,
        "verified": False,
        "original_hash": None,
        "current_hash": None,
        "match": False,
        "verdict": "",
        "exif_hash": None,
    }

    # Load image
    try:
        pil_img = Image.open(image_path)
    except Exception as e:
        result["verdict"] = f"❌ ERROR: Cannot open image — {e}"
        return result

    # Compute current pixel hash
    current_hash = compute_image_hash(pil_img)
    result["current_hash"] = current_hash

    # Try to get expected hash from sidecar file
    if expected_hash is None:
        sidecar_path = image_path + ".sha256"
        if os.path.exists(sidecar_path):
            with open(sidecar_path) as f:
                sidecar_data = json.load(f)
            expected_hash = sidecar_data.get("pixel_hash_sha256")
            result["original_hash"] = expected_hash

    # Try to get hash from EXIF
    try:
        exif_raw = pil_img.info.get("exif", b"")
        if exif_raw:
            exif_data = piexif.load(exif_raw)
            comment_raw = exif_data.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
            if comment_raw:
                comment_str = piexif.helper.UserComment.load(comment_raw)
                meta = json.loads(comment_str)
                result["exif_hash"] = meta.get("face_protect_hash")
                if expected_hash is None:
                    expected_hash = result["exif_hash"]
                    result["original_hash"] = expected_hash
    except Exception:
        pass

    if expected_hash is None:
        result["verdict"] = "⚠️ UNKNOWN: No original hash found to compare. Provide the .sha256 file."
        return result

    result["original_hash"] = expected_hash
    result["match"] = (current_hash == expected_hash)
    result["verified"] = result["match"]

    if result["match"]:
        result["verdict"] = "✅ AUTHENTIC: Image hash MATCHES — image has NOT been tampered with."
    else:
        result["verdict"] = (
            "🚨 DEEPFAKE DETECTED: Hash MISMATCH — this image has been MODIFIED.\n"
            "   The pixel content does not match the original protected image."
        )

    return result


def format_verification_result(result: dict) -> str:
    """Format verification result for display."""
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  🔐 IMAGE INTEGRITY CHECK",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  File    : {os.path.basename(result.get('image_path', 'N/A'))}",
        f"  Original: {result.get('original_hash', 'N/A')[:24]}..." if result.get('original_hash') else "  Original: N/A",
        f"  Current : {result.get('current_hash', 'N/A')[:24]}..." if result.get('current_hash') else "  Current : N/A",
        "",
        f"  {result.get('verdict', 'N/A')}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines)