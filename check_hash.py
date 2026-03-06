"""
check_hash.py
-------------
Command-line tool to verify if an image has been deepfaked.

Usage:
  python check_hash.py --image path/to/image.jpg
  python check_hash.py --image path/to/image.jpg --hash path/to/image.jpg.sha256
  python check_hash.py --image img.jpg --raw-hash abc123...

Examples:
  # Auto-detect hash from sidecar file:
  python check_hash.py --image output/protected_fawkes.jpg

  # Provide hash file explicitly:
  python check_hash.py --image image.jpg --hash image.jpg.sha256

  # Provide hash string directly:
  python check_hash.py --image image.jpg --raw-hash a3f1...
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hash_protection.hasher import verify_image_hash, format_verification_result, compute_image_hash
from PIL import Image


def main():
    parser = argparse.ArgumentParser(
        description="FaceProtect Hash Checker — Detect deepfakes via SHA-256 hash verification"
    )
    parser.add_argument("--image", required=True, help="Path to the image to verify")
    parser.add_argument("--hash", default=None, help="Path to the .sha256 sidecar file")
    parser.add_argument("--raw-hash", default=None, help="Expected SHA-256 hash string (64 hex chars)")
    parser.add_argument("--print-hash", action="store_true", help="Just print the current image hash")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)

    # Just print hash mode
    if args.print_hash:
        pil_img = Image.open(args.image)
        hash_val = compute_image_hash(pil_img)
        print(f"SHA-256 pixel hash of '{args.image}':")
        print(hash_val)
        sys.exit(0)

    # Load expected hash from file if provided
    expected_hash = args.raw_hash
    if args.hash and expected_hash is None:
        import json
        with open(args.hash) as f:
            data = json.load(f)
        expected_hash = data.get("pixel_hash_sha256")
        print(f"  Loaded hash from: {args.hash}")

    # Run verification
    result = verify_image_hash(args.image, expected_hash)
    print("\n" + format_verification_result(result))

    # Exit code: 0 = authentic, 1 = modified/deepfaked, 2 = unknown
    if result["match"]:
        sys.exit(0)
    elif result["original_hash"] is None:
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()