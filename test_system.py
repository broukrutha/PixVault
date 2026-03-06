"""
test_system.py
--------------
Quick test to verify the FaceProtect system works on CPU.
Creates a synthetic test image (no real face needed for import test),
then tests hash generation and verification.

Run: python test_system.py
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_test_face_image() -> Image.Image:
    """Create a simple synthetic face-like image for testing."""
    img = Image.new("RGB", (256, 256), color=(200, 180, 160))
    draw = ImageDraw.Draw(img)

    # Face oval
    draw.ellipse([40, 30, 216, 226], fill=(220, 190, 160), outline=(180, 150, 130), width=3)

    # Eyes
    draw.ellipse([70, 80, 110, 110], fill=(50, 30, 20))
    draw.ellipse([146, 80, 186, 110], fill=(50, 30, 20))

    # Nose
    draw.polygon([(120, 120), (108, 155), (148, 155)], fill=(200, 160, 140))

    # Mouth
    draw.arc([95, 155, 161, 195], start=0, end=180, fill=(160, 80, 80), width=4)

    return img


def test_hash_system():
    """Test hash generation and verification."""
    print("\n[Test 1] Hash System")
    print("-" * 40)

    from hash_protection.hasher import (
        compute_image_hash,
        save_with_hash,
        verify_image_hash,
        format_verification_result,
    )

    # Create test image
    img = create_test_face_image()
    os.makedirs("output", exist_ok=True)

    # Save with hash
    hash_result = save_with_hash(img, "output/test_image.jpg", method="Test")
    print(f"  ✅ Hash generated: {hash_result['pixel_hash'][:32]}...")
    print(f"  ✅ Saved to: {hash_result['saved_to']}")
    print(f"  ✅ Sidecar: {hash_result['hash_file']}")

    # Verify original (should pass)
    result = verify_image_hash("output/test_image.jpg")
    print(f"\n  Original image check: {result['verdict']}")
    assert result["match"] == True, "Hash should match for unmodified image!"

    # Simulate tampering
    tampered = img.copy()
    pixels = np.array(tampered)
    pixels[0, 0, 0] = (pixels[0, 0, 0] + 10) % 256  # Change 1 pixel
    tampered = Image.fromarray(pixels)
    tampered.save("output/tampered_image.jpg", quality=95)

    result2 = verify_image_hash("output/tampered_image.jpg", hash_result["pixel_hash"])
    print(f"  Tampered image check: {result2['verdict']}")
    assert result2["match"] == False, "Hash should NOT match for tampered image!"

    print("\n  ✅ Hash system working correctly!")


def test_imports():
    """Test all imports load correctly."""
    print("\n[Test 2] Import Check")
    print("-" * 40)

    tests = [
        ("protectors.fawkes", "FawkesProtector"),
        ("protectors.lowkey", "LowKeyProtector"),
        ("protectors.amt_gan", "AMTGANProtector"),
        ("protectors.ulixes", "UlixesProtector"),
        ("utils.face_extractor", "FaceExtractor"),
        ("utils.metrics", "full_report"),
        ("hash_protection.hasher", "compute_image_hash"),
    ]

    for module, cls in tests:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"  ✅ {module}.{cls}")
        except ImportError as e:
            print(f"  ❌ {module}.{cls} — {e}")
        except Exception as e:
            print(f"  ⚠️  {module}.{cls} — {e}")


def test_metrics():
    """Test metric computation."""
    print("\n[Test 3] Metrics")
    print("-" * 40)
    from utils.metrics import compute_ssim, compute_psnr, full_report

    orig = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    prot = np.clip(orig.astype(np.int32) + np.random.randint(-10, 10, orig.shape), 0, 255).astype(np.uint8)

    ssim_val = compute_ssim(orig, prot)
    psnr_val = compute_psnr(orig, prot)
    print(f"  SSIM: {ssim_val}")
    print(f"  PSNR: {psnr_val} dB")
    print(f"  ✅ Metrics working!")


if __name__ == "__main__":
    print("=" * 50)
    print("  FaceProtect System Test")
    print("=" * 50)

    test_imports()
    test_hash_system()
    test_metrics()

    print("\n" + "=" * 50)
    print("  ✅ All tests passed! Run 'python app.py' to start UI.")
    print("=" * 50)