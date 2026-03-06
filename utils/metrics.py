"""
metrics.py
----------
Computes image quality and protection success metrics.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import cv2


def compute_ssim(original: np.ndarray, protected: np.ndarray) -> float:
    """Structural Similarity Index [0, 1]. Higher = more visually similar."""
    # Convert to grayscale for SSIM
    if original.ndim == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        prot_gray = cv2.cvtColor(protected, cv2.COLOR_RGB2GRAY)
    else:
        orig_gray, prot_gray = original, protected
    score = ssim(orig_gray, prot_gray, data_range=255)
    return round(float(score), 4)


def compute_psnr(original: np.ndarray, protected: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB. Higher = better quality."""
    score = psnr(original, protected, data_range=255)
    return round(float(score), 2)


def compute_l2_perturbation(original: np.ndarray, protected: np.ndarray) -> float:
    """Mean L2 perturbation per pixel (how much we changed)."""
    diff = original.astype(np.float32) - protected.astype(np.float32)
    return round(float(np.mean(np.abs(diff))), 4)


def compute_embedding_shift(orig_emb: np.ndarray, prot_emb: np.ndarray) -> dict:
    """
    Measures how much the face embedding shifted.
    Larger shift = better protection.
    """
    if orig_emb is None or prot_emb is None:
        return {"cosine_similarity": None, "euclidean_distance": None, "shift_percentage": None}

    cos_sim = float(
        np.dot(orig_emb, prot_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(prot_emb) + 1e-8)
    )
    euc_dist = float(np.linalg.norm(orig_emb - prot_emb))

    # Shift percentage: how far the embedding moved relative to its norm
    shift_pct = euc_dist / (np.linalg.norm(orig_emb) + 1e-8) * 100

    return {
        "cosine_similarity": round(cos_sim, 4),       # < 0.7 means face NOT recognized
        "euclidean_distance": round(euc_dist, 4),
        "shift_percentage": round(shift_pct, 2),
    }


def protection_success(cosine_similarity: float, threshold: float = 0.7) -> bool:
    """
    Returns True if the face recognition model would FAIL to match
    the protected image against the original (i.e., protection worked).
    """
    return cosine_similarity < threshold


def full_report(
    original_arr: np.ndarray,
    protected_arr: np.ndarray,
    orig_emb: np.ndarray,
    prot_emb: np.ndarray,
    method: str = "Unknown",
) -> dict:
    """Generates a full quality + protection metrics report."""
    emb_metrics = compute_embedding_shift(orig_emb, prot_emb)
    cos_sim = emb_metrics["cosine_similarity"]
    success = protection_success(cos_sim) if cos_sim is not None else False

    report = {
        "method": method,
        "image_quality": {
            "ssim": compute_ssim(original_arr, protected_arr),
            "psnr_db": compute_psnr(original_arr, protected_arr),
            "mean_pixel_change": compute_l2_perturbation(original_arr, protected_arr),
        },
        "protection": {
            "original_vs_protected_cosine": emb_metrics["cosine_similarity"],
            "embedding_euclidean_shift": emb_metrics["euclidean_distance"],
            "embedding_shift_percent": emb_metrics["shift_percentage"],
            "face_recognition_fooled": success,
            "protection_accuracy": f"{'✅ SUCCESS' if success else '⚠️ PARTIAL'} (threshold=0.7)",
        },
    }
    return report


def format_report(report: dict) -> str:
    """Human-readable report string."""
    lines = [
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  Protection Method : {report['method']}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  📸 IMAGE QUALITY (higher = more natural looking)",
        f"     SSIM Score        : {report['image_quality']['ssim']} / 1.0",
        f"     PSNR              : {report['image_quality']['psnr_db']} dB",
        f"     Mean Pixel Change : {report['image_quality']['mean_pixel_change']} px",
        f"",
        f"  🛡️  PROTECTION STRENGTH",
        f"     Cosine Similarity : {report['protection']['original_vs_protected_cosine']}",
        f"     Embedding Shift   : {report['protection']['embedding_euclidean_shift']}",
        f"     Shift %           : {report['protection']['embedding_shift_percent']}%",
        f"     Result            : {report['protection']['protection_accuracy']}",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines)