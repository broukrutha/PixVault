"""
combined.py — All 4 methods stacked with EOT + Face-Only Protection
"""

import torch
import numpy as np
from PIL import Image
import cv2

from protectors.base import BaseProtector, create_face_blend_mask
from protectors.fawkes import FawkesProtector
from protectors.lowkey import LowKeyProtector
from protectors.amt_gan import AMTGANProtector
from protectors.ulixes import UlixesProtector
from utils.image_utils import pil_to_numpy, numpy_to_pil
from utils.metrics import full_report


class CombinedProtector(BaseProtector):
    """
    Chains all 4 protectors sequentially on the face crop only.
    Uses reduced epsilon per stage — combined effect is strong but invisible.
    """

    def __init__(self, device="cpu", epsilon_scale=0.6, steps_scale=0.6, verbose=True):
        super().__init__(device)
        self.name = "⚡ Combined (All 4 Methods)"

        # Each method uses smaller epsilon individually — combined effect stacks
        configs = [
            ("Fawkes",  FawkesProtector,  0.018, 30, 1.0),
            ("LowKey",  LowKeyProtector,  0.018, 25, 1.0),
            ("AMT-GAN", AMTGANProtector,  0.015, 25, 1.0),
            ("Ulixes",  UlixesProtector,  0.018, 30, 1.0),
        ]

        self.pipeline = []
        for name, cls, base_eps, base_steps, scale in configs:
            eps   = round(base_eps * epsilon_scale, 5)
            steps = max(10, int(base_steps * steps_scale))
            p = cls(device=device, epsilon=eps, steps=steps)
            self.pipeline.append((name, p))
            if verbose:
                print(f"  [Combined] Loaded {name} | epsilon={eps} | steps={steps}")

    def _perturb_face_crop(self, face_crop: Image.Image, aligned_tensor: torch.Tensor) -> Image.Image:
        current_crop = face_crop.copy()
        for i, (name, protector) in enumerate(self.pipeline):
            print(f"\n  [Combined] Stage {i+1}/4 — {name}")
            try:
                current_crop = protector._perturb_face_crop(current_crop, aligned_tensor)
                print(f"  [Combined] ✅ {name} done.")
            except Exception as e:
                print(f"  [Combined] ⚠️ {name} skipped: {e}")
        return current_crop

    def protect_with_stage_metrics(self, pil_image: Image.Image):
        original_arr = pil_to_numpy(pil_image)
        orig_emb = self.extractor.get_embedding(pil_image)

        aligned_tensor, box, prob = self.extractor.detect_face_with_box(pil_image)
        if aligned_tensor is None:
            print("[Combined] No face detected.")
            report = full_report(original_arr, original_arr, orig_emb, orig_emb, method=self.name)
            return pil_image, report, []

        x1, y1, x2, y2 = box
        face_crop = pil_image.crop((x1, y1, x2, y2))
        current_crop = face_crop.copy()
        stage_reports = []

        for i, (name, protector) in enumerate(self.pipeline):
            print(f"\n  [Combined] Stage {i+1}/4 — {name}")
            try:
                current_crop = protector._perturb_face_crop(current_crop, aligned_tensor)
                crop_arr = pil_to_numpy(current_crop)
                full_arr = self._paste_face_back(original_arr, crop_arr, box)
                full_pil = numpy_to_pil(full_arr)
                prot_emb = self.extractor.get_embedding(full_pil)
                r = full_report(original_arr, full_arr, orig_emb, prot_emb, method=name)
                stage_reports.append(r)
                cos = r["protection"]["original_vs_protected_cosine"]
                print(f"  [Combined] ✅ {name} | Cosine: {cos}")
            except Exception as e:
                print(f"  [Combined] ⚠️ {name} skipped: {e}")

        final_arr = self._paste_face_back(original_arr, pil_to_numpy(current_crop), box)
        final_pil  = numpy_to_pil(final_arr)
        final_emb  = self.extractor.get_embedding(final_pil)
        overall    = full_report(original_arr, final_arr, orig_emb, final_emb, method=self.name)
        return final_pil, overall, stage_reports


def format_combined_report(overall_report: dict, stage_reports: list) -> str:
    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  ⚡ COMBINED PROTECTION REPORT",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  📈 Per-Stage Cosine Similarity (↓ lower = better)",
        "",
    ]
    prev = 1.0
    for i, r in enumerate(stage_reports):
        cos = r["protection"]["original_vs_protected_cosine"]
        imp = prev - cos if cos else 0
        bar = "█" * max(1, int((1 - cos) * 20)) if cos else ""
        lines.append(f"  Stage {i+1} ({r['method'][:12]:<12}): {cos:.4f}  {bar}  (↓{imp:.4f})")
        prev = cos if cos else prev

    lines += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "  🏆 FINAL COMBINED RESULT",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  SSIM              : {overall_report['image_quality']['ssim']}",
        f"  PSNR              : {overall_report['image_quality']['psnr_db']} dB",
        f"  Cosine Similarity : {overall_report['protection']['original_vs_protected_cosine']}",
        f"  Embedding Shift   : {overall_report['protection']['embedding_euclidean_shift']}",
        f"  Shift %           : {overall_report['protection']['embedding_shift_percent']}%",
        f"  Result            : {overall_report['protection']['protection_accuracy']}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines)