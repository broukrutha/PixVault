"""
fawkes.py — Fawkes with EOT + Face-Only Protection
Invisible cloaking: strong enough to fool models, subtle enough to be imperceptible.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from protectors.base import BaseProtector
from utils.image_utils import tensor_to_numpy, numpy_to_pil, pil_to_numpy
from utils.eot import EOT


class FawkesProtector(BaseProtector):
    def __init__(self, device="cpu", epsilon=0.03, steps=40, step_size=0.002, n_eot=6):
        super().__init__(device)
        self.name = "Fawkes (Data Poisoning Cloak)"
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.eot = EOT(n_transforms=n_eot)

    def _perturb_face_crop(self, face_crop: Image.Image, aligned_tensor: torch.Tensor) -> Image.Image:
        face = aligned_tensor.unsqueeze(0)

        with torch.no_grad():
            orig_emb = self.extractor.resnet(face)

        torch.manual_seed(42)
        target_emb = F.normalize(torch.randn_like(orig_emb), dim=1)
        if F.cosine_similarity(orig_emb, target_emb).item() > -0.2:
            target_emb = -target_emb

        delta = torch.zeros_like(face, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)

        for step in range(self.steps):
            optimizer.zero_grad()
            perturbed = torch.clamp(face + delta, -1.0, 1.0)
            loss = self.eot.get_eot_loss(self.extractor.resnet, perturbed, orig_emb, target_emb)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

            if (step + 1) % 10 == 0:
                with torch.no_grad():
                    cos = F.cosine_similarity(self.extractor.resnet(torch.clamp(face + delta, -1, 1)), orig_emb).item()
                print(f"  [Fawkes+EOT] Step {step+1}/{self.steps} | cos_sim={cos:.4f}")

        with torch.no_grad():
            delta_np = delta.squeeze(0).numpy()
            delta_hwc = np.transpose(delta_np, (1, 2, 0)) * 127.5

        # Smooth the delta to reduce visible structure
        delta_hwc = cv2.GaussianBlur(delta_hwc, (3, 3), sigmaX=0.8)

        crop_w, crop_h = face_crop.size
        delta_resized = cv2.resize(delta_hwc, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        crop_arr = pil_to_numpy(face_crop).astype(np.float32)
        protected = np.clip(crop_arr + delta_resized, 0, 255).astype(np.uint8)
        return numpy_to_pil(protected)