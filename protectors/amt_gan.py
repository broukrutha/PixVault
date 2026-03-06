"""
amt_gan.py — AMT-GAN with EOT + Face-Only Protection
Purely adversarial — no color tinting, just invisible perturbation in makeup regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from protectors.base import BaseProtector
from utils.image_utils import numpy_to_pil, pil_to_numpy
from utils.eot import EOT


class AMTGANProtector(BaseProtector):
    def __init__(self, device="cpu", epsilon=0.03, steps=35, step_size=0.003, makeup_alpha=0.0, n_eot=6):
        super().__init__(device)
        self.name = "AMT-GAN (Adversarial Makeup)"
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.makeup_alpha = makeup_alpha
        self.eot = EOT(n_transforms=n_eot)

    def _create_makeup_mask_160(self) -> torch.Tensor:
        h, w = 160, 160
        mask = np.zeros((h, w), dtype=np.float32)
        mask[int(h*0.72):int(h*0.80), int(w*0.38):int(w*0.62)] = 1.0  # lips
        mask[int(h*0.30):int(h*0.40), int(w*0.15):int(w*0.42)] = 1.0  # left eye
        mask[int(h*0.30):int(h*0.40), int(w*0.58):int(w*0.85)] = 1.0  # right eye
        mask[int(h*0.52):int(h*0.60), int(w*0.08):int(w*0.22)] = 0.35  # cheeks
        mask[int(h*0.52):int(h*0.60), int(w*0.78):int(w*0.92)] = 0.35
        mask = cv2.GaussianBlur(mask, (11, 11), sigmaX=3)
        return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)

    def _perturb_face_crop(self, face_crop: Image.Image, aligned_tensor: torch.Tensor) -> Image.Image:
        face = aligned_tensor.unsqueeze(0)
        mask_tensor = self._create_makeup_mask_160()

        with torch.no_grad():
            orig_emb = self.extractor.resnet(face)

        delta = torch.zeros_like(face, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)

        for step in range(self.steps):
            optimizer.zero_grad()
            masked_delta = delta * mask_tensor
            perturbed = torch.clamp(face + masked_delta, -1.0, 1.0)
            loss = self.eot.get_eot_loss(self.extractor.resnet, perturbed, orig_emb)
            loss += 0.1 * torch.mean(torch.abs(delta * (1 - mask_tensor)))
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

            if (step + 1) % 10 == 0:
                with torch.no_grad():
                    md = (delta * mask_tensor).detach()
                    cos = F.cosine_similarity(self.extractor.resnet(torch.clamp(face + md, -1, 1)), orig_emb).item()
                print(f"  [AMT-GAN+EOT] Step {step+1}/{self.steps} | cos_sim={cos:.4f}")

        with torch.no_grad():
            final_delta = (delta * mask_tensor).detach().squeeze(0).numpy()
            delta_hwc = np.transpose(final_delta, (1, 2, 0)) * 127.5
            # Extra smooth to make invisible
            delta_hwc = cv2.GaussianBlur(delta_hwc, (5, 5), sigmaX=1.5)

        crop_w, crop_h = face_crop.size
        delta_resized = cv2.resize(delta_hwc, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        crop_arr = pil_to_numpy(face_crop).astype(np.float32)
        protected = np.clip(crop_arr + delta_resized, 0, 255).astype(np.uint8)
        return numpy_to_pil(protected)