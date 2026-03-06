"""
ulixes.py — Ulixes with EOT + Face-Only Protection
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from protectors.base import BaseProtector
from utils.image_utils import numpy_to_pil, pil_to_numpy
from utils.eot import EOT


class UlixesProtector(BaseProtector):
    def __init__(self, device="cpu", epsilon=0.03, steps=45, step_size=0.003, margin=0.5, n_eot=6):
        super().__init__(device)
        self.name = "Ulixes (Embedding Cluster Attack)"
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.margin = margin
        self.eot = EOT(n_transforms=n_eot)

    def _generate_impostor(self, orig_emb):
        torch.manual_seed(123)
        imp = F.normalize(torch.randn_like(orig_emb), dim=1)
        for _ in range(100):
            if F.cosine_similarity(orig_emb, imp).item() <= -0.2:
                break
            imp = F.normalize(torch.randn_like(orig_emb), dim=1)
        return imp

    def _triplet_loss(self, anchor, positive, negative):
        d_anchor   = 1.0 - F.cosine_similarity(anchor, positive)
        d_impostor = 1.0 - F.cosine_similarity(negative, positive)
        return torch.clamp(d_impostor - d_anchor + self.margin, min=0).mean()

    def _perturb_face_crop(self, face_crop: Image.Image, aligned_tensor: torch.Tensor) -> Image.Image:
        face = aligned_tensor.unsqueeze(0)
        with torch.no_grad():
            orig_emb = self.extractor.resnet(face)

        impostor_emb = self._generate_impostor(orig_emb)

        delta = torch.zeros_like(face, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.step_size)
        best_delta, best_dist = delta.detach().clone(), 0.0

        for step in range(self.steps):
            optimizer.zero_grad()
            perturbed = torch.clamp(face + delta, -1.0, 1.0)
            avg_emb = self.eot.get_eot_embedding(self.extractor.resnet, perturbed)
            loss = self._triplet_loss(orig_emb.detach(), avg_emb, impostor_emb.detach())
            loss += 0.5 * F.cosine_similarity(avg_emb, orig_emb.detach()).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                dist = (1.0 - F.cosine_similarity(orig_emb, self.extractor.resnet(torch.clamp(face + delta, -1, 1)))).item()
                if dist > best_dist:
                    best_dist, best_delta = dist, delta.data.clone()

            if (step + 1) % 10 == 0:
                with torch.no_grad():
                    e = self.extractor.resnet(torch.clamp(face + delta, -1, 1))
                    print(f"  [Ulixes+EOT] Step {step+1}/{self.steps} | cos_orig={F.cosine_similarity(e, orig_emb).item():.4f}")

        delta_np = best_delta.squeeze(0).numpy()
        delta_hwc = np.transpose(delta_np, (1, 2, 0)) * 127.5
        # Smooth to reduce visible structure
        delta_hwc = cv2.GaussianBlur(delta_hwc, (3, 3), sigmaX=0.8)

        crop_w, crop_h = face_crop.size
        delta_resized = cv2.resize(delta_hwc, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        crop_arr = pil_to_numpy(face_crop).astype(np.float32)
        protected = np.clip(crop_arr + delta_resized, 0, 255).astype(np.uint8)
        return numpy_to_pil(protected)