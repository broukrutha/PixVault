"""
lowkey.py — LowKey with EOT + Face-Only Protection
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

from protectors.base import BaseProtector
from utils.image_utils import tensor_to_numpy, numpy_to_pil, pil_to_numpy
from utils.eot import EOT


class LowKeyProtector(BaseProtector):
    def __init__(self, device="cpu", epsilon=0.03, steps=30, step_size=0.003, restarts=2, smooth_sigma=1.2, n_eot=6):
        super().__init__(device)
        self.name = "LowKey (Smooth Transferable Attack)"
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.restarts = restarts
        self.smooth_sigma = smooth_sigma
        self.eot = EOT(n_transforms=n_eot)

    def _smooth_perturbation(self, delta_np):
        smoothed = np.zeros_like(delta_np)
        for c in range(delta_np.shape[0]):
            smoothed[c] = gaussian_filter(delta_np[c], sigma=self.smooth_sigma)
        return smoothed

    def _run_attack(self, face, orig_emb):
        delta = torch.empty_like(face).uniform_(-self.epsilon * 0.3, self.epsilon * 0.3)
        delta.requires_grad_(True)
        optimizer = torch.optim.SGD([delta], lr=self.step_size, momentum=0.9)
        best_delta, best_loss = delta.detach().clone(), float("inf")

        for step in range(self.steps):
            optimizer.zero_grad()
            perturbed = torch.clamp(face + delta, -1.0, 1.0)
            loss = self.eot.get_eot_loss(self.extractor.resnet, perturbed, orig_emb)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_delta = delta.detach().clone()
        return best_delta

    def _perturb_face_crop(self, face_crop: Image.Image, aligned_tensor: torch.Tensor) -> Image.Image:
        face = aligned_tensor.unsqueeze(0)
        with torch.no_grad():
            orig_emb = self.extractor.resnet(face)

        all_deltas = []
        for r in range(self.restarts):
            print(f"  [LowKey+EOT] Restart {r+1}/{self.restarts}")
            all_deltas.append(self._run_attack(face.clone(), orig_emb))

        best_delta, best_cos = None, float("inf")
        with torch.no_grad():
            for d in all_deltas:
                cos = F.cosine_similarity(self.extractor.resnet(torch.clamp(face + d, -1, 1)), orig_emb).item()
                if cos < best_cos:
                    best_cos, best_delta = cos, d
        print(f"  [LowKey+EOT] Best cosine: {best_cos:.4f}")

        # Heavy smoothing — LowKey's signature
        delta_np = self._smooth_perturbation(best_delta.squeeze(0).numpy())
        delta_hwc = np.transpose(delta_np, (1, 2, 0)) * 127.5

        crop_w, crop_h = face_crop.size
        delta_resized = cv2.resize(delta_hwc, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        crop_arr = pil_to_numpy(face_crop).astype(np.float32)
        protected = np.clip(crop_arr + delta_resized, 0, 255).astype(np.uint8)
        return numpy_to_pil(protected)