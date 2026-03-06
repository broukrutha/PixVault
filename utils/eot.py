"""
eot.py
------
Expectation Over Transformations (EOT)
(from: "Synthesizing Robust Adversarial Examples", Athalye et al. 2018)

EOT makes adversarial perturbations robust across real-world transformations.
Instead of optimizing for one input, we average gradients over many randomly
transformed versions of the image. The result is a perturbation that survives:
  - JPEG compression
  - Slight rotation / zoom
  - Brightness / contrast changes
  - Gaussian blur
  - Horizontal flip

This is critical for social media protection because platforms re-compress,
resize, and process uploaded images — EOT ensures the protection survives.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np


class EOT:
    """
    Applies random transformations during adversarial optimization.

    Usage:
        eot = EOT(n_transforms=10)
        avg_emb = eot.get_eot_embedding(model, perturbed_face_tensor)
        loss = cosine_similarity(avg_emb, orig_emb)
    """

    def __init__(
        self,
        n_transforms: int = 8,       # Number of random transforms to average over
        rotation_range: float = 8.0, # Max rotation in degrees
        scale_range: tuple = (0.90, 1.10),  # Zoom range
        brightness_range: tuple = (0.85, 1.15),
        contrast_range: tuple = (0.85, 1.15),
        blur_prob: float = 0.3,      # Probability of applying Gaussian blur
        flip_prob: float = 0.5,      # Probability of horizontal flip
        noise_std: float = 0.02,     # Gaussian noise std (in [-1,1] scale)
        jpeg_prob: float = 0.3,      # Probability of simulating JPEG compression
    ):
        self.n_transforms = n_transforms
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_prob = blur_prob
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.jpeg_prob = jpeg_prob

    def _apply_single_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a random combination of transforms to a single face tensor.
        x shape: (3, H, W), values in [-1, 1]
        """
        # Convert to [0, 1] for torchvision transforms
        x01 = (x + 1.0) / 2.0
        x01 = x01.clamp(0, 1)

        # Random horizontal flip
        if random.random() < self.flip_prob:
            x01 = TF.hflip(x01)

        # Random rotation
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        x01 = TF.rotate(x01, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.5)

        # Random zoom/scale via center crop + resize
        scale = random.uniform(*self.scale_range)
        h, w = x01.shape[-2], x01.shape[-1]
        new_h = int(h * scale)
        new_w = int(w * scale)
        if scale < 1.0:
            # Zoom out — pad then resize
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            x01 = TF.center_crop(x01, (new_h, new_w))
            x01 = F.interpolate(x01.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        else:
            # Zoom in — resize then center crop
            x01 = F.interpolate(x01.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
            x01 = TF.center_crop(x01, (h, w))

        # Random brightness
        brightness_factor = random.uniform(*self.brightness_range)
        x01 = TF.adjust_brightness(x01, brightness_factor)

        # Random contrast
        contrast_factor = random.uniform(*self.contrast_range)
        x01 = TF.adjust_contrast(x01, contrast_factor)

        # Random Gaussian blur
        if random.random() < self.blur_prob:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.5, 1.5)
            x01 = TF.gaussian_blur(x01, kernel_size=kernel_size, sigma=sigma)

        # Simulate JPEG compression (quantization noise)
        if random.random() < self.jpeg_prob:
            jpeg_noise = torch.randn_like(x01) * 0.015
            x01 = (x01 + jpeg_noise).clamp(0, 1)

        # Gaussian noise
        noise = torch.randn_like(x01) * self.noise_std
        x01 = (x01 + noise).clamp(0, 1)

        # Convert back to [-1, 1]
        return x01 * 2.0 - 1.0

    def get_eot_embedding(self, model, x: torch.Tensor) -> torch.Tensor:
        """
        Average embeddings across N randomly transformed versions of x.
        x shape: (1, 3, H, W), values in [-1, 1]
        Returns averaged embedding tensor with gradients.
        """
        embeddings = []
        for _ in range(self.n_transforms):
            x_t = self._apply_single_transform(x.squeeze(0))
            x_t = x_t.unsqueeze(0)
            emb = model(x_t)
            embeddings.append(emb)

        # Stack and average — gradients flow through all transforms
        stacked = torch.stack(embeddings, dim=0)   # (N, 1, 512)
        avg_emb = stacked.mean(dim=0)               # (1, 512)
        return avg_emb

    def get_eot_loss(
        self,
        model,
        perturbed: torch.Tensor,
        orig_emb: torch.Tensor,
        target_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute EOT loss:
          - Push embedding away from orig_emb (maximize distance)
          - Optionally pull toward target_emb (impostor identity)

        Returns scalar loss tensor.
        """
        avg_emb = self.get_eot_embedding(model, perturbed)

        # Push away from original
        loss = F.cosine_similarity(avg_emb, orig_emb.detach()).mean()

        # Pull toward target if provided
        if target_emb is not None:
            loss -= 0.4 * F.cosine_similarity(avg_emb, target_emb.detach()).mean()

        return loss