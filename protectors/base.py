"""
base.py
-------
Abstract base class for all image protection methods.
Handles face-region detection, crop, protect, and paste-back
so that ONLY the face area is modified — background is untouched.
"""

from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import cv2
import torch
from utils.face_extractor import FaceExtractor
from utils.metrics import full_report, format_report
from utils.image_utils import pil_to_numpy, numpy_to_pil


def create_face_blend_mask(box: list, image_shape: tuple, feather: int = 25) -> np.ndarray:
    """
    Creates a soft feathered mask for the face bounding box.
    Values = 1.0 inside face, 0.0 outside, smoothly blended at edges.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    feather_size = feather * 2 + 1
    mask = cv2.GaussianBlur(mask, (feather_size, feather_size), sigmaX=feather * 0.5)
    return mask[:, :, np.newaxis]  # (H, W, 1)


class BaseProtector(ABC):
    """
    All protection methods inherit from this class.
    Subclasses implement `_perturb_face_crop()` which receives just the
    cropped face region. Base class handles paste-back with feathered blending
    so background/clothes are completely untouched.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.extractor = FaceExtractor(device=device)
        self.name = "BaseProtector"

    @abstractmethod
    def _perturb_face_crop(self, face_crop: Image.Image, aligned_tensor: torch.Tensor) -> Image.Image:
        """Override in subclass. Receives face crop, returns protected crop."""
        pass

    def _paste_face_back(self, original_arr: np.ndarray, protected_crop_arr: np.ndarray, box: list) -> np.ndarray:
        """Paste protected face crop back using feathered mask — background untouched."""
        x1, y1, x2, y2 = box
        h_crop, w_crop = y2 - y1, x2 - x1

        if protected_crop_arr.shape[:2] != (h_crop, w_crop):
            protected_crop_arr = cv2.resize(protected_crop_arr, (w_crop, h_crop), interpolation=cv2.INTER_LINEAR)

        mask = create_face_blend_mask(box, original_arr.shape, feather=20)

        protected_full = original_arr.copy().astype(np.float32)
        protected_full[y1:y2, x1:x2] = protected_crop_arr.astype(np.float32)

        result = mask * protected_full + (1.0 - mask) * original_arr.astype(np.float32)
        return np.clip(result, 0, 255).astype(np.uint8)

    def protect(self, pil_image: Image.Image) -> tuple:
        """Run face detection → crop → protect → paste back → compute metrics."""
        original_arr = pil_to_numpy(pil_image)
        orig_emb = self.extractor.get_embedding(pil_image)

        aligned_tensor, box, prob = self.extractor.detect_face_with_box(pil_image)

        if aligned_tensor is None:
            print(f"[{self.name}] No face detected — returning original.")
            protected_pil = pil_image.copy()
        else:
            x1, y1, x2, y2 = box
            print(f"[{self.name}] Face detected at [{x1},{y1},{x2},{y2}] prob={prob:.2f}")
            face_crop_pil = pil_image.crop((x1, y1, x2, y2))
            protected_crop_pil = self._perturb_face_crop(face_crop_pil, aligned_tensor)
            protected_crop_arr = pil_to_numpy(protected_crop_pil)
            protected_arr = self._paste_face_back(original_arr, protected_crop_arr, box)
            protected_pil = numpy_to_pil(protected_arr)

        protected_arr_final = pil_to_numpy(protected_pil)
        prot_emb = self.extractor.get_embedding(protected_pil)
        report = full_report(original_arr, protected_arr_final, orig_emb, prot_emb, method=self.name)
        return protected_pil, report