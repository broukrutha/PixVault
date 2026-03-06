"""
image_utils.py
--------------
Helper functions for image loading, saving, and conversion.
"""

import numpy as np
from PIL import Image
import cv2
import torch


def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """PIL Image → numpy uint8 RGB array."""
    return np.array(pil_img.convert("RGB"))


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """numpy uint8 RGB → PIL Image."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized face tensor [-1, 1] to uint8 numpy [0, 255].
    tensor shape: (3, H, W) or (1, 3, H, W)
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))  # CHW → HWC
    arr = (arr + 1.0) / 2.0 * 255.0  # [-1,1] → [0,255]
    return np.clip(arr, 0, 255).astype(np.uint8)


def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert uint8 numpy HWC [0,255] to float tensor CHW [-1,1].
    """
    arr = arr.astype(np.float32) / 255.0 * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC → CHW
    return tensor


def resize_to_match(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize img to match target HxW."""
    h, w = target_shape[:2]
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def blend_perturbation(original: np.ndarray, protected: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Blend original and protected images by alpha [0=original, 1=protected]."""
    blended = (1 - alpha) * original.astype(np.float32) + alpha * protected.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def load_image(path: str) -> Image.Image:
    """Load image from disk as PIL Image."""
    return Image.open(path).convert("RGB")


def save_image(pil_img: Image.Image, path: str) -> None:
    """Save PIL image to disk."""
    pil_img.save(path)