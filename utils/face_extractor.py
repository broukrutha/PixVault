"""
face_extractor.py
-----------------
Handles face detection, alignment, and embedding extraction
using facenet-pytorch (CPU-friendly, no GPU required).
"""

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2


class FaceExtractor:
    """
    Extracts 512-d face embeddings using FaceNet (InceptionResnetV1).
    Runs fully on CPU.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # MTCNN for face detection and alignment
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device,
            keep_all=False,
        )

        # InceptionResnetV1 pretrained on VGGFace2 — lightweight & CPU-friendly
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Disable gradients for inference
        for param in self.resnet.parameters():
            param.requires_grad = False

    def detect_face(self, pil_image: Image.Image):
        """Returns (aligned_tensor, prob) or (None, None) if no face found."""
        img_tensor, prob = self.mtcnn(pil_image, return_prob=True)
        if img_tensor is None or prob < 0.90:
            return None, None
        return img_tensor, prob

    def detect_face_with_box(self, pil_image):
        """
        Returns (aligned_tensor, box, prob) where box = [x1, y1, x2, y2]
        in original image pixel coordinates, or (None, None, None).
        """
        boxes, probs = self.mtcnn.detect(pil_image)
        if boxes is None or probs[0] < 0.90:
            return None, None, None

        box = boxes[0].astype(int)  # [x1, y1, x2, y2]
        # Add margin
        w = pil_image.width
        h = pil_image.height
        margin = 20
        x1 = max(0, box[0] - margin)
        y1 = max(0, box[1] - margin)
        x2 = min(w, box[2] + margin)
        y2 = min(h, box[3] + margin)
        box_padded = [x1, y1, x2, y2]

        aligned, prob = self.mtcnn(pil_image, return_prob=True)
        if aligned is None:
            return None, None, None
        return aligned, box_padded, float(probs[0])

    def get_embedding(self, pil_image: Image.Image) -> np.ndarray | None:
        """Returns 512-d embedding for the face in the image."""
        aligned, prob = self.detect_face(pil_image)
        if aligned is None:
            return None
        aligned = aligned.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.resnet(aligned)
        return embedding.squeeze().cpu().numpy()

    def get_embedding_from_tensor(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Returns embedding tensor with gradients enabled (for optimization)."""
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)
        return self.resnet(face_tensor)

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings [-1, 1], 1 = identical."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def euclidean_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Euclidean distance between two embeddings."""
        return float(np.linalg.norm(emb1 - emb2))

    def is_same_person(self, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.7) -> bool:
        """True if cosine similarity > threshold (recognized as same person)."""
        return self.cosine_similarity(emb1, emb2) > threshold