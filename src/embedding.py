"""
SigLIP ONNX embedding engine.

Loads INT8-quantized vision and text ONNX models once at startup.
Provides encode_images() and encode_texts() for the processing pipeline.
"""

import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from transformers import AutoTokenizer
import structlog

logger = structlog.get_logger()


class SigLIPEmbedder:
    """SigLIP ONNX embedding engine for images and text."""

    def __init__(self, model_dir: str, model_config):
        """
        Load ONNX models and tokenizer into memory.

        Args:
            model_dir: Path to directory containing ONNX model files.
            model_config: ModelConfig dataclass from config.py.
        """
        self.cfg = model_config
        model_path = Path(model_dir)

        # Load ONNX sessions (resident in RAM)
        vision_path = model_path / Path(self.cfg.vision_model).name
        text_path = model_path / Path(self.cfg.text_model).name

        if not vision_path.exists():
            raise FileNotFoundError(
                f"Vision model not found: {vision_path}. "
                "Run: python scripts/download_model.py"
            )
        if not text_path.exists():
            raise FileNotFoundError(
                f"Text model not found: {text_path}. "
                "Run: python scripts/download_model.py"
            )

        logger.info("loading_vision_model", path=str(vision_path))
        self.vision_session = ort.InferenceSession(
            str(vision_path), providers=["CPUExecutionProvider"]
        )

        logger.info("loading_text_model", path=str(text_path))
        self.text_session = ort.InferenceSession(
            str(text_path), providers=["CPUExecutionProvider"]
        )

        # Load tokenizer (lightweight, no PyTorch required)
        logger.info("loading_tokenizer", id=self.cfg.tokenizer_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_id)

        logger.info("embedding_engine_ready",
                     dim=self.cfg.embedding_dim,
                     image_size=self.cfg.image_size)

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess a PIL image for SigLIP vision encoder.
        Resize to model input size, normalize to [-1, 1], channels-first.
        """
        img = image.convert("RGB").resize(
            (self.cfg.image_size, self.cfg.image_size), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32)       # (H, W, 3)
        arr = arr / 127.5 - 1.0                      # Scale to [-1, 1]
        arr = arr.transpose(2, 0, 1)                  # (3, H, W)
        return arr

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """
        Encode a list of PIL images into normalized embeddings.

        Args:
            images: List of PIL Image objects.

        Returns:
            np.ndarray of shape (N, embedding_dim), L2-normalized.
        """
        pixel_values = np.stack(
            [self.preprocess_image(img) for img in images]
        ).astype(np.float32)

        outputs = self.vision_session.run(
            ["pooler_output"], {"pixel_values": pixel_values}
        )
        embeddings = outputs[0]  # (N, embedding_dim)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        return embeddings / norms

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of text strings into normalized embeddings.

        Args:
            texts: List of natural language strings.

        Returns:
            np.ndarray of shape (N, embedding_dim), L2-normalized.
        """
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.cfg.max_text_length,
            truncation=True,
            return_tensors="np",
        )
        input_ids = tokens["input_ids"].astype(np.int64)

        outputs = self.text_session.run(
            ["pooler_output"], {"input_ids": input_ids}
        )
        embeddings = outputs[0]  # (N, embedding_dim)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms
