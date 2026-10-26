import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.schemas import ClassProb, Prediction

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
EXPORT_DIR = Path(__file__).resolve().parent.parent / "intel_v1_export"
IMG_SIZE = 224
INPUT_NAME = None
OUTPUT_NAME = None
FALLBACK_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]


class Predictor:
    def __init__(self) -> None:
        self.session = None
        self.state = "unmounted"
        self.weights_file: str | None = None
        self.classes: list[str] = FALLBACK_CLASSES

    def load(self) -> None:
        candidates = [
            WEIGHTS_DIR / "best.pt",
            EXPORT_DIR / "best.pt",
            *sorted(WEIGHTS_DIR.glob("*.onnx")),
            *sorted(EXPORT_DIR.glob("*.onnx")),
        ]
        onnx = next((c for c in candidates if c.is_file() and c.suffix == ".onnx"), None)
        if onnx is None:
            return
        self.session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        n_classes = self.session.get_outputs()[0].shape[-1]
        if not (isinstance(n_classes, int) and n_classes == len(FALLBACK_CLASSES)):
            self.session = None
            return
        self.weights_file = onnx.name
        self.state = "ready"

    def predict(self, image_bytes: bytes) -> Prediction:
        started = time.perf_counter()
        image = Image.open(_bytestream(image_bytes)).convert("RGB")
        width, height = image.size

        tensor = _preprocess(image)[np.newaxis].astype(np.float32)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        probs = _softmax(logits)
        order = np.argsort(-probs)
        table = [ClassProb(name=self.classes[i], prob=float(probs[i])) for i in order]

        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return Prediction(
            top_class=table[0].name,
            confidence=table[0].prob,
            prob_table=table,
            width=width,
            height=height,
            elapsed_ms=elapsed_ms,
            model_id=self.weights_file or "unknown",
        )


def _bytestream(data: bytes):
    import io

    return io.BytesIO(data)


def _preprocess(image: Image.Image) -> np.ndarray:
    """Ultralytics classification transform: resize shorter edge to 256,
    center-crop 224, keep pixels in [0, 1]."""
    scale = int(IMG_SIZE * 1.143)
    image = image.resize(_fit(image.size, scale), Image.BILINEAR)
    left = (image.width - IMG_SIZE) // 2
    top = (image.height - IMG_SIZE) // 2
    image = image.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)


def _fit(size: tuple[int, int], target: int) -> tuple[int, int]:
    width, height = size
    if width < height:
        return target, max(1, round(height * target / width))
    return max(1, round(width * target / height)), target


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


predictor = Predictor()
