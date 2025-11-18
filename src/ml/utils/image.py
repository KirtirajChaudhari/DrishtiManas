"""Image utilities for preprocessing medical imagery."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision import transforms


def load_image_tensor(image_path: str, transform: transforms.Compose):
    """Load an image from disk and convert to tensor."""
    path = Path(image_path)
    if not path.exists():
        # Fallback to a blank image so demos can run without assets.
        image = Image.new("RGB", (224, 224), color=(0, 0, 0))
    else:
        image = Image.open(path).convert("RGB")
    return transform(image)
