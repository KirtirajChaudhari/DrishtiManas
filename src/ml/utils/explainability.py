"""Simple explainability utilities that generate faux heatmaps for demos."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance


def generate_heatmap(image_path: str, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(image_path)
    if not source.exists():
        base = Image.new("RGB", (224, 224), color=(255, 0, 0))
    else:
        base = Image.open(source).convert("RGB")
    # Apply a quick contrast boost to mimic attention overlay.
    heatmap = ImageEnhance.Color(base).enhance(1.8)
    out_path = output_dir / f"heatmap_{source.stem or 'sample'}.png"
    heatmap.save(out_path)
    return str(out_path)
