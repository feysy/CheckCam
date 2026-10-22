"""One-time SigLIP ONNX model downloader.

Downloads INT8 quantized vision and text encoders from HuggingFace
into the local models/ directory.

Usage:
    python scripts/download_model.py
"""

import sys
import shutil
from pathlib import Path

# Allow importing src from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from huggingface_hub import hf_hub_download


def download_models():
    cfg = load_config()
    model_dir = Path(cfg.storage.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    files_to_download = [cfg.model.vision_model, cfg.model.text_model]

    print(f"Model repository: {cfg.model.repo_id}")
    print(f"Target directory: {model_dir.resolve()}\n")

    for filename in files_to_download:
        target = model_dir / Path(filename).name
        if target.exists():
            print(f"  ✓ Already exists: {target}")
            continue

        print(f"  ↓ Downloading {filename} from {cfg.model.repo_id}...")
        downloaded = hf_hub_download(
            repo_id=cfg.model.repo_id,
            filename=filename,
        )
        # Copy to our local model directory
        shutil.copy2(downloaded, target)
        print(f"  ✓ Saved to: {target}")

    print("\n✅ All models ready.")


if __name__ == "__main__":
    download_models()
