
from pathlib import Path
import torch

from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT = (
    PROJECT_DIR / "results" / "Patchcore" / "surface_scratch"
    / "latest" / "weights" / "lightning" / "model.ckpt"
)

# Start with one known scratch image from your dataset.
IMAGE_PATH = PROJECT_DIR / "test_images" / "test_image.png"

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = Patchcore(
        backbone="wide_resnet50_2",
        pre_trained=True,
    )

    engine = Engine(
        accelerator="auto",
        devices=1,
    )

    print("\nLoading trained PatchCore checkpoint...")
    engine.predict(
        model=model,
        dataset=PredictDataset(path=IMAGE_PATH),
        ckpt_path=str(CHECKPOINT),
    )

    print("\nPrediction completed.")


if __name__ == "__main__":
    main()
