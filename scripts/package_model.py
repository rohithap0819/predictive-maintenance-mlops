from pathlib import Path
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)

required_files = [
    "best_model.pkl",
    "type_encoder.pkl",
    "best_model_metadata.pkl",
]

for filename in required_files:
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact not found: {path}"
        )


output_file = (
    ARTIFACT_DIR / "predictive-maintenance-model.zip"
)

with zipfile.ZipFile(
    output_file,
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as zip_file:

    for filename in required_files:
        path = MODEL_DIR / filename
        zip_file.write(
            path,
            arcname=filename,
        )

print(f"Created model artifact: {output_file}")