import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

DATASET_IDS = [
    "lesleynatrop/vehicle-sign-detection",
    "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
]

DEFAULT_CONSOLIDATED_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "consolidated"


def download_kaggle_dataset(dataset_id, download_dir=None):
    """Download a Kaggle dataset via kagglehub and return the local path."""
    return kagglehub.dataset_download(dataset_id, path=download_dir)


def consolidate_datasets(dataset_ids, output_dir=DEFAULT_CONSOLIDATED_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    consolidated_paths = {}
    for dataset_id in dataset_ids:
        source_path = download_kaggle_dataset(dataset_id)
        slug = dataset_id.split("/")[-1]
        destination = output_dir / slug
        shutil.copytree(source_path, destination, dirs_exist_ok=True)
        consolidated_paths[dataset_id] = str(destination)

    return consolidated_paths


if __name__ == "__main__":
    paths = consolidate_datasets(DATASET_IDS)
    for dataset_id, path in paths.items():
        print(f"{dataset_id} -> {path}")
    print(f"Consolidated dataset available at: {DEFAULT_CONSOLIDATED_DIR}")
