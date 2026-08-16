from pathlib import Path
from unittest.mock import patch

from src.data.dataset import download_kaggle_dataset


def test_download_kaggle_dataset_uses_kagglehub(tmp_path):
    destination = tmp_path / "downloaded"

    with patch("src.data.dataset.kagglehub.dataset_download", return_value=str(destination)) as mock_download:
        result = download_kaggle_dataset("owner/dataset", download_dir=str(destination))

    assert result == str(destination)
    mock_download.assert_called_once_with("owner/dataset", path=str(destination))
