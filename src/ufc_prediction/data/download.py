"""Dataset download and path resolution for Kaggle datasets.

Tries kagglehub for automatic download; falls back to checking local
directories for manually downloaded CSV files.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")


def ensure_dataset(dataset_slug: str, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Ensure a Kaggle dataset is available locally.

    Tries kagglehub first; falls back to checking data_dir for manually
    downloaded files. Returns the path to the dataset directory.

    Args:
        dataset_slug: Kaggle dataset identifier (e.g., "rajeevw/ufcdata")
        data_dir: Local directory to check for manually downloaded files

    Returns:
        Path to the directory containing dataset files

    Raises:
        FileNotFoundError: if dataset cannot be found or downloaded
    """
    # Try kagglehub download first
    try:
        import kagglehub

        path = kagglehub.dataset_download(dataset_slug)
        logger.info("Downloaded %s via kagglehub to %s", dataset_slug, path)
        return Path(path)
    except Exception:
        logger.warning("kagglehub download failed for %s", dataset_slug, exc_info=True)

    # Fallback: check local data directory
    # For "rajeevw/ufcdata", check data/rajeevw-ufcdata/ or data/ufcdata/
    slug_parts = dataset_slug.split("/")
    candidates = [
        data_dir / "-".join(slug_parts),  # data/rajeevw-ufcdata/
        data_dir / slug_parts[-1],  # data/ufcdata/
        data_dir,  # data/ (files directly in data dir)
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.csv")):
            logger.info("Found %s locally at %s", dataset_slug, candidate)
            return candidate

    raise FileNotFoundError(
        f"Dataset {dataset_slug} not found. Either:\n"
        f"  1. Set KAGGLE_USERNAME and KAGGLE_KEY env vars for auto-download\n"
        f"  2. Download manually from https://www.kaggle.com/datasets/{dataset_slug}\n"
        f"     and place CSV files in {data_dir}/{slug_parts[-1]}/"
    )
