"""Download and extract the Caltech-UCSD Birds-200-2011 dataset.

Usage:
    python download_data.py

The dataset is ~1.1 GB compressed. It will be saved to ./data/CUB_200_2011/.
If the data already exists, this is a no-op.

Source: Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.
"The Caltech-UCSD Birds-200-2011 Dataset." 2011.
https://www.vision.caltech.edu/datasets/cub_200_2011/
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

import config

# Official Caltech Data record. If this URL ever rots, alternatives:
#   - Kaggle: kaggle datasets download wenewone/cub2002011
#   - Hugging Face: datasets.load_dataset("Donghyun99/CUB-200-2011")
URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
TGZ_NAME = "CUB_200_2011.tgz"


def download(url: str, dest: Path) -> None:
    """Stream-download `url` to `dest`, showing a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} already present ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        bar.close()


def extract(tgz: Path, dest_dir: Path) -> None:
    """Extract `tgz` into `dest_dir`. Skips if extraction looks complete."""
    marker = dest_dir / "CUB_200_2011" / "images.txt"
    if marker.exists():
        print(f"[skip] extraction already complete at {dest_dir}")
        return
    print(f"[extract] {tgz} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        # tqdm progress over member count
        members = tf.getmembers()
        for m in tqdm(members, desc="extracting"):
            tf.extract(m, dest_dir)


def verify(cub_root: Path) -> bool:
    """Sanity-check the extracted layout."""
    required = [
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
        "classes.txt",
        "images",  # directory
    ]
    missing = [p for p in required if not (cub_root / p).exists()]
    if missing:
        print(f"[error] missing files/folders in {cub_root}: {missing}")
        return False
    # Spot-check that the images directory has 200 species subfolders
    n_classes = sum(1 for p in (cub_root / "images").iterdir() if p.is_dir())
    if n_classes != 200:
        print(f"[error] expected 200 species folders, found {n_classes}")
        return False
    print(f"[ok] verified CUB-200-2011 at {cub_root} ({n_classes} species)")
    return True


def main() -> int:
    tgz_path = config.DATA_DIR / TGZ_NAME
    download(URL, tgz_path)
    extract(tgz_path, config.DATA_DIR)
    if not verify(config.CUB_ROOT):
        return 1
    print("\nDone. Next step:  python train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
