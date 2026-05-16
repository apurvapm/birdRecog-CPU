"""PyTorch Dataset for Caltech-UCSD Birds-200-2011 (CUB-200-2011).

CUB ships several small text files that together describe the dataset:

  images.txt              <image_id> <relative_path>
  image_class_labels.txt  <image_id> <class_id>            (class_id is 1..200)
  train_test_split.txt    <image_id> <is_training_image>   (1 = train, 0 = test)
  classes.txt             <class_id> <class_name>          ("001.Black_footed_Albatross")

We parse those once into in-memory lists, then `__getitem__` opens the JPEG,
applies torchvision transforms, and returns (tensor, label).

We also expose `build_loaders()` which returns ready-to-use train and val
DataLoaders with the standard ImageNet augmentations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


# Standard ImageNet normalization stats (because EfficientNet was pretrained on
# ImageNet and our images need to look the same as what it was trained on).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _read_id_to_str(path: Path) -> dict[int, str]:
    """Read a CUB text file whose lines are '<id> <value...>'."""
    out: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, _, rest = line.partition(" ")
            out[int(idx_str)] = rest
    return out


def _read_id_to_int(path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, _, b = line.partition(" ")
            out[int(a)] = int(b)
    return out


def load_class_names(cub_root: Path = config.CUB_ROOT) -> List[str]:
    """Return the 200 class names in canonical (class_id) order.

    classes.txt entries look like "001.Black_footed_Albatross"; we strip the
    numeric prefix and replace underscores with spaces for display.
    """
    raw = _read_id_to_str(cub_root / "classes.txt")
    names = [None] * 200
    for class_id, raw_name in raw.items():
        clean = raw_name.split(".", 1)[-1].replace("_", " ")
        names[class_id - 1] = clean  # class_id is 1-indexed in the file
    if any(n is None for n in names):
        raise RuntimeError("classes.txt did not contain all 200 classes")
    return names  # type: ignore[return-value]


class CUBDataset(Dataset):
    """Caltech-UCSD Birds-200-2011 as a PyTorch Dataset."""

    def __init__(
        self,
        cub_root: Path = config.CUB_ROOT,
        train: bool = True,
        transform=None,
    ):
        self.cub_root = Path(cub_root)
        self.images_dir = self.cub_root / "images"
        self.transform = transform

        id_to_path = _read_id_to_str(self.cub_root / "images.txt")
        id_to_class = _read_id_to_int(self.cub_root / "image_class_labels.txt")
        id_to_split = _read_id_to_int(self.cub_root / "train_test_split.txt")

        wanted_split = 1 if train else 0
        self.samples: List[Tuple[Path, int]] = []
        for img_id, rel_path in id_to_path.items():
            if id_to_split.get(img_id) != wanted_split:
                continue
            # CUB classes are 1-indexed; PyTorch wants 0-indexed labels.
            label = id_to_class[img_id] - 1
            self.samples.append((self.images_dir / rel_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No samples found for train={train}. Did you run download_data.py?"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# --- transform builders ---


def train_transforms(image_size: int = config.IMAGE_SIZE):
    """Augmentations for training: random crop, flip, color jitter."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms(image_size: int = config.IMAGE_SIZE):
    """Deterministic transforms for validation / inference."""
    resize = int(image_size * 256 / 224)  # standard 224 -> 256 ratio
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for CUB-200-2011."""
    train_ds = CUBDataset(train=True, transform=train_transforms())
    val_ds = CUBDataset(train=False, transform=eval_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    # Quick smoke check
    names = load_class_names()
    print(f"Loaded {len(names)} class names. First 5: {names[:5]}")
    ds = CUBDataset(train=True, transform=eval_transforms())
    print(f"Train set size: {len(ds)}")
    img, label = ds[0]
    print(f"Sample 0: tensor shape={tuple(img.shape)}  label={label} ({names[label]})")
