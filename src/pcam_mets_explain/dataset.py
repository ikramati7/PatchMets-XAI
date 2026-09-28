"""HDF5 PatchCamelyon dataset for PyTorch training."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from pcam_mets_explain.config import DATA_RAW, IMAGE_SIZE, SEED


def h5_paths(raw_dir: Path, split: str) -> tuple[Path, Path]:
    stem = f"camelyonpatch_level_2_split_{split}"
    return raw_dir / f"{stem}_x.h5", raw_dir / f"{stem}_y.h5"


def require_split(raw_dir: Path, split: str) -> tuple[Path, Path]:
    x_path, y_path = h5_paths(raw_dir, split)
    missing = [str(path) for path in (x_path, y_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing PCam files:\n  "
            + "\n  ".join(missing)
            + "\nRun: python -m pcam_mets_explain.download   # full train images needed for training"
        )
    return x_path, y_path


class PCamH5Dataset(Dataset):
    """Lazy HDF5 reader. Keeps file handles open for fast indexed access."""

    def __init__(
        self,
        raw_dir: Path,
        split: str,
        *,
        transform=None,
        max_samples: int | None = None,
        seed: int = SEED,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.split = split
        self.transform = transform
        self.x_path, self.y_path = require_split(self.raw_dir, split)

        with h5py.File(self.y_path, "r") as handle:
            labels = np.asarray(handle["y"]).reshape(-1).astype(np.int64)
        with h5py.File(self.x_path, "r") as handle:
            n_images = int(handle["x"].shape[0])
        if n_images != labels.size:
            raise ValueError(f"{split}: image count {n_images} != label count {labels.size}")

        indices = np.arange(labels.size)
        if max_samples is not None and max_samples < labels.size:
            rng = np.random.default_rng(seed)
            # Keep class balance in the subset.
            pos = indices[labels == 1]
            neg = indices[labels == 0]
            n_each = max_samples // 2
            chosen = np.concatenate(
                [
                    rng.choice(neg, size=min(n_each, neg.size), replace=False),
                    rng.choice(pos, size=min(n_each, pos.size), replace=False),
                ]
            )
            indices = np.sort(chosen)

        self.indices = indices
        self.labels = labels[indices]
        self._x_file: h5py.File | None = None
        self._y_file: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def _ensure_open(self) -> None:
        if self._x_file is None:
            self._x_file = h5py.File(self.x_path, "r")
        if self._y_file is None:
            self._y_file = h5py.File(self.y_path, "r")

    def __getitem__(self, index: int):
        self._ensure_open()
        assert self._x_file is not None
        row = int(self.indices[index])
        image = np.asarray(self._x_file["x"][row])
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"Unexpected shape {image.shape} at {self.split}[{row}]")
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, torch.tensor(label, dtype=torch.float32)

    def close(self) -> None:
        if self._x_file is not None:
            self._x_file.close()
            self._x_file = None
        if self._y_file is not None:
            self._y_file.close()
            self._y_file = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_transforms(train: bool):
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_dataloaders(
    raw_dir: Path = DATA_RAW,
    *,
    batch_size: int = 64,
    num_workers: int = 2,
    max_train: int | None = None,
    max_valid: int | None = None,
):
    from torch.utils.data import DataLoader

    train_ds = PCamH5Dataset(
        raw_dir, "train", transform=make_transforms(True), max_samples=max_train
    )
    valid_ds = PCamH5Dataset(
        raw_dir, "valid", transform=make_transforms(False), max_samples=max_valid
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader, train_ds, valid_ds
