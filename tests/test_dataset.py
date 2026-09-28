from pathlib import Path

import h5py
import numpy as np
import pytest

from pcam_mets_explain.dataset import PCamH5Dataset, h5_paths, require_split


def _write_fake_split(raw_dir: Path, split: str, n: int = 20) -> None:
    images = np.zeros((n, 96, 96, 3), dtype=np.uint8)
    images[..., 0] = 120
    labels = np.zeros((n, 1, 1, 1), dtype=np.uint8)
    labels[n // 2 :] = 1
    x_path, y_path = h5_paths(raw_dir, split)
    with h5py.File(x_path, "w") as handle:
        handle.create_dataset("x", data=images)
    with h5py.File(y_path, "w") as handle:
        handle.create_dataset("y", data=labels)


def test_require_split_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing PCam"):
        require_split(tmp_path, "train")


def test_dataset_subset_is_balanced(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_fake_split(raw, "train", n=40)
    ds = PCamH5Dataset(raw, "train", max_samples=10, seed=0)
    assert len(ds) == 10
    labels = [int(ds[i][1].item()) for i in range(len(ds))]
    assert labels.count(0) == 5
    assert labels.count(1) == 5
    image, label = ds[0]
    assert image.shape[0] == 3
    assert label.ndim == 0
    ds.close()
