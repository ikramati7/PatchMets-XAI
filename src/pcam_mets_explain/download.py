"""Download PCam into data/raw from Hugging Face (Google Drive is often blocked)."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from pcam_mets_explain.config import DATA_RAW, IMAGE_SIZE

HF_DATASET = "1aurent/PatchCamelyon"
SPLITS = ("train", "valid", "test")


def h5_path(raw_dir: Path, split: str, kind: str) -> Path:
    return raw_dir / f"camelyonpatch_level_2_split_{split}_{kind}.h5"


def h5_is_complete(path: Path, dataset: str, expected: int | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r") as handle:
            if dataset not in handle:
                return False
            count = int(handle[dataset].shape[0])
    except OSError:
        return False
    if count <= 0:
        return False
    return expected is None or count == expected


def _as_rgb(image) -> np.ndarray:
    array = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected image shape {array.shape}")
    return array.astype(np.uint8, copy=False)


def write_split_from_hf(split: str, raw_dir: Path, *, write_x: bool, write_y: bool, batch_size: int = 256) -> None:
    from datasets import load_dataset

    need_x = write_x and not h5_is_complete(h5_path(raw_dir, split, "x"), "x")
    need_y = write_y and not h5_is_complete(h5_path(raw_dir, split, "y"), "y")
    if not need_x and not need_y:
        print(f"Already complete: {split}")
        return

    print(f"Loading Hugging Face split '{split}' ({HF_DATASET}) …")
    dataset = load_dataset(HF_DATASET, split=split)
    count = len(dataset)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if need_y:
        labels = np.zeros((count, 1, 1, 1), dtype=np.uint8)
        for start in tqdm(range(0, count, batch_size), desc=f"{split} labels"):
            stop = min(start + batch_size, count)
            batch = dataset[start:stop]
            values = np.asarray(batch["label"], dtype=np.uint8).reshape(-1)
            labels[start:stop, 0, 0, 0] = values
        y_path = h5_path(raw_dir, split, "y")
        tmp = y_path.with_suffix(".h5.tmp")
        with h5py.File(tmp, "w") as handle:
            handle.create_dataset("y", data=labels, compression="gzip")
        tmp.replace(y_path)
        print(f"Wrote {y_path}  n={count}")

    if need_x:
        x_path = h5_path(raw_dir, split, "x")
        tmp = x_path.with_suffix(".h5.tmp")
        with h5py.File(tmp, "w") as handle:
            images = handle.create_dataset(
                "x",
                shape=(count, IMAGE_SIZE, IMAGE_SIZE, 3),
                dtype=np.uint8,
                chunks=(min(32, count), IMAGE_SIZE, IMAGE_SIZE, 3),
            )
            for start in tqdm(range(0, count, batch_size), desc=f"{split} images"):
                stop = min(start + batch_size, count)
                batch = dataset[start:stop]
                stacked = np.stack([_as_rgb(image) for image in batch["image"]])
                images[start:stop] = stacked
        tmp.replace(x_path)
        print(f"Wrote {x_path}  n={count}")


def download_files(raw_dir: Path, light: bool) -> None:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_x = not (light and split == "train")
        write_split_from_hf(split, raw_dir, write_x=write_x, write_y=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--light",
        action="store_true",
        help="Skip train images (~6 GB). Still downloads train labels plus valid and test.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW)
    args = parser.parse_args(argv)
    download_files(args.raw_dir, light=args.light)
    print("Ready. Next: python -m pcam_mets_explain.preview")


if __name__ == "__main__":
    main()
