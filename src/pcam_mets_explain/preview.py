"""Inspect PCam splits and save an 8-patch figure (4 normal, 4 metastasis)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pcam_mets_explain.config import (
    CLASS_NAMES,
    DATA_RAW,
    EXPECTED_SPLIT_SIZES,
    IMAGE_SIZE,
    REPORTS,
    SEED,
)

LABEL_VALUE = {"no_metastasis": 0, "metastasis": 1}


def _h5_path(raw_dir: Path, split: str, kind: str) -> Path:
    return raw_dir / f"camelyonpatch_level_2_split_{split}_{kind}.h5"


def _squeeze_labels(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1).astype(int)


def read_labels(raw_dir: Path, split: str) -> np.ndarray:
    path = _h5_path(raw_dir, split, "y")
    if not path.is_file():
        raise FileNotFoundError(f"Missing labels: {path}")
    with h5py.File(path, "r") as handle:
        return _squeeze_labels(handle["y"][:])


def read_images(raw_dir: Path, split: str, indices: np.ndarray) -> np.ndarray:
    path = _h5_path(raw_dir, split, "x")
    if not path.is_file():
        raise FileNotFoundError(f"Missing images: {path}")
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["x"][sorted(indices)])


def split_summary(raw_dir: Path) -> dict[str, dict[str, int | str | bool]]:
    summary = {}
    for split, expected in EXPECTED_SPLIT_SIZES.items():
        y_path = _h5_path(raw_dir, split, "y")
        x_path = _h5_path(raw_dir, split, "x")
        if not y_path.is_file():
            summary[split] = {"status": "missing_labels", "n": 0, "expected": expected}
            continue
        labels = read_labels(raw_dir, split)
        n_x = None
        if x_path.is_file():
            with h5py.File(x_path, "r") as handle:
                n_x = int(handle["x"].shape[0])
        counts = np.bincount(labels, minlength=2)
        summary[split] = {
            "status": "ok",
            "n": int(labels.size),
            "expected": expected,
            "matches_published_size": int(labels.size) == expected,
            "n_images": n_x,
            "images_match_labels": n_x == int(labels.size) if n_x is not None else False,
            CLASS_NAMES[0]: int(counts[0]),
            CLASS_NAMES[1]: int(counts[1]),
        }
    return summary


def pick_example_indices(labels: np.ndarray, per_class: int = 4, seed: int = SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    chosen = {}
    for name, value in LABEL_VALUE.items():
        pool = np.flatnonzero(labels == value)
        if pool.size < per_class:
            raise ValueError(f"Need {per_class} '{name}' patches, found {pool.size}")
        chosen[name] = np.sort(rng.choice(pool, size=per_class, replace=False))
    return chosen


def save_example_grid(images: np.ndarray, titles: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    for axis, image, title in zip(axes.ravel(), images, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
        if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
            raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} patches, got {image.shape}")
    figure.suptitle("PCam examples (research only, not a diagnosis)", fontsize=11)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def preview(raw_dir: Path = DATA_RAW, reports_dir: Path = REPORTS, split: str = "valid") -> dict:
    summary = split_summary(raw_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    counts_path = reports_dir / "split_counts.json"
    counts_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    labels = read_labels(raw_dir, split)
    chosen = pick_example_indices(labels)
    order = (
        [(idx, "no_metastasis") for idx in chosen["no_metastasis"]]
        + [(idx, "metastasis") for idx in chosen["metastasis"]]
    )
    indices = np.array([item[0] for item in order], dtype=int)
    images = read_images(raw_dir, split, indices)
    titles = [f"{name}\n#{idx}" for idx, name in order]
    figure_path = reports_dir / "figures" / "pcam_examples.png"
    save_example_grid(images, titles, figure_path)
    return {"summary": summary, "figure": str(figure_path), "counts": str(counts_path), "split": split}


def main() -> None:
    result = preview()
    print(json.dumps(result["summary"], indent=2))
    print(f"Example grid: {result['figure']}")
    print(f"Split counts: {result['counts']}")
    print(f"Plotted from split: {result['split']}")


if __name__ == "__main__":
    main()
