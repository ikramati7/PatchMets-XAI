from pathlib import Path

import h5py
import numpy as np

from pcam_mets_explain.preview import preview, split_summary


def _write_split(raw_dir: Path, split: str, n: int) -> None:
    images = np.zeros((n, 96, 96, 3), dtype=np.uint8)
    images[:, :, :, 0] = 180
    labels = np.zeros((n, 1, 1, 1), dtype=np.uint8)
    labels[n // 2 :, ...] = 1
    with h5py.File(raw_dir / f"camelyonpatch_level_2_split_{split}_x.h5", "w") as handle:
        handle.create_dataset("x", data=images)
    with h5py.File(raw_dir / f"camelyonpatch_level_2_split_{split}_y.h5", "w") as handle:
        handle.create_dataset("y", data=labels)


def test_preview_writes_figure_and_counts(tmp_path: Path):
    raw = tmp_path / "raw"
    reports = tmp_path / "reports"
    raw.mkdir()
    for split, n in (("train", 20), ("valid", 16), ("test", 16)):
        _write_split(raw, split, n)

    result = preview(raw, reports, split="valid")
    summary = split_summary(raw)
    assert summary["valid"]["n"] == 16
    assert summary["valid"]["no_metastasis"] == 8
    assert (reports / "figures" / "pcam_examples.png").is_file()
    assert Path(result["counts"]).is_file()
