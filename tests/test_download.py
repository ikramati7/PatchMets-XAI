from pathlib import Path

import h5py
import numpy as np

from pcam_mets_explain.download import h5_is_complete, h5_path


def test_h5_is_complete(tmp_path: Path):
    path = h5_path(tmp_path, "valid", "y")
    assert not h5_is_complete(path, "y")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("y", data=np.zeros((8, 1, 1, 1), dtype=np.uint8))
    assert h5_is_complete(path, "y")
    assert h5_is_complete(path, "y", expected=8)
    assert not h5_is_complete(path, "y", expected=16)
    assert not h5_is_complete(path, "x")
