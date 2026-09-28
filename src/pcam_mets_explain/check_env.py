"""Print an environment report. Safe to run before PyTorch is installed."""

from __future__ import annotations

import platform
import sys
from importlib import import_module, metadata

from pcam_mets_explain.config import DATA_PROCESSED, DATA_RAW, REPORTS, ROOT


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


DIST_NAMES = {
    "PIL": "pillow",
    "sklearn": "scikit-learn",
}


def main() -> None:
    print("PatchMets-XAI environment")
    print(f"  python     {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f"  project    {ROOT}")
    print(f"  data/raw   {DATA_RAW}  exists={DATA_RAW.is_dir()}")
    print(f"  processed  {DATA_PROCESSED}  exists={DATA_PROCESSED.is_dir()}")
    print(f"  reports    {REPORTS}  exists={REPORTS.is_dir()}")
    for package in ("numpy", "PIL", "matplotlib", "sklearn", "tqdm", "torch", "torchvision", "streamlit"):
        module_name = "PIL" if package == "PIL" else package
        dist_name = DIST_NAMES.get(package, package)
        try:
            import_module(module_name)
            status = _version(dist_name)
        except ImportError:
            status = "not installed"
        print(f"  {package:<12} {status}")
    try:
        import torch

        print(f"  cuda        {torch.cuda.is_available()}")
    except ImportError:
        print("  cuda        n/a (install torch for training or the demo)")
    print("Environment check finished. Next: download PCam into data/raw.")


if __name__ == "__main__":
    main()
