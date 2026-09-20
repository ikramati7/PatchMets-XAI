from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "reports" / "checkpoints"

IMAGE_SIZE = 96
NUM_CLASSES = 2
CLASS_NAMES = ("no_metastasis", "metastasis")
SEED = 42
