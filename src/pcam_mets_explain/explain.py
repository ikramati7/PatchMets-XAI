"""Day 3: ROC, confusion matrix, and Grad-CAM on successes and failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from pcam_mets_explain.config import CHECKPOINTS, CLASS_NAMES, DATA_RAW, REPORTS, SEED
from pcam_mets_explain.dataset import PCamH5Dataset, make_transforms
from pcam_mets_explain.gradcam import GradCAM, overlay_cam_on_rgb, resnet18_target_layer
from pcam_mets_explain.model import load_checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    labels_all = []
    probs_all = []
    indices_all = []
    for batch_index, (images, labels) in enumerate(tqdm(loader, desc="predict", leave=False)):
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_all.append(labels.numpy())
        probs_all.append(probs)
        start = batch_index * loader.batch_size
        indices_all.append(np.arange(start, start + len(labels)))
    return {
        "labels": np.concatenate(labels_all),
        "probs": np.concatenate(probs_all),
        "loader_indices": np.concatenate(indices_all),
    }


def plot_roc(labels: np.ndarray, probs: np.ndarray, output: Path, auc: float, split: str) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"PCam {split} ROC (research only)")
    axis.legend(loc="lower right")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def plot_confusion(labels: np.ndarray, preds: np.ndarray, output: Path) -> None:
    matrix = confusion_matrix(labels, preds)
    figure, axis = plt.subplots(figsize=(5, 4.5))
    display = ConfusionMatrixDisplay(matrix, display_labels=list(CLASS_NAMES))
    display.plot(ax=axis, colorbar=False, cmap="Blues")
    axis.set_title("Confusion matrix (threshold 0.5)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def pick_case_indices(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, per_bucket: int = 2) -> dict[str, list[int]]:
    rng = np.random.default_rng(SEED)
    buckets = {
        "true_negative": np.flatnonzero((labels == 0) & (preds == 0)),
        "true_positive": np.flatnonzero((labels == 1) & (preds == 1)),
        "false_positive": np.flatnonzero((labels == 0) & (preds == 1)),
        "false_negative": np.flatnonzero((labels == 1) & (preds == 0)),
    }
    chosen: dict[str, list[int]] = {}
    for name, pool in buckets.items():
        if pool.size == 0:
            chosen[name] = []
            continue
        # Prefer confident mistakes / clear successes.
        if name.startswith("true"):
            scores = probs[pool] if name == "true_positive" else 1.0 - probs[pool]
        else:
            scores = probs[pool] if name == "false_positive" else 1.0 - probs[pool]
        order = np.argsort(-scores)
        top = pool[order[: max(per_bucket * 3, per_bucket)]]
        take = min(per_bucket, top.size)
        chosen[name] = sorted(rng.choice(top, size=take, replace=False).tolist())
    return chosen


def denormalize_imagenet(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    image = (tensor * std + mean).clamp(0, 1)
    return np.uint8(image.permute(1, 2, 0).cpu().numpy() * 255)


def write_gradcam_grid(
    model,
    dataset: PCamH5Dataset,
    case_indices: dict[str, list[int]],
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    device,
    output: Path,
) -> list[dict]:
    model.eval()
    cam_engine = GradCAM(model, resnet18_target_layer(model))
    panels = []
    notes = []
    order = [
        ("true_negative", "TN"),
        ("true_positive", "TP"),
        ("false_positive", "FP"),
        ("false_negative", "FN"),
    ]
    for bucket, short in order:
        for local_index in case_indices.get(bucket, []):
            image_t, _ = dataset[local_index]
            batch = image_t.unsqueeze(0).to(device)
            model.zero_grad(set_to_none=True)
            cam = cam_engine(batch)[0]
            rgb = denormalize_imagenet(image_t)
            overlay = overlay_cam_on_rgb(rgb, cam)
            title = (
                f"{short}  p={probs[local_index]:.2f}\n"
                f"true={CLASS_NAMES[int(labels[local_index])]}"
            )
            note = {
                "bucket": bucket,
                "dataset_index": int(local_index),
                "h5_index": int(dataset.indices[local_index]),
                "true_label": CLASS_NAMES[int(labels[local_index])],
                "pred_label": CLASS_NAMES[int(preds[local_index])],
                "prob_metastasis": float(probs[local_index]),
                "comment": _auto_comment(bucket, float(probs[local_index])),
            }
            notes.append(note)
            panels.append((rgb, overlay, title))

    cam_engine.close()
    if not panels:
        raise RuntimeError("No Grad-CAM panels selected")

    cols = 4
    rows = int(np.ceil(len(panels) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(11, 2.8 * rows))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.axis("off")
    for i, (rgb, overlay, title) in enumerate(panels):
        axis = axes[i // cols, i % cols]
        # Show overlay (explanation) as the main image.
        axis.imshow(overlay)
        axis.set_title(title, fontsize=8)
        axis.axis("off")
    figure.suptitle("Grad-CAM examples (research only, not a diagnosis)", fontsize=11)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return notes


def _auto_comment(bucket: str, prob: float) -> str:
    if bucket == "true_positive":
        return "Correct metastasis call; heatmap should focus on dense atypical cells."
    if bucket == "true_negative":
        return "Correct normal call; heatmap should avoid blank/fat-only regions if possible."
    if bucket == "false_positive":
        return f"False alarm (p={prob:.2f}); check stain, lymphoid density, or artifacts."
    if bucket == "false_negative":
        return f"Missed metastasis (p={prob:.2f}); tumor may be small or near the patch edge."
    return ""


def explain(
    *,
    checkpoint: Path = CHECKPOINTS / "best.pt",
    raw_dir: Path = DATA_RAW,
    split: str = "valid",
    max_samples: int | None = 4_000,
    batch_size: int = 64,
    num_workers: int = 2,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint, map_location=device)
    model.to(device)

    dataset = PCamH5Dataset(
        raw_dir,
        split,
        transform=make_transforms(False),
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    preds_pack = collect_predictions(model, loader, device)
    labels = preds_pack["labels"].astype(int)
    probs = preds_pack["probs"]
    preds = (probs >= 0.5).astype(int)
    auc = float(roc_auc_score(labels, probs))
    acc = float(accuracy_score(labels, preds))

    figures = REPORTS / "figures"
    roc_path = figures / f"roc_{split}.png"
    cm_path = figures / f"confusion_matrix_{split}.png"
    cam_path = figures / f"gradcam_examples_{split}.png"
    plot_roc(labels, probs, roc_path, auc, split=split)
    plot_confusion(labels, preds, cm_path)

    cases = pick_case_indices(labels, preds, probs, per_bucket=2)
    notes = write_gradcam_grid(model, dataset, cases, labels, probs, preds, device, cam_path)

    summary = {
        "split": split,
        "n": int(labels.size),
        "auc": auc,
        "accuracy": acc,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "checkpoint": str(checkpoint),
        "checkpoint_val_auc": payload.get("val_auc"),
        "figures": {
            "roc": str(roc_path),
            "confusion_matrix": str(cm_path),
            "gradcam": str(cam_path),
        },
        "case_notes": notes,
        "disclaimer": "Research demo only. Patch-level, not a diagnosis.",
    }
    out_json = REPORTS / "day3_explain.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Merge into metrics.json if present.
    metrics_path = REPORTS / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = {}
    metrics["day3"] = {
        "split": split,
        "auc": auc,
        "accuracy": acc,
        "figures": summary["figures"],
        "explain_json": str(out_json),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    dataset.close()
    print(json.dumps({"auc": auc, "accuracy": acc, "figures": summary["figures"]}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS / "best.pt")
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--split", default="valid", choices=["valid", "test", "train"])
    parser.add_argument("--max-samples", type=int, default=4_000, help="0 = full split")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    max_samples = None if args.max_samples == 0 else args.max_samples
    explain(
        checkpoint=args.checkpoint,
        raw_dir=args.raw_dir,
        split=args.split,
        max_samples=max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
