"""Fine-tune ResNet18 on PatchCamelyon with validation AUC."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from tqdm import tqdm

from pcam_mets_explain.config import CHECKPOINTS, DATA_RAW, REPORTS, SEED
from pcam_mets_explain.dataset import build_dataloaders
from pcam_mets_explain.model import build_resnet18


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    all_labels: list[float] = []
    all_probs: list[float] = []
    criterion = nn.BCEWithLogitsLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        losses.append(float(loss.item()))
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    preds = [1.0 if p >= 0.5 else 0.0 for p in all_probs]
    auc = float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else float("nan")
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "auc": auc,
        "accuracy": float(accuracy_score(all_labels, preds)),
        "n": float(len(all_labels)),
    }


def train_one_epoch(model, loader, optimizer, device, epoch: int) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running = 0.0
    steps = 0
    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{running / steps:.4f}")
    return running / max(steps, 1)


def train(
    *,
    raw_dir: Path = DATA_RAW,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
    num_workers: int = 2,
    max_train: int | None = 20_000,
    max_valid: int | None = 4_000,
    pretrained: bool = True,
) -> dict:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, valid_loader, train_ds, valid_ds = build_dataloaders(
        raw_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_train=max_train,
        max_valid=max_valid,
    )
    print(f"Train patches: {len(train_ds)}  Valid patches: {len(valid_ds)}")

    model = build_resnet18(pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINTS / "best.pt"
    history = []
    best_auc = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        metrics = evaluate(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_auc={metrics['auc']:.4f}  "
            f"val_acc={metrics['accuracy']:.4f}"
        )
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": best_auc,
                    "val_accuracy": metrics["accuracy"],
                    "class_names": ["no_metastasis", "metastasis"],
                    "max_train": max_train,
                    "max_valid": max_valid,
                },
                best_path,
            )
            print(f"  saved {best_path} (best val AUC so far)")

    final = {
        "task": "pcam_metastasis_binary",
        "model": "resnet18",
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_train": max_train,
        "max_valid": max_valid,
        "train_n": len(train_ds),
        "valid_n": len(valid_ds),
        "best_val_auc": best_auc,
        "best_checkpoint": str(best_path),
        "history": history,
        "disclaimer": "Research demo only. Patch-level, not a diagnosis.",
    }
    metrics_path = REPORTS / "metrics.json"
    metrics_path.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")

    train_ds.close()
    valid_ds.close()
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATA_RAW)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--max-train",
        type=int,
        default=20_000,
        help="Balanced train subset size (default 20000). Use 0 for full train set.",
    )
    parser.add_argument(
        "--max-valid",
        type=int,
        default=4_000,
        help="Balanced valid subset size (default 4000). Use 0 for full valid set.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    max_train = None if args.max_train == 0 else args.max_train
    max_valid = None if args.max_valid == 0 else args.max_valid
    train(
        raw_dir=args.raw_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        max_train=max_train,
        max_valid=max_valid,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
