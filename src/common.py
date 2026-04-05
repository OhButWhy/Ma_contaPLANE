import csv
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(payload: Any, path: Path) -> None:
    ensure_parent(path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_run_config(cfg: TrainConfig, cli_name: str, path: Path) -> None:
    cfg_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(cfg).items()
    }
    torch_version_module = getattr(torch, "version", None)
    cuda_version = getattr(torch_version_module, "cuda", None)

    save_json(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": cfg.seed,
            "script": cli_name,
            "environment": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": cuda_version,
            },
            "config": cfg_payload,
        },
        path,
    )


def save_history(history: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(history, out_dir / "train_history.json")

    if not history:
        return

    csv_path = out_dir / "train_history.csv"
    fieldnames = list(history[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_plots(history: List[Dict[str, float]], plots_dir: Path) -> None:
    if not history:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib недоступен, графики не сохранены.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_precision = [float(row["val_precision"]) for row in history]
    val_recall = [float(row["val_recall"]) for row in history]
    val_f1 = [float(row["val_f1"]) for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", linewidth=2)
    plt.title("Train Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_precision, marker="o", linewidth=2, label="Precision")
    plt.plot(epochs, val_recall, marker="o", linewidth=2, label="Recall")
    plt.plot(epochs, val_f1, marker="o", linewidth=2, label="F1")
    plt.title("Validation Metrics by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_curve.png", dpi=160)
    plt.close()


def draw_predictions(
    image_path: Path,
    pred: Dict[str, torch.Tensor],
    out_path: Path,
    score_threshold: float,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    boxes = pred.get("boxes", torch.zeros((0, 4))).cpu()
    scores = pred.get("scores", torch.zeros((0,))).cpu()

    for box, score in zip(boxes, scores):
        if float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        draw.text((x1 + 2, y1 + 2), f"plane {score:.2f}", fill=(255, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
