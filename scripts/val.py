import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import get_default_config  # noqa: E402
from src.data_utils import (  # noqa: E402
    YoloDetectionDataset,
    create_dataloader,
    detection_map,
    detection_prf1,
)
from src.model import create_model  # noqa: E402
from src.common import draw_predictions, save_json  # noqa: E402


def main() -> None:
    cfg = get_default_config()
    ckpt_path = cfg.checkpoint_dir / cfg.best_checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg.val_metrics_out.parent.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    val_dataset = YoloDetectionDataset(cfg.val_split, cfg.data_dir)
    val_loader = create_dataloader(
        split_file=cfg.val_split,
        data_dir=cfg.data_dir,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    try:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model = create_model(
        num_classes=cfg.num_classes,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            outputs = model(images)
            pred_cpu = [
                {k: v.detach().cpu() for k, v in out.items()}
                for out in outputs
            ]
            target_cpu = [
                {k: v.detach().cpu() for k, v in target.items()}
                for target in targets
            ]

            all_preds.extend(pred_cpu)
            all_targets.extend(target_cpu)

            if cfg.save_visualizations and idx < cfg.viz_count:
                image_path = val_dataset.image_paths[idx]
                out_path = (
                    cfg.artifacts_dir
                    / "val_predictions"
                    / f"{image_path.stem}_pred.jpg"
                )
                draw_predictions(
                    image_path=image_path,
                    pred=pred_cpu[0],
                    out_path=out_path,
                    score_threshold=cfg.score_threshold,
                )

    metrics = detection_prf1(
        predictions=all_preds,
        targets=all_targets,
        score_threshold=cfg.score_threshold,
        iou_threshold=0.5,
    )
    map_metrics = detection_map(
        predictions=all_preds,
        targets=all_targets,
    )

    print("Split: val")
    print(f"Samples: {len(val_dataset)}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"mAP50:     {map_metrics['map50']:.4f}")
    print(f"mAP50-95:  {map_metrics['map50_95']:.4f}")
    print(
        f"TP={int(metrics['tp'])} "
        f"FP={int(metrics['fp'])} "
        f"FN={int(metrics['fn'])}"
    )

    payload = {
        "split": "val",
        "samples": len(val_dataset),
        "checkpoint": str(ckpt_path.as_posix()),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "map50": float(map_metrics["map50"]),
        "map50_95": float(map_metrics["map50_95"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "config": {
            "seed": cfg.seed,
            "score_threshold": cfg.score_threshold,
            "num_workers": cfg.num_workers,
        },
    }
    save_json(payload, cfg.val_metrics_out)
    print(f"Metrics saved to: {cfg.val_metrics_out}")


if __name__ == "__main__":
    main()
