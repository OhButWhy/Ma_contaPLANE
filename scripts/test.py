import sys
from pathlib import Path

import torch
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import get_default_config  # noqa: E402
from src.data_utils import (  # noqa: E402
    YoloDetectionDataset,
    create_dataloader,
    detection_prf1,
)
from src.model import create_model  # noqa: E402
from src.common import draw_predictions, save_json  # noqa: E402


def _match_counts(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int, int]:
    if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
        return 0, 0, 0
    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0])

    ious = box_iou(pred_boxes, gt_boxes)
    matched_gt = set()
    matched_pred = set()

    flat = []
    for i in range(ious.shape[0]):
        for j in range(ious.shape[1]):
            flat.append((float(ious[i, j]), i, j))
    flat.sort(key=lambda x: x[0], reverse=True)

    for iou, i, j in flat:
        if iou < iou_threshold:
            break
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)

    tp = len(matched_pred)
    fp = int(pred_boxes.shape[0] - tp)
    fn = int(gt_boxes.shape[0] - len(matched_gt))
    return tp, fp, fn


def main() -> None:
    cfg = get_default_config()
    ckpt_path = cfg.checkpoint_dir / cfg.best_checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg.test_metrics_out.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_visualizations:
        (cfg.artifacts_dir / "predictions").mkdir(parents=True, exist_ok=True)

    test_dataset = YoloDetectionDataset(cfg.test_split, cfg.data_dir)
    test_loader = create_dataloader(
        split_file=cfg.test_split,
        data_dir=cfg.data_dir,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = create_model(
        num_classes=cfg.num_classes,
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds = []
    all_targets = []
    per_image_stats = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_loader):
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

            pred_boxes = pred_cpu[0].get("boxes", torch.zeros((0, 4)))
            pred_scores = pred_cpu[0].get("scores", torch.zeros((0,)))
            if pred_boxes.numel() > 0:
                keep = pred_scores >= cfg.score_threshold
                pred_boxes = pred_boxes[keep]

            gt_boxes = target_cpu[0]["boxes"]
            tp_i, fp_i, fn_i = _match_counts(
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
                iou_threshold=0.5,
            )
            image_path = test_dataset.image_paths[idx]
            per_image_stats.append(
                {
                    "image": str(image_path.as_posix()),
                    "tp": int(tp_i),
                    "fp": int(fp_i),
                    "fn": int(fn_i),
                    "gt_count": int(gt_boxes.shape[0]),
                    "pred_count": int(pred_boxes.shape[0]),
                    "error_score": int(fp_i + fn_i),
                }
            )

            if cfg.save_visualizations and idx < cfg.viz_count:
                out_path = (
                    cfg.artifacts_dir
                    / "predictions"
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

    print("Split: test")
    print(f"Samples: {len(test_dataset)}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(
        f"TP={int(metrics['tp'])} "
        f"FP={int(metrics['fp'])} "
        f"FN={int(metrics['fn'])}"
    )

    payload = {
        "split": "test",
        "samples": len(test_dataset),
        "checkpoint": str(ckpt_path.as_posix()),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "config": {
            "seed": cfg.seed,
            "score_threshold": cfg.score_threshold,
            "num_workers": cfg.num_workers,
        },
    }
    save_json(payload, cfg.test_metrics_out)
    print(f"Metrics saved to: {cfg.test_metrics_out}")

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    worst_cases = sorted(
        per_image_stats,
        key=lambda x: (x["error_score"], x["fn"], x["fp"]),
        reverse=True,
    )[:20]
    detailed_report_path = cfg.reports_dir / "test_detailed_report.json"
    worst_cases_path = cfg.reports_dir / "test_worst_cases.json"
    text_report_path = cfg.reports_dir / "test_report.txt"

    save_json(
        {
            "summary": payload,
            "per_image": per_image_stats,
        },
        detailed_report_path,
    )
    save_json(
        {
            "summary": payload,
            "worst_cases": worst_cases,
        },
        worst_cases_path,
    )
    text_report_path.write_text(
        "\n".join(
            [
                "Test report",
                f"Samples: {len(test_dataset)}",
                f"Precision: {metrics['precision']:.4f}",
                f"Recall: {metrics['recall']:.4f}",
                f"F1: {metrics['f1']:.4f}",
                (
                    f"TP={int(metrics['tp'])} "
                    f"FP={int(metrics['fp'])} "
                    f"FN={int(metrics['fn'])}"
                ),
                "",
                "Top-20 worst cases (by FP+FN):",
            ]
            + [
                (
                    f"{idx + 1}. {item['image']} | "
                    f"tp={item['tp']} fp={item['fp']} fn={item['fn']}"
                )
                for idx, item in enumerate(worst_cases)
            ]
        ),
        encoding="utf-8",
    )

    print(f"Detailed report saved to: {detailed_report_path}")
    print(f"Worst-cases saved to: {worst_cases_path}")
    print(f"Text report saved to: {text_report_path}")


if __name__ == "__main__":
    main()
