from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou
from torchvision.transforms import functional as F


def _resolve_image_path(entry: str, data_dir: Path) -> Path:
    """Resolve split-file path to the current local dataset layout."""
    normalized = entry.strip().replace("\\", "/")
    if not normalized:
        return Path("")

    candidate = Path(normalized)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    basename = Path(normalized).name
    candidates = [
        data_dir / "img" / basename,
        data_dir / basename,
        Path(".") / normalized,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()

    # Fallback to expected location for this project.
    return (data_dir / "img" / basename).resolve()


def read_split(split_file: Path, data_dir: Path) -> List[Path]:
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    paths: List[Path] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        img_path = _resolve_image_path(line, data_dir)
        if img_path.exists():
            paths.append(img_path)
    return paths


def load_yolo_labels(
    label_path: Path,
    image_w: int,
    image_h: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    boxes: List[List[float]] = []
    labels: List[int] = []

    if not label_path.exists():
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue

        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            continue

        x1 = max((cx - w / 2.0) * image_w, 0.0)
        y1 = max((cy - h / 2.0) * image_h, 0.0)
        x2 = min((cx + w / 2.0) * image_w, float(image_w - 1))
        y2 = min((cy + h / 2.0) * image_h, float(image_h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        # In torchvision detection, class 0 is background.
        labels.append(class_id + 1)

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int64),
    )


class YoloDetectionDataset(Dataset):
    def __init__(self, split_file: Path, data_dir: Path):
        self.image_paths = read_split(split_file=split_file, data_dir=data_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label_path = image_path.with_suffix(".txt")
        boxes, labels = load_yolo_labels(label_path, width, height)

        if boxes.numel() > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }

        image_tensor = F.pil_to_tensor(image).float() / 255.0
        return image_tensor, target


def collate_fn(
    batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
):
    images, targets = zip(*batch)
    return list(images), list(targets)


def create_dataloader(
    split_file: Path,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = YoloDetectionDataset(split_file=split_file, data_dir=data_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )


def detection_prf1(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute object-level precision/recall/F1 with greedy IoU matching."""
    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(predictions, targets):
        gt_boxes = target["boxes"].detach().cpu()

        pred_boxes = pred.get("boxes", torch.zeros((0, 4))).detach().cpu()
        pred_scores = pred.get("scores", torch.zeros((0,))).detach().cpu()
        if pred_boxes.numel() > 0:
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]

        if gt_boxes.numel() == 0 and pred_boxes.numel() == 0:
            continue
        if gt_boxes.numel() == 0:
            fp += int(pred_boxes.shape[0])
            continue
        if pred_boxes.numel() == 0:
            fn += int(gt_boxes.shape[0])
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        flat_scores = []
        for i in range(ious.shape[0]):
            for j in range(ious.shape[1]):
                flat_scores.append((float(ious[i, j]), i, j))
        flat_scores.sort(key=lambda row: row[0], reverse=True)

        for iou, i, j in flat_scores:
            if iou < iou_threshold:
                break
            if i in matched_pred or j in matched_gt:
                continue
            matched_pred.add(i)
            matched_gt.add(j)

        tp += len(matched_pred)
        fp += int(pred_boxes.shape[0] - len(matched_pred))
        fn += int(gt_boxes.shape[0] - len(matched_gt))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
