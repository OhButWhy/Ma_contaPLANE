import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

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
from src.common import (  # noqa: E402
    save_history,
    save_plots,
    save_json,
    save_run_config,
    set_seed,
)


def main() -> None:
    cfg = get_default_config()

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.train_history_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    save_run_config(cfg, cli_name="train", path=cfg.run_config_out)

    set_seed(cfg.seed)
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    train_dataset = YoloDetectionDataset(cfg.train_split, cfg.data_dir)
    val_dataset = YoloDetectionDataset(cfg.val_split, cfg.data_dir)

    train_loader = create_dataloader(
        split_file=cfg.train_split,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
    )
    val_loader = create_dataloader(
        split_file=cfg.val_split,
        data_dir=cfg.data_dir,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    model = create_model(
        num_classes=cfg.num_classes,
        pretrained=cfg.use_pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_reduce_factor,
        patience=cfg.lr_reduce_patience,
        min_lr=cfg.lr_reduce_min,
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(
            train_loader,
            desc=f"Train epoch {epoch}/{cfg.epochs}",
            unit="batch",
        )
        for step, (images, targets) in enumerate(train_pbar, start=1):
            images = [img.to(device) for img in images]
            moved_targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            loss_dict = model(images, moved_targets)
            loss = torch.stack(list(loss_dict.values())).sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            train_pbar.set_postfix(loss=f"{(running_loss / step):.4f}")

        train_loss = running_loss / max(len(train_loader), 1)

        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            val_pbar = tqdm(
                val_loader,
                desc=f"Val epoch {epoch}/{cfg.epochs}",
                unit="batch",
                leave=False,
            )
            for images, targets in val_pbar:
                moved_images = [img.to(device) for img in images]
                outputs = model(moved_images)

                cpu_outputs = [
                    {k: v.detach().cpu() for k, v in out.items()}
                    for out in outputs
                ]
                cpu_targets = [
                    {k: v.detach().cpu() for k, v in target.items()}
                    for target in targets
                ]

                all_preds.extend(cpu_outputs)
                all_targets.extend(cpu_targets)

        val_metrics = detection_prf1(
            predictions=all_preds,
            targets=all_targets,
            score_threshold=cfg.score_threshold,
            iou_threshold=0.5,
        )
        scheduler.step(val_metrics["f1"])

        print(
            f"[epoch {epoch}] loss={train_loss:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_precision": float(val_metrics["precision"]),
                "val_recall": float(val_metrics["recall"]),
                "val_f1": float(val_metrics["f1"]),
            }
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "val_metrics": val_metrics,
            "config": {
                "num_classes": cfg.num_classes,
                "score_threshold": cfg.score_threshold,
                "seed": cfg.seed,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "num_workers": cfg.num_workers,
            },
        }
        last_checkpoint_path = cfg.checkpoint_dir / cfg.last_checkpoint_name
        torch.save(checkpoint_payload, last_checkpoint_path)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0
            checkpoint_payload["best_f1"] = best_f1
            checkpoint_payload["best_epoch"] = best_epoch
            torch.save(
                checkpoint_payload,
                cfg.checkpoint_dir / cfg.best_checkpoint_name,
            )
            print(f"Saved new best checkpoint with F1={best_f1:.4f}")
        else:
            improvement = val_metrics["f1"] - best_f1
            if improvement < cfg.early_stopping_min_delta:
                patience_counter += 1

        print(
            f"[epoch {epoch}] lr={optimizer.param_groups[0]['lr']:.6f} "
            f"patience={patience_counter}/{cfg.early_stopping_patience}"
        )

        if patience_counter >= cfg.early_stopping_patience:
            print(
                "Early stopping: "
                f"нет улучшений F1 >= {cfg.early_stopping_min_delta} "
                f"{cfg.early_stopping_patience} эпох"
            )
            break

    save_history(history, cfg.train_history_dir)
    save_plots(history, cfg.plots_dir)

    final_summary = {
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "epochs_completed": len(history),
        "checkpoint_best": str(
            (cfg.checkpoint_dir / cfg.best_checkpoint_name).as_posix()
        ),
        "checkpoint_last": str(
            (cfg.checkpoint_dir / cfg.last_checkpoint_name).as_posix()
        ),
    }
    save_json(final_summary, cfg.train_history_dir / "final_summary.json")

    print("Training completed.")


if __name__ == "__main__":
    main()
