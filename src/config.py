from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    root_dir: Path
    data_dir: Path
    train_split: Path
    val_split: Path
    test_split: Path
    checkpoint_dir: Path
    artifacts_dir: Path
    train_history_dir: Path
    plots_dir: Path
    reports_dir: Path
    val_metrics_out: Path
    test_metrics_out: Path
    run_config_out: Path
    train_log_name: str
    val_log_name: str
    device: str = "cpu"
    seed: int = 42
    num_classes: int = 2  # background + airplane
    epochs: int = 10
    batch_size: int = 1
    lr: float = 2e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    score_threshold: float = 0.35
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 2e-3
    lr_reduce_patience: int = 1
    lr_reduce_factor: float = 0.5
    lr_reduce_min: float = 1e-6
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    use_pretrained_backbone: bool = True
    save_visualizations: bool = False
    viz_count: int = 20
    val_split_name: str = "val"
    test_split_name: str = "test"


def get_default_config() -> TrainConfig:
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "src" / "data"
    checkpoint_dir = root_dir / "checkpoints"
    artifacts_dir = root_dir / "outputs"

    return TrainConfig(
        root_dir=root_dir,
        data_dir=data_dir,
        train_split=data_dir / "train.txt",
        val_split=data_dir / "validation.txt",
        test_split=data_dir / "test.txt",
        checkpoint_dir=checkpoint_dir,
        artifacts_dir=artifacts_dir,
        train_history_dir=artifacts_dir / "metrics",
        plots_dir=artifacts_dir / "plots",
        reports_dir=artifacts_dir / "reports",
        val_metrics_out=artifacts_dir / "metrics" / "val_metrics.json",
        test_metrics_out=artifacts_dir / "metrics" / "test_metrics.json",
        run_config_out=artifacts_dir / "metrics" / "run_config.json",
        train_log_name="train.log",
        val_log_name="val.log",
    )
