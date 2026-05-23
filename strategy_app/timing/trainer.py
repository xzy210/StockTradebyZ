from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import TimingDataset
from .model import TCNAttentionClassifier, TCNAttentionConfig


@dataclass(frozen=True)
class TimingTrainConfig:
    """训练参数配置。"""

    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    device: str = "auto"
    use_class_weight: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimingTrainResult:
    model: TCNAttentionClassifier
    best_state_dict: Dict[str, torch.Tensor]
    history: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def train_timing_model(
    dataset: TimingDataset,
    model_config: TCNAttentionConfig,
    train_config: TimingTrainConfig | None = None,
    progress_callback: Callable[[int, int, dict], None] | None = None,
) -> TimingTrainResult:
    """训练 TCN + Attention 三分类模型。"""

    cfg = train_config or TimingTrainConfig()
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    model = TCNAttentionClassifier(model_config).to(device)

    train_loader = _build_loader(dataset.x_train, dataset.y_train, cfg.batch_size, shuffle=True)
    val_loader = _build_loader(dataset.x_val, dataset.y_val, cfg.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(weight=_class_weight(dataset.y_train, device) if cfg.use_class_weight else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        if progress_callback is not None:
            progress_callback(epoch, cfg.epochs, dict(row))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    test_loader = _build_loader(dataset.x_test, dataset.y_test, cfg.batch_size, shuffle=False)
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, device)
    metrics = {
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_class_metrics": evaluate_class_metrics(model, dataset.x_test, dataset.y_test, device=device),
    }
    return TimingTrainResult(model=model, best_state_dict=best_state, history=history, metrics=metrics)


@torch.no_grad()
def evaluate_class_metrics(
    model: TCNAttentionClassifier,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> dict:
    model.eval()
    tensor_x = torch.as_tensor(x, dtype=torch.float32, device=device)
    logits = model(tensor_x)
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    labels = np.asarray(y)

    result = {}
    for cls in (0, 1, 2):
        true_positive = int(((pred == cls) & (labels == cls)).sum())
        predicted = int((pred == cls).sum())
        actual = int((labels == cls).sum())
        result[str(cls)] = {
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / actual if actual else 0.0,
            "support": actual,
        }
    return result


def _build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(
    model: TCNAttentionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        if is_train:
            loss.backward()
            optimizer.step()

        count = int(batch_y.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_correct += int((torch.argmax(logits, dim=1) == batch_y).sum().detach().cpu())
        total_count += count

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def _class_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
