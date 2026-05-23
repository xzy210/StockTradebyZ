from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .dataset import StandardScaler
from .features import TimingFeatureConfig
from .labels import TripleBarrierConfig
from .model import TCNAttentionConfig
from .trainer import TimingTrainConfig, TimingTrainResult


@dataclass(frozen=True)
class TimingModelManifest:
    """模型版本清单。"""

    model_version: str
    created_at: str
    symbols: list[str]
    frequency: str = "1d"
    data_start: str = ""
    data_end: str = ""
    feature_names: list[str] = field(default_factory=list)
    feature_config: dict = field(default_factory=dict)
    label_config: dict = field(default_factory=dict)
    dataset_config: dict = field(default_factory=dict)
    model_config: dict = field(default_factory=dict)
    train_config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    label_distribution: dict = field(default_factory=dict)
    dataset_artifact: dict = field(default_factory=dict)
    schema_version: str = "timing_model_manifest.v1"

    def to_dict(self) -> dict:
        return asdict(self)


def save_timing_model(
    *,
    output_dir: str | Path,
    train_result: TimingTrainResult,
    scaler: StandardScaler,
    feature_names: list[str],
    feature_config: TimingFeatureConfig,
    label_config: TripleBarrierConfig,
    dataset_config: Any,
    model_config: TCNAttentionConfig,
    train_config: TimingTrainConfig,
    symbols: list[str],
    frequency: str = "1d",
    data_start: str = "",
    data_end: str = "",
    label_distribution: dict | None = None,
    dataset_artifact: dict | None = None,
) -> Path:
    """保存模型权重、scaler、配置和训练指标。"""

    model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir = Path(output_dir) / model_version
    version_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_result.best_state_dict, version_dir / "model.pt")
    with (version_dir / "scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)

    _write_json(version_dir / "feature_config.json", feature_config.to_dict())
    _write_json(version_dir / "label_config.json", label_config.to_dict())
    _write_json(version_dir / "model_config.json", model_config.to_dict())
    _write_json(version_dir / "train_config.json", train_config.to_dict())
    _write_json(version_dir / "metrics.json", train_result.metrics)
    _write_json(version_dir / "history.json", train_result.history)

    manifest = TimingModelManifest(
        model_version=model_version,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        symbols=symbols,
        frequency=frequency,
        data_start=data_start,
        data_end=data_end,
        feature_names=feature_names,
        feature_config=feature_config.to_dict(),
        label_config=label_config.to_dict(),
        dataset_config=dataset_config.to_dict() if hasattr(dataset_config, "to_dict") else dict(dataset_config or {}),
        model_config=model_config.to_dict(),
        train_config=train_config.to_dict(),
        metrics=train_result.metrics,
        label_distribution=dict(label_distribution or {}),
        dataset_artifact=dict(dataset_artifact or {}),
    )
    _write_json(version_dir / "manifest.json", manifest.to_dict())
    return version_dir


def load_scaler(path: str | Path) -> StandardScaler:
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(payload), file, ensure_ascii=False, indent=2)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return value
