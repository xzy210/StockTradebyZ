from __future__ import annotations

import json
import pickle
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QDate, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from strategy_app.backtest import BacktestConfig, UnifiedBacktestEngine
from strategy_app.strategies.tcn_attention_timing_strategy import TCNAttentionTimingStrategy
from strategy_app.timing import (
    TimingDatasetConfig,
    TimingFeatureConfig,
    TripleBarrierConfig,
    build_timing_dataset,
    build_timing_features,
    build_triple_barrier_labels,
)
from strategy_app.timing.data_loader import load_timing_bars
from strategy_app.timing.dataset import TimingDataset, describe_labels
from strategy_app.timing.model import TCNAttentionConfig
from strategy_app.timing.model_store import load_scaler, save_timing_model
from strategy_app.timing.trainer import TimingTrainConfig, train_timing_model

MODEL_FREQUENCY_ROLE = Qt.ItemDataRole.UserRole.value + 1


class TimingTrainingThread(QThread):
    info_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, dict)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            dataset_dir = Path(self.params["dataset_dir"])
            self.info_signal.emit(f"加载已构建数据集: {dataset_dir}")
            dataset, dataset_summary = _load_timing_dataset_artifact(dataset_dir)
            feature_names = list(dataset_summary.get("feature_names") or dataset.feature_names)
            feature_config = TimingFeatureConfig(**_tuple_fields(dataset_summary.get("feature_config") or {}, ("momentum_windows", "ma_windows", "extra_feature_columns")))
            label_config = TripleBarrierConfig(**(dataset_summary.get("label_config") or {}))
            dataset_config = TimingDatasetConfig(**(dataset_summary.get("dataset_config") or {}))
            train_config = TimingTrainConfig(
                epochs=self.params["epochs"],
                batch_size=self.params["batch_size"],
                learning_rate=self.params["learning_rate"],
                weight_decay=self.params["weight_decay"],
                patience=self.params["patience"],
                device=self.params["device"],
                use_class_weight=self.params["use_class_weight"],
            )
            self.info_signal.emit(
                f"样本集: train={len(dataset.y_train)}, val={len(dataset.y_val)}, test={len(dataset.y_test)}"
            )

            model_config = TCNAttentionConfig(
                input_dim=dataset.num_features,
                channels=tuple(self.params["channels"]),
                kernel_size=self.params["kernel_size"],
                dropout=self.params["dropout"],
                attention_dim=self.params["attention_dim"],
            )
            self.info_signal.emit("开始训练 TCN + Attention 模型...")
            result = train_timing_model(
                dataset,
                model_config,
                train_config,
                progress_callback=lambda current, total, row: self.progress_signal.emit(current, total, row),
            )
            label_distribution = describe_labels(
                pd.concat(
                    [
                        pd.Series(dataset.y_train),
                        pd.Series(dataset.y_val),
                        pd.Series(dataset.y_test),
                    ],
                    ignore_index=True,
                ).to_numpy()
            )
            model_dir = save_timing_model(
                output_dir=self.params["output_dir"],
                train_result=result,
                scaler=dataset.scaler,
                feature_names=feature_names,
                feature_config=feature_config,
                label_config=label_config,
                dataset_config=dataset_config,
                model_config=model_config,
                train_config=train_config,
                symbols=list(dataset_summary.get("symbols") or []),
                frequency=str(dataset_summary.get("frequency") or ""),
                data_start=str(dataset_summary.get("start_date") or ""),
                data_end=str(dataset_summary.get("end_date") or ""),
                label_distribution=label_distribution,
                dataset_artifact={
                    "dataset_dir": str(dataset_dir),
                    "dataset_version": dataset_dir.name,
                    "schema_version": dataset_summary.get("schema_version", ""),
                },
            )
            self.finished_signal.emit(str(model_dir))
        except Exception as exc:
            traceback.print_exc()
            self.error_signal.emit(str(exc))


class TimingDataBuildThread(QThread):
    info_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            symbols = self.params["symbols"]
            data_dir = Path(self.params["data_dir"])
            frames = []
            feature_names = []
            feature_config = TimingFeatureConfig(
                momentum_windows=tuple(self.params["momentum_windows"]),
                ma_windows=tuple(self.params["ma_windows"]),
                volatility_window=self.params["volatility_window"],
            )
            label_config = TripleBarrierConfig(
                horizon=self.params["horizon"],
                up_mult=self.params["up_mult"],
                down_mult=self.params["down_mult"],
                volatility_window=self.params["volatility_window"],
            )
            dataset_config = TimingDatasetConfig(
                lookback=self.params["lookback"],
                train_ratio=self.params["train_ratio"],
                val_ratio=self.params["val_ratio"],
            )

            for index, symbol in enumerate(symbols, start=1):
                self.info_signal.emit(f"构建数据: 加载并处理 {symbol} ({index}/{len(symbols)})")
                raw = load_timing_bars(
                    data_dir,
                    symbol,
                    frequency=self.params["frequency"],
                    start_date=self.params["start_date"],
                    end_date=self.params["end_date"],
                    auto_fetch=True,
                    log_callback=self.info_signal.emit,
                )
                features, names = build_timing_features(raw, feature_config)
                labeled = build_triple_barrier_labels(features, label_config)
                labeled["symbol"] = symbol
                frames.append(labeled)
                if not feature_names:
                    feature_names = names

            all_data = pd.concat(frames, ignore_index=True)
            self.info_signal.emit("构造滑动窗口训练数据...")
            dataset = build_timing_dataset(all_data, feature_names, dataset_config)

            dataset_dir = Path(self.params["output_dir"]) / "datasets" / datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_dir.mkdir(parents=True, exist_ok=True)
            all_data.to_parquet(dataset_dir / "labeled_data.parquet", index=False)
            dataset.metadata.to_parquet(dataset_dir / "metadata.parquet", index=False)
            with (dataset_dir / "scaler.pkl").open("wb") as file:
                pickle.dump(dataset.scaler, file)
            np.savez_compressed(
                dataset_dir / "dataset.npz",
                x_train=dataset.x_train,
                y_train=dataset.y_train,
                x_val=dataset.x_val,
                y_val=dataset.y_val,
                x_test=dataset.x_test,
                y_test=dataset.y_test,
            )

            label_distribution = describe_labels(
                pd.concat(
                    [
                        pd.Series(dataset.y_train),
                        pd.Series(dataset.y_val),
                        pd.Series(dataset.y_test),
                    ],
                    ignore_index=True,
                ).to_numpy()
            )
            split_summary = {
                "train_samples": int(len(dataset.y_train)),
                "val_samples": int(len(dataset.y_val)),
                "test_samples": int(len(dataset.y_test)),
                "feature_count": int(dataset.num_features),
                "feature_names": feature_names,
                "label_distribution": label_distribution,
                "symbols": symbols,
                "frequency": self.params["frequency"],
                "start_date": self.params["start_date"],
                "end_date": self.params["end_date"],
                "feature_config": feature_config.to_dict(),
                "label_config": label_config.to_dict(),
                "dataset_config": dataset_config.to_dict(),
                "schema_version": "timing_dataset_artifact.v1",
            }
            with (dataset_dir / "summary.json").open("w", encoding="utf-8") as file:
                json.dump(split_summary, file, ensure_ascii=False, indent=2)

            self.finished_signal.emit({"dataset_dir": str(dataset_dir), **split_summary})
        except Exception as exc:
            traceback.print_exc()
            self.error_signal.emit(str(exc))


class TimingBacktestThread(QThread):
    info_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            symbol = self.params["symbol"]
            self.info_signal.emit(f"加载 {symbol} {self.params['frequency']} 回测数据...")
            frame = load_timing_bars(
                data_dir=self.params["data_dir"],
                symbol=symbol,
                frequency=self.params["frequency"],
                start_date=self.params["start_date"],
                end_date=self.params["end_date"],
                auto_fetch=True,
                log_callback=self.info_signal.emit,
            )
            if frame.empty:
                raise ValueError(f"未加载到 {symbol} 的有效数据")

            strategy = TCNAttentionTimingStrategy()
            strategy.set_params(
                {
                    "model_dir": self.params["model_dir"],
                    "device": self.params["device"],
                    "up_threshold": self.params["up_threshold"],
                    "down_threshold": self.params["down_threshold"],
                    "direction_margin": self.params["direction_margin"],
                    "target_percent": self.params["target_percent"],
                    "frequency": self.params["frequency"],
                }
            )
            engine = UnifiedBacktestEngine(BacktestConfig(initial_cash=self.params["initial_cash"], mode="bar"))
            self.info_signal.emit("运行统一回测引擎...")
            result = engine.run(strategy, frame, code=symbol, mode="bar")
            self.finished_signal.emit(result)
        except Exception as exc:
            traceback.print_exc()
            self.error_signal.emit(str(exc))


class TimingStrategyWidget(QWidget):
    """时序策略训练与回测研究页。"""

    def __init__(self, data_dir: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.data_dir = data_dir
        self.project_root = Path(__file__).resolve().parents[2]
        self.models_dir = self.project_root / "models" / "timing" / "tcn_attention"
        self.training_thread: TimingTrainingThread | None = None
        self.data_build_thread: TimingDataBuildThread | None = None
        self.backtest_thread: TimingBacktestThread | None = None
        self._last_labeled_df: pd.DataFrame | None = None
        self._barrier_overlay_items: list = []
        self._training_history: list[dict] = []
        self._setup_ui()
        self.refresh_datasets()
        self.refresh_models()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.log_edit = QTextEdit(self)
        self.log_edit.setReadOnly(True)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_data_build_tab(), "数据构建")
        tabs.addTab(self._build_train_tab(), "训练")
        tabs.addTab(self._build_backtest_tab(), "回测")
        tabs.addTab(self._build_label_viz_tab(), "标签可视化")
        layout.addWidget(tabs, 1)

    def _build_data_build_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(tab)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(scroll)
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 8, 8, 8)

        data_group = QGroupBox("数据、标签与样本", inner)
        data_form = QFormLayout(data_group)
        data_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.train_symbols_edit = QLineEdit("000001", data_group)
        self.train_frequency_combo = _frequency_combo(data_group)
        self.train_start_edit = _date_edit(QDate(2024, 1, 1), data_group)
        self.train_end_edit = _date_edit(QDate.currentDate(), data_group)
        self.lookback_spin = _spin(2, 1000, 60, data_group)
        self.horizon_spin = _spin(1, 240, 12, data_group)
        self.train_volatility_window_spin = _spin(5, 240, 20, data_group)
        self.train_up_mult_spin = _float_spin(0.1, 10.0, 1.5, data_group, decimals=2, step=0.1)
        self.train_down_mult_spin = _float_spin(0.1, 10.0, 1.0, data_group, decimals=2, step=0.1)
        self.train_ratio_spin = _prob_spin(0.7, data_group)
        self.val_ratio_spin = _prob_spin(0.15, data_group)
        data_form.addRow("标的代码", self.train_symbols_edit)
        data_form.addRow("K线周期", self.train_frequency_combo)
        data_form.addRow("开始日期", self.train_start_edit)
        data_form.addRow("结束日期", self.train_end_edit)
        data_form.addRow("lookback", self.lookback_spin)
        data_form.addRow("horizon", self.horizon_spin)
        data_form.addRow("波动率窗口", self.train_volatility_window_spin)
        data_form.addRow("上障碍倍数", self.train_up_mult_spin)
        data_form.addRow("下障碍倍数", self.train_down_mult_spin)
        data_form.addRow("训练集比例", self.train_ratio_spin)
        data_form.addRow("验证集比例", self.val_ratio_spin)

        self.build_data_button = QPushButton("构建数据", data_group)
        self.build_data_button.setToolTip("生成特征、三障碍标签和训练/验证/测试数据，不训练模型")
        self.build_data_button.clicked.connect(self.start_data_build)
        data_form.addRow(self.build_data_button)

        inner_layout.addWidget(data_group)
        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)
        return tab

    def _build_train_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, tab)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        scroll = QScrollArea(splitter)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget(scroll)
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 8, 8, 8)

        dataset_group = QGroupBox("训练数据集（只读）", inner)
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_row = QHBoxLayout()
        self.dataset_combo = QComboBox(dataset_group)
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_selection_changed)
        refresh_dataset_btn = QPushButton("刷新数据集", dataset_group)
        refresh_dataset_btn.clicked.connect(self.refresh_datasets)
        dataset_row.addWidget(self.dataset_combo, 1)
        dataset_row.addWidget(refresh_dataset_btn)
        dataset_layout.addLayout(dataset_row)

        self.dataset_summary_edit = QTextEdit(dataset_group)
        self.dataset_summary_edit.setReadOnly(True)
        self.dataset_summary_edit.setMinimumHeight(160)
        self.dataset_summary_edit.setPlaceholderText("请选择或先构建一个数据集。")
        dataset_layout.addWidget(self.dataset_summary_edit)

        model_group = QGroupBox("模型结构", inner)
        model_form = QFormLayout(model_group)
        model_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.channels_edit = QLineEdit("64,64,64", model_group)
        self.channels_edit.setToolTip("TCN 每层通道数，使用英文逗号分隔，例如 64,64,64 或 32,32")
        self.kernel_size_spin = _spin(1, 15, 3, model_group)
        self.dropout_spin = _prob_spin(0.2, model_group)
        self.attention_dim_spin = _spin(1, 1024, 64, model_group)
        model_form.addRow("channels", self.channels_edit)
        model_form.addRow("kernel size", self.kernel_size_spin)
        model_form.addRow("dropout", self.dropout_spin)
        model_form.addRow("attention dim", self.attention_dim_spin)

        train_group = QGroupBox("训练过程", inner)
        train_form = QFormLayout(train_group)
        train_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.epochs_spin = _spin(1, 500, 10, train_group)
        self.batch_spin = _spin(8, 2048, 128, train_group)
        self.learning_rate_spin = _decimal_spin(0.000001, 1.0, 0.001, train_group)
        self.weight_decay_spin = _decimal_spin(0.0, 1.0, 0.0001, train_group)
        self.patience_spin = _spin(1, 200, 5, train_group)
        self.use_class_weight_check = QCheckBox("启用类别权重", train_group)
        self.use_class_weight_check.setChecked(True)
        self.train_button = QPushButton("开始训练", train_group)
        self.train_button.clicked.connect(self.start_training)
        train_form.addRow("epochs", self.epochs_spin)
        train_form.addRow("batch size", self.batch_spin)
        train_form.addRow("learning rate", self.learning_rate_spin)
        train_form.addRow("weight decay", self.weight_decay_spin)
        train_form.addRow("patience", self.patience_spin)
        train_form.addRow("类别权重", self.use_class_weight_check)
        train_form.addRow(self.train_button)

        inner_layout.addWidget(dataset_group)
        inner_layout.addWidget(model_group)
        inner_layout.addWidget(train_group)
        inner_layout.addStretch(1)
        scroll.setWidget(inner)
        splitter.addWidget(scroll)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        progress_group = QGroupBox("训练进度与曲线", right_panel)
        progress_layout = QVBoxLayout(progress_group)
        self.train_progress_bar = QProgressBar(progress_group)
        self.train_progress_bar.setRange(0, 100)
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setFormat("等待训练")
        progress_layout.addWidget(self.train_progress_bar)

        self.train_chart = pg.GraphicsLayoutWidget(progress_group)
        self.train_chart.setMinimumHeight(420)
        self.train_loss_plot = self.train_chart.addPlot(row=0, col=0, title="Loss 曲线")
        self.train_loss_plot.showGrid(x=True, y=True, alpha=0.25)
        self.train_accuracy_plot = self.train_chart.addPlot(row=1, col=0, title="Accuracy 曲线")
        self.train_accuracy_plot.showGrid(x=True, y=True, alpha=0.25)
        progress_layout.addWidget(self.train_chart, 1)
        right_layout.addWidget(progress_group, 1)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1000])
        return tab

    def _build_backtest_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, tab)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        left_panel = QWidget(splitter)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(520)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 6, 8)

        param_scroll = QScrollArea(left_panel)
        param_scroll.setWidgetResizable(True)
        param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        param_scroll.setMinimumHeight(260)
        param_inner = QWidget(param_scroll)
        param_layout = QVBoxLayout(param_inner)
        param_layout.setContentsMargins(0, 0, 0, 0)

        form_group = QGroupBox("回测参数", tab)
        form = QFormLayout(form_group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo = QComboBox(form_group)
        self.model_combo.currentIndexChanged.connect(self._sync_backtest_frequency_from_model)
        refresh_btn = QPushButton("刷新模型", form_group)
        refresh_btn.clicked.connect(self.refresh_models)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(refresh_btn)

        self.backtest_symbol_edit = QLineEdit("000001", form_group)
        self.backtest_frequency_combo = _frequency_combo(form_group)
        self.backtest_start_edit = _date_edit(QDate(2024, 1, 1), form_group)
        self.backtest_end_edit = _date_edit(QDate.currentDate(), form_group)
        self.initial_cash_spin = _double_spin(10000, 100000000, 100000, form_group)
        self.up_threshold_spin = _prob_spin(0.55, form_group)
        self.down_threshold_spin = _prob_spin(0.55, form_group)
        self.margin_spin = _prob_spin(0.15, form_group)
        self.target_percent_spin = _prob_spin(0.5, form_group)
        self.backtest_button = QPushButton("开始回测", form_group)
        self.backtest_button.clicked.connect(self.start_backtest)

        form.addRow("模型版本", model_row)
        form.addRow("标的代码", self.backtest_symbol_edit)
        form.addRow("K线周期", self.backtest_frequency_combo)
        form.addRow("开始日期", self.backtest_start_edit)
        form.addRow("结束日期", self.backtest_end_edit)
        form.addRow("初始资金", self.initial_cash_spin)
        form.addRow("看多阈值", self.up_threshold_spin)
        form.addRow("看空阈值", self.down_threshold_spin)
        form.addRow("方向差阈值", self.margin_spin)
        form.addRow("目标仓位", self.target_percent_spin)
        form.addRow(self.backtest_button)
        param_layout.addWidget(form_group)
        param_layout.addStretch(1)
        param_scroll.setWidget(param_inner)
        left_layout.addWidget(param_scroll, 0)

        self.result_label = QLabel("暂无回测结果", left_panel)
        self.result_label.setWordWrap(True)
        left_layout.addWidget(self.result_label, 0)

        self.log_edit.setParent(left_panel)
        self.log_edit.setMinimumHeight(160)
        left_layout.addWidget(self.log_edit, 1)
        splitter.addWidget(left_panel)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 8, 8, 8)

        self.backtest_chart = pg.GraphicsLayoutWidget(right_panel)
        self.backtest_chart.setMinimumHeight(420)
        self.backtest_equity_plot = self.backtest_chart.addPlot(row=0, col=0, title="资产曲线 / 收盘价")
        self.backtest_equity_plot.showGrid(x=True, y=True, alpha=0.25)
        self.backtest_signal_plot = self.backtest_chart.addPlot(row=1, col=0, title="模型概率与阈值")
        self.backtest_signal_plot.showGrid(x=True, y=True, alpha=0.25)
        self.backtest_signal_plot.setYRange(0, 1)
        self.backtest_position_plot = self.backtest_chart.addPlot(row=2, col=0, title="持仓数量")
        self.backtest_position_plot.showGrid(x=True, y=True, alpha=0.25)
        right_layout.addWidget(self.backtest_chart, 1)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1200])
        return tab

    def _build_label_viz_tab(self) -> QWidget:
        tab = QWidget(self)
        root = QHBoxLayout(tab)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, tab)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # —— 左：窄参数栏（可滚动） ——
        left_scroll = QScrollArea(splitter)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(260)
        left_scroll.setMaximumWidth(380)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setContentsMargins(4, 4, 4, 4)

        form_group = QGroupBox("数据与三障碍参数\n（与训练页一致）", left_inner)
        form_group.setStyleSheet("QGroupBox { font-size: 11px; }")
        form = QFormLayout(form_group)
        form.setFieldGrowthPolicy(form.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.viz_symbol_edit = QLineEdit("000001", form_group)
        self.viz_frequency_combo = _frequency_combo(form_group)
        self.viz_start_edit = _date_edit(QDate(2024, 1, 1), form_group)
        self.viz_end_edit = _date_edit(QDate.currentDate(), form_group)
        self.viz_horizon_spin = _spin(1, 240, 12, form_group)
        self.viz_vol_spin = _spin(5, 120, 20, form_group)
        self.viz_up_mult_spin = QDoubleSpinBox(form_group)
        self.viz_up_mult_spin.setRange(0.1, 10.0)
        self.viz_up_mult_spin.setDecimals(2)
        self.viz_up_mult_spin.setSingleStep(0.1)
        self.viz_up_mult_spin.setValue(1.5)
        self.viz_down_mult_spin = QDoubleSpinBox(form_group)
        self.viz_down_mult_spin.setRange(0.1, 10.0)
        self.viz_down_mult_spin.setDecimals(2)
        self.viz_down_mult_spin.setSingleStep(0.1)
        self.viz_down_mult_spin.setValue(1.0)

        btn_sync = QPushButton("同步训练页", form_group)
        btn_sync.clicked.connect(self._sync_label_viz_from_train)
        btn_refresh = QPushButton("刷新图表", form_group)
        btn_refresh.clicked.connect(self._refresh_label_chart)
        btn_export = QPushButton("导出 CSV", form_group)
        btn_export.clicked.connect(self._export_label_csv)
        btn_clear_box = QPushButton("清除框选", form_group)
        btn_clear_box.setToolTip("去掉当前在图表上绘制的三障碍矩形与退出点标记")
        btn_clear_box.clicked.connect(self._clear_barrier_overlays)
        btn_col = QVBoxLayout()
        btn_col.addWidget(btn_sync)
        btn_col.addWidget(btn_refresh)
        btn_col.addWidget(btn_export)
        btn_col.addWidget(btn_clear_box)
        btn_wrap = QWidget(form_group)
        btn_wrap.setLayout(btn_col)

        form.addRow("标的代码", self.viz_symbol_edit)
        form.addRow("K线周期", self.viz_frequency_combo)
        form.addRow("开始日期", self.viz_start_edit)
        form.addRow("结束日期", self.viz_end_edit)
        form.addRow("horizon", self.viz_horizon_spin)
        form.addRow("波动率窗口", self.viz_vol_spin)
        form.addRow("上障碍倍数", self.viz_up_mult_spin)
        form.addRow("下障碍倍数", self.viz_down_mult_spin)
        form.addRow(btn_wrap)

        self.viz_stats_label = QLabel(
            "统计：刷新图表后显示。图例：浅蓝=收盘；绿/黄/红=标签 1/0/-1。",
            form_group,
        )
        self.viz_stats_label.setWordWrap(True)
        form.addRow(self.viz_stats_label)

        tip = QLabel(
            "图表操作：滚轮缩放；按住左键拖动平移；可拖动中间分隔条调左右宽度。"
            "单击任一标签散点可绘制该根 K 线的三障碍矩形（止盈/止损价与 horizon 时间窗）。",
            form_group,
        )
        tip.setWordWrap(True)
        tip.setProperty("class", "description")
        form.addRow(tip)

        left_layout.addWidget(form_group)
        left_layout.addStretch(1)
        left_scroll.setWidget(left_inner)
        splitter.addWidget(left_scroll)

        # —— 右：大图区域 ——
        self.label_plot = pg.PlotWidget(splitter)
        self.label_plot.setLabel("left", "价格")
        self.label_plot.setLabel("bottom", "K 线序号（按日期升序）")
        self.label_plot.showGrid(x=True, y=True, alpha=0.3)
        self.label_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label_plot.setMinimumHeight(320)
        self._configure_label_plot_interaction()
        splitter.addWidget(self.label_plot)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 1200])
        return tab

    def _configure_label_plot_interaction(self) -> None:
        """左键拖动平移；滚轮缩放 XY（可按需只缩放一个方向）。"""
        vb = self.label_plot.getPlotItem().getViewBox()
        vb.setMouseMode(pg.ViewBox.PanMode)
        vb.setMouseEnabled(x=True, y=True)
        vb.setAspectLocked(False)

    def _reset_label_plot(self) -> None:
        """Clear plot contents and reset the legend so repeated refreshes do not duplicate entries."""
        self.label_plot.clear()
        plot_item = self.label_plot.getPlotItem()
        if plot_item.legend is None:
            self.label_plot.addLegend(offset=(10, 10))
        else:
            plot_item.legend.clear()

    def _clear_barrier_overlays(self) -> None:
        """移除三障碍高亮图形（不清除主曲线与散点）。"""
        for item in self._barrier_overlay_items:
            try:
                self.label_plot.removeItem(item)
            except Exception:
                pass
        self._barrier_overlay_items.clear()

    def _on_label_point_clicked(self, *args) -> None:
        points = None
        if len(args) == 3:
            _, points, _ = args
        elif len(args) == 2:
            _, points = args
        elif len(args) == 1:
            points = args[0]
        if not points or self._last_labeled_df is None or self._last_labeled_df.empty:
            return
        spot = points[0]
        pos = spot.pos()
        row = int(round(float(pos.x())))
        self._draw_barrier_quad_for_row(row)

    def _draw_barrier_quad_for_row(self, row: int) -> None:
        """绘制索引 row 处的三障碍区间：左界=标签观测 bar，右界=+horizon；上下界为止盈/止损价。"""
        df = self._last_labeled_df
        if df is None or df.empty:
            return
        if row < 0 or row >= len(df):
            self._log(f"无效 K 线索引: {row}")
            return
        label = df.at[row, "tb_label"]
        if not np.isfinite(label):
            QMessageBox.information(
                self,
                "无标签",
                f"索引 {row} 处没有有效的 triple-barrier 标签（多为区间末尾未标注）。",
            )
            return

        upper = float(df.at[row, "tb_upper_price"])
        lower = float(df.at[row, "tb_lower_price"])
        horizon = int(df.at[row, "tb_horizon"])
        if not np.isfinite(upper) or not np.isfinite(lower) or horizon <= 0:
            self._log(f"索引 {row} 缺少障碍价格或 horizon")
            return

        x0 = float(row)
        x1 = float(row + horizon)
        self._clear_barrier_overlays()

        dash_pen = pg.mkPen("#eceff4", width=1.2, style=Qt.PenStyle.DashLine)

        # 水平条带 + FillBetween 得到与 horizon、止盈/止损一致的矩形内部填充
        pen_hidden = pg.mkPen(width=0)
        pen_hidden.setStyle(Qt.PenStyle.NoPen)
        c_up = pg.PlotDataItem([x0, x1], [upper, upper], pen=pen_hidden)
        c_lo = pg.PlotDataItem([x0, x1], [lower, lower], pen=pen_hidden)
        fill = pg.FillBetweenItem(c_lo, c_up, brush=pg.mkBrush(136, 192, 208, 55))
        self.label_plot.addItem(c_up)
        self.label_plot.addItem(c_lo)
        self.label_plot.addItem(fill)
        self._barrier_overlay_items.extend([c_up, c_lo, fill])

        # 矩形四边描边
        top = pg.PlotDataItem([x0, x1], [upper, upper], pen=dash_pen)
        bot = pg.PlotDataItem([x0, x1], [lower, lower], pen=dash_pen)
        self.label_plot.addItem(top)
        self.label_plot.addItem(bot)
        self._barrier_overlay_items.extend([top, bot])

        # horizon 对应的左右垂直边界
        for xv in (x0, x1):
            line = pg.PlotDataItem([xv, xv], [lower, upper], pen=dash_pen)
            self.label_plot.addItem(line)
            self._barrier_overlay_items.append(line)

        exit_idx = int(df.at[row, "tb_exit_index"])
        exit_price = float(df.at[row, "tb_exit_price"])
        trig = str(df.at[row, "tb_trigger_type"] or "")
        if exit_idx >= 0 and exit_idx < len(df) and np.isfinite(exit_price):
            exit_scatter = pg.ScatterPlotItem(
                pos=np.array([[float(exit_idx), exit_price]]),
                size=14,
                pen=pg.mkPen("#d08770", width=2),
                brush=pg.mkBrush("#d08770"),
                symbol="star",
            )
            self.label_plot.addItem(exit_scatter)
            self._barrier_overlay_items.append(exit_scatter)

        names = {-1: "下看空", 0: "震荡", 1: "上看多"}
        self._log(
            f"三障碍框 索引={row} 标签={names.get(int(label), label)} horizon={horizon} "
            f"止盈={upper:.4f} 止损={lower:.4f} 时间窗[{int(x0)},{int(x1)}] "
            f"实际退出 bar={exit_idx} 类型={trig} 价={exit_price:.4f}"
        )

    def _sync_label_viz_from_train(self) -> None:
        syms = _parse_symbols(self.train_symbols_edit.text())
        self.viz_symbol_edit.setText(syms[0] if syms else "000001")
        self.viz_frequency_combo.setCurrentIndex(self.train_frequency_combo.currentIndex())
        self.viz_start_edit.setDate(self.train_start_edit.date())
        self.viz_end_edit.setDate(self.train_end_edit.date())
        self.viz_horizon_spin.setValue(self.horizon_spin.value())
        self.viz_vol_spin.setValue(self.train_volatility_window_spin.value())
        self.viz_up_mult_spin.setValue(self.train_up_mult_spin.value())
        self.viz_down_mult_spin.setValue(self.train_down_mult_spin.value())
        self._log("标签可视化：已从训练页同步标的、日期与标签参数。")

    def _refresh_label_chart(self) -> None:
        try:
            syms = _parse_symbols(self.viz_symbol_edit.text())
            if not syms:
                QMessageBox.warning(self, "参数错误", "请填写标的代码")
                return
            symbol = syms[0]
            start = self.viz_start_edit.date().toString("yyyy-MM-dd")
            end = self.viz_end_edit.date().toString("yyyy-MM-dd")
            raw = load_timing_bars(
                self.data_dir,
                symbol,
                frequency=self.viz_frequency_combo.currentData(),
                start_date=start,
                end_date=end,
                auto_fetch=True,
                log_callback=self._log,
            )
            if raw.empty:
                QMessageBox.warning(self, "无数据", "该区间无 K 线，请调整日期或标的")
                return
            feature_config = TimingFeatureConfig(
                momentum_windows=(3, 5, 15),
                ma_windows=(20,),
                volatility_window=self.viz_vol_spin.value(),
            )
            label_config = TripleBarrierConfig(
                horizon=self.viz_horizon_spin.value(),
                up_mult=float(self.viz_up_mult_spin.value()),
                down_mult=float(self.viz_down_mult_spin.value()),
                volatility_window=self.viz_vol_spin.value(),
            )
            features, _ = build_timing_features(raw, feature_config)
            labeled = build_triple_barrier_labels(features, label_config)
            labeled["symbol"] = symbol
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "生成标签失败", str(exc))
            return

        self._last_labeled_df = labeled
        counts = labeled["tb_label"].value_counts(dropna=True)
        n1 = int(counts.get(1, 0))
        n0 = int(counts.get(0, 0))
        nm1 = int(counts.get(-1, 0))
        nan_ct = int(labeled["tb_label"].isna().sum())
        self.viz_stats_label.setText(
            f"标的 {symbol} | K 线 {len(labeled)} | "
            f"上看多(1): {n1} | 震荡(0): {n0} | 下看空(-1): {nm1} | 末尾未标注(NaN): {nan_ct}"
        )

        self._reset_label_plot()
        self._barrier_overlay_items.clear()
        x = np.arange(len(labeled), dtype=float)
        close = labeled["close"].to_numpy(dtype=float)
        self.label_plot.plot(x, close, pen=pg.mkPen("#88c0d0", width=1.2), name="收盘")

        palette = {
            1: ("#a3be8c", "o", "上看多(1)"),
            0: ("#ebcb8b", "s", "震荡(0)"),
            -1: ("#bf616a", "t", "下看空(-1)"),
        }
        for lab, (color, symb, _) in palette.items():
            mask = labeled["tb_label"] == lab
            idx = np.flatnonzero(mask.to_numpy())
            if len(idx) == 0:
                continue
            y = close[idx]
            scatter = pg.ScatterPlotItem(
                x=idx.astype(float),
                y=y,
                size=9,
                pen=pg.mkPen(color),
                brush=pg.mkBrush(color),
                symbol=symb,
                hoverable=True,
            )
            if hasattr(scatter, "setClickable"):
                scatter.setClickable(True)
            scatter.sigClicked.connect(self._on_label_point_clicked)
            self.label_plot.addItem(scatter)

        legend = self.label_plot.getPlotItem().legend
        if legend is not None:
            for color, symb, title in (
                ("#a3be8c", "o", "看多 (+1)"),
                ("#ebcb8b", "s", "震荡 (0)"),
                ("#bf616a", "t", "看空 (-1)"),
            ):
                legend.addItem(
                    pg.ScatterPlotItem(
                        size=9,
                        pen=pg.mkPen(color),
                        brush=pg.mkBrush(color),
                        symbol=symb,
                    ),
                    title,
                )

        vb = self.label_plot.getPlotItem().getViewBox()
        vb.autoRange(padding=0.05)
        self._log(f"标签图表已刷新: {symbol} ({start} ~ {end})")

    def _export_label_csv(self) -> None:
        if self._last_labeled_df is None or self._last_labeled_df.empty:
            QMessageBox.information(self, "提示", "请先点击「刷新图表」生成标签")
            return
        default_name = (
            f"{self._last_labeled_df['symbol'].iloc[-1]}_timing_labels.csv"
            if "symbol" in self._last_labeled_df.columns
            else "timing_labels.csv"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出标签明细",
            str(Path(self.data_dir).parent / "exports" / default_name),
            "CSV (*.csv)",
        )
        if not path:
            return
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            c
            for c in (
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tb_label",
                "tb_trigger_type",
                "tb_exit_index",
                "tb_exit_time",
                "tb_exit_price",
                "tb_upper_price",
                "tb_lower_price",
                "tb_horizon",
            )
            if c in self._last_labeled_df.columns
        ]
        self._last_labeled_df[cols].to_csv(export_path, index=False, encoding="utf-8-sig")
        self._log(f"已导出: {export_path}")
        QMessageBox.information(self, "完成", f"已保存\n{export_path}")

    def _collect_data_build_params(self) -> dict | None:
        train_ratio = self.train_ratio_spin.value()
        val_ratio = self.val_ratio_spin.value()
        if train_ratio + val_ratio >= 0.95:
            QMessageBox.warning(self, "参数错误", "训练集比例 + 验证集比例需要小于 0.95，给测试集保留足够样本")
            return None
        params = {
            "symbols": _parse_symbols(self.train_symbols_edit.text()),
            "data_dir": self.data_dir,
            "output_dir": str(self.models_dir),
            "frequency": self.train_frequency_combo.currentData(),
            "start_date": self.train_start_edit.date().toString("yyyy-MM-dd"),
            "end_date": self.train_end_edit.date().toString("yyyy-MM-dd"),
            "lookback": self.lookback_spin.value(),
            "horizon": self.horizon_spin.value(),
            "momentum_windows": (3, 5, 15),
            "ma_windows": (20,),
            "volatility_window": self.train_volatility_window_spin.value(),
            "up_mult": self.train_up_mult_spin.value(),
            "down_mult": self.train_down_mult_spin.value(),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        }
        if not params["symbols"]:
            QMessageBox.warning(self, "参数错误", "请至少输入一个标的代码")
            return None
        return params

    def _collect_training_params(self) -> dict | None:
        try:
            channels = _parse_channels(self.channels_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "参数错误", str(exc))
            return None
        dataset_dir = self.dataset_combo.currentData()
        if not dataset_dir:
            QMessageBox.warning(self, "缺少数据集", "请先点击「构建数据」或选择一个已构建数据集")
            return None
        return {
            "dataset_dir": dataset_dir,
            "output_dir": str(self.models_dir),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "learning_rate": self.learning_rate_spin.value(),
            "weight_decay": self.weight_decay_spin.value(),
            "patience": self.patience_spin.value(),
            "use_class_weight": self.use_class_weight_check.isChecked(),
            "channels": channels,
            "kernel_size": self.kernel_size_spin.value(),
            "dropout": self.dropout_spin.value(),
            "attention_dim": self.attention_dim_spin.value(),
            "device": "auto",
        }

    def start_data_build(self) -> None:
        if self.data_build_thread and self.data_build_thread.isRunning():
            QMessageBox.warning(self, "构建中", "已有数据构建任务正在运行")
            return
        params = self._collect_data_build_params()
        if params is None:
            return
        self.build_data_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self._log("启动时序策略数据构建...")
        self.data_build_thread = TimingDataBuildThread(params)
        self.data_build_thread.info_signal.connect(self._log)
        self.data_build_thread.finished_signal.connect(self._on_data_build_finished)
        self.data_build_thread.error_signal.connect(self._on_data_build_error)
        self.data_build_thread.start()

    def start_training(self) -> None:
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "训练中", "已有训练任务正在运行")
            return
        params = self._collect_training_params()
        if params is None:
            return
        self.train_button.setEnabled(False)
        self.build_data_button.setEnabled(False)
        self._training_history = []
        self.train_progress_bar.setRange(0, int(params.get("epochs") or 1))
        self.train_progress_bar.setValue(0)
        self.train_progress_bar.setFormat("训练中: 0/%m")
        self._refresh_training_charts(self._training_history)
        self._log("启动时序策略训练...")
        self.training_thread = TimingTrainingThread(params)
        self.training_thread.info_signal.connect(self._log)
        self.training_thread.progress_signal.connect(self._on_training_progress)
        self.training_thread.finished_signal.connect(self._on_training_finished)
        self.training_thread.error_signal.connect(self._on_training_error)
        self.training_thread.start()

    def start_backtest(self) -> None:
        if self.backtest_thread and self.backtest_thread.isRunning():
            QMessageBox.warning(self, "回测中", "已有回测任务正在运行")
            return
        model_dir = self.model_combo.currentData()
        if not model_dir:
            QMessageBox.warning(self, "缺少模型", "请先训练或选择模型版本")
            return
        params = {
            "symbol": _parse_symbols(self.backtest_symbol_edit.text())[0],
            "data_dir": self.data_dir,
            "model_dir": model_dir,
            "frequency": self.backtest_frequency_combo.currentData(),
            "start_date": self.backtest_start_edit.date().toString("yyyy-MM-dd"),
            "end_date": self.backtest_end_edit.date().toString("yyyy-MM-dd"),
            "initial_cash": self.initial_cash_spin.value(),
            "up_threshold": self.up_threshold_spin.value(),
            "down_threshold": self.down_threshold_spin.value(),
            "direction_margin": self.margin_spin.value(),
            "target_percent": self.target_percent_spin.value(),
            "device": "auto",
        }
        self.backtest_button.setEnabled(False)
        self._log(f"启动回测，模型: {model_dir}")
        self.backtest_thread = TimingBacktestThread(params)
        self.backtest_thread.info_signal.connect(self._log)
        self.backtest_thread.finished_signal.connect(self._on_backtest_finished)
        self.backtest_thread.error_signal.connect(self._on_backtest_error)
        self.backtest_thread.start()

    def refresh_models(self) -> None:
        current_path = self.model_combo.currentData()
        self.model_combo.clear()
        if not self.models_dir.exists():
            return
        for path in sorted(self.models_dir.iterdir(), reverse=True):
            if path.is_dir() and (path / "manifest.json").exists():
                frequency = _read_model_frequency(path)
                suffix = f" [{frequency}]" if frequency else ""
                self.model_combo.addItem(f"{path.name}{suffix}", str(path))
                self.model_combo.setItemData(self.model_combo.count() - 1, frequency, MODEL_FREQUENCY_ROLE)
                if current_path and str(path) == str(current_path):
                    self.model_combo.setCurrentIndex(self.model_combo.count() - 1)
        self._sync_backtest_frequency_from_model()

    def refresh_datasets(self, select_path: str | None = None) -> None:
        current_path = select_path or self.dataset_combo.currentData()
        self.dataset_combo.clear()
        datasets_dir = self.models_dir / "datasets"
        if not datasets_dir.exists():
            self._on_dataset_selection_changed()
            return
        for path in sorted(datasets_dir.iterdir(), reverse=True):
            if not path.is_dir() or not (path / "summary.json").exists():
                continue
            label = _read_dataset_label(path)
            self.dataset_combo.addItem(label, str(path))
            if current_path and str(path) == str(current_path):
                self.dataset_combo.setCurrentIndex(self.dataset_combo.count() - 1)
        self._on_dataset_selection_changed()

    def _on_dataset_selection_changed(self, *_args) -> None:
        if not hasattr(self, "dataset_summary_edit"):
            return
        dataset_dir = self.dataset_combo.currentData() if hasattr(self, "dataset_combo") else None
        if not dataset_dir:
            self.dataset_summary_edit.setPlainText("暂无可用数据集。请先在「数据构建」页生成数据集。")
            return
        try:
            with (Path(dataset_dir) / "summary.json").open("r", encoding="utf-8") as file:
                summary = json.load(file)
            self.dataset_summary_edit.setPlainText(_format_dataset_summary(summary, Path(dataset_dir)))
        except Exception as exc:
            self.dataset_summary_edit.setPlainText(f"读取数据集摘要失败: {exc}")

    def _sync_backtest_frequency_from_model(self, *_args) -> None:
        frequency = self.model_combo.currentData(MODEL_FREQUENCY_ROLE)
        if not frequency:
            return
        _set_combo_data(self.backtest_frequency_combo, str(frequency))

    def _on_training_finished(self, model_dir: str) -> None:
        self.train_button.setEnabled(True)
        self.build_data_button.setEnabled(True)
        self._log(f"训练完成: {model_dir}")
        history_path = Path(model_dir) / "history.json"
        if history_path.exists():
            try:
                with history_path.open("r", encoding="utf-8") as file:
                    self._training_history = list(json.load(file) or [])
                self._refresh_training_charts(self._training_history)
                self.train_progress_bar.setRange(0, max(len(self._training_history), 1))
                self.train_progress_bar.setValue(len(self._training_history))
                self.train_progress_bar.setFormat(f"训练完成: {len(self._training_history)}轮")
            except Exception as exc:
                self._log(f"读取训练曲线失败: {exc}")
        self.refresh_models()

    def _on_training_error(self, message: str) -> None:
        self.train_button.setEnabled(True)
        self.build_data_button.setEnabled(True)
        self.train_progress_bar.setFormat("训练失败")
        self._log(f"训练失败: {message}")
        QMessageBox.critical(self, "训练失败", message)

    def _on_training_progress(self, current: int, total: int, row: dict) -> None:
        self.train_progress_bar.setRange(0, max(int(total or 1), 1))
        self.train_progress_bar.setValue(int(current or 0))
        self.train_progress_bar.setFormat(f"训练中: {int(current or 0)}/{int(total or 0)}")
        self._training_history.append(dict(row or {}))
        self._refresh_training_charts(self._training_history)

    def _refresh_training_charts(self, history: list[dict]) -> None:
        if not hasattr(self, "train_loss_plot") or not hasattr(self, "train_accuracy_plot"):
            return
        for plot in (self.train_loss_plot, self.train_accuracy_plot):
            plot.clear()
            if plot.legend is None:
                plot.addLegend(offset=(10, 10))
            else:
                plot.legend.clear()
            plot.showGrid(x=True, y=True, alpha=0.25)

        frame = pd.DataFrame(history or [])
        if frame.empty or "epoch" not in frame.columns:
            return
        x = pd.to_numeric(frame["epoch"], errors="coerce").to_numpy(dtype=float)
        loss_values = []
        for column, color, name in (
            ("train_loss", "#88c0d0", "train loss"),
            ("val_loss", "#bf616a", "val loss"),
        ):
            if column not in frame.columns:
                continue
            y = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            loss_values.append(y)
            _plot_finite(self.train_loss_plot, x, y, pen=pg.mkPen(color, width=1.3), name=name)
        if loss_values:
            _set_numeric_range(self.train_loss_plot, x, np.concatenate(loss_values), y_floor=0.0)

        acc_values = []
        for column, color, name in (
            ("train_accuracy", "#a3be8c", "train accuracy"),
            ("val_accuracy", "#ebcb8b", "val accuracy"),
        ):
            if column not in frame.columns:
                continue
            y = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            acc_values.append(y)
            _plot_finite(self.train_accuracy_plot, x, y, pen=pg.mkPen(color, width=1.3), name=name)
        if acc_values:
            self.train_accuracy_plot.setXRange(float(np.nanmin(x)), float(np.nanmax(x)) if len(x) > 1 else float(np.nanmin(x)) + 1, padding=0.01)
            self.train_accuracy_plot.setYRange(0, 1, padding=0)

    def _on_data_build_finished(self, summary: dict) -> None:
        self.build_data_button.setEnabled(True)
        self.train_button.setEnabled(True)
        dataset_dir = str(summary.get("dataset_dir") or "")
        text = (
            f"数据构建完成: {dataset_dir}\n"
            f"样本: train={summary.get('train_samples', 0)}, "
            f"val={summary.get('val_samples', 0)}, test={summary.get('test_samples', 0)} | "
            f"特征数={summary.get('feature_count', 0)} | "
            f"标签分布={summary.get('label_distribution', {})}"
        )
        self._log(text)
        self.refresh_datasets(select_path=dataset_dir)
        QMessageBox.information(self, "数据构建完成", text)

    def _on_data_build_error(self, message: str) -> None:
        self.build_data_button.setEnabled(True)
        self.train_button.setEnabled(True)
        self._log(f"数据构建失败: {message}")
        QMessageBox.critical(self, "数据构建失败", message)

    def _on_backtest_finished(self, result: dict) -> None:
        self.backtest_button.setEnabled(True)
        metrics = result.get("metrics") or {}
        final_value = float(result.get("final_value") or 0.0)
        trades = len(result.get("trades") or [])
        text = f"最终资产: {final_value:.2f} | 交易数: {trades} | 指标: {metrics}"
        diagnostic = self._backtest_diagnostic_text(result)
        if diagnostic:
            text = f"{text}\n{diagnostic}"
        self.result_label.setText(text)
        self._log(text)
        self._refresh_backtest_charts(result)

    def _backtest_diagnostic_text(self, result: dict) -> str:
        trace = _as_dataframe(result.get("timing_signal_trace"))
        if trace.empty:
            return "诊断：本次没有可用的模型预测轨迹，可能是 K 线数量不足 lookback 或推理窗口含 NaN。"

        trades = len(result.get("trades") or [])
        buy_hits = int(trace.get("buy_threshold_hit", pd.Series(dtype=bool)).fillna(False).sum())
        sell_hits = int(trace.get("sell_threshold_hit", pd.Series(dtype=bool)).fillna(False).sum())
        buy_actions = int((trace.get("action", pd.Series(dtype=str)) == "buy").sum())
        sell_actions = int((trace.get("action", pd.Series(dtype=str)) == "sell").sum())
        max_up = float(pd.to_numeric(trace.get("p_up"), errors="coerce").max())
        max_down = float(pd.to_numeric(trace.get("p_down"), errors="coerce").max())
        if trades == 0 and buy_hits == 0:
            cause = "主要原因：没有任何 K 线同时满足看多阈值和方向差阈值，所以策略从未建仓。"
        elif trades == 0 and buy_actions == 0:
            cause = "主要原因：虽有阈值命中，但未形成可执行买入，可能受已有仓位、允许买入或一手数量限制影响。"
        elif trades == 0:
            cause = "主要原因：策略产生过买卖信号，但订单未成交，可查看成交约束、价格区间和资金/一手限制。"
        else:
            cause = ""
        return (
            f"诊断：预测点 {len(trace)} 个，买阈值命中 {buy_hits} 次，卖阈值命中 {sell_hits} 次，"
            f"实际买信号 {buy_actions} 次，实际卖信号 {sell_actions} 次，"
            f"max p_up={max_up:.3f}，max p_down={max_down:.3f}。{cause}"
        )

    def _refresh_backtest_charts(self, result: dict) -> None:
        trace = _as_dataframe(result.get("timing_signal_trace"))
        equity = _as_dataframe(result.get("equity_curve"))
        for plot in (self.backtest_equity_plot, self.backtest_signal_plot, self.backtest_position_plot):
            plot.clear()
            if plot.legend is None:
                plot.addLegend(offset=(10, 10))
            else:
                plot.legend.clear()
            plot.showGrid(x=True, y=True, alpha=0.25)

        if not equity.empty and {"total_asset", "close"}.issubset(equity.columns):
            x = np.arange(len(equity), dtype=float)
            asset = pd.to_numeric(equity["total_asset"], errors="coerce").to_numpy(dtype=float)
            close = pd.to_numeric(equity["close"], errors="coerce").to_numpy(dtype=float)
            asset_norm = _normalize_series(asset)
            close_norm = _normalize_series(close)
            _plot_finite(self.backtest_equity_plot, x, asset_norm, pen=pg.mkPen("#a3be8c", width=1.5), name="资产(归一化)")
            _plot_finite(self.backtest_equity_plot, x, close_norm, pen=pg.mkPen("#88c0d0", width=1.2), name="收盘(归一化)")
            _set_numeric_range(self.backtest_equity_plot, x, np.concatenate([asset_norm, close_norm]))

        if not trace.empty:
            x = np.arange(len(trace), dtype=float)
            for column, color, name in (
                ("p_up", "#a3be8c", "p_up"),
                ("p_flat", "#ebcb8b", "p_flat"),
                ("p_down", "#bf616a", "p_down"),
            ):
                y = pd.to_numeric(trace.get(column), errors="coerce").to_numpy(dtype=float)
                _plot_finite(self.backtest_signal_plot, x, y, pen=pg.mkPen(color, width=1.2), name=name)

            up_threshold = pd.to_numeric(trace.get("up_threshold"), errors="coerce").dropna()
            down_threshold = pd.to_numeric(trace.get("down_threshold"), errors="coerce").dropna()
            if not up_threshold.empty:
                y = float(up_threshold.iloc[-1])
                self.backtest_signal_plot.plot([0, len(trace) - 1], [y, y], pen=pg.mkPen("#a3be8c", style=Qt.PenStyle.DashLine), name="看多阈值")
            if not down_threshold.empty:
                y = float(down_threshold.iloc[-1])
                self.backtest_signal_plot.plot([0, len(trace) - 1], [y, y], pen=pg.mkPen("#bf616a", style=Qt.PenStyle.DashLine), name="看空阈值")

            close = pd.to_numeric(trace.get("close"), errors="coerce").to_numpy(dtype=float)
            close_norm = _normalize_series(close)
            marker_x = _trace_x_on_equity(equity, trace)
            for action, color, symbol, name in (
                ("buy", "#a3be8c", "t1", "买信号"),
                ("sell", "#bf616a", "t", "卖信号"),
            ):
                mask = (trace.get("action", pd.Series(dtype=str)) == action).to_numpy()
                idx = np.flatnonzero(mask)
                if len(idx):
                    y = close_norm[idx]
                    x_marker = marker_x[idx] if len(marker_x) == len(trace) else idx.astype(float)
                    valid = np.isfinite(x_marker) & np.isfinite(y)
                    if not valid.any():
                        continue
                    self.backtest_equity_plot.addItem(
                        pg.ScatterPlotItem(
                            x=x_marker[valid].astype(float),
                            y=y[valid],
                            size=12,
                            pen=pg.mkPen(color),
                            brush=pg.mkBrush(color),
                            symbol=symbol,
                            name=name,
                        )
                    )

            qty = pd.to_numeric(trace.get("position_qty"), errors="coerce").fillna(0).to_numpy(dtype=float)
            _plot_finite(self.backtest_position_plot, x, qty, pen=pg.mkPen("#d08770", width=1.2), name="持仓数量")
            self.backtest_signal_plot.setXRange(0, max(len(trace) - 1, 1), padding=0.01)
            self.backtest_signal_plot.setYRange(0, 1, padding=0)
            _set_numeric_range(self.backtest_position_plot, x, qty, y_floor=0.0)

    def _on_backtest_error(self, message: str) -> None:
        self.backtest_button.setEnabled(True)
        self._log(f"回测失败: {message}")
        QMessageBox.critical(self, "回测失败", message)

    def _log(self, message: str) -> None:
        self.log_edit.append(message)


def _as_dataframe(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _normalize_series(values: np.ndarray) -> np.ndarray:
    data = values.astype(float, copy=True)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data, dtype=float)
    base = data[finite][0]
    if not np.isfinite(base) or abs(base) < 1e-12:
        base = 1.0
    return data / base


def _plot_finite(plot, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return
    plot.plot(x[mask], y[mask], **kwargs)


def _set_numeric_range(plot, x: np.ndarray, y: np.ndarray, *, y_floor: float | None = None) -> None:
    valid_x = x[np.isfinite(x)]
    valid_y = y[np.isfinite(y)]
    if len(valid_x) == 0 or len(valid_y) == 0:
        return
    x_min = float(valid_x.min())
    x_max = float(valid_x.max())
    if x_max <= x_min:
        x_max = x_min + 1.0

    y_min = float(valid_y.min())
    y_max = float(valid_y.max())
    if y_floor is not None:
        y_min = min(float(y_floor), y_min)
    if y_max <= y_min:
        padding = max(abs(y_max) * 0.05, 1.0)
        y_min -= padding
        y_max += padding
    else:
        padding = max((y_max - y_min) * 0.08, 0.02)
        y_min -= padding
        y_max += padding

    plot.setXRange(x_min, x_max, padding=0.01)
    plot.setYRange(y_min, y_max, padding=0)


def _trace_x_on_equity(equity: pd.DataFrame, trace: pd.DataFrame) -> np.ndarray:
    fallback = np.arange(len(trace), dtype=float)
    if equity.empty or trace.empty or "date" not in equity.columns or "date" not in trace.columns:
        return fallback
    equity_dates = pd.to_datetime(equity["date"], errors="coerce")
    trace_dates = pd.to_datetime(trace["date"], errors="coerce")
    if equity_dates.isna().all() or trace_dates.isna().all():
        return fallback
    index_by_date = {value: float(index) for index, value in enumerate(equity_dates)}
    mapped = trace_dates.map(index_by_date).to_numpy(dtype=float)
    return np.where(np.isfinite(mapped), mapped, fallback)


def _parse_symbols(text: str) -> list[str]:
    return [item.strip().upper().split(".", 1)[0] for item in text.replace("，", ",").split(",") if item.strip()]


def _parse_channels(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(text or "").replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError("channels 必须是用英文逗号分隔的正整数，例如 64,64,64") from exc
        if value <= 0:
            raise ValueError("channels 中的每个通道数都必须大于 0")
        values.append(value)
    if not values:
        raise ValueError("请填写至少一层 channels，例如 64,64,64")
    return tuple(values)


def _frequency_combo(parent: QWidget) -> QComboBox:
    combo = QComboBox(parent)
    for label, value in (
        ("日线 1d", "1d"),
        ("1分钟 1m", "1m"),
        ("5分钟 5m", "5m"),
        ("15分钟 15m", "15m"),
        ("30分钟 30m", "30m"),
        ("小时线 60m", "60m"),
    ):
        combo.addItem(label, value)
    return combo


def _set_combo_data(combo: QComboBox, value: str) -> None:
    for index in range(combo.count()):
        if str(combo.itemData(index)) == str(value):
            combo.setCurrentIndex(index)
            return


def _read_model_frequency(model_dir: Path) -> str:
    try:
        with (model_dir / "manifest.json").open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        return str(manifest.get("frequency") or "")
    except Exception:
        return ""


def _read_dataset_label(dataset_dir: Path) -> str:
    try:
        with (dataset_dir / "summary.json").open("r", encoding="utf-8") as file:
            summary = json.load(file)
        symbols = ",".join(list(summary.get("symbols") or [])[:3])
        if len(summary.get("symbols") or []) > 3:
            symbols += "..."
        return (
            f"{dataset_dir.name} [{summary.get('frequency', '')} | {symbols} | "
            f"L{(summary.get('dataset_config') or {}).get('lookback', '')} | "
            f"H{(summary.get('label_config') or {}).get('horizon', '')}]"
        )
    except Exception:
        return dataset_dir.name


def _format_dataset_summary(summary: dict, dataset_dir: Path) -> str:
    feature_config = summary.get("feature_config") or {}
    label_config = summary.get("label_config") or {}
    dataset_config = summary.get("dataset_config") or {}
    symbols = ", ".join(summary.get("symbols") or [])
    lines = [
        f"数据集目录: {dataset_dir}",
        f"标的: {symbols}",
        f"周期: {summary.get('frequency', '')}",
        f"日期: {summary.get('start_date', '')} ~ {summary.get('end_date', '')}",
        "",
        "样本切分:",
        f"  train={summary.get('train_samples', 0)} | val={summary.get('val_samples', 0)} | test={summary.get('test_samples', 0)}",
        f"  train_ratio={dataset_config.get('train_ratio', '')} | val_ratio={dataset_config.get('val_ratio', '')}",
        f"  lookback={dataset_config.get('lookback', '')}",
        "",
        "标签参数:",
        f"  horizon={label_config.get('horizon', '')}",
        f"  volatility_window={label_config.get('volatility_window', '')}",
        f"  up_mult={label_config.get('up_mult', '')}",
        f"  down_mult={label_config.get('down_mult', '')}",
        "",
        "特征参数:",
        f"  feature_count={summary.get('feature_count', 0)}",
        f"  momentum_windows={feature_config.get('momentum_windows', [])}",
        f"  ma_windows={feature_config.get('ma_windows', [])}",
        f"  rsi_window={feature_config.get('rsi_window', '')}",
        f"  macd=({feature_config.get('macd_fast', '')}, {feature_config.get('macd_slow', '')}, {feature_config.get('macd_signal', '')})",
        "",
        f"标签分布: {summary.get('label_distribution', {})}",
    ]
    return "\n".join(lines)


def _load_timing_dataset_artifact(dataset_dir: Path) -> tuple[TimingDataset, dict]:
    summary_path = dataset_dir / "summary.json"
    data_path = dataset_dir / "dataset.npz"
    metadata_path = dataset_dir / "metadata.parquet"
    scaler_path = dataset_dir / "scaler.pkl"
    missing = [str(path) for path in (summary_path, data_path, metadata_path, scaler_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"数据集产物缺少文件: {missing}")
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    arrays = np.load(data_path)
    metadata = pd.read_parquet(metadata_path)
    scaler = load_scaler(scaler_path)
    dataset_config = TimingDatasetConfig(**(summary.get("dataset_config") or {}))
    feature_names = list(summary.get("feature_names") or [])
    dataset = TimingDataset(
        x_train=arrays["x_train"],
        y_train=arrays["y_train"],
        x_val=arrays["x_val"],
        y_val=arrays["y_val"],
        x_test=arrays["x_test"],
        y_test=arrays["y_test"],
        metadata=metadata,
        feature_names=feature_names,
        scaler=scaler,
        config=dataset_config,
    )
    return dataset, summary


def _tuple_fields(payload: dict, fields: tuple[str, ...]) -> dict:
    result = dict(payload or {})
    for field in fields:
        if field in result:
            result[field] = tuple(result[field] or ())
    return result


def _date_edit(value: QDate, parent: QWidget) -> QDateEdit:
    widget = QDateEdit(value, parent)
    widget.setCalendarPopup(True)
    return widget


def _spin(minimum: int, maximum: int, value: int, parent: QWidget) -> QSpinBox:
    widget = QSpinBox(parent)
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


def _double_spin(minimum: float, maximum: float, value: float, parent: QWidget) -> QDoubleSpinBox:
    widget = QDoubleSpinBox(parent)
    widget.setRange(minimum, maximum)
    widget.setDecimals(2)
    widget.setValue(value)
    return widget


def _float_spin(
    minimum: float,
    maximum: float,
    value: float,
    parent: QWidget,
    *,
    decimals: int = 2,
    step: float = 0.1,
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox(parent)
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setValue(value)
    return widget


def _decimal_spin(minimum: float, maximum: float, value: float, parent: QWidget) -> QDoubleSpinBox:
    widget = QDoubleSpinBox(parent)
    widget.setRange(minimum, maximum)
    widget.setDecimals(6)
    widget.setSingleStep(value if value > 0 else 0.0001)
    widget.setValue(value)
    return widget


def _prob_spin(value: float, parent: QWidget) -> QDoubleSpinBox:
    widget = QDoubleSpinBox(parent)
    widget.setRange(0.0, 1.0)
    widget.setSingleStep(0.01)
    widget.setDecimals(2)
    widget.setValue(value)
    return widget
