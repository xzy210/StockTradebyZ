from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from common.io_utils import atomic_write_json
from trading_app.services.execution.trade_record_service import TradeRecordService


class LiveStrategyFeeSettingsDialog(QDialog):
    """全局交易费用设置弹窗。"""

    _CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "trade_fee_config.json"
    _DEFAULTS = {
        "commission_rate": 0.0001,
        "min_commission": 5.0,
        "stamp_tax_rate": 0.0005,
        "transfer_fee_rate": 0.00001,
        "etf_exempt_stamp_tax": True,
        "etf_exempt_transfer_fee": True,
        "etf_code_prefixes": ["51", "56", "58", "15", "16"],
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("交易费用设置（所有策略共用）")
        self.setModal(True)
        self.setMinimumWidth(460)
        self._build_ui()
        self._load_config()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        hint = QLabel(
            "这里维护 AI实盘决策、ETF轮动实盘、实盘收益共用的手续费估算口径。\n"
            "保存后会立即刷新内存配置，新的估算与统计会统一使用这里的规则。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94A3B8;font-size:11px;")
        root.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.spin_commission = QDoubleSpinBox()
        self.spin_commission.setRange(0.0, 10.0)
        self.spin_commission.setDecimals(2)
        self.spin_commission.setSingleStep(0.1)
        self.spin_commission.setSuffix(" 万分之")
        self.spin_commission.setToolTip("所有策略默认佣金率；例如 1.00 表示万分之一。")
        form.addRow("基础佣金:", self.spin_commission)

        self.spin_min_commission = QDoubleSpinBox()
        self.spin_min_commission.setRange(0.0, 100.0)
        self.spin_min_commission.setDecimals(2)
        self.spin_min_commission.setSingleStep(1.0)
        self.spin_min_commission.setSuffix(" 元")
        form.addRow("最低佣金:", self.spin_min_commission)

        self.spin_stamp_tax = QDoubleSpinBox()
        self.spin_stamp_tax.setRange(0.0, 20.0)
        self.spin_stamp_tax.setDecimals(2)
        self.spin_stamp_tax.setSingleStep(0.1)
        self.spin_stamp_tax.setSuffix(" 万分之")
        self.spin_stamp_tax.setToolTip("卖出时印花税率。")
        form.addRow("卖出印花税:", self.spin_stamp_tax)

        self.spin_transfer_fee = QDoubleSpinBox()
        self.spin_transfer_fee.setRange(0.0, 20.0)
        self.spin_transfer_fee.setDecimals(3)
        self.spin_transfer_fee.setSingleStep(0.01)
        self.spin_transfer_fee.setSuffix(" 万分之")
        self.spin_transfer_fee.setToolTip("双边过户费率。")
        form.addRow("过户费:", self.spin_transfer_fee)

        self.chk_etf_exempt_stamp = QCheckBox("ETF 默认免印花税")
        form.addRow("", self.chk_etf_exempt_stamp)

        self.chk_etf_exempt_transfer = QCheckBox("ETF 默认免过户费")
        form.addRow("", self.chk_etf_exempt_transfer)

        self.edit_etf_prefixes = QLineEdit()
        self.edit_etf_prefixes.setPlaceholderText("例如 51,56,58,15,16")
        self.edit_etf_prefixes.setToolTip("用于识别 ETF 代码前缀，多个前缀用英文逗号分隔。")
        form.addRow("ETF 代码前缀:", self.edit_etf_prefixes)

        root.addLayout(form)

        self.lbl_path = QLabel(str(self._CONFIG_PATH))
        self.lbl_path.setStyleSheet("color:#64748B;font-size:11px;")
        self.lbl_path.setWordWrap(True)
        root.addWidget(self.lbl_path)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).setText("恢复默认")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._load_defaults
        )
        root.addWidget(buttons)

    def _load_config(self) -> None:
        cfg = dict(self._DEFAULTS)
        try:
            if self._CONFIG_PATH.exists():
                with open(self._CONFIG_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f) or {}
                for key in cfg:
                    if key in raw:
                        cfg[key] = raw[key]
        except Exception as exc:
            QMessageBox.warning(self, "交易费用设置", f"读取配置失败，将回退默认值：{exc}")
        self._apply_config(cfg)

    def _load_defaults(self) -> None:
        self._apply_config(dict(self._DEFAULTS))

    def _apply_config(self, cfg: dict) -> None:
        self.spin_commission.setValue(float(cfg.get("commission_rate", 0.0) or 0.0) * 10000)
        self.spin_min_commission.setValue(float(cfg.get("min_commission", 0.0) or 0.0))
        self.spin_stamp_tax.setValue(float(cfg.get("stamp_tax_rate", 0.0) or 0.0) * 10000)
        self.spin_transfer_fee.setValue(float(cfg.get("transfer_fee_rate", 0.0) or 0.0) * 10000)
        self.chk_etf_exempt_stamp.setChecked(bool(cfg.get("etf_exempt_stamp_tax", True)))
        self.chk_etf_exempt_transfer.setChecked(bool(cfg.get("etf_exempt_transfer_fee", True)))
        prefixes = cfg.get("etf_code_prefixes") or []
        self.edit_etf_prefixes.setText(",".join(str(item).strip() for item in prefixes if str(item).strip()))

    def _on_accept(self) -> None:
        prefixes = [
            item.strip()
            for item in str(self.edit_etf_prefixes.text() or "").split(",")
            if item.strip()
        ]
        if not prefixes:
            QMessageBox.warning(self, "交易费用设置", "ETF 代码前缀不能为空。")
            return

        payload = {
            "_doc": "交易手续费配置。所有模块（strategy_budget / trade_record / 下单路径估算）共用。",
            "commission_rate": round(self.spin_commission.value() / 10000, 8),
            "min_commission": round(self.spin_min_commission.value(), 2),
            "stamp_tax_rate": round(self.spin_stamp_tax.value() / 10000, 8),
            "transfer_fee_rate": round(self.spin_transfer_fee.value() / 10000, 8),
            "etf_exempt_stamp_tax": self.chk_etf_exempt_stamp.isChecked(),
            "etf_exempt_transfer_fee": self.chk_etf_exempt_transfer.isChecked(),
            "etf_code_prefixes": prefixes,
        }
        try:
            atomic_write_json(self._CONFIG_PATH, payload)
            TradeRecordService.reload_fee_config()
        except Exception as exc:
            QMessageBox.critical(self, "交易费用设置", f"保存失败：{exc}")
            return
        QMessageBox.information(self, "交易费用设置", "全局交易费用配置已保存。")
        self.accept()
