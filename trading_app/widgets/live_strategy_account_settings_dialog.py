"""账户级（全局网关）设置对话框。

收敛那些会影响**所有策略**订单的 :class:`AutoTradeConfig` 字段（`auto_trade_mode`
由状态栏快速切换，这里聚焦剩余 5 个没有 UI 入口的字段）：

- ``manual_orders_enabled``        手动下单总开关
- ``require_trading_time``         是否强制交易时段内才放行
- ``duplicate_window_seconds``     同委托去重时长
- ``status_poll_seconds``          委托提交后轮询成交的总时长
- ``status_poll_interval_seconds`` 轮询间隔

这是 "增量统一 Step A" 的 UI 载体：账户级配置共用同一份 JSON，ETF / AI 任何一条
订单都会经这些字段校验，因此面板放在实盘策略中枢状态栏边，供两个实盘策略面板共用。
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from trading_app.services.ai.auto_trade_config_service import (
    AutoTradeConfig,
    AutoTradeConfigService,
    get_auto_trade_config_service,
)
from trading_app.widgets.live_strategy_fee_settings_dialog import LiveStrategyFeeSettingsDialog


class LiveStrategyAccountSettingsDialog(QDialog):
    """账户级（全局网关）行为设置弹窗。

    仅承载"所有策略共用"的网关级开关；策略专属规则各自的 Tab 去管。
    """

    _DEFAULTS = AutoTradeConfig()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        service: Optional[AutoTradeConfigService] = None,
    ) -> None:
        super().__init__(parent)
        self._service = service or get_auto_trade_config_service()
        self.setWindowTitle("账户级设置（所有策略共用）")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._build_ui()
        self._load_from_service()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        hint = QLabel(
            "这些开关作用于**所有实盘策略**的统一执行网关。\n"
            "执行模式（实盘 / 影子 / 关闭）请在状态栏顶部的下拉切换；手续费规则在下方单独维护。"
        )
        hint.setStyleSheet("color:#94A3B8;font-size:11px;")
        hint.setWordWrap(True)
        root.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.chk_manual = QCheckBox("允许手动下单")
        self.chk_manual.setToolTip(
            "关闭后，所有 trigger=manual 的订单（含 AI 面板、ETF 手动按钮、异常订单重试）均被网关拦截"
        )
        form.addRow("手动下单:", self.chk_manual)

        self.chk_trading_time = QCheckBox("仅交易时段内放行")
        self.chk_trading_time.setToolTip(
            "启用后，网关会在非交易时段（含午休 / 休市 / 法定假日）直接拦截；开发 / 夜间回放测试时可关闭"
        )
        form.addRow("交易时段闸:", self.chk_trading_time)

        self.spin_dup_window = QSpinBox()
        self.spin_dup_window.setRange(1, 3600)
        self.spin_dup_window.setSingleStep(5)
        self.spin_dup_window.setSuffix(" 秒")
        self.spin_dup_window.setToolTip(
            "相同 (策略+代码+方向+数量+价格) 的订单，在此窗口内重复提交会被拦截，返回"
            "「近期已有相同委托记录，已拦截重复报单」"
        )
        form.addRow("去重窗口:", self.spin_dup_window)

        self.spin_poll_total = QDoubleSpinBox()
        self.spin_poll_total.setRange(0.5, 120.0)
        self.spin_poll_total.setDecimals(1)
        self.spin_poll_total.setSingleStep(1.0)
        self.spin_poll_total.setSuffix(" 秒")
        self.spin_poll_total.setToolTip(
            "提交委托后轮询券商查询成交状态的最长等待时间；超时后标记为「已提交待确认」"
        )
        form.addRow("成交轮询总时长:", self.spin_poll_total)

        self.spin_poll_step = QDoubleSpinBox()
        self.spin_poll_step.setRange(0.2, 10.0)
        self.spin_poll_step.setDecimals(1)
        self.spin_poll_step.setSingleStep(0.2)
        self.spin_poll_step.setSuffix(" 秒")
        self.spin_poll_step.setToolTip("轮询间隔；过短会加重 miniQMT 负载，建议 1.0 秒左右")
        form.addRow("成交轮询间隔:", self.spin_poll_step)

        root.addLayout(form)

        self._current_mode_lbl = QLabel()
        self._current_mode_lbl.setStyleSheet("color:#64748B;font-size:11px;")
        root.addWidget(self._current_mode_lbl)

        self._fee_settings_btn = QDialogButtonBox(parent=self)
        self._fee_settings_btn.addButton("交易费用设置…", QDialogButtonBox.ButtonRole.ActionRole)
        fee_button = self._fee_settings_btn.buttons()[0]
        fee_button.clicked.connect(self._open_fee_settings_dialog)
        root.addWidget(self._fee_settings_btn)

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
            self._on_restore_defaults
        )
        root.addWidget(buttons)

    def _load_from_service(self) -> None:
        cfg = self._service.get_config()
        self.chk_manual.setChecked(bool(cfg.manual_orders_enabled))
        self.chk_trading_time.setChecked(bool(cfg.require_trading_time))
        self.spin_dup_window.setValue(int(cfg.duplicate_window_seconds))
        self.spin_poll_total.setValue(float(cfg.status_poll_seconds))
        self.spin_poll_step.setValue(float(cfg.status_poll_interval_seconds))

        mode_label = {"off": "关闭", "shadow": "影子", "paper": "纸面", "live": "实盘"}.get(
            cfg.auto_trade_mode, cfg.auto_trade_mode
        )
        self._current_mode_lbl.setText(
            f"当前执行模式：{mode_label}（在顶部状态栏切换）"
        )

    def _on_restore_defaults(self) -> None:
        d = self._DEFAULTS
        self.chk_manual.setChecked(bool(d.manual_orders_enabled))
        self.chk_trading_time.setChecked(bool(d.require_trading_time))
        self.spin_dup_window.setValue(int(d.duplicate_window_seconds))
        self.spin_poll_total.setValue(float(d.status_poll_seconds))
        self.spin_poll_step.setValue(float(d.status_poll_interval_seconds))

    def _open_fee_settings_dialog(self) -> None:
        dialog = LiveStrategyFeeSettingsDialog(self)
        dialog.exec()

    def _on_accept(self) -> None:
        if self.spin_poll_step.value() > self.spin_poll_total.value():
            QMessageBox.warning(
                self,
                "账户级设置",
                "成交轮询间隔不能大于总时长，请调整后再保存。",
            )
            return
        try:
            self._service.update_config(
                manual_orders_enabled=self.chk_manual.isChecked(),
                require_trading_time=self.chk_trading_time.isChecked(),
                duplicate_window_seconds=int(self.spin_dup_window.value()),
                status_poll_seconds=float(self.spin_poll_total.value()),
                status_poll_interval_seconds=float(self.spin_poll_step.value()),
            )
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.critical(self, "账户级设置", f"保存失败: {exc}")
            return
        self.accept()
