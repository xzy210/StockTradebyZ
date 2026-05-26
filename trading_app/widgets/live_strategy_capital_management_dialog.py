from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from trading_app.services.strategy_budget_service import get_strategy_budget_service
from trading_app.services.strategy_constants import (
    AI_STOCK_STRATEGY_ID,
    AI_STOCK_STRATEGY_NAME,
    AI_STOCK_VIRTUAL_ACCOUNT_ID,
)


class LiveStrategyCapitalManagementDialog(QDialog):
    """统一管理 AI 实盘决策 / ETF 轮动实盘启动资金。"""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        ai_panel=None,
        etf_panel=None,
    ) -> None:
        super().__init__(parent)
        self.ai_panel = ai_panel
        self.etf_panel = etf_panel
        self.strategy_budget = get_strategy_budget_service()
        self.setWindowTitle("实盘资金管理")
        self.setModal(True)
        self.setMinimumWidth(480)
        self._build_ui()
        self._load_current_values()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        hint = QLabel(
            "这里维护实盘收益使用的实盘策略启动资金口径。\n"
            "未勾选“同步重置账本”时按新旧启动资金差额划拨现金；勾选后会把该策略主账本现金校正到新的启动资金，但保留当前持仓。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94A3B8;font-size:11px;")
        root.addWidget(hint)

        self.ai_group, self.ai_spin, self.ai_reset_cb, self.ai_status = self._build_strategy_group(
            "AI实盘决策",
            allow_zero=True,
            tooltip="0 表示继续使用主账本的自动剩余额度推导。",
        )
        root.addWidget(self.ai_group)

        self.etf_group, self.etf_spin, self.etf_reset_cb, self.etf_status = self._build_strategy_group(
            "ETF轮动实盘",
            allow_zero=False,
            tooltip="ETF轮动实盘通常使用显式启动资金；修改后会同步更新策略配置与主账本。",
        )
        root.addWidget(self.etf_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_strategy_group(
        self,
        title: str,
        *,
        allow_zero: bool,
        tooltip: str,
    ) -> tuple[QGroupBox, QDoubleSpinBox, QCheckBox, QLabel]:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        spin = QDoubleSpinBox()
        spin.setRange(0.0 if allow_zero else 1000.0, 10_000_000.0)
        spin.setDecimals(0)
        spin.setSingleStep(10000.0)
        spin.setSuffix(" 元")
        spin.setToolTip(tooltip)
        form.addRow("启动资金:", spin)

        reset_cb = QCheckBox("保存时同步重置账本现金")
        reset_cb.setToolTip("不勾选时按差额划拨现金；勾选时用于手动校正账本偏差，会保留当前持仓，仅重算现金与资金上限。")
        form.addRow("", reset_cb)

        status = QLabel("-")
        status.setWordWrap(True)
        status.setStyleSheet("color:#64748B;font-size:11px;")
        form.addRow("当前状态:", status)
        return group, spin, reset_cb, status

    def _load_current_values(self) -> None:
        ai_snapshot = self.strategy_budget.build_account_snapshot(
            AI_STOCK_STRATEGY_ID,
            strategy_name=AI_STOCK_STRATEGY_NAME,
            virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
        )
        ai_capital = float(ai_snapshot.get("capital_limit", 0.0) or 0.0)
        self.ai_spin.setValue(ai_capital)
        self.ai_status.setText(
            f"当前启动资金 ¥{ai_capital:,.0f}；可用现金 ¥{float(ai_snapshot.get('available_cash', 0.0) or 0.0):,.2f}；"
            f"总盈亏 ¥{float(ai_snapshot.get('total_pnl', 0.0) or 0.0):,.2f}"
        )

        if self.etf_panel is None:
            self.etf_group.setEnabled(False)
            self.etf_status.setText("当前未挂载 ETF 面板。")
            return

        strategy_id, strategy_name, virtual_account_id = self.etf_panel._etf_strategy_identity()  # noqa: SLF001
        etf_snapshot = self.strategy_budget.build_account_snapshot(
            strategy_id,
            strategy_name=strategy_name,
            virtual_account_id=virtual_account_id,
        )
        etf_capital = float(getattr(self.etf_panel.engine.config, "dedicated_capital", 0.0) or 0.0)
        self.etf_spin.setValue(etf_capital)
        self.etf_status.setText(
            f"当前启动资金 ¥{etf_capital:,.0f}；可用现金 ¥{float(etf_snapshot.get('available_cash', 0.0) or 0.0):,.2f}；"
            f"总盈亏 ¥{float(etf_snapshot.get('total_pnl', 0.0) or 0.0):,.2f}"
        )

    def _apply_ai_capital(self, capital_limit: float, *, reset_ledger: bool) -> None:
        self.strategy_budget.upsert_strategy_config(
            strategy_id=AI_STOCK_STRATEGY_ID,
            strategy_name=AI_STOCK_STRATEGY_NAME,
            virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
            capital_limit=capital_limit,
            enabled=True,
        )
        if reset_ledger:
            self.strategy_budget.reset_strategy_account(
                strategy_id=AI_STOCK_STRATEGY_ID,
                strategy_name=AI_STOCK_STRATEGY_NAME,
                virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
                capital_limit=capital_limit,
                cash_balance=capital_limit,
                preserve_positions=True,
            )
        else:
            self.strategy_budget.adjust_strategy_capital_limit(
                strategy_id=AI_STOCK_STRATEGY_ID,
                strategy_name=AI_STOCK_STRATEGY_NAME,
                virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
                new_capital_limit=capital_limit,
            )

    def _apply_etf_capital(self, capital_limit: float, *, reset_ledger: bool) -> None:
        if self.etf_panel is None:
            return
        strategy_id, strategy_name, virtual_account_id = self.etf_panel._etf_strategy_identity()  # noqa: SLF001
        cfg = self.etf_panel.engine.config
        cfg.dedicated_capital = capital_limit
        self.etf_panel.engine.update_config(cfg)
        self.etf_panel._sync_etf_strategy_profile()  # noqa: SLF001
        if reset_ledger:
            self.etf_panel.engine.reset_dedicated_capital(capital_limit)
            self.strategy_budget.reset_strategy_account(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                capital_limit=capital_limit,
                cash_balance=capital_limit,
                preserve_positions=True,
            )
        else:
            self.strategy_budget.adjust_strategy_capital_limit(
                strategy_id=strategy_id,
                strategy_name=strategy_name,
                virtual_account_id=virtual_account_id,
                new_capital_limit=capital_limit,
            )
        try:
            self.etf_panel._refresh_status()  # noqa: SLF001
        except Exception:
            pass

    def _on_accept(self) -> None:
        ai_capital = round(float(self.ai_spin.value() or 0.0), 2)
        etf_capital = round(float(self.etf_spin.value() or 0.0), 2)
        if self.etf_panel is not None and etf_capital <= 0:
            QMessageBox.warning(self, "实盘资金管理", "ETF轮动实盘的启动资金必须大于 0。")
            return
        try:
            self._apply_ai_capital(ai_capital, reset_ledger=self.ai_reset_cb.isChecked())
            self._apply_etf_capital(etf_capital, reset_ledger=self.etf_reset_cb.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "实盘资金管理", f"保存失败：{exc}")
            return
        QMessageBox.information(self, "实盘资金管理", "实盘策略启动资金已保存。")
        self.accept()
