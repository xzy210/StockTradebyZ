# -*- coding: utf-8 -*-
"""AI ????????????????????"""
from __future__ import annotations

import logging
import math
from typing import Callable, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.broker_session_service import get_broker_session_service
from common.strategy_panel_context import StrategyPanelContext
from trading_app.services.strategy.strategy_constants import (
    AI_STOCK_STRATEGY_ID,
    AI_STOCK_STRATEGY_NAME,
    AI_STOCK_VIRTUAL_ACCOUNT_ID,
)
from trading_app.services.ai.trade_decision_models import TradeAction, TradeDecision
from trading_app.services.execution.trade_execution_service import ExecutionRequest, get_trade_execution_service
from trading_app.services.execution.trade_record_service import TradeSource

logger = logging.getLogger(__name__)


class OrderExecutionPanel(QWidget):
    """Reusable manual order placement panel."""

    order_executed = pyqtSignal(bool, bool, str, int, float)  # success, filled_confirmed, message, order_id, price

    def __init__(
        self,
        parent=None,
        *,
        strategy_context: Optional[StrategyPanelContext] = None,
        symbol_name_resolver: Optional[Callable[[str], str]] = None,
    ):
        super().__init__(parent)
        self.broker = get_broker_session_service()
        self.execution_service = get_trade_execution_service()
        self._strategy_context = strategy_context or StrategyPanelContext(
            strategy_id=AI_STOCK_STRATEGY_ID,
            strategy_name=AI_STOCK_STRATEGY_NAME,
            virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
            owner_type="ai",
        )
        self._symbol_name_resolver = symbol_name_resolver
        self._decision_context: Optional[dict] = None
        self._current_code = ""
        self._current_name = ""
        self._current_direction = "buy"
        self._form_updating = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel("手动委托")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        order_form = QFormLayout()
        order_form.setSpacing(6)
        self.code_input = QLineEdit()
        self.code_input.setPlaceholderText("输入股票代码，如 000001 或 000001.SZ")
        self.lbl_order_name = QLabel("-")
        self.direction_combo = QComboBox()
        self.direction_combo.addItem("买入", "buy")
        self.direction_combo.addItem("卖出", "sell")
        self.lbl_order_confidence = QLabel("-")
        self.lbl_order_risk = QLabel("-")
        self.price_input = QLineEdit()
        self.price_input.setPlaceholderText("委托价格")
        self.fetch_tick_price_btn = QPushButton("取一档价")
        self.fetch_tick_price_btn.setToolTip("按当前委托方向自动填入买一/卖一价格")
        self.volume_input = QLineEdit()
        self.volume_input.setPlaceholderText("委托数量(手,1手=100股)")
        self.amount_label = QLabel("-")
        price_row = QWidget()
        price_layout = QHBoxLayout(price_row)
        price_layout.setContentsMargins(0, 0, 0, 0)
        price_layout.setSpacing(6)
        price_layout.addWidget(self.price_input, stretch=1)
        price_layout.addWidget(self.fetch_tick_price_btn)
        order_form.addRow("代码:", self.code_input)
        order_form.addRow("名称:", self.lbl_order_name)
        order_form.addRow("方向:", self.direction_combo)
        order_form.addRow("置信度:", self.lbl_order_confidence)
        order_form.addRow("风控:", self.lbl_order_risk)
        order_form.addRow("价格:", price_row)
        order_form.addRow("数量(手):", self.volume_input)
        order_form.addRow("委托金额:", self.amount_label)
        layout.addLayout(order_form)

        self.decision_note = QPlainTextEdit()
        self.decision_note.setReadOnly(True)
        self.decision_note.setPlaceholderText("这里会展示当前 AI 实盘决策的委托说明。")
        self.decision_note.setMaximumHeight(180)
        layout.addWidget(self.decision_note)

        btn_row = QHBoxLayout()
        self.clear_btn = QPushButton("清空委托")
        self.clear_btn.clicked.connect(self._clear_order_form)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        self.exec_btn = QPushButton("提交委托")
        self.exec_btn.setFixedHeight(38)
        self.exec_btn.setStyleSheet(
            "QPushButton { background-color: #0078d4; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #106ebe; }"
            "QPushButton:disabled { background-color: #999999; }"
        )
        self.exec_btn.clicked.connect(self._on_execute)
        btn_row.addWidget(self.exec_btn)
        layout.addLayout(btn_row)
        layout.addStretch()

        self.code_input.textChanged.connect(self._on_code_changed)
        self.direction_combo.currentIndexChanged.connect(self._on_direction_changed)
        self.price_input.textChanged.connect(self._update_amount)
        self.volume_input.textChanged.connect(self._update_amount)
        self.fetch_tick_price_btn.clicked.connect(self._fill_price_from_tick)

    def _set_direction(self, direction: str) -> None:
        idx = self.direction_combo.findData("buy" if direction == "buy" else "sell")
        if idx < 0:
            idx = 0
        self.direction_combo.setCurrentIndex(idx)

    def _selected_direction(self) -> str:
        return str(self.direction_combo.currentData() or "buy")

    def _resolve_symbol_name(self, code: str) -> str:
        code = str(code or "").strip()
        if not code:
            return ""
        resolver = self._symbol_name_resolver
        if callable(resolver):
            try:
                resolved = resolver(code)
                if resolved:
                    return str(resolved).strip()
            except Exception:
                pass
        return ""

    def _switch_to_manual_mode(self, note: str = "") -> None:
        self.clear_decision_context()
        self.lbl_order_confidence.setText("-")
        self.lbl_order_risk.setText("手动委托")
        self.decision_note.setPlainText(note or "该委托来自手动录入，不绑定 AI 实盘决策记录。")
        self.exec_btn.setEnabled(True)

    def _on_code_changed(self, text: str) -> None:
        code = str(text or "").strip().upper()
        self._current_code = code
        resolved_name = self._resolve_symbol_name(code)
        self._current_name = resolved_name or code
        self.lbl_order_name.setText(resolved_name or (code or "-"))
        if self._form_updating:
            return
        if self._decision_context is not None:
            self._switch_to_manual_mode("该委托已切换为手动录入，不再绑定巡检/AI 实盘决策记录。")
        if not code:
            self.volume_input.clear()
            return
        suggested_lots = self._suggest_lots_for_manual(code, self._selected_direction())
        if suggested_lots:
            self.volume_input.setText(suggested_lots)

    def _on_direction_changed(self, _index: int) -> None:
        self._current_direction = self._selected_direction()
        if self._form_updating:
            return
        if self._decision_context is not None:
            self._switch_to_manual_mode("该委托已切换为手动录入，不再绑定巡检/AI 实盘决策记录。")
        code = self.code_input.text().strip()
        if not code:
            return
        suggested_lots = self._suggest_lots_for_manual(code, self._current_direction)
        if suggested_lots:
            self.volume_input.setText(suggested_lots)

    def fill_from_decision(
        self,
        decision: TradeDecision,
        *,
        risk_result=None,
        decision_record_id: str = "",
    ):
        self._decision_context = {
            "decision": decision,
            "risk_result": risk_result,
            "decision_record_id": decision_record_id,
        }
        direction = "buy" if decision.action in (TradeAction.BUY.value, TradeAction.ADD.value) else "sell"
        self._form_updating = True
        self._current_code = decision.symbol_code
        self._current_name = decision.symbol_name
        self._current_direction = direction
        self.code_input.setText(decision.symbol_code)
        self.lbl_order_name.setText(decision.symbol_name)
        self._set_direction(direction)
        self.lbl_order_confidence.setText(f"{decision.confidence:.0%}")
        risk_text = "通过" if getattr(risk_result, "passed", False) else "待确认"
        if getattr(risk_result, "blocked_reasons", None):
            risk_text = " / ".join(list(risk_result.blocked_reasons)[:2])
        self.lbl_order_risk.setText(risk_text)
        self.price_input.setText(f"{decision.current_price:.2f}" if decision.current_price > 0 else "")
        self.volume_input.setText(self._suggest_lots_for_decision(decision))
        note_parts = [
            f"操作建议: {decision.action_label}",
            f"目标价: {decision.target_price:.2f}" if decision.target_price > 0 else "目标价: -",
            f"止损价: {decision.stop_loss_price:.2f}" if decision.stop_loss_price > 0 else "止损价: -",
            f"建议仓位: {decision.position_pct:.0%}" if decision.position_pct > 0 else "建议仓位: -",
        ]
        if decision.reasoning:
            note_parts.append(f"理由: {decision.reasoning}")
        if getattr(risk_result, "warnings", None):
            note_parts.append("风险提示: " + "；".join(list(risk_result.warnings)[:3]))
        if not decision.is_actionable:
            note_parts.append("当前结论为非执行类建议，默认不提交委托。")
        self.decision_note.setPlainText("\n".join(note_parts))
        self.exec_btn.setEnabled(bool(decision.is_actionable))
        self._form_updating = False
        self._update_amount()

    def fill_order(self, code: str, direction: str, price: float):
        self._form_updating = True
        self._switch_to_manual_mode("该委托来自当前持仓/账户操作，不绑定 AI 实盘决策记录。")
        self._current_code = str(code or "").strip().upper()
        self._current_name = self._resolve_symbol_name(self._current_code) or self._current_code
        self._current_direction = "buy" if direction == "buy" else "sell"
        self.code_input.setText(self._current_code)
        self.lbl_order_name.setText(self._current_name or (self._current_code or "-"))
        self._set_direction(self._current_direction)
        if price > 0:
            self.price_input.setText(f"{price:.2f}")
        else:
            self.price_input.clear()
        self.volume_input.setText(self._suggest_lots_for_manual(code, self._current_direction))
        self._form_updating = False
        self._update_amount()

    def clear_decision_context(self):
        self._decision_context = None

    def _clear_order_form(self):
        self.clear_decision_context()
        self._current_code = ""
        self._current_name = ""
        self._current_direction = "buy"
        self._form_updating = True
        self.code_input.clear()
        self.lbl_order_name.setText("-")
        self._set_direction("buy")
        self.lbl_order_confidence.setText("-")
        self.lbl_order_risk.setText("-")
        self.price_input.clear()
        self.volume_input.clear()
        self.amount_label.setText("-")
        self.decision_note.clear()
        self.exec_btn.setEnabled(True)
        self._form_updating = False

    def _suggest_lots_for_decision(self, decision: TradeDecision) -> str:
        try:
            if decision.action in (TradeAction.SELL.value, TradeAction.REDUCE.value):
                volume = self.execution_service.estimate_volume_for_decision(decision)
                return str(max(int(volume / 100), 0))
            if self.broker.is_connected and decision.current_price > 0:
                asset = self.broker.query_stock_asset()
                cash = float(getattr(asset, "cash", 0) or 0)
                amount = cash * max(float(decision.position_pct or 0.0), 0.0)
                lots = int(math.floor(amount / (decision.current_price * 100)))
                return str(max(lots, 1))
        except Exception:
            pass
        return ""

    def _suggest_lots_for_manual(self, code: str, direction: str) -> str:
        if not self.broker.is_connected or direction != "sell":
            return ""
        try:
            positions = self.broker.query_stock_positions() or []
            code_plain = (code or "").strip().upper().split(".")[0]
            for pos in positions:
                pos_code = str(getattr(pos, "stock_code", "") or "").strip().upper().split(".")[0]
                if pos_code != code_plain:
                    continue
                can_use = int(getattr(pos, "can_use_volume", 0) or 0)
                return str(max(int(can_use / 100), 0))
        except Exception:
            pass
        return ""

    @staticmethod
    def _normalize_xt_code(code: str) -> str:
        value = str(code or "").strip().upper()
        if not value:
            return ""
        if "." in value:
            return value
        if value.startswith(("5", "6", "9")):
            return f"{value}.SH"
        if value.startswith(("0", "1", "2", "3")):
            return f"{value}.SZ"
        return value

    def _resolve_level1_price_from_tick(self, code: str, direction: str) -> tuple[float, str]:
        xt_code = self._normalize_xt_code(code)
        if not xt_code:
            return 0.0, "请先输入有效的证券代码"
        try:
            from xtquant import xtdata
        except Exception as exc:
            logger.exception("导入 xtdata 失败")
            return 0.0, f"行情接口不可用: {exc}"
        try:
            full_tick = xtdata.get_full_tick([xt_code]) or {}
        except Exception as exc:
            logger.exception("获取 tick 失败: %s", xt_code)
            return 0.0, f"获取实时行情失败: {exc}"
        tick = full_tick.get(xt_code)
        if not isinstance(tick, dict) or not tick:
            return 0.0, f"{xt_code} 未返回有效 tick 数据"
        price_key = "askPrice" if direction == "buy" else "bidPrice"
        price_label = "卖一价" if direction == "buy" else "买一价"
        prices = list(tick.get(price_key, []) or [])
        if prices:
            try:
                level1 = float(prices[0] or 0.0)
            except Exception:
                level1 = 0.0
            if level1 > 0:
                return level1, price_label
        fallback = float(tick.get("lastPrice", 0) or 0.0)
        if fallback > 0:
            return fallback, f"{price_label}缺失，已回退到最新价"
        return 0.0, f"{xt_code} 未返回可用的{price_label}或最新价"

    def _fill_price_from_tick(self) -> None:
        code = self.code_input.text().strip().upper()
        if not code:
            QMessageBox.warning(self, "提示", "请先输入证券代码")
            return
        price, message = self._resolve_level1_price_from_tick(code, self._selected_direction())
        if price <= 0:
            QMessageBox.warning(self, "提示", message)
            return
        self.price_input.setText(f"{price:.3f}".rstrip("0").rstrip("."))
        self._update_amount()
        resolved_name = self._resolve_symbol_name(code)
        if resolved_name:
            self._current_name = resolved_name
            self.lbl_order_name.setText(resolved_name)

    def _update_amount(self):
        try:
            price = float(self.price_input.text())
            lots = int(self.volume_input.text())
            amount = price * lots * 100
            self.amount_label.setText(f"¥{amount:,.2f}")
        except (ValueError, TypeError):
            self.amount_label.setText("-")

    def _on_execute(self):
        code = self.code_input.text().strip().upper()
        if not code:
            QMessageBox.warning(self, "提示", "当前没有可提交的委托")
            return
        if not self.broker.is_connected:
            QMessageBox.warning(self, "提示", "券商未连接")
            return
        try:
            price = float(self.price_input.text())
            lots = int(self.volume_input.text())
        except (ValueError, TypeError):
            QMessageBox.warning(self, "提示", "请输入有效的价格和数量")
            return
        volume = lots * 100
        if volume <= 0:
            QMessageBox.warning(self, "提示", "委托数量必须大于0")
            return
        self._current_code = code
        self._current_direction = self._selected_direction()
        resolved_name = self._resolve_symbol_name(code)
        self._current_name = resolved_name or self._current_name or code
        self.lbl_order_name.setText(self._current_name or code)
        order_type = 23 if self._current_direction == "buy" else 24
        action_label = "买入" if self._current_direction == "buy" else "卖出"
        confirm = QMessageBox.question(
            self,
            "委托确认",
            f"确认{action_label} {code} {lots}手(={volume}股) @ ¥{price:.2f}？\n"
            f"委托金额: ¥{price * volume:,.2f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            decision = self._decision_context.get("decision") if self._decision_context else None
            risk_result = self._decision_context.get("risk_result") if self._decision_context else None
            decision_record_id = str(self._decision_context.get("decision_record_id", "")) if self._decision_context else ""
            if decision is not None and not decision.is_actionable:
                QMessageBox.information(self, "提示", "当前 AI 结论不是可执行委托，无需提交下单。")
                return
            result = self.execution_service.execute(
                ExecutionRequest(
                    stock_code=code,
                    stock_name=self._current_name or code,
                    order_type=order_type,
                    order_volume=volume,
                    price_type=5,
                    price=price,
                    source=TradeSource.MANUAL.value,
                    trigger="manual",
                    strategy_name=self._strategy_context.strategy_name,
                    strategy_id=self._strategy_context.strategy_id,
                    virtual_account_id=self._strategy_context.virtual_account_id,
                    remark=f"{self._strategy_context.strategy_name}手动委托下单",
                    decision=decision,
                    risk_result=risk_result,
                    decision_record_id=decision_record_id,
                    require_approval=False,
                    approved=False,
                    metadata={
                        "owner_type": self._strategy_context.owner_type,
                        **dict(self._strategy_context.metadata or {}),
                    },
                )
            )
            self.order_executed.emit(
                result.success,
                result.filled_confirmed,
                result.message,
                result.broker_order_id,
                price,
            )
            if result.success and decision is not None:
                self.clear_decision_context()
        except Exception as exc:
            msg = f"下单失败: {exc}"
            self.order_executed.emit(False, False, msg, -1, 0.0)


# ───────────────────────────────────────────────────────────────────────────
#  Center panel: AI Decision analysis
# ───────────────────────────────────────────────────────────────────────────

