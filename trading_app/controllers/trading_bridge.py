from __future__ import annotations

import logging
import math
from typing import Tuple

from PyQt6.QtCore import QObject, Qt
from PyQt6.QtWidgets import QMainWindow

from common.broker_session_service import get_broker_session_service
from trading_app.services.execution.auto_stop_loss_service import get_auto_stop_loss_service
from trading_app.services.execution.conditional_order_service import get_conditional_order_service
from trading_app.services.execution.trade_execution_service import get_trade_execution_service
from trading_app.services.execution.trade_record_service import (
    TradeDirection,
    TradeSource,
    get_trade_record_service,
    set_auto_stop_loss_service_getter,
)
from trading_app.widgets.broker_account_widget import BrokerAccountWidget

logger = logging.getLogger(__name__)


class TradingBridge(QObject):
    """Bridges trading UI, broker session, and trading-related services."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.broker_session_service = get_broker_session_service()
        self.execution_service = get_trade_execution_service()
        self.broker_window = None
        self._broker_widget = None
        self._conditional_log_connected = False
        self._auto_stop_log_connected = False

    @property
    def broker_widget(self):
        return self._broker_widget

    def initialize(self):
        conditional_service = get_conditional_order_service()
        conditional_service.set_trade_executor(self.execute_conditional_order)
        conditional_service.start_monitoring()
        if not self._conditional_log_connected:
            conditional_service.log_message.connect(
                lambda msg: self.main_window.statusBar().showMessage(msg, 5000)
            )
            self._conditional_log_connected = True

        auto_stop_loss_service = get_auto_stop_loss_service()
        auto_stop_loss_service.set_conditional_order_service(conditional_service)
        if not self._auto_stop_log_connected:
            auto_stop_loss_service.log_message.connect(
                lambda msg: self.main_window.statusBar().showMessage(msg, 5000)
            )
            self._auto_stop_log_connected = True
        set_auto_stop_loss_service_getter(get_auto_stop_loss_service)

        pending_count = conditional_service.pending_count
        if pending_count > 0:
            self.main_window.statusBar().showMessage(
                f"✓ 条件单监控已启动，{pending_count}个待触发",
                5000,
            )
        else:
            self.main_window.statusBar().showMessage("✓ 条件单监控已启动", 3000)

    def shutdown(self):
        if self.broker_window and self.broker_window.isVisible():
            try:
                self.broker_window.close()
            except Exception:
                pass
        try:
            self.broker_session_service.disconnect()
        except Exception:
            pass

    def on_quote_updated(self, quote_data):
        if quote_data.last_price <= 0:
            return
        if not bool(getattr(quote_data, "is_fresh", False)):
            logger.debug(
                "忽略非新鲜行情，不触发条件单: %s source_time=%s age=%s",
                getattr(quote_data, "code", ""),
                getattr(quote_data, "source_time", None),
                getattr(quote_data, "age_seconds", None),
            )
            return
        conditional_service = get_conditional_order_service()
        if conditional_service.is_monitoring:
            conditional_service.check_single_quote(
                quote_data.simple_code,
                quote_data.last_price,
                is_fresh=True,
            )

    def open_broker_account(self, stock_code: str = None):
        if self.broker_window and self.broker_window.isVisible():
            self.broker_window.activateWindow()
            self.broker_window.raise_()
            if stock_code and self._broker_widget:
                self._broker_widget.set_stock_code(stock_code)
            return

        self.broker_window = QMainWindow(self.main_window)
        self.broker_window.setWindowTitle("交易")
        self.broker_window.resize(1200, 800)
        self.broker_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        combined_name_map = {**self.main_window.name_map, **self.main_window.etf_name_map}
        self._broker_widget = BrokerAccountWidget(
            name_map=combined_name_map,
            broker_session_service=self.broker_session_service,
        )
        if stock_code:
            self._broker_widget.set_stock_code(stock_code)

        self._broker_widget.positionsUpdated.connect(self.sync_broker_positions)
        self.broker_window.setCentralWidget(self._broker_widget)
        self.broker_window.destroyed.connect(self._on_broker_window_destroyed)
        self.broker_window.show()

    def _on_broker_window_destroyed(self):
        self.broker_window = None
        self._broker_widget = None

    def sync_broker_positions(self, position_codes: list):
        success, msg = self.main_window.watchlist_manager.update_broker_positions(position_codes)
        if success:
            self.main_window.stock_list_widget.update_group_combo()
            self.main_window.etf_list_widget.update_group_combo()
            if self.main_window.stock_list_widget.get_current_group() == "中金持仓":
                self.main_window.stock_list_widget.on_group_combo_changed(
                    self.main_window.stock_list_widget.group_combo.currentIndex()
                )
            self.main_window.statusBar().showMessage(msg)

    def execute_agent_decision(self, decision, *, risk_result=None, decision_record_id: str = ""):
        """Execute a trade decision from the AI agent.

        Returns ExecutionResult.
        """
        return self.execution_service.execute_agent_decision(
            decision,
            stock_name=decision.symbol_name,
            decision_record_id=decision_record_id,
            risk_result=risk_result,
        )

    def execute_conditional_order(self, stock_code, order_type, order_volume, price_type, price):
        stock_name = stock_code.split(".")[0] if stock_code else ""
        return self.execution_service.execute_conditional_order(
            stock_code=stock_code,
            stock_name=stock_name,
            order_type=order_type,
            order_volume=order_volume,
            price_type=price_type,
            price=price,
            strategy_name="ConditionalOrder",
            remark="条件单自动执行",
        )

    def _calc_buy_volume(self, decision) -> int:
        """Calculate buy volume rounded down to nearest 100 shares."""
        try:
            broker = self.broker_session_service
            # Query available cash
            assets = broker.query_stock_asset()
            if not assets:
                return 0
            available = float(getattr(assets, "cash", 0) or 0)
            if available <= 0:
                return 0

            price = decision.current_price
            if price <= 0:
                return 0

            target_amount = available * decision.position_pct
            raw_volume = target_amount / price
            volume = int(math.floor(raw_volume / 100)) * 100
            return max(volume, 0)
        except Exception as exc:
            logger.warning("Failed to calculate buy volume: %s", exc)
            return 0

    def _calc_sell_volume(self, decision) -> int:
        """Calculate sell volume from current holdings."""
        try:
            broker = self.broker_session_service
            positions = broker.query_stock_positions()
            if not positions:
                return 0

            code = decision.symbol_code
            for pos in positions:
                pos_code = getattr(pos, "stock_code", "") or ""
                if pos_code == code or pos_code.endswith(code) or code.endswith(pos_code):
                    can_sell = int(getattr(pos, "can_use_volume", 0) or 0)
                    if decision.action == "reduce":
                        sell_vol = int(math.floor(can_sell * 0.5 / 100)) * 100
                        return max(sell_vol, 0)
                    return can_sell
            return 0
        except Exception as exc:
            logger.warning("Failed to calculate sell volume: %s", exc)
            return 0
