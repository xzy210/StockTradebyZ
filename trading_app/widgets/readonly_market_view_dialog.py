# -*- coding: utf-8 -*-
"""Read-only market view used by the live strategy center."""

from __future__ import annotations

from datetime import date, timedelta

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from common.data_portal import get_data_portal
from common.indicators import attach_all_indicators
from common.market_data_policy import is_etf_like_code, normalize_symbol_code
from trading_app.services.market_data.quote_service import QuoteData, get_quote_service
from trading_app.widgets.kline_widget import KLineWidget
from trading_app.widgets.order_book_widget import OrderBookWidget
from trading_app.widgets.timeshare_widget import TimeShareWidget


class ReadOnlyMarketViewDialog(QDialog):
    """K线/分时/盘口查看弹窗。

    该弹窗只做行情展示，不暴露下单、条件单或止损入口。
    """

    def __init__(self, parent=None, *, symbol_name_resolver=None) -> None:
        super().__init__(parent)
        self.symbol_name_resolver = symbol_name_resolver
        self.symbol_code = ""
        self.symbol_name = ""
        self._timeshare_loaded = False
        self._timeshare_context: tuple[str, str, str, float] | None = None
        self._quote_owner_id = f"readonly_market_view:{id(self)}"
        self._quote_service = get_quote_service()

        self.setWindowTitle("只读行情视图")
        self.resize(1180, 760)
        self._setup_ui()
        self._quote_service.quote_updated.connect(self._on_quote_updated)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.title_label = QLabel("请选择标的", self)
        self.title_label.setStyleSheet("font-size:16px;font-weight:bold;color:#f3f4f6;")
        header.addWidget(self.title_label, 1)

        self.status_label = QLabel("只读：不会触发任何下单或条件单操作", self)
        self.status_label.setStyleSheet("color:#9ca3af;font-size:12px;")
        header.addWidget(self.status_label)

        refresh_btn = QPushButton("刷新", self)
        refresh_btn.clicked.connect(lambda: self.load_symbol(self.symbol_code, self.symbol_name))
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.tabs = QTabWidget(self)
        self.kline_widget = KLineWidget(self.tabs)
        self.kline_widget.set_indicators(show_volume=True, show_macd=True, show_kdj=False)
        self.kline_widget.set_ma_windows([5, 10, 20, 60])
        self.tabs.addTab(self.kline_widget, "K线")

        self.timeshare_widget = TimeShareWidget(self.tabs)
        self.tabs.addTab(self.timeshare_widget, "分时")

        order_tab = QWidget(self.tabs)
        order_layout = QVBoxLayout(order_tab)
        order_layout.setContentsMargins(0, 0, 0, 0)
        self.order_book_widget = OrderBookWidget(order_tab)
        order_layout.addWidget(self.order_book_widget)
        order_layout.addStretch(1)
        self.tabs.addTab(order_tab, "盘口")

        layout.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._on_tab_changed)

    def load_symbol(self, code: str, name: str = "") -> bool:
        normalized = normalize_symbol_code(code).zfill(6)
        if not normalized or normalized == "000000":
            QMessageBox.information(self, "只读行情", "请先输入或选择一个有效标的代码。")
            return False

        self.symbol_code = normalized
        self.symbol_name = name or self._resolve_symbol_name(normalized)
        self._timeshare_loaded = False
        self._timeshare_context = None
        self.title_label.setText(f"{self.symbol_code} {self.symbol_name}".strip())
        self.status_label.setText("正在加载本地日线数据...")

        end_date = date.today()
        start_date = end_date - timedelta(days=520)
        asset_type = "etf" if is_etf_like_code(normalized) else "stock"
        df = get_data_portal().get_daily_bars(
            normalized,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            asset_type=asset_type,
            use_cache=True,
        )
        if df is None or df.empty:
            self.status_label.setText("未找到本地日线数据")
            self.kline_widget.info_label.setText(f"未找到 {normalized} 的本地日线数据")
            self.timeshare_widget.info_label.setText(f"未找到 {normalized} 的本地日线数据")
            self.timeshare_widget.setup_plots()
            self.order_book_widget.clear_data()
            return False

        df = attach_all_indicators(
            df,
            ma_windows=[5, 10, 20, 60],
            include_macd=True,
            include_kdj=False,
            include_bbi=False,
            vol_ma_window=5,
        )
        self.kline_widget.set_data(df, normalized, self.symbol_name)

        latest_date = df["date"].max()
        latest_date_text = latest_date.strftime("%Y-%m-%d")
        prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else float(df["open"].iloc[-1])
        self._prepare_timeshare_lazy_load(
            normalized,
            latest_date_text,
            str(get_data_portal().default_data_dir),
            prev_close,
        )

        self._quote_service.replace_subscription(self._quote_owner_id, [normalized], start_service=True)
        cached_quote = self._quote_service.get_quote(normalized)
        if cached_quote is not None:
            self._apply_quote(cached_quote)
        else:
            self.order_book_widget.clear_data()
        self.status_label.setText(f"日线范围: {df['date'].min().strftime('%Y-%m-%d')} ~ {latest_date_text}")
        return True

    def _prepare_timeshare_lazy_load(self, code: str, date_text: str, data_dir: str, prev_close: float) -> None:
        self._timeshare_loaded = False
        self._timeshare_context = (code, date_text, data_dir, prev_close)
        self.timeshare_widget.code = code
        self.timeshare_widget.date_str = date_text
        self.timeshare_widget.data_dir = data_dir
        self.timeshare_widget.prev_close = prev_close
        self.timeshare_widget.data = None
        self.timeshare_widget.setup_plots()
        self.timeshare_widget.info_label.setText(
            f"{code} {date_text} 分时图未加载，切换到分时页后再获取数据"
        )
        if self.tabs.currentWidget() is self.timeshare_widget:
            self._load_timeshare_if_needed()

    def _on_tab_changed(self, _index: int) -> None:
        if not hasattr(self, "timeshare_widget"):
            return
        if self.tabs.currentWidget() is self.timeshare_widget:
            self._load_timeshare_if_needed()

    def _load_timeshare_if_needed(self) -> None:
        if self._timeshare_loaded or self._timeshare_context is None:
            return
        code, date_text, data_dir, prev_close = self._timeshare_context
        self._timeshare_loaded = True
        self.timeshare_widget.load_data(code, date_text, data_dir, prev_close=prev_close)

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self._quote_service.clear_owner_subscription(self._quote_owner_id)
        except Exception:
            pass
        super().closeEvent(event)

    def _resolve_symbol_name(self, code: str) -> str:
        if callable(self.symbol_name_resolver):
            try:
                return str(self.symbol_name_resolver(code) or "")
            except Exception:
                return ""
        return ""

    def _on_quote_updated(self, quote: QuoteData) -> None:
        if not self.symbol_code:
            return
        if normalize_symbol_code(getattr(quote, "code", "")) != self.symbol_code:
            return
        self._apply_quote(quote)

    def _apply_quote(self, quote: QuoteData) -> None:
        self.order_book_widget.update_data(quote.to_dict(), prev_close=float(quote.prev_close or 0.0))
        if getattr(quote, "last_price", 0.0):
            self.status_label.setText(
                f"最新价 {quote.last_price:.2f} / 涨跌幅 {quote.change_pct:+.2f}% / "
                f"{quote.received_time.strftime('%H:%M:%S')}"
            )


__all__ = ["ReadOnlyMarketViewDialog"]
