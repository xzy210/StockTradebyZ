# -*- coding: utf-8 -*-
"""AI decision research perspective for historical LLM prompt replay."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from PyQt6.QtCore import QDate, Qt
from PyQt6.QtWidgets import (
    QDateEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from common.agent import (
    AgentContextService,
    AgentEvidenceService,
    AgentPromptBuilder,
    DecisionRunContext,
    EvidenceBundle,
    EvidenceItem,
    TASK_MODE_TRADE_DECISION,
)
from common.data_portal import get_data_portal
from common.events import BacktestEvent, EventBus
from common.experiment_store import ExperimentStore
from strategy_app.strategies import AIStockStrategyParams


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"
_PERSISTED_EVENT_TYPES = {
    "run_requested",
    "run_started",
    "symbol_replayed",
    "run_completed",
    "run_failed",
    "experiment_saved",
    "experiment_save_failed",
}


def _make_event(
    event_type: str,
    message: str,
    *,
    progress_current: int | None = None,
    progress_total: int | None = None,
    run_id: str = "",
    payload: dict[str, Any] | None = None,
) -> BacktestEvent:
    return BacktestEvent(
        date=None,
        bars={},
        history={},
        prices={},
        valid_symbols=[],
        event_type=event_type,
        message=message,
        progress_current=progress_current,
        progress_total=progress_total,
        mode="ai_decision_research",
        run_id=run_id,
        payload=payload or {},
    )


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model or {})


class AIDecisionResearchWidget(QWidget):
    """Replay historical AI decision prompts and persist experiment records."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        event_bus: EventBus | None = None,
        experiment_store: ExperimentStore | None = None,
        on_experiment_saved: Callable[[], None] | None = None,
        data_dir: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.event_bus = event_bus or EventBus()
        self.experiment_store = experiment_store
        self.on_experiment_saved = on_experiment_saved
        self.data_dir = Path(data_dir or _DEFAULT_DATA_DIR)
        self.evidence_service = AgentEvidenceService(_PROJECT_ROOT / "data" / "ai_decision_research_evidence")
        self._run_events: list[BacktestEvent] = []
        self._last_result: dict[str, Any] | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.symbol_edit = QTextEdit(self)
        self.symbol_edit.setPlaceholderText("每行或逗号分隔输入股票/ETF代码，例如：000001.SZ, 510300")
        self.symbol_edit.setPlainText("000001.SZ")
        self.symbol_edit.setMaximumHeight(90)

        self.replay_date_edit = QDateEdit(QDate.currentDate(), self)
        self.replay_date_edit.setCalendarPopup(True)
        self.replay_date_edit.setDisplayFormat("yyyy-MM-dd")

        self.lookback_spin = QSpinBox(self)
        self.lookback_spin.setRange(30, 1000)
        self.lookback_spin.setValue(180)
        self.lookback_spin.setSuffix(" 日")

        self.model_edit = QLineEdit(self)
        self.model_edit.setPlaceholderText("例如 kimi-k3 / glm-5.3-flash / gemini-3-flash-preview")

        self.prompt_edit = QTextEdit(self)
        self.prompt_edit.setPlaceholderText("留空时使用 AIStockStrategyParams 默认系统提示词")
        self.prompt_edit.setMaximumHeight(110)

        self.run_btn = QPushButton("生成历史回放并保存实验", self)
        self.run_btn.clicked.connect(self.run_replay)

        form = QFormLayout()
        form.addRow("标的列表", self.symbol_edit)
        form.addRow("回放日期", self.replay_date_edit)
        form.addRow("回看窗口", self.lookback_spin)
        form.addRow("模型名称", self.model_edit)
        form.addRow("系统提示词", self.prompt_edit)

        config_group = QGroupBox("AI 决策研究参数", self)
        config_layout = QVBoxLayout(config_group)
        config_layout.addLayout(form)
        config_layout.addWidget(self.run_btn)

        self.result_table = QTableWidget(0, 7, self)
        self.result_table.setHorizontalHeaderLabels(["代码", "名称", "日期", "收盘", "涨跌幅", "证据", "Prompt长度"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.result_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.result_table.itemSelectionChanged.connect(self._render_selected_detail)

        self.detail_edit = QTextEdit(self)
        self.detail_edit.setReadOnly(True)
        self.detail_edit.setPlaceholderText("选择结果行查看 prompt / evidence / experiment payload。")

        right_splitter = QSplitter(Qt.Orientation.Vertical, self)
        right_splitter.addWidget(self.result_table)
        right_splitter.addWidget(self.detail_edit)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 2)

        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.addWidget(config_group)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        self.status_label = QLabel("就绪：生成回放包会保存到实验记录，不会调用实盘下单。", self)
        self.status_label.setProperty("class", "description")

        layout = QVBoxLayout(self)
        layout.addWidget(main_splitter, 1)
        layout.addWidget(self.status_label)

    def run_replay(self) -> None:
        symbols = self._parse_symbols(self.symbol_edit.toPlainText())
        if not symbols:
            QMessageBox.warning(self, "参数缺失", "请至少输入一个标的代码。")
            return

        replay_date = self.replay_date_edit.date().toString("yyyy-MM-dd")
        lookback_days = int(self.lookback_spin.value())
        run_id = f"ai_decision_research_{uuid4().hex[:12]}"
        params = self._build_params()
        params_hash = params.params_hash()
        self._run_events = []
        self._last_result = None
        self.result_table.setRowCount(0)
        self.detail_edit.clear()

        self._publish("run_requested", f"AI 决策研究回放请求：{len(symbols)} 个标的", run_id=run_id)
        self._publish("run_started", f"开始生成历史回放：{replay_date}", run_id=run_id)

        try:
            rows: list[dict[str, Any]] = []
            for index, symbol in enumerate(symbols, start=1):
                row = self._replay_symbol(
                    symbol=symbol,
                    replay_date=replay_date,
                    lookback_days=lookback_days,
                    params=params,
                    run_id=run_id,
                )
                rows.append(row)
                self._append_result_row(row)
                self._publish(
                    "symbol_replayed",
                    f"已生成 {symbol} 回放包",
                    progress_current=index,
                    progress_total=len(symbols),
                    run_id=run_id,
                    payload={"symbol": symbol, "evidence_report_path": row.get("evidence_report_path", "")},
                )
        except Exception as exc:
            self._publish("run_failed", f"AI 决策研究回放失败：{exc}", run_id=run_id, payload={"error": str(exc)})
            QMessageBox.critical(self, "回放失败", str(exc))
            return

        result = {
            "schema_version": "ai_decision_research_result.v1",
            "run_id": run_id,
            "strategy_id": params.strategy_id or "ai_stock",
            "params_hash": params_hash,
            "mode": "ai_decision_research",
            "engine_version": "ai_decision_research.v1",
            "data_version": replay_date,
            "final_value": None,
            "replay_date": replay_date,
            "lookback_days": lookback_days,
            "symbols": symbols,
            "results": rows,
        }
        self._last_result = result
        self._publish("run_completed", f"AI 决策研究回放完成：{len(rows)} 条", run_id=run_id)
        self._save_experiment_record(result, params)
        self.status_label.setText(f"完成：{run_id} / params_hash={params_hash}")
        if self.result_table.rowCount() > 0:
            self.result_table.selectRow(0)

    def _build_params(self) -> AIStockStrategyParams:
        return AIStockStrategyParams.from_mapping(
            {
                "model_name": self.model_edit.text().strip(),
                "system_prompt": self.prompt_edit.toPlainText().strip(),
                "prompt_template_version": "ai_decision_research_v1",
            }
        )

    def _replay_symbol(
        self,
        *,
        symbol: str,
        replay_date: str,
        lookback_days: int,
        params: AIStockStrategyParams,
        run_id: str,
    ) -> dict[str, Any]:
        start_date = (datetime.strptime(replay_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        df = get_data_portal().get_daily_bars(
            symbol,
            start=start_date,
            end=replay_date,
            data_dir=self.data_dir,
            use_cache=False,
        )
        if df is None or df.empty:
            raise ValueError(f"{symbol} 在 {start_date}~{replay_date} 无可用日线数据")

        df = df.sort_values("date").reset_index(drop=True)
        latest = df.iloc[-1]
        prev_close = float(df["close"].iloc[-2]) if len(df) >= 2 else float(latest["close"])
        close = float(latest["close"])
        change_pct = ((close / prev_close) - 1.0) * 100.0 if prev_close else 0.0
        latest_date = str(pd.to_datetime(latest["date"]).date())
        name = self._resolve_name(symbol)
        run_context = DecisionRunContext(
            run_at=f"{replay_date} 15:10:00",
            trading_day=replay_date,
            is_trading_day=True,
            session_phase="postmarket",
            prefer_realtime=False,
            daily_bar_as_of=latest_date,
            realtime_as_of="",
        )
        raw_context = {
            "symbol": {
                "code": symbol,
                "name": name,
                "asset_type": "auto",
                "current_view": "research_replay",
                "latest_close": close,
                "latest_change_pct": change_pct,
                "latest_volume": float(latest.get("volume", 0.0) or 0.0),
                "data_points": int(len(df)),
                "date_start": str(pd.to_datetime(df["date"].iloc[0]).date()),
                "date_end": latest_date,
                "indicators": ["close", "volume", "return"],
            },
            "broker": {"connected": False},
            "decision_run_context": run_context.to_dict(),
        }
        context = AgentContextService.from_raw(raw_context)
        system_prompt = AgentPromptBuilder.build_system_prompt(
            params.system_prompt,
            context,
            task_mode=TASK_MODE_TRADE_DECISION,
        )
        user_prompt = AgentPromptBuilder.build_quick_task_prompt(TASK_MODE_TRADE_DECISION, context)
        evidence_item = EvidenceItem(
            tool_name="historical_daily_bars",
            title=f"{name or symbol} 历史日线摘要",
            summary=f"{latest_date} 收盘 {close:.3f}，涨跌幅 {change_pct:.2f}%，样本 {len(df)} 条",
            content=self._build_bar_summary(df),
            metadata={
                "symbol": symbol,
                "replay_date": replay_date,
                "latest_date": latest_date,
                "lookback_days": lookback_days,
            },
        )
        bundle = EvidenceBundle(
            run_id=f"{run_id}_{symbol.replace('.', '')}",
            task_mode=TASK_MODE_TRADE_DECISION,
            user_input=user_prompt,
            created_at=self.evidence_service.now_iso(),
            context_summary=context.to_summary_lines(),
            items=[evidence_item],
        )
        evidence_report_path = self.evidence_service.save_bundle(bundle)
        return {
            "symbol": symbol,
            "name": name,
            "replay_date": replay_date,
            "latest_date": latest_date,
            "close": close,
            "change_pct": change_pct,
            "data_points": int(len(df)),
            "params_hash": params.params_hash(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "evidence_report_path": evidence_report_path,
            "evidence_summary": evidence_item.summary,
            "context": raw_context,
        }

    def _save_experiment_record(self, result: dict[str, Any], params: AIStockStrategyParams) -> None:
        if self.experiment_store is None:
            return
        try:
            events = [event for event in self._run_events if event.event_type in _PERSISTED_EVENT_TYPES]
            record = self.experiment_store.save(result, events=events, params=_model_to_dict(params))
        except Exception as exc:
            self._publish("experiment_save_failed", f"AI 决策研究实验记录保存失败：{exc}", payload={"error": str(exc)})
            self.status_label.setText(f"实验记录保存失败：{exc}")
            return

        self._publish(
            "experiment_saved",
            f"AI 决策研究实验记录已保存：{record.run_id}",
            run_id=record.run_id,
            payload={"path": record.path},
        )
        if self.on_experiment_saved is not None:
            self.on_experiment_saved()

    def _append_result_row(self, row: dict[str, Any]) -> None:
        table_row = self.result_table.rowCount()
        self.result_table.insertRow(table_row)
        values = [
            row.get("symbol", ""),
            row.get("name", ""),
            row.get("latest_date", ""),
            f"{float(row.get('close', 0.0) or 0.0):.3f}",
            f"{float(row.get('change_pct', 0.0) or 0.0):.2f}%",
            row.get("evidence_report_path", ""),
            str(len(row.get("system_prompt", "")) + len(row.get("user_prompt", ""))),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.ItemDataRole.UserRole, row)
            self.result_table.setItem(table_row, col, item)

    def _render_selected_detail(self) -> None:
        items = self.result_table.selectedItems()
        if not items:
            return
        row = items[0].data(Qt.ItemDataRole.UserRole)
        if not isinstance(row, dict):
            return
        text = (
            f"## {row.get('name') or row.get('symbol')} ({row.get('symbol')})\n\n"
            f"- 回放日期: {row.get('replay_date')}\n"
            f"- 数据日期: {row.get('latest_date')}\n"
            f"- 证据文件: {row.get('evidence_report_path')}\n"
            f"- params_hash: {row.get('params_hash')}\n\n"
            "## 用户任务\n\n"
            f"{row.get('user_prompt', '')}\n\n"
            "## 系统提示词\n\n"
            f"{row.get('system_prompt', '')}\n"
        )
        self.detail_edit.setPlainText(text)

    def _publish(self, event_type: str, message: str, **kwargs: Any) -> None:
        event = _make_event(event_type, message, **kwargs)
        self._run_events.append(event)
        self.event_bus.publish(event)
        self.status_label.setText(message)

    def _resolve_name(self, symbol: str) -> str:
        portal = get_data_portal()
        for asset_type in ("stock", "etf", "index"):
            try:
                name = portal.get_name_map(asset_type=asset_type).get(portal.normalize_symbol(symbol), "")
            except Exception:
                name = ""
            if name:
                return str(name)
        return ""

    @staticmethod
    def _parse_symbols(text: str) -> list[str]:
        raw = text.replace(",", "\n").replace("，", "\n").replace(";", "\n").replace("；", "\n")
        symbols = []
        seen = set()
        for part in raw.splitlines():
            value = part.strip().upper()
            if not value or value in seen:
                continue
            symbols.append(value)
            seen.add(value)
        return symbols

    @staticmethod
    def _build_bar_summary(df: pd.DataFrame) -> str:
        cols = [col for col in ["date", "open", "high", "low", "close", "volume"] if col in df.columns]
        tail = df[cols].tail(10).copy()
        if "date" in tail.columns:
            tail["date"] = pd.to_datetime(tail["date"]).dt.strftime("%Y-%m-%d")
        returns = df["close"].pct_change().dropna()
        summary = [
            f"- 样本数: {len(df)}",
            f"- 区间: {pd.to_datetime(df['date'].iloc[0]).date()} ~ {pd.to_datetime(df['date'].iloc[-1]).date()}",
            f"- 近20日收益: {AIDecisionResearchWidget._window_return(df, 20):.2f}%",
            f"- 近60日收益: {AIDecisionResearchWidget._window_return(df, 60):.2f}%",
            f"- 日收益波动率: {(returns.std() * 100.0 if not returns.empty else 0.0):.2f}%",
            "",
            "最近10根日线:",
            tail.to_string(index=False),
        ]
        return "\n".join(summary)

    @staticmethod
    def _window_return(df: pd.DataFrame, window: int) -> float:
        if len(df) < 2:
            return 0.0
        start_idx = max(0, len(df) - window)
        start_close = float(df["close"].iloc[start_idx] or 0.0)
        end_close = float(df["close"].iloc[-1] or 0.0)
        return ((end_close / start_close) - 1.0) * 100.0 if start_close else 0.0


def create_ai_decision_research_tab(
    parent: QWidget | None = None,
    *,
    event_bus: EventBus | None = None,
    experiment_store: ExperimentStore | None = None,
    on_experiment_saved: Callable[[], None] | None = None,
) -> QWidget:
    return AIDecisionResearchWidget(
        parent,
        event_bus=event_bus,
        experiment_store=experiment_store,
        on_experiment_saved=on_experiment_saved,
    )


__all__ = ["AIDecisionResearchWidget", "create_ai_decision_research_tab"]
