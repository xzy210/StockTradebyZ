from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, Optional, Protocol, runtime_checkable

from common.execution_contract import OrderExecutionReport, OrderIntent, RebalanceIntent, StrategySignal, TargetPortfolio
from trading_app.services.ops.live_strategy_end_of_day_service import StrategyEndOfDayResult

logger = logging.getLogger(__name__)


@runtime_checkable
class LiveStrategyAdapter(Protocol):
    """Explicit integration contract between the live strategy center and a strategy."""

    @property
    def strategy_id(self) -> str:
        ...

    @property
    def strategy_name(self) -> str:
        ...

    @property
    def virtual_account_id(self) -> str:
        ...

    @property
    def widget(self):
        ...

    def get_status_summary(self) -> dict:
        ...

    def get_task_summaries(self) -> list[dict]:
        ...

    def get_task_summary(self, task_key: str) -> dict:
        ...

    def pause_automation(self) -> str:
        ...

    def resume_automation(self) -> str:
        ...

    def is_automation_paused(self) -> bool:
        ...

    def run_end_of_day(self, snapshot_date: str) -> StrategyEndOfDayResult:
        ...

    def refresh_after_eod(self) -> None:
        ...

    def get_rotation_pool(self) -> list[str]:
        ...

    def generate_live_signals(self, payload: Optional[dict] = None) -> list[StrategySignal]:
        ...

    def generate_live_order_intents(self, payload: Optional[dict] = None) -> list[OrderIntent]:
        ...

    def generate_live_rebalance_intent(self, payload: Optional[dict] = None) -> Optional[RebalanceIntent]:
        ...

    def execute_live_order_intents(
        self,
        intents: list[OrderIntent],
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        ...

    def execute_live_rebalance_intent(
        self,
        rebalance_intent: RebalanceIntent,
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        ...

    def execute_live_signals(
        self,
        signals: list[StrategySignal],
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        ...


@dataclass
class PanelLiveStrategyAdapter:
    """Adapter for existing strategy panels that already expose center methods."""

    strategy_id_value: str
    strategy_name_value: str
    panel: object
    virtual_account_id_value: str = ""
    automation_paused_provider: Optional[Callable[[], bool]] = None
    rotation_pool_provider: Optional[Callable[[], list[str]]] = None

    @classmethod
    def from_panel(
        cls,
        panel: object,
        *,
        strategy_id: str = "",
        strategy_name: str = "",
        virtual_account_id: str = "",
        automation_paused_provider: Optional[Callable[[], bool]] = None,
        rotation_pool_provider: Optional[Callable[[], list[str]]] = None,
    ) -> "PanelLiveStrategyAdapter":
        resolved_id = str(strategy_id or "").strip()
        resolved_name = str(strategy_name or "").strip()
        resolved_virtual_account_id = str(virtual_account_id or "").strip()
        identity_method = getattr(panel, "_etf_strategy_identity", None)
        if callable(identity_method) and (not resolved_id or not resolved_name or not resolved_virtual_account_id):
            try:
                legacy_id, legacy_name, legacy_virtual_account_id = identity_method()
                resolved_id = resolved_id or str(legacy_id or "").strip()
                resolved_name = resolved_name or str(legacy_name or "").strip()
                resolved_virtual_account_id = resolved_virtual_account_id or str(legacy_virtual_account_id or "").strip()
            except Exception as exc:
                logger.debug("读取 legacy 策略身份失败: %s", exc)
        return cls(
            strategy_id_value=resolved_id,
            strategy_name_value=resolved_name or resolved_id,
            panel=panel,
            virtual_account_id_value=resolved_virtual_account_id,
            automation_paused_provider=automation_paused_provider,
            rotation_pool_provider=rotation_pool_provider,
        )

    @property
    def strategy_id(self) -> str:
        return str(self.strategy_id_value or "").strip()

    @property
    def strategy_name(self) -> str:
        return str(self.strategy_name_value or self.strategy_id or "").strip()

    @property
    def virtual_account_id(self) -> str:
        return str(self.virtual_account_id_value or "").strip()

    @property
    def widget(self):
        return self.panel

    def get_status_summary(self) -> dict:
        payload = self._call_dict("get_center_status_summary")
        payload.setdefault("strategy_id", self.strategy_id)
        payload.setdefault("strategy_name", self.strategy_name)
        if self.virtual_account_id:
            payload.setdefault("virtual_account_id", self.virtual_account_id)
        payload.setdefault("automation_paused", self.is_automation_paused())
        return payload

    def get_task_summaries(self) -> list[dict]:
        method = getattr(self.panel, "get_center_task_summaries", None)
        if not callable(method):
            return []
        try:
            raw_rows = method() or []
        except Exception as exc:
            logger.warning("读取策略任务摘要失败 strategy_id=%s err=%s", self.strategy_id, exc)
            return []
        rows: list[dict] = []
        for item in raw_rows:
            row = dict(item or {})
            row.setdefault("strategy_id", self.strategy_id)
            row.setdefault("strategy_name", self.strategy_name)
            if self.virtual_account_id:
                row.setdefault("virtual_account_id", self.virtual_account_id)
            rows.append(row)
        return rows

    def get_task_summary(self, task_key: str) -> dict:
        method = getattr(self.panel, "get_center_task_summary", None)
        if callable(method):
            try:
                row = dict(method(task_key) or {})
            except Exception as exc:
                logger.warning("读取策略任务摘要失败 strategy_id=%s task_key=%s err=%s", self.strategy_id, task_key, exc)
                row = {}
            if row:
                row.setdefault("strategy_id", self.strategy_id)
                row.setdefault("strategy_name", self.strategy_name)
                if self.virtual_account_id:
                    row.setdefault("virtual_account_id", self.virtual_account_id)
                return row
        for row in self.get_task_summaries():
            if str(row.get("task_key", "") or "") == str(task_key or ""):
                return dict(row)
        return {}

    def pause_automation(self) -> str:
        return self._call_text("pause_center_automation", default=f"{self.strategy_name} 不支持暂停自动化")

    def resume_automation(self) -> str:
        return self._call_text("resume_center_automation", default=f"{self.strategy_name} 不支持恢复自动化")

    def is_automation_paused(self) -> bool:
        if self.automation_paused_provider is not None:
            try:
                return bool(self.automation_paused_provider())
            except Exception as exc:
                logger.debug("读取策略暂停状态失败 strategy_id=%s err=%s", self.strategy_id, exc)
        status = self._call_dict("get_center_status_summary")
        return bool(status.get("automation_paused", False))

    def run_end_of_day(self, snapshot_date: str) -> StrategyEndOfDayResult:
        method = getattr(self.panel, "run_end_of_day_tasks", None)
        if not callable(method):
            return StrategyEndOfDayResult(
                strategy_id=self.strategy_id,
                strategy_name=self.strategy_name,
                success=True,
                message=f"{self.strategy_name} 无日终钩子",
            )
        return method(snapshot_date)

    def refresh_after_eod(self) -> None:
        method = getattr(self.panel, "refresh_end_of_day_ui", None)
        if callable(method):
            method()

    def get_rotation_pool(self) -> list[str]:
        if self.rotation_pool_provider is None:
            return []
        try:
            return list(self.rotation_pool_provider() or [])
        except Exception as exc:
            logger.debug("读取策略轮动池失败 strategy_id=%s err=%s", self.strategy_id, exc)
            return []

    def generate_live_signals(self, payload: Optional[dict] = None) -> list[StrategySignal]:
        """Generate unified live signals from an adapted strategy panel when supported."""
        method = getattr(self.panel, "generate_live_signals", None)
        if not callable(method):
            return []
        try:
            raw_signals = list(method(dict(payload or {})) or [])
        except TypeError:
            try:
                raw_signals = list(method() or [])
            except Exception as exc:
                logger.warning("生成实盘策略中枢统一信号失败 strategy_id=%s err=%s", self.strategy_id, exc)
                return []
        except Exception as exc:
            logger.warning("生成实盘策略中枢统一信号失败 strategy_id=%s err=%s", self.strategy_id, exc)
            return []
        return [self._with_strategy_identity(signal) for signal in raw_signals if isinstance(signal, StrategySignal)]

    def generate_live_order_intents(self, payload: Optional[dict] = None) -> list[OrderIntent]:
        """Generate native OrderIntent outputs from an adapted strategy panel when supported."""
        method = getattr(self.panel, "generate_live_order_intents", None)
        if callable(method):
            try:
                raw_intents = list(method(dict(payload or {})) or [])
            except TypeError:
                try:
                    raw_intents = list(method() or [])
                except Exception as exc:
                    logger.warning("生成实盘策略中枢订单意图失败 strategy_id=%s err=%s", self.strategy_id, exc)
                    return []
            except Exception as exc:
                logger.warning("生成实盘策略中枢订单意图失败 strategy_id=%s err=%s", self.strategy_id, exc)
                return []
            return [self._with_order_identity(intent) for intent in raw_intents if isinstance(intent, OrderIntent)]

        rebalance_intent = self.generate_live_rebalance_intent(payload)
        if rebalance_intent is not None:
            return list(rebalance_intent.order_intents or ())
        return []

    def generate_live_rebalance_intent(self, payload: Optional[dict] = None) -> Optional[RebalanceIntent]:
        """Generate a native RebalanceIntent from an adapted strategy panel when supported."""
        method = getattr(self.panel, "generate_live_rebalance_intent", None)
        if callable(method):
            try:
                raw_intent = method(dict(payload or {}))
            except TypeError:
                try:
                    raw_intent = method()
                except Exception as exc:
                    logger.warning("生成实盘策略中枢组合意图失败 strategy_id=%s err=%s", self.strategy_id, exc)
                    return None
            except Exception as exc:
                logger.warning("生成实盘策略中枢组合意图失败 strategy_id=%s err=%s", self.strategy_id, exc)
                return None
            return self._coerce_rebalance_intent(raw_intent)

        check_method = getattr(self.panel, "run_signal_check", None)
        if not callable(check_method):
            return None
        try:
            raw_result = check_method(schedule_context=dict(payload or {}).get("schedule_context"))
        except TypeError:
            try:
                raw_result = check_method()
            except Exception as exc:
                logger.warning("生成实盘策略中枢组合意图失败 strategy_id=%s err=%s", self.strategy_id, exc)
                return None
        except Exception as exc:
            logger.warning("生成实盘策略中枢组合意图失败 strategy_id=%s err=%s", self.strategy_id, exc)
            return None
        if isinstance(raw_result, dict):
            return self._coerce_rebalance_intent(raw_result.get("rebalance_intent"))
        return None

    def execute_live_order_intents(
        self,
        intents: list[OrderIntent],
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Execute native OrderIntent outputs through TradeExecutionService."""
        normalized = [self._with_order_identity(intent) for intent in list(intents or [])]
        if not normalized:
            return []
        panel_execute = getattr(self.panel, "execute_live_order_intents", None)
        if callable(panel_execute):
            return list(
                panel_execute(
                    normalized,
                    execution_service=execution_service,
                    stock_name_map=stock_name_map or {},
                )
                or []
            )
        service = execution_service
        if service is None:
            from trading_app.services.execution.trade_execution_service import get_trade_execution_service
            service = get_trade_execution_service()
        return list(service.execute_order_intents(normalized, stock_name_map=stock_name_map or {}))

    def execute_live_rebalance_intent(
        self,
        rebalance_intent: RebalanceIntent,
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Execute a native RebalanceIntent through TradeExecutionService."""
        normalized = self._with_rebalance_identity(rebalance_intent)
        if normalized is None:
            return []
        panel_execute = getattr(self.panel, "execute_live_rebalance_intent", None)
        if callable(panel_execute):
            return list(
                panel_execute(
                    normalized,
                    execution_service=execution_service,
                    stock_name_map=stock_name_map or {},
                )
                or []
            )
        service = execution_service
        if service is None:
            from trading_app.services.execution.trade_execution_service import get_trade_execution_service
            service = get_trade_execution_service()
        return list(service.execute_rebalance_intent(normalized, stock_name_map=stock_name_map or {}))

    def execute_live_signals(
        self,
        signals: list[StrategySignal],
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Execute unified live signals through TradeExecutionService."""
        normalized = [self._with_strategy_identity(signal) for signal in list(signals or [])]
        if not normalized:
            return []
        panel_execute = getattr(self.panel, "execute_live_signals", None)
        if callable(panel_execute):
            return list(
                panel_execute(
                    normalized,
                    execution_service=execution_service,
                    stock_name_map=stock_name_map or {},
                )
                or []
            )
        service = execution_service
        if service is None:
            from trading_app.services.execution.trade_execution_service import get_trade_execution_service
            service = get_trade_execution_service()
        return list(service.execute_signals(normalized, stock_name_map=stock_name_map or {}))

    def _with_strategy_identity(self, signal: StrategySignal) -> StrategySignal:
        metadata = dict(signal.metadata or {})
        if self.virtual_account_id:
            metadata.setdefault("virtual_account_id", self.virtual_account_id)
        metadata.setdefault("source", "live_strategy_center")
        metadata.setdefault("trigger", "strategy_center")
        return replace(
            signal,
            strategy_id=signal.strategy_id or self.strategy_id,
            strategy_name=signal.strategy_name or self.strategy_name,
            metadata=metadata,
        )

    def _with_order_identity(self, intent: OrderIntent) -> OrderIntent:
        metadata = dict(intent.metadata or {})
        if self.virtual_account_id:
            metadata.setdefault("virtual_account_id", self.virtual_account_id)
        source = str(metadata.get("source", "") or intent.source or "").strip()
        trigger = str(metadata.get("trigger", "") or intent.trigger or "").strip()
        if source in {"", "strategy"}:
            source = "live_strategy_center"
        if trigger in {"", "auto"}:
            trigger = "strategy_center"
        metadata["source"] = source
        metadata["trigger"] = trigger
        virtual_account_id = intent.virtual_account_id or metadata.get("virtual_account_id", "")
        return replace(
            intent,
            strategy_id=intent.strategy_id or self.strategy_id,
            strategy_name=intent.strategy_name or self.strategy_name,
            virtual_account_id=str(virtual_account_id or ""),
            source=source,
            trigger=trigger,
            metadata=metadata,
        )

    def _with_rebalance_identity(self, rebalance_intent: Optional[RebalanceIntent]) -> Optional[RebalanceIntent]:
        if rebalance_intent is None:
            return None
        target = rebalance_intent.target_portfolio
        target_metadata = dict(target.metadata or {})
        if self.virtual_account_id:
            target_metadata.setdefault("virtual_account_id", self.virtual_account_id)
        normalized_target = replace(
            target,
            strategy_id=target.strategy_id or self.strategy_id,
            strategy_name=target.strategy_name or self.strategy_name,
            metadata=target_metadata,
        )
        return replace(
            rebalance_intent,
            target_portfolio=normalized_target,
            order_intents=tuple(self._with_order_identity(intent) for intent in rebalance_intent.order_intents or ()),
        )

    def _coerce_rebalance_intent(self, raw_intent) -> Optional[RebalanceIntent]:
        if isinstance(raw_intent, RebalanceIntent):
            return self._with_rebalance_identity(raw_intent)
        if not isinstance(raw_intent, dict):
            return None
        target_payload = raw_intent.get("target_portfolio")
        if isinstance(target_payload, TargetPortfolio):
            target_portfolio = target_payload
        elif isinstance(target_payload, dict):
            target_portfolio = TargetPortfolio(**target_payload)
        else:
            return None
        order_intents = []
        for item in list(raw_intent.get("order_intents", []) or []):
            if isinstance(item, OrderIntent):
                order_intents.append(item)
            elif isinstance(item, dict):
                order_intents.append(OrderIntent(**item))
        return self._with_rebalance_identity(
            RebalanceIntent(
                target_portfolio=target_portfolio,
                order_intents=tuple(order_intents),
                current_positions=dict(raw_intent.get("current_positions", {}) or {}),
                prices=dict(raw_intent.get("prices", {}) or {}),
                total_asset=float(raw_intent.get("total_asset", 0.0) or 0.0),
                available_cash=float(raw_intent.get("available_cash", 0.0) or 0.0),
                intent_id=str(raw_intent.get("intent_id", "") or ""),
                reason=str(raw_intent.get("reason", "") or ""),
                timestamp=raw_intent.get("timestamp"),
                metadata=dict(raw_intent.get("metadata", {}) or {}),
                schema_version=str(raw_intent.get("schema_version", "rebalance_intent.v1") or "rebalance_intent.v1"),
            )
        )

    def _call_dict(self, method_name: str) -> dict:
        method = getattr(self.panel, method_name, None)
        if not callable(method):
            return {}
        try:
            return dict(method() or {})
        except Exception as exc:
            logger.warning("调用策略方法失败 strategy_id=%s method=%s err=%s", self.strategy_id, method_name, exc)
            return {}

    def _call_text(self, method_name: str, *, default: str = "") -> str:
        method = getattr(self.panel, method_name, None)
        if not callable(method):
            return default
        try:
            return str(method() or "")
        except Exception as exc:
            logger.warning("调用策略方法失败 strategy_id=%s method=%s err=%s", self.strategy_id, method_name, exc)
            return f"{self.strategy_name} 操作失败: {exc}"