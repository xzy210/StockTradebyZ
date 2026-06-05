from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional, Sequence

from PyQt6.QtCore import QObject

from common.execution_contract import OrderExecutionReport, OrderIntent, RebalanceIntent
from trading_app.services.strategy.strategy_spec_service import get_strategy_spec_service

logger = logging.getLogger(__name__)


class LiveStrategyHubController(QObject):
    """Non-visual orchestration for the live strategy hub."""

    CENTER_STRATEGY_ID = "center"
    CENTER_STRATEGY_NAME = "实盘策略中枢"

    def __init__(
        self,
        *,
        task_service,
        hub_state_service,
        eod_service,
        strategy_adapters: Optional[Sequence[object]] = None,
        startup_orchestrator=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.task_service = task_service
        self.hub_state_service = hub_state_service
        self.eod_service = eod_service
        self.strategy_adapters = list(strategy_adapters or [])
        self.startup_orchestrator = startup_orchestrator
        self.strategy_spec_service = get_strategy_spec_service()
        self.ai_strategy_spec = self.strategy_spec_service.ai_stock()
        self.unmanaged_strategy_spec = self.strategy_spec_service.unmanaged()
        self.ai_strategy_adapter = self._find_adapter(self.ai_strategy_spec.strategy_id)
        self.etf_strategy_adapter = self._find_first_non_system_adapter()
        self._startup_message_provider: Callable[[], str] = lambda: ""

    def register_center_tasks(
        self,
        *,
        startup_action: Callable[[], None],
        morning_freshness_action: Callable[[], None],
        end_of_day_action: Callable[[], None],
        ai_task_action: Optional[Callable[[], None]] = None,
        unmanaged_scan_action: Optional[Callable[[], None]] = None,
        etf_scan_action: Optional[Callable[[], None]] = None,
        etf_execute_action: Optional[Callable[[], None]] = None,
        startup_message_provider: Optional[Callable[[], str]] = None,
        strategy_task_specs: Optional[Iterable[object]] = None,
    ) -> None:
        if startup_message_provider is not None:
            self._startup_message_provider = startup_message_provider

        self.task_service.register_task(
            task_key="startup_check",
            task_type="system",
            title="启动自检",
            provider=self._task_provider_startup,
            strategy_id=self.CENTER_STRATEGY_ID,
            strategy_name=self.CENTER_STRATEGY_NAME,
            actions={"立即执行": lambda: self._run_action(startup_action, "已触发启动自检")},
        )
        self.task_service.register_task(
            task_key="morning_freshness",
            task_type="system",
            title="数据新鲜度检查",
            provider=self._task_provider_morning_freshness,
            strategy_id=self.CENTER_STRATEGY_ID,
            strategy_name=self.CENTER_STRATEGY_NAME,
            actions={"立即执行": lambda: self._run_action(morning_freshness_action, "已触发盘中新鲜度检查")},
        )
        self.task_service.register_task(
            task_key="end_of_day_cycle",
            task_type="eod",
            title="统一日终流程",
            provider=self._task_provider_end_of_day,
            strategy_id=self.CENTER_STRATEGY_ID,
            strategy_name=self.CENTER_STRATEGY_NAME,
            actions={"立即执行": lambda: self._run_action(end_of_day_action, "已触发统一日终流程")},
        )
        self._register_strategy_task_specs(
            strategy_task_specs,
            ai_task_action=ai_task_action,
            unmanaged_scan_action=unmanaged_scan_action,
            etf_scan_action=etf_scan_action,
            etf_execute_action=etf_execute_action,
        )

    def _register_strategy_task_specs(
        self,
        task_specs: Optional[Iterable[object]],
        *,
        ai_task_action: Optional[Callable[[], None]] = None,
        unmanaged_scan_action: Optional[Callable[[], None]] = None,
        etf_scan_action: Optional[Callable[[], None]] = None,
        etf_execute_action: Optional[Callable[[], None]] = None,
    ) -> None:
        specs = list(task_specs or [])
        if not specs:
            specs = self._build_legacy_strategy_task_specs(
                ai_task_action=ai_task_action,
                unmanaged_scan_action=unmanaged_scan_action,
                etf_scan_action=etf_scan_action,
                etf_execute_action=etf_execute_action,
            )
        for spec in specs:
            self.task_service.register_task(
                task_key=str(getattr(spec, "task_key", "") or ""),
                task_type=str(getattr(spec, "task_type", "") or ""),
                title=str(getattr(spec, "title", "") or ""),
                provider=getattr(spec, "provider"),
                strategy_id=str(getattr(spec, "strategy_id", "") or ""),
                strategy_name=str(getattr(spec, "strategy_name", "") or ""),
                actions=dict(getattr(spec, "actions", {}) or {}),
            )

    def _build_legacy_strategy_task_specs(
        self,
        *,
        ai_task_action: Optional[Callable[[], None]] = None,
        unmanaged_scan_action: Optional[Callable[[], None]] = None,
        etf_scan_action: Optional[Callable[[], None]] = None,
        etf_execute_action: Optional[Callable[[], None]] = None,
    ) -> list[object]:
        from .strategy_plugin import LiveStrategyTaskSpec

        ai_actions = {
            "暂停调度": self._pause_ai_automation,
            "恢复调度": self._resume_ai_automation,
        }
        if ai_task_action is not None:
            ai_actions["立即执行"] = lambda: self._run_action(ai_task_action, "已触发 AI 定时任务")

        unmanaged_actions = {}
        if unmanaged_scan_action is not None:
            unmanaged_actions["立即执行"] = lambda: self._run_action(unmanaged_scan_action, "已触发未管理持仓 AI 巡检")

        etf_actions = {
            "暂停调度": self._pause_etf_automation,
            "恢复调度": self._resume_etf_automation,
        }
        if etf_scan_action is not None:
            etf_actions["仅检查信号"] = lambda: self._run_action(etf_scan_action, "已触发 ETF 信号检查")
        if etf_execute_action is not None:
            etf_actions["检查并执行"] = lambda: self._run_action(etf_execute_action, "已触发 ETF 信号检查并执行")

        return [
            LiveStrategyTaskSpec(
                task_key="daily_ai_strategy_cycle",
                task_type="ai",
                    title="每日 AI 实盘决策任务",
                provider=self._task_provider_ai_scheduler,
                strategy_id=self._adapter_strategy_id(self.ai_strategy_adapter, self.ai_strategy_spec.strategy_id),
                strategy_name=self._adapter_strategy_name(self.ai_strategy_adapter, self.ai_strategy_spec.strategy_name),
                actions=ai_actions,
                order=10,
            ),
            LiveStrategyTaskSpec(
                task_key="daily_unmanaged_position_scan",
                task_type="review",
                title="未管理持仓 AI 巡检",
                provider=self._task_provider_unmanaged_ai_scheduler,
                strategy_id=self.unmanaged_strategy_spec.strategy_id,
                strategy_name=self.unmanaged_strategy_spec.strategy_name,
                actions=unmanaged_actions,
                order=20,
            ),
            LiveStrategyTaskSpec(
                task_key="etf_rotation_auto_check",
                task_type="etf",
                title="ETF 自动轮动检查",
                provider=self._task_provider_etf_rotation,
                strategy_id=self._adapter_strategy_id(self.etf_strategy_adapter, ""),
                strategy_name=self._adapter_strategy_name(self.etf_strategy_adapter, self.strategy_spec_service.etf_rotation().strategy_name),
                actions=etf_actions,
                order=30,
            ),
        ]

    def refresh_public_views(self, refreshers: Iterable[Callable[[], None]] = ()) -> str:
        self.refresh_state()
        for refresher in list(refreshers or []):
            try:
                refresher()
            except Exception as exc:
                logger.debug("刷新中心公共视图失败: %s", exc)
        return "中心公共视图已刷新"

    def toggle_center_automation(self, resume: bool) -> str:
        message = self.resume_center_automation() if resume else self.pause_center_automation()
        self.refresh_state()
        return message

    def pause_center_automation(self) -> str:
        messages = [self._call_adapter_text(adapter, "pause_automation") for adapter in self.strategy_adapters]
        return "；".join([item for item in messages if item])

    def resume_center_automation(self) -> str:
        messages = [self._call_adapter_text(adapter, "resume_automation") for adapter in self.strategy_adapters]
        return "；".join([item for item in messages if item])

    def refresh_state(self) -> None:
        try:
            self.hub_state_service.refresh_state()
        except Exception as exc:
            logger.debug("刷新中心状态失败: %s", exc)

    def refresh_strategies_after_eod(self) -> None:
        for adapter in list(self.strategy_adapters or []):
            try:
                adapter.refresh_after_eod()
            except Exception as exc:
                logger.debug("刷新策略日终 UI 失败 strategy_id=%s err=%s", getattr(adapter, "strategy_id", ""), exc)

    def collect_rotation_pool(self) -> list[str]:
        pool: list[str] = []
        for adapter in list(self.strategy_adapters or []):
            try:
                symbols = adapter.get_rotation_pool()
            except Exception as exc:
                logger.debug("读取策略轮动池失败 strategy_id=%s err=%s", getattr(adapter, "strategy_id", ""), exc)
                symbols = []
            for symbol in list(symbols or []):
                if symbol and symbol not in pool:
                    pool.append(symbol)
        return pool

    def sync_rotation_pool(self) -> None:
        if self.eod_service is None:
            return
        self.eod_service.set_rotation_etf_pool(self.collect_rotation_pool())

    def execute_adapter_signals(
        self,
        adapter,
        *,
        payload: Optional[dict] = None,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Generate and execute native rebalance/order intents for one adapter, with StrategySignal fallback."""
        if adapter is None:
            return []

        rebalance_intent = self._generate_adapter_rebalance_intent(adapter, payload=payload)
        if rebalance_intent is not None:
            if not rebalance_intent.order_intents:
                return []
            execute_rebalance = getattr(adapter, "execute_live_rebalance_intent", None)
            if callable(execute_rebalance):
                return list(
                    execute_rebalance(
                        rebalance_intent,
                        execution_service=execution_service,
                        stock_name_map=stock_name_map or {},
                    )
                )
            service = self._resolve_execution_service(execution_service)
            return list(service.execute_rebalance_intent(rebalance_intent, stock_name_map=stock_name_map or {}))

        order_intents = self._generate_adapter_order_intents(adapter, payload=payload)
        if order_intents:
            execute_intents = getattr(adapter, "execute_live_order_intents", None)
            if callable(execute_intents):
                return list(
                    execute_intents(
                        order_intents,
                        execution_service=execution_service,
                        stock_name_map=stock_name_map or {},
                    )
                )
            service = self._resolve_execution_service(execution_service)
            return list(service.execute_order_intents(order_intents, stock_name_map=stock_name_map or {}))

        generate = getattr(adapter, "generate_live_signals", None)
        if not callable(generate):
            return []
        signals = list(generate(dict(payload or {})) or [])
        execute = getattr(adapter, "execute_live_signals", None)
        if callable(execute):
            return list(
                execute(
                    signals,
                    execution_service=execution_service,
                    stock_name_map=stock_name_map or {},
                )
            )
        if not signals:
            return []
        service = self._resolve_execution_service(execution_service)
        return list(service.execute_signals(signals, stock_name_map=stock_name_map or {}))

    def execute_adapter_rebalance_intent(
        self,
        adapter,
        rebalance_intent: RebalanceIntent,
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Execute one adapter's native RebalanceIntent through the live gateway."""
        if adapter is None or rebalance_intent is None:
            return []
        execute = getattr(adapter, "execute_live_rebalance_intent", None)
        if callable(execute):
            return list(
                execute(
                    rebalance_intent,
                    execution_service=execution_service,
                    stock_name_map=stock_name_map or {},
                )
            )
        service = self._resolve_execution_service(execution_service)
        return list(service.execute_rebalance_intent(rebalance_intent, stock_name_map=stock_name_map or {}))

    def execute_adapter_order_intents(
        self,
        adapter,
        order_intents: list[OrderIntent],
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Execute one adapter's native OrderIntent outputs through the live gateway."""
        if adapter is None or not order_intents:
            return []
        execute = getattr(adapter, "execute_live_order_intents", None)
        if callable(execute):
            return list(
                execute(
                    order_intents,
                    execution_service=execution_service,
                    stock_name_map=stock_name_map or {},
                )
            )
        service = self._resolve_execution_service(execution_service)
        return list(service.execute_order_intents(order_intents, stock_name_map=stock_name_map or {}))

    def execute_strategy_rebalance_intent(
        self,
        strategy_id: str,
        rebalance_intent: RebalanceIntent,
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Execute a native RebalanceIntent by strategy id."""
        adapter = self._find_adapter(strategy_id)
        return self.execute_adapter_rebalance_intent(
            adapter,
            rebalance_intent,
            execution_service=execution_service,
            stock_name_map=stock_name_map,
        )

    def execute_strategy_order_intents(
        self,
        strategy_id: str,
        order_intents: list[OrderIntent],
        *,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Execute native OrderIntent outputs by strategy id."""
        adapter = self._find_adapter(strategy_id)
        return self.execute_adapter_order_intents(
            adapter,
            order_intents,
            execution_service=execution_service,
            stock_name_map=stock_name_map,
        )

    def execute_strategy_signals(
        self,
        strategy_id: str,
        *,
        payload: Optional[dict] = None,
        execution_service=None,
        stock_name_map: Optional[dict[str, str]] = None,
    ) -> list[OrderExecutionReport]:
        """Generate and execute unified StrategySignal outputs by strategy id."""
        adapter = self._find_adapter(strategy_id)
        return self.execute_adapter_signals(
            adapter,
            payload=payload,
            execution_service=execution_service,
            stock_name_map=stock_name_map,
        )

    def _task_provider_startup(self) -> dict:
        return {
            "status": "running" if bool(getattr(self.startup_orchestrator, "is_running", False)) else "idle",
            "message": self._safe_startup_message(),
            "last_run": "",
            "schedule_time": "启动后自动 / 手动触发",
        }

    @staticmethod
    def _task_provider_morning_freshness() -> dict:
        return {
            "status": "scheduled",
            "message": "交易日 09:35 自动检查",
            "schedule_time": "09:35",
        }

    def _task_provider_end_of_day(self) -> dict:
        cycle_state = self._get_eod_cycle_state()
        return {
            "status": str(cycle_state.get("status", "") or "idle"),
            "message": str(cycle_state.get("last_error", "") or cycle_state.get("updated_at", "") or ""),
            "last_run": str(cycle_state.get("completed_at", "") or cycle_state.get("updated_at", "") or ""),
            "schedule_time": "收盘后 / 手动触发",
        }

    def _task_provider_ai_scheduler(self) -> dict:
        adapter = self.ai_strategy_adapter
        if adapter is None:
            return {}
        try:
            rows = adapter.get_task_summaries()
        except Exception as exc:
            logger.debug("读取 AI 实盘决策任务失败: %s", exc)
            return {}
            rows = []
        return dict(rows[0] if rows else {})

    def _task_provider_unmanaged_ai_scheduler(self) -> dict:
        adapter = self.ai_strategy_adapter
        if adapter is None:
            return {}
        try:
            return dict(adapter.get_task_summary("daily_unmanaged_position_scan") or {})
        except Exception as exc:
            logger.debug("读取未管理持仓巡检任务失败: %s", exc)
            return {}

    def _task_provider_etf_rotation(self) -> dict:
        adapter = self.etf_strategy_adapter
        if adapter is None:
            return {}
        try:
            rows = adapter.get_task_summaries()
        except Exception as exc:
            logger.debug("读取 ETF 轮动实盘任务失败: %s", exc)
            return {}
            rows = []
        return dict(rows[0] if rows else {})

    def _pause_ai_automation(self) -> str:
        return self._call_adapter_text(self.ai_strategy_adapter, "pause_automation")

    def _resume_ai_automation(self) -> str:
        return self._call_adapter_text(self.ai_strategy_adapter, "resume_automation")

    def _pause_etf_automation(self) -> str:
        return self._call_adapter_text(self.etf_strategy_adapter, "pause_automation")

    def _resume_etf_automation(self) -> str:
        return self._call_adapter_text(self.etf_strategy_adapter, "resume_automation")

    @staticmethod
    def _run_action(callback: Callable[[], None], message: str) -> str:
        callback()
        return message

    def _safe_startup_message(self) -> str:
        try:
            return str(self._startup_message_provider() or "")
        except Exception:
            return ""

    def _get_eod_cycle_state(self) -> dict:
        method = getattr(self.eod_service, "_get_cycle_state", None)
        if not callable(method):
            return {}
        try:
            return dict(method() or {})
        except Exception as exc:
            logger.debug("读取日终状态失败: %s", exc)
            return {}

    def _find_adapter(self, strategy_id: str):
        target = str(strategy_id or "").strip()
        for adapter in self.strategy_adapters:
            if str(getattr(adapter, "strategy_id", "") or "").strip() == target:
                return adapter
        return None

    def _find_first_non_system_adapter(self):
        excluded = {self.ai_strategy_spec.strategy_id, self.unmanaged_strategy_spec.strategy_id}
        for adapter in self.strategy_adapters:
            strategy_id = str(getattr(adapter, "strategy_id", "") or "").strip()
            if strategy_id and strategy_id not in excluded:
                return adapter
        return None

    @staticmethod
    def _adapter_strategy_id(adapter, default: str = "") -> str:
        return str(getattr(adapter, "strategy_id", "") or default or "").strip()

    @staticmethod
    def _adapter_strategy_name(adapter, default: str = "") -> str:
        return str(getattr(adapter, "strategy_name", "") or default or "").strip()

    @staticmethod
    def _call_adapter_text(adapter, method_name: str) -> str:
        if adapter is None:
            return ""
        method = getattr(adapter, method_name, None)
        if not callable(method):
            return ""
        try:
            return str(method() or "")
        except Exception as exc:
            logger.warning("执行策略控制失败 strategy_id=%s method=%s err=%s", getattr(adapter, "strategy_id", ""), method_name, exc)
            return f"{getattr(adapter, 'strategy_name', '策略')} 操作失败: {exc}"

    @staticmethod
    def _resolve_execution_service(execution_service):
        if execution_service is not None:
            return execution_service
        from trading_app.services.execution.trade_execution_service import get_trade_execution_service
        return get_trade_execution_service()

    @staticmethod
    def _generate_adapter_rebalance_intent(adapter, *, payload: Optional[dict] = None) -> Optional[RebalanceIntent]:
        generate = getattr(adapter, "generate_live_rebalance_intent", None)
        if not callable(generate):
            return None
        try:
            result = generate(dict(payload or {}))
        except TypeError:
            result = generate()
        return result if isinstance(result, RebalanceIntent) else None

    @staticmethod
    def _generate_adapter_order_intents(adapter, *, payload: Optional[dict] = None) -> list[OrderIntent]:
        generate = getattr(adapter, "generate_live_order_intents", None)
        if not callable(generate):
            return []
        try:
            result = list(generate(dict(payload or {})) or [])
        except TypeError:
            result = list(generate() or [])
        return [item for item in result if isinstance(item, OrderIntent)]
