from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Any, Dict, Optional
from uuid import uuid4

from common.broker_interface import BrokerOrderRequest, BrokerProtocol, LiveBrokerAdapter
from common.broker_session_service import BrokerSessionService, get_broker_session_service
from common.execution_contract import FillReport, OrderExecutionReport, OrderIntent, RebalanceIntent, StrategySignal
from live_rotation.holiday_calendar import get_non_trading_reason, is_trading_day

from common.agent.agent_context_service import BrokerContext
from trading_app.services.ai.ai_stock_risk_policy import AIStockRiskPolicy
from trading_app.services.ai.auto_trade_config_service import get_auto_trade_config_service
from trading_app.services.execution.risk_guard_service import RiskGuardService
from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service
from trading_app.services.strategy.strategy_risk import (
    StrategyRiskContext,
    get_strategy_risk_registry,
)
from trading_app.services.strategy.strategy_constants import (
    AI_STOCK_STRATEGY_ID,
    AI_STOCK_STRATEGY_NAME,
    AI_STOCK_VIRTUAL_ACCOUNT_ID,
    OWNER_TYPE_AI,
    OWNER_TYPE_OTHER,
)
from trading_app.services.strategy.strategy_registry_service import get_strategy_registry_service
from trading_app.services.ai.trade_decision_models import RiskCheckResult, TradeAction, TradeDecision
from trading_app.services.execution.trade_record_service import TradeDirection, TradeSource, get_trade_record_service
from trading_app.services.market_data.market_data_policy import is_etf_like_code
from trading_app.services.market_data.market_data_status_service import get_market_data_status_service
from trading_app.services.execution.order_execution_event_service import OrderExecutionEvent, get_order_execution_event_service
from trading_app.services.execution.order_state_machine import OrderLifecycle, OrderStateSnapshot, normalize_order_state

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRequest:
    stock_code: str
    stock_name: str
    order_type: int
    order_volume: int
    price_type: int
    price: float
    source: str
    trigger: str
    strategy_name: str = ""
    strategy_id: str = ""
    virtual_account_id: str = ""
    intent_id: str = ""
    remark: str = ""
    decision: Optional[TradeDecision] = None
    risk_result: Optional[RiskCheckResult] = None
    decision_record_id: str = ""
    require_approval: bool = False
    approved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    message: str
    broker_order_id: int = -1
    request_id: str = ""
    execution_mode: str = "live"
    order_status: str = ""
    live_submitted: bool = False
    filled_confirmed: bool = False
    submitted_only: bool = False
    shadow: bool = False
    blocked: bool = False
    trade_record_id: int = 0
    order_record_id: int = 0


class TradeExecutionService:
    """Unified execution gateway for all real orders."""

    def __init__(self, broker_service: Optional[BrokerSessionService] = None, broker: Optional[BrokerProtocol] = None):
        if broker is None:
            self.broker_service = broker_service or get_broker_session_service()
            if broker_service is None:
                self.broker = self._resolve_default_live_broker(self.broker_service)
            else:
                self.broker = LiveBrokerAdapter(self.broker_service)
        else:
            self.broker_service = broker_service
            self.broker = broker
        self.trade_service = get_trade_record_service()
        self.risk_guard = RiskGuardService()
        self.config_service = get_auto_trade_config_service()
        self.strategy_registry = get_strategy_registry_service()
        self.strategy_budget = get_strategy_budget_service()
        self.strategy_risk_registry = get_strategy_risk_registry()
        # AI 风控通过 policy 形式挂到 registry，确保 AI 订单默认就有风控兜底
        self._ai_stock_policy = AIStockRiskPolicy(risk_guard=self.risk_guard)
        self.strategy_risk_registry.register(self._ai_stock_policy, override=True)
        self._recent_fingerprints: dict[str, float] = {}
        self._event_storage: Any = get_order_execution_event_service()
        self.pending_order_lifecycles: dict[str, OrderLifecycle] = self._rebuild_pending_order_lifecycles()

    def _resolve_default_live_broker(self, broker_service: BrokerSessionService) -> BrokerProtocol:
        try:
            from common.trading_runtime import get_live_trading_engine

            engine = get_live_trading_engine(broker_service=broker_service)
            gateway = engine.get_gateway()
            if gateway is not None:
                return gateway
        except Exception:
            logger.debug("初始化 LiveTradingEngine 默认网关失败，回退 LiveBrokerAdapter", exc_info=True)
        return LiveBrokerAdapter(broker_service)

    def _rebuild_pending_order_lifecycles(self) -> dict[str, OrderLifecycle]:
        try:
            if hasattr(self._event_storage, "query_open_orders"):
                return dict(self._event_storage.query_open_orders())
        except Exception:
            logger.debug("启动时重建未终结订单状态失败", exc_info=True)
        return {}

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        cfg = self.config_service.get_config()
        request_id = uuid4().hex[:16]
        request = self._normalize_request(request)
        mode = self._resolve_execution_mode(request, cfg.auto_trade_mode, cfg.manual_orders_enabled)
        fingerprint = self._build_fingerprint(request)
        validation_error = self._validate_request(request, cfg, fingerprint)
        order_record = self.trade_service.add_order_record(
            request_id=request_id,
            stock_code=request.stock_code,
            stock_name=request.stock_name,
            direction=self._direction_from_order_type(request.order_type),
            order_volume=request.order_volume,
            price=request.price,
            price_type=request.price_type,
            source=request.source,
            trigger=request.trigger,
            strategy_name=request.strategy_name,
            strategy_id=request.strategy_id,
            virtual_account_id=request.virtual_account_id,
            intent_id=request.intent_id or request.decision_record_id or request_id,
            execution_mode=mode,
            status="validating",
            broker_order_id=-1,
            fingerprint=fingerprint,
            decision_record_id=request.decision_record_id,
            remark=request.remark,
            validation_message=validation_error or "开始执行校验",
        )
        order_record_id = getattr(order_record, "id", 0)
        self._record_order_event(
            event_type="OrderRequested",
            request_id=request_id,
            request=request,
            mode=mode,
            order_record_id=order_record_id,
            title="订单请求已进入统一执行网关",
            message=validation_error or "订单请求已创建，开始执行校验",
            level="warning" if validation_error else "info",
            status="open" if validation_error else "resolved",
            payload={"fingerprint": fingerprint},
        )

        if validation_error:
            self.trade_service.update_order_record(
                request_id,
                status="blocked",
                validation_message=validation_error,
            )
            self._record_order_event(
                event_type="OrderBlocked",
                request_id=request_id,
                request=request,
                mode=mode,
                order_record_id=order_record_id,
                title="订单被统一执行网关拦截",
                message=validation_error,
                level="warning",
                status="open",
                payload={"fingerprint": fingerprint},
            )
            return ExecutionResult(
                success=False,
                blocked=True,
                message=validation_error,
                request_id=request_id,
                execution_mode=mode,
                order_record_id=getattr(order_record, "id", 0),
            )

        intent_id = request.intent_id or request.decision_record_id or request_id
        reserved_budget = False
        if request.strategy_id and request.order_type == 23:
            reserve_error = self._reserve_strategy_budget(request, intent_id)
            if reserve_error:
                self.trade_service.update_order_record(
                    request_id,
                    status="blocked",
                    validation_message=reserve_error,
                )
                self._record_order_event(
                    event_type="OrderBlocked",
                    request_id=request_id,
                    request=request,
                    mode=mode,
                    order_record_id=order_record_id,
                    title="订单因策略预算不足被拦截",
                    message=reserve_error,
                    level="warning",
                    status="open",
                    payload={"fingerprint": fingerprint, "intent_id": intent_id},
                )
                return ExecutionResult(
                    success=False,
                    blocked=True,
                    message=reserve_error,
                    request_id=request_id,
                    execution_mode=mode,
                    order_record_id=getattr(order_record, "id", 0),
                )
            reserved_budget = True
            self._record_order_event(
                event_type="BudgetReserved",
                request_id=request_id,
                request=request,
                mode=mode,
                order_record_id=order_record_id,
                title="策略预算已预占",
                message=f"已为 {request.strategy_name or request.strategy_id} 预占买入预算",
                status="resolved",
                payload={"intent_id": intent_id},
            )

        if mode in {"shadow", "paper"}:
            message = f"影子模式记录成功: {request.stock_name or request.stock_code} {request.order_volume}股"
            self.trade_service.update_order_record(
                request_id,
                status="shadow",
                validation_message=message,
            )
            self._apply_strategy_execution(
                request,
                intent_id=intent_id,
                executed_price=request.price,
                executed_volume=request.order_volume,
                fallback_reserved=reserved_budget,
            )
            self._remember_fingerprint(fingerprint)
            self._record_order_event(
                event_type="OrderShadowRecorded",
                request_id=request_id,
                request=request,
                mode=mode,
                order_record_id=order_record_id,
                title="影子订单已记录",
                message=message,
                status="resolved",
                payload={"intent_id": intent_id, "reserved_budget": reserved_budget},
            )
            return ExecutionResult(
                success=True,
                shadow=True,
                message=message,
                request_id=request_id,
                execution_mode=mode,
                order_status="shadow",
                filled_confirmed=True,
                order_record_id=getattr(order_record, "id", 0),
            )

        try:
            submit_result = self.broker.submit(
                BrokerOrderRequest(
                    stock_code=request.stock_code,
                    order_type=request.order_type,
                    order_volume=request.order_volume,
                    price_type=request.price_type,
                    price=request.price,
                    strategy_name=request.strategy_name,
                    remark=request.remark,
                    request_id=request_id,
                    metadata={
                        "source": request.source,
                        "trigger": request.trigger,
                        "strategy_id": request.strategy_id,
                        "virtual_account_id": request.virtual_account_id,
                        "intent_id": intent_id,
                    },
                )
            )
            broker_order_id = submit_result.broker_order_id
        except Exception as exc:
            message = f"下单异常: {exc}"
            if reserved_budget:
                self.strategy_budget.release_reservation(strategy_id=request.strategy_id, intent_id=intent_id)
            self.trade_service.update_order_record(
                request_id,
                status="failed",
                validation_message=message,
            )
            self._record_order_event(
                event_type="OrderSubmitFailed",
                request_id=request_id,
                request=request,
                mode=mode,
                order_record_id=order_record_id,
                title="券商下单异常",
                message=message,
                level="error",
                status="open",
                payload={"intent_id": intent_id, "reserved_budget_released": reserved_budget},
            )
            return ExecutionResult(
                success=False,
                message=message,
                request_id=request_id,
                execution_mode=mode,
                order_record_id=getattr(order_record, "id", 0),
            )

        broker_order_id = int(broker_order_id) if isinstance(broker_order_id, (int, float)) else -1
        if broker_order_id <= 0:
            message = "券商未返回有效委托号"
            if reserved_budget:
                self.strategy_budget.release_reservation(strategy_id=request.strategy_id, intent_id=intent_id)
            self.trade_service.update_order_record(
                request_id,
                status="failed",
                validation_message=message,
            )
            self._record_order_event(
                event_type="OrderSubmitFailed",
                request_id=request_id,
                request=request,
                mode=mode,
                order_record_id=order_record_id,
                title="券商未返回有效委托号",
                message=message,
                level="error",
                status="open",
                payload={"intent_id": intent_id, "reserved_budget_released": reserved_budget},
            )
            return ExecutionResult(
                success=False,
                message=message,
                request_id=request_id,
                execution_mode=mode,
                order_record_id=getattr(order_record, "id", 0),
            )

        self.trade_service.update_order_record(
            request_id,
            broker_order_id=broker_order_id,
            status="submitted",
            validation_message="委托已提交",
        )
        self._remember_fingerprint(fingerprint)
        self._record_order_event(
            event_type="OrderSubmitted",
            request_id=request_id,
            request=request,
            mode=mode,
            broker_order_id=broker_order_id,
            order_record_id=order_record_id,
            title="订单已提交券商",
            message=f"委托已提交，单号 {broker_order_id}",
            status="resolved",
            payload={"intent_id": intent_id, "reserved_budget": reserved_budget},
        )

        return self._poll_order_status(request_id, broker_order_id, request, mode, getattr(order_record, "id", 0))

    def execute_order_intent(self, intent: OrderIntent, *, stock_name: str = "") -> OrderExecutionReport:
        """Execute the shared order intent contract through the live gateway."""
        request = ExecutionRequest(**intent.to_execution_request_kwargs(stock_name=stock_name))
        result = self.execute(request)
        fills = []
        if result.trade_record_id:
            record = self.trade_service.get_record_by_id(result.trade_record_id)
            if record is not None:
                fills.append(FillReport.from_live_trade_record(record))
        elif result.broker_order_id and result.filled_confirmed:
            record = self.trade_service.get_latest_record_by_broker_order_id(result.broker_order_id)
            if record is not None:
                fills.append(FillReport.from_live_trade_record(record))
        return OrderExecutionReport.from_live_execution_result(result, intent=intent, fills=fills)

    def execute_order_intents(self, intents: Optional[list[OrderIntent]], *, stock_name_map: Optional[Dict[str, str]] = None) -> list[OrderExecutionReport]:
        """Execute ordered OrderIntent contracts through the live unified gateway."""
        reports: list[OrderExecutionReport] = []
        names = dict(stock_name_map or {})
        for intent in intents or []:
            if intent is None or int(intent.quantity or 0) <= 0:
                continue
            report = self.execute_order_intent(intent, stock_name=names.get(self._plain_code(intent.symbol), ""))
            reports.append(report)
        return reports

    def execute_rebalance_intent(self, rebalance_intent: Optional[RebalanceIntent], *, stock_name_map: Optional[Dict[str, str]] = None) -> list[OrderExecutionReport]:
        """Execute a portfolio-level RebalanceIntent without converting it back to StrategySignal."""
        if rebalance_intent is None:
            return []
        return self.execute_order_intents(list(rebalance_intent.order_intents or ()), stock_name_map=stock_name_map)

    def execute_signals(self, signals: Optional[list[StrategySignal]], *, stock_name_map: Optional[Dict[str, str]] = None) -> list[OrderExecutionReport]:
        """Execute generated strategy signals through the live unified gateway."""
        reports: list[OrderExecutionReport] = []
        names = dict(stock_name_map or {})
        for signal in signals or []:
            report = self.execute_signal(signal, stock_name=names.get(self._plain_code(signal.symbol), ""))
            if report is not None:
                reports.append(report)
        return reports

    def execute_signal(self, signal: StrategySignal, *, stock_name: str = "") -> Optional[OrderExecutionReport]:
        """Convert one StrategySignal to OrderIntent and submit it to the live gateway."""
        if signal is None or signal.action == "hold":
            return None
        intent = self._signal_to_order_intent(signal)
        if intent is None or intent.quantity <= 0:
            return None
        return self.execute_order_intent(intent, stock_name=stock_name)

    def _signal_to_order_intent(self, signal: StrategySignal) -> Optional[OrderIntent]:
        price = float(signal.price or 0.0)
        if price <= 0:
            return None
        quantity = self._resolve_signal_order_quantity(signal, price)
        if quantity == 0:
            return None
        side = "buy" if quantity > 0 else "sell"
        return OrderIntent(
            symbol=signal.symbol,
            side=side,
            quantity=abs(quantity),
            price=price,
            intent_type=self._signal_intent_type(signal),
            strategy_id=signal.strategy_id,
            strategy_name=signal.strategy_name,
            virtual_account_id=str(signal.metadata.get("virtual_account_id", "") or ""),
            signal_id=signal.signal_id,
            reason=signal.reason,
            source=str(signal.metadata.get("source", "strategy") or "strategy"),
            trigger=str(signal.metadata.get("trigger", "auto") or "auto"),
            price_type=int(signal.metadata.get("price_type", 5) or 5),
            metadata=dict(signal.metadata),
        )

    def _resolve_signal_order_quantity(self, signal: StrategySignal, price: float) -> int:
        plain_code = self._plain_code(signal.symbol)
        current_qty, sellable_qty = self._query_position_quantity(plain_code)
        lot_size = 100

        if signal.target_percent is not None:
            target_percent = max(float(signal.target_percent or 0.0), 0.0)
            total_asset = self._get_total_asset()
            target_quantity = int(math.floor((total_asset * target_percent) / price / lot_size)) * lot_size
            return target_quantity - current_qty

        if signal.target_quantity is not None:
            requested = int(signal.target_quantity or 0)
            if signal.metadata.get("quantity_mode") == "delta":
                signed = requested if signal.action == "buy" else -requested
                return self._normalize_lot_quantity(signed, lot_size=lot_size)
            target_quantity = max(requested, 0) // lot_size * lot_size
            return target_quantity - current_qty

        metadata_quantity = signal.metadata.get("quantity")
        if metadata_quantity is not None:
            signed = int(metadata_quantity or 0)
            if signed > 0 and signal.action == "sell":
                signed = -signed
            return self._normalize_lot_quantity(signed, lot_size=lot_size)

        if signal.action == "sell" and sellable_qty > 0:
            return -sellable_qty
        return 0

    @staticmethod
    def _signal_intent_type(signal: StrategySignal) -> str:
        if signal.target_percent is not None:
            return "target_percent"
        if signal.target_quantity is not None:
            return "target_quantity"
        return "quantity"

    def _query_position_quantity(self, plain_code: str) -> tuple[int, int]:
        try:
            pos = self._query_broker_position(plain_code)
            if pos is None:
                return 0, 0
            quantity = int(getattr(pos, "volume", 0) or getattr(pos, "current_amount", 0) or getattr(pos, "position_volume", 0) or 0)
            sellable = int(getattr(pos, "can_use_volume", 0) or getattr(pos, "enable_amount", 0) or quantity)
            return quantity, sellable
        except Exception:
            logger.debug("查询持仓数量失败 code=%s", plain_code, exc_info=True)
        return 0, 0

    @staticmethod
    def _normalize_lot_quantity(quantity: int, *, lot_size: int = 100) -> int:
        if quantity == 0:
            return 0
        sign = 1 if quantity > 0 else -1
        lots = abs(int(quantity)) // max(int(lot_size or 1), 1)
        return sign * lots * max(int(lot_size or 1), 1)

    def execute_agent_decision(
        self,
        decision: TradeDecision,
        *,
        stock_name: str = "",
        decision_record_id: str = "",
        risk_result: Optional[RiskCheckResult] = None,
    ) -> ExecutionResult:
        action = decision.action
        if action in (TradeAction.BUY.value, TradeAction.ADD.value):
            order_type = 23
            volume = self._calc_buy_volume(decision)
        elif action in (TradeAction.SELL.value, TradeAction.REDUCE.value):
            order_type = 24
            volume = self._calc_sell_volume(decision)
        else:
            return ExecutionResult(False, f"不可执行的操作类型: {action}")

        if volume <= 0:
            return ExecutionResult(False, "计算委托数量为0")

        return self.execute(
            ExecutionRequest(
                stock_code=decision.symbol_code,
                stock_name=stock_name or decision.symbol_name,
                order_type=order_type,
                order_volume=volume,
                price_type=5,
                price=float(decision.current_price or 0),
                source=TradeSource.AI_AGENT.value,
                trigger="manual",
                strategy_name=AI_STOCK_STRATEGY_NAME,
                strategy_id=AI_STOCK_STRATEGY_ID,
                virtual_account_id=AI_STOCK_VIRTUAL_ACCOUNT_ID,
                intent_id=decision_record_id or uuid4().hex[:16],
                remark=f"AI实盘决策: {decision.reasoning[:50]}" if decision.reasoning else "AI实盘决策",
                decision=decision,
                risk_result=risk_result,
                decision_record_id=decision_record_id,
                require_approval=True,
                approved=True,
            )
        )

    def estimate_volume_for_decision(self, decision: TradeDecision) -> int:
        action = decision.action
        if action in (TradeAction.BUY.value, TradeAction.ADD.value):
            return self._calc_buy_volume(decision)
        if action in (TradeAction.SELL.value, TradeAction.REDUCE.value):
            return self._calc_sell_volume(decision)
        return 0

    def execute_conditional_order(
        self,
        *,
        stock_code: str,
        stock_name: str,
        order_type: int,
        order_volume: int,
        price_type: int,
        price: float,
        strategy_name: str = "ConditionalOrder",
        remark: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        return self.execute(
            ExecutionRequest(
                stock_code=stock_code,
                stock_name=stock_name,
                order_type=order_type,
                order_volume=order_volume,
                price_type=price_type,
                price=price,
                source=TradeSource.CONDITIONAL.value,
                trigger="auto",
                strategy_name=strategy_name,
                remark=remark or "条件单自动执行",
                metadata=dict(metadata or {}),
            )
        )

    def _normalize_request(self, request: ExecutionRequest) -> ExecutionRequest:
        code = (request.stock_code or "").strip().upper()
        if "." not in code and len(code) == 6:
            if code.startswith(("5", "6", "9")):
                code = f"{code}.SH"
            else:
                code = f"{code}.SZ"
        request.stock_code = code
        request.stock_name = self.trade_service.normalize_stock_name(code, request.stock_name or code).strip()
        request.price = float(request.price or 0)
        request.order_volume = int(request.order_volume or 0)
        request.strategy_name = (request.strategy_name or "").strip()
        request.strategy_id = (request.strategy_id or request.metadata.get("strategy_id", "")).strip()
        request.virtual_account_id = (
            request.virtual_account_id or request.metadata.get("virtual_account_id", "")
        ).strip()
        request.intent_id = (request.intent_id or request.metadata.get("intent_id", "")).strip()
        if request.strategy_id == AI_STOCK_STRATEGY_ID and not request.strategy_name:
            request.strategy_name = AI_STOCK_STRATEGY_NAME
        if request.strategy_id == AI_STOCK_STRATEGY_ID and not request.virtual_account_id:
            request.virtual_account_id = AI_STOCK_VIRTUAL_ACCOUNT_ID
        request.remark = (request.remark or "").strip()
        return request

    def _resolve_execution_mode(self, request: ExecutionRequest, auto_mode: str, manual_enabled: bool) -> str:
        if request.trigger == "manual":
            return "live" if manual_enabled else "off"
        return auto_mode

    def _validate_request(self, request: ExecutionRequest, cfg, fingerprint: str) -> str:
        if request.trigger == "manual" and not cfg.manual_orders_enabled:
            return "当前配置禁止手动下单"
        if request.trigger != "manual" and cfg.auto_trade_mode == "off":
            return "自动交易未开启"
        if not self._is_broker_connected():
            return "券商未连接"
        if request.require_approval and not request.approved:
            return "该决策尚未批准，不能执行"
        if request.order_volume <= 0 or request.order_volume % 100 != 0:
            return "委托数量必须为100股的整数倍"
        if request.order_type not in (23, 24):
            return f"不支持的委托方向: {request.order_type}"
        if request.price_type in (0, 5) and request.price <= 0:
            return "委托价格必须大于0"
        market_data_error = self._validate_market_data_status(request)
        if market_data_error:
            return market_data_error
        if cfg.require_trading_time:
            time_error = self._validate_trading_time()
            if time_error:
                return time_error
        duplicate_error = self._check_duplicate(fingerprint, cfg.duplicate_window_seconds)
        if duplicate_error:
            return duplicate_error
        if request.risk_result is not None and not request.risk_result.passed:
            return "风控未通过，禁止执行"
        strategy_error = self._validate_strategy_constraints(request)
        if strategy_error:
            return strategy_error
        broker_error = self._validate_broker_constraints(request)
        if broker_error:
            return broker_error
        policy_error = self._validate_strategy_risk_policy(request)
        if policy_error:
            return policy_error
        return ""

    def _validate_market_data_status(self, request: ExecutionRequest) -> str:
        plain_code = self._plain_code(request.stock_code)
        if not plain_code:
            return "行情数据状态未知: 缺少证券代码"
        is_etf = is_etf_like_code(plain_code)
        try:
            status = get_market_data_status_service().check_status(
                stock_codes=[] if is_etf else [plain_code],
                etf_codes=[plain_code] if is_etf else [],
                index_codes=[],
                realtime_probe_codes=[plain_code],
                require_minute_freshness=False,
            )
        except Exception as exc:
            logger.exception("交易前行情状态检查异常")
            return f"行情数据状态检查异常: {exc}"
        if status.can_run_live_strategy:
            return ""
        return f"行情数据未就绪: {status.summary}"

    def _validate_trading_time(self) -> str:
        now = datetime.now()
        if not is_trading_day(now.date()):
            return get_non_trading_reason(now.date()) or "当前不是交易日"
        current = now.time()
        morning_start = dt_time(9, 30)
        morning_end = dt_time(11, 30)
        afternoon_start = dt_time(13, 0)
        afternoon_end = dt_time(15, 0)
        if morning_start <= current <= morning_end or afternoon_start <= current <= afternoon_end:
            return ""
        return "当前不在交易时段"

    def _validate_broker_constraints(self, request: ExecutionRequest) -> str:
        try:
            plain_code = self._plain_code(request.stock_code)
            if request.order_type == 23:
                asset = self._query_broker_asset()
                available_cash = float(getattr(asset, "cash", 0) or getattr(asset, "available_cash", 0) or 0)
                total_asset = float(getattr(asset, "total_asset", 0) or 0)
                estimated_price = request.price if request.price > 0 else 0
                required_cash = estimated_price * request.order_volume
                if required_cash <= 0:
                    return "无法计算有效委托金额"
                if available_cash + 1e-6 < required_cash:
                    return f"可用资金不足，需 {required_cash:,.2f}，可用 {available_cash:,.2f}"

                if total_asset > 0:
                    single_limit = self.risk_guard.config.get("max_single_position_pct", 0.30)
                    total_limit = self.risk_guard.config.get("max_total_position_pct", 0.90)
                    projected_single = required_cash / total_asset
                    projected_total = (total_asset - available_cash + required_cash) / total_asset
                    if projected_single > single_limit + 1e-6:
                        return f"单笔仓位 {projected_single:.0%} 超过上限 {single_limit:.0%}"
                    if projected_total > total_limit + 1e-6:
                        return f"总仓位 {projected_total:.0%} 超过上限 {total_limit:.0%}"
            else:
                pos = self._query_broker_position(plain_code)
                can_use = int(getattr(pos, "can_use_volume", 0) or getattr(pos, "enable_amount", 0) or 0) if pos is not None else 0
                if can_use < request.order_volume:
                    return f"可卖数量不足，需 {request.order_volume}，可卖 {can_use}"
        except Exception as exc:
            return f"交易前校验失败: {exc}"
        return ""

    def _validate_strategy_constraints(self, request: ExecutionRequest) -> str:
        if not request.strategy_id:
            return ""
        owner_type = str(request.metadata.get("owner_type", "") or "").strip() or self._resolve_owner_type(request)
        ok, message, _ = self.strategy_registry.validate_or_claim(
            request.stock_code,
            strategy_id=request.strategy_id,
            strategy_name=request.strategy_name,
            virtual_account_id=request.virtual_account_id,
            owner_type=owner_type,
            auto_claim=True,
        )
        if not ok:
            return message
        if request.order_type != 23:
            return ""
        strategy_total_asset = self._get_total_asset()
        snapshot = self.strategy_budget.get_strategy_snapshot(
            request.strategy_id,
            strategy_name=request.strategy_name,
            virtual_account_id=request.virtual_account_id,
            real_total_asset=strategy_total_asset,
        )
        required_cash = round(float(request.price or 0.0) * int(request.order_volume or 0), 2)
        available_cash = float(snapshot.get("available_cash", 0.0) or 0.0)
        if required_cash <= 0:
            return "无法计算策略预算占用金额"
        if available_cash + 1e-6 < required_cash:
            return f"策略预算不足，需 {required_cash:,.2f}，可用 {available_cash:,.2f}"
        return ""

    def _validate_strategy_risk_policy(self, request: ExecutionRequest) -> str:
        strategy_id = (request.strategy_id or "").strip()
        if not strategy_id:
            return ""
        if not self.strategy_risk_registry.has(strategy_id):
            return ""
        context = self._build_strategy_risk_context(request)
        decision = self.strategy_risk_registry.evaluate(request, context)
        if not decision.passed:
            reason = decision.reason or "策略风控未通过"
            # 把 rule 标签拼进 message，便于后续对 order_record.validation_message
            # 做按规则维度的统计 / 告警分桶。
            rule = ""
            if decision.metadata:
                rule = str(decision.metadata.get("rule", "") or "").strip()
            tag = f"[policy:{strategy_id}:{rule}]" if rule else f"[policy:{strategy_id}]"
            return f"{tag} {reason}".strip()
        return ""

    def _build_strategy_risk_context(self, request: ExecutionRequest) -> StrategyRiskContext:
        broker_ctx = self._build_broker_context()
        budget_snapshot: Dict[str, Any] = {}
        if request.strategy_id:
            try:
                snapshot = self.strategy_budget.get_strategy_snapshot(
                    request.strategy_id,
                    strategy_name=request.strategy_name,
                    virtual_account_id=request.virtual_account_id,
                    real_total_asset=self._get_total_asset(),
                )
                budget_snapshot = dict(snapshot or {})
            except Exception:  # pragma: no cover - defensive
                logger.debug(
                    "构建策略风控上下文时查询预算快照失败 strategy_id=%s",
                    request.strategy_id,
                    exc_info=True,
                )
        return StrategyRiskContext(
            broker=broker_ctx,
            budget_snapshot=budget_snapshot,
            request_extras={
                "trigger": request.trigger,
                "source": request.source,
                "virtual_account_id": request.virtual_account_id,
                "intent_id": request.intent_id,
            },
        )

    def _build_broker_context(self) -> BrokerContext:
        try:
            assets = self._query_broker_asset()
            positions = self._query_broker_positions()
            return BrokerContext(
                connected=self._is_broker_connected(),
                total_asset=float(getattr(assets, "total_asset", 0) or 0),
                available_cash=float(getattr(assets, "cash", 0) or getattr(assets, "available_cash", 0) or 0),
                position_count=len(positions),
                top_positions=[],
            )
        except Exception:
            return BrokerContext(connected=self._is_broker_connected())

    def _check_duplicate(self, fingerprint: str, window_seconds: int) -> str:
        now = time.time()
        expired = [key for key, ts in self._recent_fingerprints.items() if now - ts > window_seconds]
        for key in expired:
            self._recent_fingerprints.pop(key, None)
        if fingerprint in self._recent_fingerprints:
            return "短时间内检测到重复委托，请稍后再试"
        recent = self.trade_service.find_recent_order_record(fingerprint, within_seconds=window_seconds)
        if recent:
            return "近期已有相同委托记录，已拦截重复报单"
        return ""

    def _remember_fingerprint(self, fingerprint: str) -> None:
        self._recent_fingerprints[fingerprint] = time.time()

    def _build_fingerprint(self, request: ExecutionRequest) -> str:
        price = round(float(request.price or 0), 3)
        direction = self._direction_from_order_type(request.order_type)
        return "|".join([
            request.trigger,
            request.source,
            request.strategy_id or "no_strategy",
            self._plain_code(request.stock_code),
            direction,
            str(int(request.order_volume)),
            str(int(request.price_type)),
            f"{price:.3f}",
        ])

    def _record_order_event(
        self,
        *,
        event_type: str,
        request_id: str,
        request: ExecutionRequest,
        mode: str,
        title: str,
        message: str,
        level: str = "info",
        status: str = "resolved",
        broker_order_id: int = 0,
        order_record_id: int = 0,
        dedupe_key: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            if self._event_storage is None:
                self._event_storage = get_order_execution_event_service()
            direction = self._direction_from_order_type(request.order_type)
            broker_event_key = str(int(broker_order_id or 0)) if broker_order_id else "pre_submit"
            event_scope = dedupe_key.strip() if dedupe_key else event_type
            event_payload: Dict[str, Any] = {
                "event_type": event_type,
                "event_scope": event_scope,
                "execution_mode": mode,
                "direction": direction,
                "order_type": request.order_type,
                "order_volume": request.order_volume,
                "price_type": request.price_type,
                "price": request.price,
                "trigger": request.trigger,
                "source": request.source,
                "strategy_name": request.strategy_name,
                "virtual_account_id": request.virtual_account_id,
                "intent_id": request.intent_id,
                "decision_record_id": request.decision_record_id,
                "order_record_id": order_record_id,
                "remark": request.remark,
            }
            if request.metadata:
                event_payload["metadata"] = dict(request.metadata)
            if payload:
                event_payload.update(payload)
            event = OrderExecutionEvent(
                event_id=f"order:{request_id}:{broker_event_key}:{event_type}:{event_scope}",
                level=level,
                category="order_execution",
                source=request.source or "trade_execution",
                strategy_id=request.strategy_id,
                symbol=self._plain_code(request.stock_code),
                request_id=request_id,
                broker_order_id=broker_order_id,
                title=title,
                message=message,
                status=status,
                payload=event_payload,
            )
            self._event_storage.add_event(event)
            self._publish_order_execution_event(event)
        except Exception:
            logger.debug("记录订单执行事件失败 request_id=%s event_type=%s", request_id, event_type, exc_info=True)

    def _publish_order_execution_event(self, event: OrderExecutionEvent) -> None:
        try:
            from common.trading_runtime import EVENT_ORDER_EXECUTION, LiveEvent, get_live_trading_engine

            engine = get_live_trading_engine(start_default_gateway=False)
            engine.event_engine.put(
                LiveEvent(
                    EVENT_ORDER_EXECUTION,
                    event,
                    gateway_name="trade_execution",
                    symbol=event.symbol,
                    key=event.event_id,
                    message=event.message,
                    level=event.level,
                    metadata={"category": event.category, "request_id": event.request_id},
                )
            )
        except Exception:
            logger.debug("发布订单执行 live event 失败 event_id=%s", getattr(event, "event_id", ""), exc_info=True)

    def _poll_order_status(
        self,
        request_id: str,
        broker_order_id: int,
        request: ExecutionRequest,
        mode: str,
        order_record_id: int,
    ) -> ExecutionResult:
        cfg = self.config_service.get_config()
        deadline = time.time() + cfg.status_poll_seconds
        latest_status = "submitted"

        while time.time() < deadline:
            try:
                order = self.broker.query_order(broker_order_id)
            except Exception:
                order = None

            if order is not None:
                state = normalize_order_state(order)
                latest_status = state.status_text

                if state.has_trade:
                    return self._handle_filled_order_state(
                        request_id=request_id,
                        broker_order_id=broker_order_id,
                        request=request,
                        mode=mode,
                        order_record_id=order_record_id,
                        order=order,
                        state=state,
                    )

                if state.is_rejected_terminal:
                    return self._handle_rejected_order_state(
                        request_id=request_id,
                        broker_order_id=broker_order_id,
                        request=request,
                        mode=mode,
                        order_record_id=order_record_id,
                        state=state,
                    )

                self._update_submitted_order_state(request_id, state)

            time.sleep(cfg.status_poll_interval_seconds)

        return self._handle_pending_order_confirmation(
            request_id=request_id,
            broker_order_id=broker_order_id,
            request=request,
            mode=mode,
            order_record_id=order_record_id,
            latest_status=latest_status,
        )

    def _handle_filled_order_state(
        self,
        *,
        request_id: str,
        broker_order_id: int,
        request: ExecutionRequest,
        mode: str,
        order_record_id: int,
        order: Any,
        state: OrderStateSnapshot,
    ) -> ExecutionResult:
        self.trade_service.sync_from_orders(
            [order],
            source=request.source,
            name_map={self._plain_code(request.stock_code): request.stock_name},
            strategy_id=request.strategy_id,
            virtual_account_id=request.virtual_account_id,
            intent_id=request.intent_id,
        )
        self._apply_strategy_execution(
            request,
            intent_id=request.intent_id or request_id,
            executed_price=state.traded_price or request.price,
            executed_volume=state.traded_volume or request.order_volume,
        )
        trade = self.trade_service.get_latest_record_by_broker_order_id(broker_order_id)
        trade_record_id = getattr(trade, "id", 0) if trade else 0
        message = f"委托已成交，单号 {broker_order_id}"
        self.trade_service.update_order_record(
            request_id,
            status=state.trade_record_status,
            validation_message=message,
            order_status_code=state.status_code,
            order_status_text=state.status_text,
            executed_price=state.traded_price,
            executed_volume=state.traded_volume,
            linked_trade_record_id=trade_record_id,
        )
        self._record_order_event(
            event_type=state.fill_event_type,
            request_id=request_id,
            request=request,
            mode=mode,
            broker_order_id=broker_order_id,
            order_record_id=order_record_id,
            dedupe_key=f"status:{state.status_code}:volume:{state.traded_volume}",
            title=state.fill_event_title,
            message=message,
            status="resolved",
            payload={
                "order_status_code": state.status_code,
                "order_status_text": state.status_text,
                "executed_price": state.traded_price,
                "executed_volume": state.traded_volume,
                "trade_record_id": trade_record_id,
            },
        )
        return ExecutionResult(
            success=True,
            message=message,
            broker_order_id=broker_order_id,
            request_id=request_id,
            execution_mode=mode,
            order_status=state.status_text,
            live_submitted=True,
            filled_confirmed=True,
            trade_record_id=trade_record_id,
            order_record_id=order_record_id,
        )

    def _handle_rejected_order_state(
        self,
        *,
        request_id: str,
        broker_order_id: int,
        request: ExecutionRequest,
        mode: str,
        order_record_id: int,
        state: OrderStateSnapshot,
    ) -> ExecutionResult:
        message = state.status_message or f"委托状态: {state.status_text}"
        reserved_budget_released = bool(request.strategy_id and request.order_type == 23)
        if reserved_budget_released:
            self.strategy_budget.release_reservation(
                strategy_id=request.strategy_id,
                intent_id=request.intent_id or request_id,
            )
        self.trade_service.update_order_record(
            request_id,
            status=state.trade_record_status,
            validation_message=message,
            order_status_code=state.status_code,
            order_status_text=state.status_text,
        )
        self._record_order_event(
            event_type="OrderRejected",
            request_id=request_id,
            request=request,
            mode=mode,
            broker_order_id=broker_order_id,
            order_record_id=order_record_id,
            dedupe_key=f"status:{state.status_code}",
            title="订单进入终态未成交",
            message=message,
            level="warning",
            status="open",
            payload={
                "order_status_code": state.status_code,
                "order_status_text": state.status_text,
                "reserved_budget_released": reserved_budget_released,
            },
        )
        return ExecutionResult(
            success=False,
            message=message,
            broker_order_id=broker_order_id,
            request_id=request_id,
            execution_mode=mode,
            order_status=state.status_text,
            live_submitted=False,
            order_record_id=order_record_id,
        )

    def _update_submitted_order_state(self, request_id: str, state: OrderStateSnapshot) -> None:
        message = state.status_message or f"委托状态: {state.status_text}"
        self.trade_service.update_order_record(
            request_id,
            status=state.trade_record_status,
            validation_message=message,
            order_status_code=state.status_code,
            order_status_text=state.status_text,
            executed_price=state.traded_price,
            executed_volume=state.traded_volume,
        )

    def _handle_pending_order_confirmation(
        self,
        *,
        request_id: str,
        broker_order_id: int,
        request: ExecutionRequest,
        mode: str,
        order_record_id: int,
        latest_status: str,
    ) -> ExecutionResult:
        message = f"委托已提交，待成交确认（最新状态: {latest_status}，单号 {broker_order_id}）"
        self.trade_service.update_order_record(
            request_id,
            status="submitted",
            validation_message=message,
        )
        self._record_order_event(
            event_type="OrderPendingConfirmation",
            request_id=request_id,
            request=request,
            mode=mode,
            broker_order_id=broker_order_id,
            order_record_id=order_record_id,
            dedupe_key="timeout",
            title="订单已提交但未确认成交",
            message=message,
            level="warning",
            status="open",
            payload={"latest_status": latest_status},
        )
        return ExecutionResult(
            success=True,
            message=message,
            broker_order_id=broker_order_id,
            request_id=request_id,
            execution_mode=mode,
            order_status=latest_status,
            live_submitted=True,
            filled_confirmed=False,
            submitted_only=True,
            order_record_id=order_record_id,
        )

    def _reserve_strategy_budget(self, request: ExecutionRequest, intent_id: str) -> str:
        if not request.strategy_id or request.order_type != 23:
            return ""
        total_asset = self._get_total_asset()
        ok, message = self.strategy_budget.reserve_cash(
            strategy_id=request.strategy_id,
            intent_id=intent_id,
            amount=round(float(request.price or 0.0) * int(request.order_volume or 0), 2),
            strategy_name=request.strategy_name,
            virtual_account_id=request.virtual_account_id,
            real_total_asset=total_asset,
        )
        return "" if ok else message

    def _apply_strategy_execution(
        self,
        request: ExecutionRequest,
        *,
        intent_id: str,
        executed_price: float,
        executed_volume: int,
        fallback_reserved: bool = False,
    ) -> None:
        if not request.strategy_id:
            return
        total_asset = self._get_total_asset()
        direction = "buy" if request.order_type == 23 else "sell"
        amount = round(float(executed_price or 0.0) * int(executed_volume or 0), 2)
        # 统一走 TradeRecordService.estimate_trade_fees（和 sync_from_orders / rehydrate 同款公式），
        # 保证主账本 cash/realized_pnl 与成交记录手续费口径完全一致
        fees = self.trade_service.estimate_trade_fees(
            direction=direction,
            amount=amount,
            stock_code=request.stock_code,
        )
        if request.order_type == 23:
            self.strategy_budget.commit_buy(
                strategy_id=request.strategy_id,
                symbol_code=request.stock_code,
                price=executed_price,
                volume=executed_volume,
                intent_id=intent_id if (intent_id or fallback_reserved) else "",
                strategy_name=request.strategy_name,
                virtual_account_id=request.virtual_account_id,
                real_total_asset=total_asset,
                commission=fees["commission"],
                stamp_tax=fees["stamp_tax"],
                transfer_fee=fees["transfer_fee"],
            )
            return
        self.strategy_budget.commit_sell(
            strategy_id=request.strategy_id,
            symbol_code=request.stock_code,
            price=executed_price,
            volume=executed_volume,
            strategy_name=request.strategy_name,
            virtual_account_id=request.virtual_account_id,
            real_total_asset=total_asset,
            commission=fees["commission"],
            stamp_tax=fees["stamp_tax"],
            transfer_fee=fees["transfer_fee"],
        )

    def _resolve_owner_type(self, request: ExecutionRequest) -> str:
        source = (request.source or "").lower()
        if "etf" in source:
            return "etf_rotation"
        if request.strategy_id == AI_STOCK_STRATEGY_ID or source == TradeSource.AI_AGENT.value:
            return OWNER_TYPE_AI
        return OWNER_TYPE_OTHER

    def _is_broker_connected(self) -> bool:
        return bool(getattr(self.broker, "is_connected", False))

    def _query_broker_asset(self) -> Any:
        return self.broker.query_asset()

    def _query_broker_positions(self) -> list[Any]:
        positions = self.broker.query_position("")
        if positions is None:
            return []
        if isinstance(positions, list):
            return positions
        if isinstance(positions, tuple):
            return list(positions)
        return [positions]

    def _query_broker_position(self, plain_code: str) -> Any:
        normalized = self._plain_code(plain_code)
        position = self.broker.query_position(normalized)
        if position is None:
            return None
        if isinstance(position, list) or isinstance(position, tuple):
            for item in position:
                pos_code = self._plain_code(getattr(item, "stock_code", "") or "")
                if pos_code == normalized:
                    return item
            return None
        return position

    def _get_total_asset(self) -> float:
        try:
            asset = self._query_broker_asset()
            return float(getattr(asset, "total_asset", 0) or 0.0)
        except Exception:
            return 0.0

    def _calc_buy_volume(self, decision: TradeDecision) -> int:
        try:
            assets = self._query_broker_asset()
            available = float(getattr(assets, "cash", 0) or getattr(assets, "available_cash", 0) or 0)
            if available <= 0 or decision.current_price <= 0:
                return 0
            target_amount = available * max(float(decision.position_pct or 0), 0.0)
            return max(int(math.floor(target_amount / decision.current_price / 100)) * 100, 0)
        except Exception:
            return 0

    def _calc_sell_volume(self, decision: TradeDecision) -> int:
        try:
            plain_code = self._plain_code(decision.symbol_code)
            pos = self._query_broker_position(plain_code)
            if pos is None:
                return 0
            can_sell = int(getattr(pos, "can_use_volume", 0) or getattr(pos, "enable_amount", 0) or 0)
            if decision.action == TradeAction.REDUCE.value:
                return max(int(math.floor(can_sell * 0.5 / 100)) * 100, 0)
            return can_sell
        except Exception:
            return 0

    @staticmethod
    def _direction_from_order_type(order_type: int) -> str:
        return TradeDirection.BUY.value if int(order_type) == 23 else TradeDirection.SELL.value

    @staticmethod
    def _plain_code(code: str) -> str:
        value = (code or "").strip().upper()
        return value.split(".")[0] if "." in value else value


_trade_execution_service: Optional[TradeExecutionService] = None


def get_trade_execution_service() -> TradeExecutionService:
    global _trade_execution_service
    if _trade_execution_service is None:
        _trade_execution_service = TradeExecutionService()
    return _trade_execution_service
