from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from trading_app.services.ai.daily_auto_trade_service import DailyAutoTradeService, get_daily_auto_trade_service
from trading_app.services.market_data.kline_full_refresh_service import KlineFullRefreshService

logger = logging.getLogger(__name__)


@dataclass
class StrategyEndOfDayResult:
    strategy_id: str
    strategy_name: str
    success: bool
    message: str
    details: Dict[str, object] = field(default_factory=dict)


class LiveStrategyEndOfDayService(QObject):
    """Coordinate the unified end-of-day workflow for the live strategy center.

    Unified lifecycle:
      Phase 0  – pause center automation
      Phase 1  – shared reconciliation (orders / trades / assets / positions / broker snapshots)
      Phase 2  – per-strategy post-close hooks
      Phase 3  – portfolio and strategy ledger snapshots
      Phase 4  – full K-line refresh (forward adjusted)
      Phase 5  – unlock next-day workflow marker
    """

    status_changed = pyqtSignal(str)
    cycle_finished = pyqtSignal(bool, str, dict)
    _CYCLE_STATE_KEY = "live_strategy_center_eod"
    _UNIFIED_PHASE_KEYS = (
        "phase0_pause_automation",
        "phase1_shared_reconcile",
        "phase2_strategy_hooks",
        "phase3_portfolio_snapshots",
        "phase4_kline_refresh",
        "phase5_next_day_unlock",
    )

    def __init__(
        self,
        daily_auto_trade: Optional[DailyAutoTradeService] = None,
        parent=None,
        *,
        rotation_etf_pool: Optional[List[str]] = None,
        portfolio_service: Optional[object] = None,
    ) -> None:
        super().__init__(parent)
        self.daily_auto_trade = daily_auto_trade or get_daily_auto_trade_service()
        self.portfolio_service = portfolio_service
        self._strategy_hooks: Dict[str, Callable[[str], StrategyEndOfDayResult]] = {}
        self._strategy_names: Dict[str, str] = {}
        self._rotation_etf_pool: List[str] = list(rotation_etf_pool or [])
        self._automation_pauser: Optional[Callable[[], str]] = None
        self._automation_resumer: Optional[Callable[[], str]] = None
        self._suppress_shared_reconcile_callback = False
        self.daily_auto_trade.reconcile_finished.connect(self._on_shared_reconcile_finished)

    def set_rotation_etf_pool(self, pool: List[str]) -> None:
        self._rotation_etf_pool = list(pool or [])

    def set_portfolio_service(self, portfolio_service: object) -> None:
        self.portfolio_service = portfolio_service

    def set_automation_controls(
        self,
        *,
        pause: Optional[Callable[[], str]] = None,
        resume: Optional[Callable[[], str]] = None,
    ) -> None:
        self._automation_pauser = pause
        self._automation_resumer = resume

    def register_strategy(
        self,
        strategy_id: str,
        strategy_name: str,
        hook: Callable[[str], StrategyEndOfDayResult],
    ) -> None:
        self._strategy_hooks[strategy_id] = hook
        self._strategy_names[strategy_id] = strategy_name

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _get_cycle_state(self, snapshot_date: Optional[str] = None) -> dict:
        return self.daily_auto_trade.get_day_state_section(
            self._CYCLE_STATE_KEY,
            day=snapshot_date or self._today(),
        )

    def _update_cycle_state(self, snapshot_date: Optional[str] = None, **fields) -> None:
        self.daily_auto_trade.update_day_state_section(
            self._CYCLE_STATE_KEY,
            day=snapshot_date or self._today(),
            **fields,
        )

    def _update_cycle_phase(
        self,
        *,
        phase_key: str,
        status: str,
        snapshot_date: Optional[str] = None,
        trigger: str = "",
        message: str = "",
        details: Optional[dict] = None,
    ) -> None:
        target_day = snapshot_date or self._today()
        cycle_state = self._get_cycle_state(target_day)
        phases = dict(cycle_state.get("phases", {}) or {})
        phase_state = dict(phases.get(phase_key, {}) or {})
        now = self._now()
        phase_state.update({
            "status": status,
            "message": message,
            "updated_at": now,
        })
        if trigger:
            phase_state["trigger"] = trigger
        if details is not None:
            phase_state["details"] = dict(details or {})
        if status == "running":
            phase_state["started_at"] = now
            phase_state["completed_at"] = ""
        else:
            phase_state.setdefault("started_at", now)
            phase_state["completed_at"] = now
        phases[phase_key] = phase_state
        self._update_cycle_state(
            target_day,
            phases=phases,
            last_trigger=trigger or cycle_state.get("last_trigger", ""),
            updated_at=now,
        )
        self._refresh_cycle_overall_state(snapshot_date=target_day, trigger=trigger)

    def _refresh_cycle_overall_state(self, *, snapshot_date: Optional[str] = None, trigger: str = "") -> dict:
        target_day = snapshot_date or self._today()
        cycle_state = self._get_cycle_state(target_day)
        phases = dict(cycle_state.get("phases", {}) or {})
        reconcile_state = self.daily_auto_trade.get_day_state_section("reconcile", day=target_day)
        shared_status = str(reconcile_state.get("status", "") or "")
        if shared_status and "phase1_shared_reconcile" not in phases:
            phases["phase1_shared_reconcile"] = {
                "status": shared_status,
                "message": str(reconcile_state.get("summary_message", "") or reconcile_state.get("error", "") or ""),
                "updated_at": str(reconcile_state.get("completed_at", "") or reconcile_state.get("started_at", "") or ""),
            }

        if "phase4_kline_refresh" not in phases and "phase0_kline_refresh" in phases:
            phases["phase4_kline_refresh"] = dict(phases.get("phase0_kline_refresh", {}) or {})
        phase_statuses = {
            key: str((phases.get(key, {}) or {}).get("status", "") or "")
            for key in self._UNIFIED_PHASE_KEYS
        }

        if "running" in set(phase_statuses.values()):
            overall_status = "running"
        elif phase_statuses and all(phase_statuses.get(key) == "completed" for key in self._UNIFIED_PHASE_KEYS):
            overall_status = "completed"
        elif "failed" in set(phase_statuses.values()):
            overall_status = "partial_failed"
        elif shared_status == "completed" or any(phase_statuses.values()):
            overall_status = "partial"
        else:
            overall_status = str(cycle_state.get("status", "") or "")

        payload = {
            "status": overall_status,
            "updated_at": self._now(),
            "shared_reconcile_status": shared_status,
            "phase_statuses": phase_statuses,
        }
        if trigger:
            payload["last_trigger"] = trigger
        if overall_status == "completed":
            payload["completed_at"] = self._now()
            payload["last_error"] = ""
        elif overall_status in ("partial_failed", "failed"):
            failed_messages = [
                str((phases.get(key, {}) or {}).get("message", "") or "")
                for key, status in phase_statuses.items()
                if status == "failed"
            ]
            payload["last_error"] = " | ".join([item for item in failed_messages if item])
        self._update_cycle_state(target_day, **payload)
        merged = self._get_cycle_state(target_day)
        merged["shared_reconcile_status"] = shared_status
        return merged

    # ------------------------------------------------------------------
    # Unified workflow phases
    # ------------------------------------------------------------------

    def _run_pause_automation_phase(self, *, snapshot_date: str, trigger: str) -> tuple[bool, str]:
        self._update_cycle_phase(
            phase_key="phase0_pause_automation",
            status="running",
            snapshot_date=snapshot_date,
            trigger=trigger,
            message="开始暂停中心自动化",
        )
        if self._automation_pauser is None:
            message = "未配置中心自动化暂停钩子，跳过"
            self._update_cycle_phase(
                phase_key="phase0_pause_automation",
                status="completed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=message,
            )
            return True, message
        try:
            message = self._automation_pauser() or "中心自动化已暂停"
            self._update_cycle_phase(
                phase_key="phase0_pause_automation",
                status="completed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=message,
            )
            return True, message
        except Exception as exc:
            logger.exception("暂停中心自动化失败")
            message = f"暂停中心自动化失败: {exc}"
            self._update_cycle_phase(
                phase_key="phase0_pause_automation",
                status="failed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=message,
            )
            return False, message

    def _mark_shared_reconcile_phase(
        self,
        *,
        snapshot_date: str,
        trigger: str,
        status: str,
        message: str,
    ) -> None:
        reconcile_state = self.daily_auto_trade.get_day_state_section("reconcile", day=snapshot_date)
        self._update_cycle_phase(
            phase_key="phase1_shared_reconcile",
            status=status,
            snapshot_date=snapshot_date,
            trigger=trigger,
            message=message,
            details={"reconcile_state": reconcile_state},
        )

    def _run_portfolio_snapshot_phase(self, *, snapshot_date: str, trigger: str) -> tuple[bool, str, dict]:
        self._update_cycle_phase(
            phase_key="phase3_portfolio_snapshots",
            status="running",
            snapshot_date=snapshot_date,
            trigger=trigger,
            message="开始固化组合与策略账本快照",
        )
        if self.portfolio_service is None:
            message = "未配置组合快照服务，跳过组合快照固化"
            self._update_cycle_phase(
                phase_key="phase3_portfolio_snapshots",
                status="completed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=message,
                details={"snapshot_count": 0, "results": []},
            )
            return True, message, {"snapshot_count": 0, "results": []}
        try:
            results = list(self.portfolio_service.finalize_day_snapshots(remark="日终统一快照") or [])
            details = {"snapshot_count": len(results), "results": results}
            message = f"组合与策略账本快照已固化 {len(results)} 组"
            self._update_cycle_phase(
                phase_key="phase3_portfolio_snapshots",
                status="completed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=message,
                details=details,
            )
            return True, message, details
        except Exception as exc:
            logger.exception("组合与策略账本快照固化失败")
            message = f"组合与策略账本快照固化失败: {exc}"
            details = {"snapshot_count": 0, "results": [], "error": str(exc)}
            self._update_cycle_phase(
                phase_key="phase3_portfolio_snapshots",
                status="failed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=message,
                details=details,
            )
            return False, message, details

    def _run_next_day_unlock_phase(self, *, snapshot_date: str, trigger: str) -> tuple[bool, str]:
        message = "次日任务解锁标记已更新"
        self._update_cycle_phase(
            phase_key="phase5_next_day_unlock",
            status="running",
            snapshot_date=snapshot_date,
            trigger=trigger,
            message="开始更新次日任务解锁标记",
        )
        if self._automation_resumer is not None:
            try:
                resume_message = self._automation_resumer() or "中心自动化已恢复"
                message = f"{message}；{resume_message}"
            except Exception as exc:
                logger.exception("恢复中心自动化失败")
                message = f"次日任务解锁完成，但恢复中心自动化失败: {exc}"
                self._update_cycle_phase(
                    phase_key="phase5_next_day_unlock",
                    status="failed",
                    snapshot_date=snapshot_date,
                    trigger=trigger,
                    message=message,
                )
                return False, message
        self._update_cycle_phase(
            phase_key="phase5_next_day_unlock",
            status="completed",
            snapshot_date=snapshot_date,
            trigger=trigger,
            message=message,
        )
        return True, message

    def _run_kline_full_refresh(self) -> tuple[bool, str]:
        """Phase 4: full (forward adjusted) K-line refresh for all assets."""
        self.status_changed.emit("Phase 4: 开始全量K线数据刷新...")
        svc = KlineFullRefreshService(rotation_etf_pool=self._rotation_etf_pool)
        result = svc.run_full_refresh_result(status_cb=lambda msg: self.status_changed.emit(msg))
        return result.to_legacy_tuple()

    # ------------------------------------------------------------------
    # Public entry-points
    # ------------------------------------------------------------------

    def run_manual_cycle(self) -> tuple[bool, str, dict]:
        return self.run_unified_cycle(trigger="manual", run_hooks=True, run_refresh=True)

    def run_unified_cycle(
        self,
        *,
        trigger: str = "manual",
        run_hooks: bool = True,
        run_refresh: bool = True,
    ) -> tuple[bool, str, dict]:
        snapshot_date = self._today()
        self.status_changed.emit("开始执行中心统一日终流程...")
        self._update_cycle_state(
            snapshot_date,
            status="running",
            started_at=self._now(),
            completed_at="",
            last_trigger=trigger,
            last_error="",
            workflow_version="unified_v1",
        )
        self.status_changed.emit("Phase 0: 暂停中心自动化...")
        pause_success, pause_message = self._run_pause_automation_phase(
            snapshot_date=snapshot_date,
            trigger=trigger,
        )
        if not pause_success:
            return self._finalize_preflight_failure(
                snapshot_date=snapshot_date,
                trigger=trigger,
                pause_message=pause_message,
            )

        self.status_changed.emit("Phase 1: 执行共享日终对账...")
        self._mark_shared_reconcile_phase(
            snapshot_date=snapshot_date,
            trigger=trigger,
            status="running",
            message="开始执行共享日终对账",
        )
        self._suppress_shared_reconcile_callback = True
        try:
            success, message = self.daily_auto_trade.run_end_of_day_reconcile(slot=trigger, snapshot_date=snapshot_date)
        finally:
            self._suppress_shared_reconcile_callback = False
        self._mark_shared_reconcile_phase(
            snapshot_date=snapshot_date,
            trigger=trigger,
            status="completed" if success else "failed",
            message=message,
        )
        if not success:
            return self._finalize_shared_failure(
                snapshot_date=snapshot_date,
                trigger=trigger,
                shared_message=message,
            )
        return self._run_post_reconcile_phases(
            snapshot_date=snapshot_date,
            shared_message=message,
            trigger=trigger,
            run_hooks=run_hooks,
            run_refresh=run_refresh,
            pause_message=pause_message,
        )

    def run_catchup_if_needed(self) -> tuple[bool, str]:
        snapshot_date = self._today()
        cycle_state = self._get_cycle_state(snapshot_date)
        phases = dict(cycle_state.get("phases", {}) or {})
        reconcile_state = self.daily_auto_trade.get_day_state_section("reconcile", day=snapshot_date)
        reconcile_status = str(reconcile_state.get("status", "") or "")
        phase_status = {
            key: str((phases.get(key, {}) or {}).get("status", "") or "")
            for key in self._UNIFIED_PHASE_KEYS
        }
        should_run, reason = self.daily_auto_trade.should_run_reconcile_catchup()
        missing_after_reconcile = reconcile_status == "completed" and any(
            phase_status.get(key) != "completed"
            for key in (
                "phase2_strategy_hooks",
                "phase3_portfolio_snapshots",
                "phase4_kline_refresh",
                "phase5_next_day_unlock",
            )
        )
        logger.info("检查是否需要补跑日终流程: should_run=%s reason=%s missing_after_reconcile=%s", should_run, reason, missing_after_reconcile)
        if not should_run and not missing_after_reconcile:
            return False, reason
        self._update_cycle_state(
            snapshot_date,
            status="running",
            started_at=cycle_state.get("started_at", "") or self._now(),
            completed_at="",
            last_trigger="catchup",
            workflow_version="unified_v1",
        )
        if not should_run and reconcile_status == "completed":
            shared_message = str(reconcile_state.get("summary_message", "") or "今日日终对账已完成")
            pause_message = str((phases.get("phase0_pause_automation", {}) or {}).get("message", "") or "")
            if phase_status.get("phase0_pause_automation") != "completed":
                pause_success, pause_message = self._run_pause_automation_phase(
                    snapshot_date=snapshot_date,
                    trigger="catchup",
                )
                if not pause_success:
                    final_success, final_message, _ = self._finalize_preflight_failure(
                        snapshot_date=snapshot_date,
                        trigger="catchup",
                        pause_message=pause_message,
                    )
                    return final_success, final_message
            self._mark_shared_reconcile_phase(
                snapshot_date=snapshot_date,
                trigger="catchup",
                status="completed",
                message=shared_message,
            )
            run_hooks = phase_status.get("phase2_strategy_hooks") != "completed"
            run_refresh = phase_status.get("phase4_kline_refresh") != "completed"
            logger.info(
                "补跑日终流程继续: shared_done hooks_missing=%s portfolio_missing=%s refresh_missing=%s unlock_missing=%s",
                run_hooks,
                phase_status.get("phase3_portfolio_snapshots") != "completed",
                run_refresh,
                phase_status.get("phase5_next_day_unlock") != "completed",
            )
            final_success, final_message, _ = self._run_post_reconcile_phases(
                snapshot_date=snapshot_date,
                shared_message=shared_message,
                trigger="catchup",
                run_hooks=run_hooks,
                run_portfolio=phase_status.get("phase3_portfolio_snapshots") != "completed",
                run_refresh=run_refresh,
                run_unlock=phase_status.get("phase5_next_day_unlock") != "completed",
                pause_message=pause_message,
            )
            logger.info("补跑日终流程结束: success=%s message=%s", final_success, final_message)
            return final_success, final_message
        self.status_changed.emit("检测到缺失的日终流程，执行统一日终补跑...")
        final_success, final_message, _ = self.run_unified_cycle(trigger="catchup", run_hooks=True, run_refresh=True)
        logger.info("补跑日终流程结束: success=%s message=%s", final_success, final_message)
        return final_success, final_message

    def _on_shared_reconcile_finished(self, success: bool, message: str) -> None:
        if self._suppress_shared_reconcile_callback:
            logger.info("跳过共享对账完成回调：由手动/补跑流程接管后续阶段")
            return
        snapshot_date = self._today()
        self._update_cycle_state(
            snapshot_date,
            status="running",
            started_at=self._get_cycle_state(snapshot_date).get("started_at", "") or self._now(),
            completed_at="",
            last_trigger="scheduled",
            last_error="",
            workflow_version="unified_v1",
        )
        self._update_cycle_phase(
            phase_key="phase0_pause_automation",
            status="completed",
            snapshot_date=snapshot_date,
            trigger="scheduled",
            message="定时共享对账已先行触发，暂停中心自动化阶段跳过",
        )
        self._mark_shared_reconcile_phase(
            snapshot_date=snapshot_date,
            trigger="scheduled",
            status="completed" if success else "failed",
            message=message,
        )
        if not success:
            self._finalize_shared_failure(
                snapshot_date=snapshot_date,
                trigger="scheduled",
                shared_message=message,
            )
            return
        self._run_post_reconcile_phases(
            snapshot_date=snapshot_date,
            shared_message=message,
            trigger="scheduled",
            run_hooks=True,
            run_refresh=True,
            pause_message="定时共享对账已先行触发，暂停中心自动化阶段跳过",
        )

    def _execute_strategy_hooks(
        self,
        *,
        snapshot_date: str,
        trigger: str,
    ) -> tuple[bool, Dict[str, Dict[str, object]], List[str]]:
        strategy_results: Dict[str, Dict[str, object]] = {}
        all_success = True
        self._update_cycle_state(
            snapshot_date,
            status="running",
            started_at=self._get_cycle_state(snapshot_date).get("started_at", "") or self._now(),
            completed_at="",
            last_trigger=trigger,
        )
        self._update_cycle_phase(
            phase_key="phase2_strategy_hooks",
            status="running",
            snapshot_date=snapshot_date,
            trigger=trigger,
            message="开始执行策略日终钩子",
        )
        logger.info("开始执行策略日终钩子: trigger=%s strategy_count=%d", trigger, len(self._strategy_hooks))

        for strategy_id, hook in self._strategy_hooks.items():
            strategy_name = self._strategy_names.get(strategy_id, strategy_id)
            logger.info("执行策略日终钩子: trigger=%s strategy_id=%s strategy_name=%s", trigger, strategy_id, strategy_name)
            try:
                result = hook(snapshot_date)
            except Exception as exc:
                logger.exception("Strategy end-of-day hook failed: %s", strategy_id)
                result = StrategyEndOfDayResult(
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    success=False,
                    message=f"{strategy_name} 日终流程异常: {exc}",
                )
            logger.info(
                "策略日终钩子完成: trigger=%s strategy_id=%s success=%s message=%s",
                trigger,
                strategy_id,
                result.success,
                result.message,
            )
            strategy_results[strategy_id] = {
                "strategy_name": result.strategy_name,
                "success": result.success,
                "message": result.message,
                "details": dict(result.details or {}),
            }
            all_success = all_success and result.success
        self._update_cycle_phase(
            phase_key="phase2_strategy_hooks",
            status="completed" if all_success else "failed",
            snapshot_date=snapshot_date,
            trigger=trigger,
            message=" | ".join(
                [
                    f"{result['strategy_name']}: {result['message']}"
                    for result in strategy_results.values()
                ]
            ),
            details={"strategy_results": strategy_results},
        )
        logger.info("策略日终钩子汇总完成: trigger=%s success=%s", trigger, all_success)
        summary_parts = [
            f"{result['strategy_name']}: {result['message']}"
            for result in strategy_results.values()
        ]
        return all_success, strategy_results, summary_parts

    def _run_kline_refresh_phase(
        self,
        *,
        snapshot_date: str,
        trigger: str,
    ) -> tuple[bool, str]:
        self._update_cycle_phase(
            phase_key="phase4_kline_refresh",
            status="running",
            snapshot_date=snapshot_date,
            trigger=trigger,
            message="开始全量K线数据刷新",
        )
        refresh_ok, refresh_msg = self._run_kline_full_refresh()
        self._update_cycle_phase(
            phase_key="phase4_kline_refresh",
            status="completed" if refresh_ok else "failed",
            snapshot_date=snapshot_date,
            trigger=trigger,
            message=refresh_msg,
        )
        if refresh_ok:
            logger.info("%s 模式下 K线全量刷新完成: %s", trigger, refresh_msg)
        else:
            logger.warning("%s 模式下 K线全量刷新有部分失败: %s", trigger, refresh_msg)
        return refresh_ok, refresh_msg

    def _run_post_reconcile_phases(
        self,
        *,
        snapshot_date: str,
        shared_message: str,
        trigger: str,
        run_hooks: bool,
        run_refresh: bool,
        run_portfolio: bool = True,
        run_unlock: bool = True,
        pause_message: str = "",
    ) -> tuple[bool, str, dict]:
        strategy_success = True
        strategy_results: Dict[str, Dict[str, object]] = {}
        strategy_summary_parts: List[str] = []
        refresh_ok = True
        refresh_msg = "未执行全量K线刷新"

        if run_hooks:
            self.status_changed.emit("Phase 2: 执行策略日终钩子...")
            strategy_success, strategy_results, strategy_summary_parts = self._execute_strategy_hooks(
                snapshot_date=snapshot_date,
                trigger=trigger,
            )
        else:
            self._update_cycle_phase(
                phase_key="phase2_strategy_hooks",
                status="completed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message="策略日终钩子已在前序流程完成",
                details={"strategy_results": strategy_results},
            )

        if run_portfolio:
            self.status_changed.emit("Phase 3: 固化组合与策略账本快照...")
            portfolio_success, portfolio_message, portfolio_details = self._run_portfolio_snapshot_phase(
                snapshot_date=snapshot_date,
                trigger=trigger,
            )
        else:
            portfolio_success = True
            portfolio_message = "组合与策略账本快照已在前序流程完成"
            portfolio_details = dict(
                ((self._get_cycle_state(snapshot_date).get("phases", {}) or {}).get("phase3_portfolio_snapshots", {}) or {}).get("details", {}) or {}
            )
            self._update_cycle_phase(
                phase_key="phase3_portfolio_snapshots",
                status="completed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=portfolio_message,
                details=portfolio_details,
            )

        if run_refresh:
            self.status_changed.emit("Phase 4: 执行全量K线数据刷新...")
            refresh_ok, refresh_msg = self._run_kline_refresh_phase(
                snapshot_date=snapshot_date,
                trigger=trigger,
            )
        else:
            refresh_msg = "全量K线刷新已在前序流程完成"
            self._update_cycle_phase(
                phase_key="phase4_kline_refresh",
                status="completed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=refresh_msg,
            )

        if run_unlock:
            self.status_changed.emit("Phase 5: 解锁次日任务...")
            unlock_success, unlock_message = self._run_next_day_unlock_phase(
                snapshot_date=snapshot_date,
                trigger=trigger,
            )
        else:
            unlock_success = True
            unlock_message = "次日任务已在前序流程解锁"
            self._update_cycle_phase(
                phase_key="phase5_next_day_unlock",
                status="completed",
                snapshot_date=snapshot_date,
                trigger=trigger,
                message=unlock_message,
            )

        return self._finalize_cycle(
            snapshot_date=snapshot_date,
            trigger=trigger,
            pause_message=pause_message,
            shared_success=True,
            shared_message=shared_message,
            strategy_results=strategy_results,
            strategy_success=strategy_success,
            strategy_summary_parts=strategy_summary_parts,
            portfolio_success=portfolio_success,
            portfolio_message=portfolio_message,
            portfolio_details=portfolio_details,
            kline_refresh_success=refresh_ok,
            kline_refresh_message=refresh_msg,
            unlock_success=unlock_success,
            unlock_message=unlock_message,
        )

    def _finalize_preflight_failure(
        self,
        *,
        snapshot_date: str,
        trigger: str,
        pause_message: str,
    ) -> tuple[bool, str, dict]:
        final_message = f"统一日终预检失败: {pause_message}"
        payload = self._build_cycle_payload(
            pause_success=False,
            pause_message=pause_message,
            shared_success=False,
            shared_message="未执行共享日终对账",
            strategy_results={},
            trigger=trigger,
            portfolio_success=False,
            portfolio_message="未执行组合快照固化",
            portfolio_details={},
            kline_refresh_success=False,
            kline_refresh_message="未执行全量K线刷新",
            unlock_success=False,
            unlock_message="未解锁次日任务",
        )
        self._update_cycle_state(
            snapshot_date,
            status="failed",
            completed_at=self._now(),
            last_trigger=trigger,
            last_error=final_message,
        )
        self.status_changed.emit(final_message)
        self.cycle_finished.emit(False, final_message, payload)
        return False, final_message, payload

    def _finalize_shared_failure(
        self,
        *,
        snapshot_date: str,
        trigger: str,
        shared_message: str,
    ) -> tuple[bool, str, dict]:
        final_message = f"共享日终对账失败: {shared_message}"
        payload = self._build_cycle_payload(
            pause_success=True,
            pause_message=str((self._get_cycle_state(snapshot_date).get("phases", {}) or {}).get("phase0_pause_automation", {}).get("message", "") or ""),
            shared_success=False,
            shared_message=shared_message,
            strategy_results={},
            trigger=trigger,
            portfolio_success=False,
            portfolio_message="未执行组合快照固化",
            portfolio_details={},
            kline_refresh_success=False,
            kline_refresh_message="未执行全量K线刷新",
            unlock_success=False,
            unlock_message="未解锁次日任务",
        )
        self._update_cycle_state(
            snapshot_date,
            status="failed",
            completed_at=self._now(),
            last_trigger=trigger,
            last_error=final_message,
        )
        self.status_changed.emit(final_message)
        self.cycle_finished.emit(False, final_message, payload)
        return False, final_message, payload

    def _finalize_cycle(
        self,
        *,
        snapshot_date: str,
        trigger: str,
        pause_message: str,
        shared_success: bool,
        shared_message: str,
        strategy_results: Dict[str, Dict[str, object]],
        strategy_success: bool,
        strategy_summary_parts: List[str],
        portfolio_success: bool,
        portfolio_message: str,
        portfolio_details: dict,
        kline_refresh_success: bool,
        kline_refresh_message: str,
        unlock_success: bool,
        unlock_message: str,
    ) -> tuple[bool, str, dict]:
        summary_parts = [shared_message]
        summary_parts.extend(strategy_summary_parts)
        if portfolio_message:
            summary_parts.append(portfolio_message)
        if kline_refresh_message:
            summary_parts.append(f"K线全量刷新: {kline_refresh_message}")
        if unlock_message:
            summary_parts.append(unlock_message)
        final_success = shared_success and strategy_success and portfolio_success and kline_refresh_success and unlock_success
        final_message = " | ".join([part for part in summary_parts if part])
        payload = self._build_cycle_payload(
            pause_success=True,
            pause_message=pause_message,
            shared_success=shared_success,
            shared_message=shared_message,
            strategy_results=strategy_results,
            trigger=trigger,
            portfolio_success=portfolio_success,
            portfolio_message=portfolio_message,
            portfolio_details=portfolio_details,
            kline_refresh_success=kline_refresh_success,
            kline_refresh_message=kline_refresh_message,
            unlock_success=unlock_success,
            unlock_message=unlock_message,
        )
        self._refresh_cycle_overall_state(snapshot_date=snapshot_date, trigger=trigger)
        if not final_success:
            self._update_cycle_state(snapshot_date, last_error=final_message)
        self.status_changed.emit(final_message)
        self.cycle_finished.emit(final_success, final_message, payload)
        return final_success, final_message, payload

    @staticmethod
    def _build_cycle_payload(
        *,
        pause_success: bool,
        pause_message: str,
        shared_success: bool,
        shared_message: str,
        strategy_results: Dict[str, Dict[str, object]],
        trigger: str = "",
        portfolio_success: bool = False,
        portfolio_message: str = "",
        portfolio_details: Optional[dict] = None,
        kline_refresh_success: bool = False,
        kline_refresh_message: str = "",
        unlock_success: bool = False,
        unlock_message: str = "",
    ) -> dict:
        return {
            "pause_automation": {
                "success": pause_success,
                "message": pause_message,
            },
            "shared_reconcile": {
                "success": shared_success,
                "message": shared_message,
            },
            "strategy_results": strategy_results,
            "portfolio_snapshots": {
                "success": portfolio_success,
                "message": portfolio_message,
                "details": dict(portfolio_details or {}),
            },
            "kline_refresh": {
                "success": kline_refresh_success,
                "message": kline_refresh_message,
            },
            "next_day_unlock": {
                "success": unlock_success,
                "message": unlock_message,
            },
            "trigger": trigger,
            "workflow_version": "unified_v1",
        }
