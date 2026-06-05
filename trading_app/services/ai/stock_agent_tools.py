from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

from common.agent.agent_context_service import AgentRuntimeContext
from trading_app.services.ai.agent_watchlist_scan_service import AgentWatchlistScanService
from common.agent.decision_run_context import DecisionRunContext, build_decision_run_context
from trading_app.services.market_data.market_context_service import MarketContextService
from trading_app.services.execution.portfolio_risk_service import PortfolioRiskService
from trading_app.services.market_data.realtime_snapshot_service import load_symbol_view
from trading_app.services.ai.stock_fundamental_service import StockFundamentalService
from trading_app.services.ai.stock_news_service import StockNewsService
from trading_app.services.ai.stock_analyzer import get_analyzer

logger = logging.getLogger(__name__)


@dataclass
class AgentToolExecutionContext:
    runtime_context: AgentRuntimeContext
    raw_context: Dict[str, Any]
    user_text: str = ""


@dataclass
class AgentToolResult:
    tool_name: str
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentToolDefinition:
    name: str
    description: str
    handler: Callable[..., AgentToolResult]


class StockAgentToolRegistry:
    """Simple name-to-handler registry for domain tools."""

    def __init__(self) -> None:
        self._definitions: Dict[str, AgentToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[..., AgentToolResult],
    ) -> None:
        self._definitions[name] = AgentToolDefinition(
            name=name,
            description=description,
            handler=handler,
        )

    def dispatch(
        self,
        name: str,
        execution_context: AgentToolExecutionContext,
        **kwargs,
    ) -> AgentToolResult | None:
        definition = self._definitions.get(name)
        if not definition:
            return None
        return definition.handler(execution_context, **kwargs)

    @property
    def names(self) -> List[str]:
        return list(self._definitions.keys())


def build_default_stock_tool_registry() -> StockAgentToolRegistry:
    registry = StockAgentToolRegistry()
    registry.register("context_snapshot", "汇总当前运行上下文", _context_snapshot_tool)
    registry.register("symbol_technical_snapshot", "生成当前标的技术摘要", _symbol_technical_snapshot_tool)
    registry.register("symbol_news_snapshot", "生成当前标的消息面摘要", _symbol_news_snapshot_tool)
    registry.register("symbol_fundamental_snapshot", "生成当前标的基本面摘要", _symbol_fundamental_snapshot_tool)
    registry.register("symbol_analysis_packet", "生成当前标的的深度分析资料", _symbol_analysis_packet_tool)
    registry.register("current_kline_image", "截取当前 K 线图像", _current_kline_image_tool)
    registry.register("watchlist_snapshot", "生成当前分组技术快照", _watchlist_snapshot_tool)
    registry.register("position_snapshot", "生成当前持仓摘要", _position_snapshot_tool)
    registry.register("compare_symbols", "比较多个股票或 ETF", _compare_symbols_tool)
    registry.register("market_context_snapshot", "采集大盘环境快照", _market_context_snapshot_tool)
    registry.register("portfolio_industry_exposure", "分析当前持仓行业暴露", _portfolio_industry_exposure_tool)
    return registry


def extract_symbol_codes(text: str) -> List[str]:
    codes = re.findall(r"\b\d{6}\b", text or "")
    unique_codes: List[str] = []
    for code in codes:
        if code not in unique_codes:
            unique_codes.append(code)
    return unique_codes


def _shared_cache_parts(raw_context: Dict[str, Any]) -> tuple[Any, Dict[str, AgentToolResult]] | tuple[None, None]:
    cache = raw_context.get("_agent_shared_evidence_cache") if isinstance(raw_context, dict) else None
    if not isinstance(cache, dict):
        return None, None
    results = cache.get("results")
    if not isinstance(results, dict):
        results = {}
        cache["results"] = results
    return cache.get("lock"), results


def _copy_tool_result(result: AgentToolResult) -> AgentToolResult:
    return copy.deepcopy(result)


def _get_cached_tool_result(execution_context: AgentToolExecutionContext, tool_name: str) -> AgentToolResult | None:
    lock, results = _shared_cache_parts(execution_context.raw_context)
    if results is None:
        return None
    if lock is not None:
        lock.acquire()
    try:
        cached = results.get(tool_name)
        return _copy_tool_result(cached) if isinstance(cached, AgentToolResult) else None
    finally:
        if lock is not None:
            lock.release()


def _store_cached_tool_result(
    execution_context: AgentToolExecutionContext,
    tool_name: str,
    result: AgentToolResult,
) -> AgentToolResult:
    lock, results = _shared_cache_parts(execution_context.raw_context)
    if results is None:
        return result
    if lock is not None:
        lock.acquire()
    try:
        results[tool_name] = _copy_tool_result(result)
    finally:
        if lock is not None:
            lock.release()
    return result


def _context_snapshot_tool(execution_context: AgentToolExecutionContext) -> AgentToolResult:
    lines = execution_context.runtime_context.to_summary_lines()
    content = "\n".join(f"- {line}" for line in lines) if lines else "- 当前无可用运行上下文"
    return AgentToolResult(
        tool_name="context_snapshot",
        title="当前运行上下文摘要",
        summary=f"已汇总 {len(lines)} 条上下文信息",
        content=content,
        metadata={"line_count": len(lines)},
    )


def _symbol_technical_snapshot_tool(
    execution_context: AgentToolExecutionContext,
    code: str | None = None,
) -> AgentToolResult:
    runtime_symbol = execution_context.runtime_context.symbol
    symbol_code = code or runtime_symbol.code
    if not symbol_code:
        return AgentToolResult(
            tool_name="symbol_technical_snapshot",
            title="当前标的技术摘要",
            summary="当前没有选中的标的",
            content="- 当前上下文缺少标的代码，无法生成技术摘要。",
        )

    symbol_name = runtime_symbol.name or symbol_code
    asset_type = runtime_symbol.asset_type or "股票"
    df = _load_symbol_df(symbol_code, execution_context.raw_context, asset_type=asset_type)
    if df is None or df.empty:
        return AgentToolResult(
            tool_name="symbol_technical_snapshot",
            title=f"{symbol_name}({symbol_code}) 技术摘要",
            summary="未找到本地行情数据",
            content="- 本地数据目录中没有可用的 K 线数据，无法生成技术摘要。",
            metadata={"code": symbol_code},
        )

    summary = _build_symbol_summary(df, symbol_code, symbol_name, asset_type)
    return AgentToolResult(
        tool_name="symbol_technical_snapshot",
        title=f"{symbol_name}({symbol_code}) 技术摘要",
        summary=summary["summary"],
        content=summary["content"],
        metadata={"code": symbol_code, "name": symbol_name, "asset_type": asset_type},
    )


def _symbol_analysis_packet_tool(
    execution_context: AgentToolExecutionContext,
    max_days: int = 750,
) -> AgentToolResult:
    runtime_symbol = execution_context.runtime_context.symbol
    symbol_code = runtime_symbol.code
    if not symbol_code:
        return AgentToolResult(
            tool_name="symbol_analysis_packet",
            title="当前标的深度分析资料",
            summary="当前没有选中的标的",
            content="- 当前上下文缺少标的代码，无法生成深度分析资料。",
        )

    symbol_name = runtime_symbol.name or symbol_code
    asset_type = runtime_symbol.asset_type or "股票"
    df = _get_current_symbol_dataframe(execution_context, symbol_code, asset_type)
    if df is None or df.empty:
        return AgentToolResult(
            tool_name="symbol_analysis_packet",
            title=f"{symbol_name}({symbol_code}) 深度分析资料",
            summary="未找到可分析的 K 线数据",
            content="- 无法获取当前标的的 K 线数据，不能生成深度分析资料。",
            metadata={"code": symbol_code},
        )

    analyzer = get_analyzer()
    resolved_max_days = int(max_days or 0)
    kline_text = analyzer.format_kline_data(
        df,
        symbol_code,
        symbol_name,
        max_days=resolved_max_days,
    )
    guide = analyzer.load_guide()
    effective_days = len(df) if resolved_max_days == 0 else min(len(df), resolved_max_days)
    range_text = "全部数据" if resolved_max_days == 0 else f"最近 {effective_days} 个交易日"
    content = "\n".join([
        f"- 标的: {symbol_name}({symbol_code}) / {asset_type}",
        f"- 分析范围: {range_text}",
        "",
        "## 分析指导手册",
        guide,
        "",
        "## K线资料",
        kline_text,
        "",
        "## 使用建议",
        "- 请优先基于以上指导手册与 K 线资料完成结构化技术分析。",
        "- 回答时请引用证据，并说明风险点与失效条件。",
    ])
    return AgentToolResult(
        tool_name="symbol_analysis_packet",
        title=f"{symbol_name}({symbol_code}) 深度分析资料",
        summary=f"已生成 {range_text} 的分析资料",
        content=content,
        metadata={
            "code": symbol_code,
            "name": symbol_name,
            "max_days": resolved_max_days,
            "record_count": effective_days,
        },
    )


def _symbol_news_snapshot_tool(
    execution_context: AgentToolExecutionContext,
    code: str | None = None,
    lookback_days: int = 7,
    limit: int = 8,
) -> AgentToolResult:
    runtime_symbol = execution_context.runtime_context.symbol
    symbol_code = code or runtime_symbol.code
    symbol_name = runtime_symbol.name or symbol_code
    if not symbol_code:
        return AgentToolResult(
            tool_name="symbol_news_snapshot",
            title="当前标的消息面摘要",
            summary="当前没有选中的标的",
            content="- 当前上下文缺少标的代码，无法生成消息面摘要。",
        )

    service = StockNewsService()
    snapshot = service.build_snapshot(
        code=symbol_code,
        name=symbol_name,
        lookback_days=lookback_days,
        limit=limit,
    )
    metadata = dict(snapshot.metadata)
    metadata.update({"code": symbol_code, "name": symbol_name})
    return AgentToolResult(
        tool_name="symbol_news_snapshot",
        title=f"{symbol_name}({symbol_code}) 消息面摘要",
        summary=snapshot.summary,
        content=snapshot.content,
        metadata=metadata,
    )


def _symbol_fundamental_snapshot_tool(
    execution_context: AgentToolExecutionContext,
    code: str | None = None,
) -> AgentToolResult:
    runtime_symbol = execution_context.runtime_context.symbol
    symbol_code = code or runtime_symbol.code
    symbol_name = runtime_symbol.name or symbol_code
    if not symbol_code:
        return AgentToolResult(
            tool_name="symbol_fundamental_snapshot",
            title="当前标的基本面摘要",
            summary="当前没有选中的标的",
            content="- 当前上下文缺少标的代码，无法生成基本面摘要。",
        )

    service = StockFundamentalService()
    snapshot = service.build_snapshot(
        code=symbol_code,
        name=symbol_name,
        raw_context=execution_context.raw_context,
    )
    metadata = dict(snapshot.metadata)
    metadata.update({"code": symbol_code, "name": symbol_name})
    return AgentToolResult(
        tool_name="symbol_fundamental_snapshot",
        title=f"{symbol_name}({symbol_code}) 基本面摘要",
        summary=snapshot.summary,
        content=snapshot.content,
        metadata=metadata,
    )


def _current_kline_image_tool(execution_context: AgentToolExecutionContext) -> AgentToolResult:
    runtime_symbol = execution_context.runtime_context.symbol
    symbol_code = runtime_symbol.code
    symbol_name = runtime_symbol.name or symbol_code or "当前标的"
    asset_type = runtime_symbol.asset_type or "股票"
    capture_hook = _get_tool_hook(execution_context.raw_context, "capture_current_kline_image")
    if not callable(capture_hook):
        return AgentToolResult(
            tool_name="current_kline_image",
            title="当前 K 线截图",
            summary="当前运行环境不支持截图工具",
            content="- 当前环境未注入 K 线截图能力，无法自动附加图像。",
        )

    try:
        image_path = capture_hook(symbol_code, asset_type)
    except TypeError:
        image_path = capture_hook()
    if not image_path:
        return AgentToolResult(
            tool_name="current_kline_image",
            title="当前 K 线截图",
            summary="未能获取当前 K 线截图",
            content="- 当前没有可用的 K 线图像，或截图保存失败。",
        )

    path_obj = Path(image_path)
    return AgentToolResult(
        tool_name="current_kline_image",
        title=f"{symbol_name} K 线截图",
        summary=f"已截取当前 K 线图: {path_obj.name}",
        content="\n".join([
            f"- 标的: {symbol_name}({symbol_code or '-'})",
            f"- 截图路径: {image_path}",
            "- 该截图将作为多模态输入附加给模型，用于图形/形态分析。",
        ]),
        metadata={"image_path": image_path, "code": symbol_code, "name": symbol_name},
    )


def _watchlist_snapshot_tool(
    execution_context: AgentToolExecutionContext,
    limit: int = 8,
) -> AgentToolResult:
    watchlist = execution_context.raw_context.get("watchlist", {}) or {}
    visible_items = list(watchlist.get("visible_items", []) or [])
    if not visible_items:
        return AgentToolResult(
            tool_name="watchlist_snapshot",
            title="当前分组技术快照",
            summary="当前分组没有可巡检标的",
            content="- 当前上下文缺少分组标的，无法生成巡检摘要。",
        )

    summaries = []
    data_dir = execution_context.raw_context.get("data_dir", "")
    for item in visible_items[: max(1, int(limit))]:
        summary = AgentWatchlistScanService._summarize_symbol(
            code=item.get("code", ""),
            name=item.get("name", "") or item.get("code", ""),
            asset_type=item.get("asset_type", "股票"),
            data_dir=data_dir,
        )
        if summary:
            summaries.append(summary)

    if not summaries:
        return AgentToolResult(
            tool_name="watchlist_snapshot",
            title="当前分组技术快照",
            summary="可见标的历史数据不足",
            content="- 当前分组的可见标的缺少足够历史数据，无法生成巡检快照。",
        )

    group_name = watchlist.get("group_name") or watchlist.get("source_tab") or "当前分组"
    table = AgentWatchlistScanService._build_summary_table(summaries)
    strongest = sorted(
        summaries,
        key=lambda item: (item.pct_60d, item.volume_ratio_5_20, -item.price_percentile_120d),
        reverse=True,
    )[:3]
    weakest = sorted(
        summaries,
        key=lambda item: (item.pct_60d, item.volume_ratio_5_20, item.price_percentile_120d),
    )[:3]
    content_lines = [
        f"- 分组: {group_name}",
        f"- 纳入样本: {len(summaries)} 只",
        "",
        "## 偏强样本",
        *[
            f"- {item.name}({item.code}): {item.trend_label} / {item.ma_structure} / {item.risk_label}"
            for item in strongest
        ],
        "",
        "## 偏弱样本",
        *[
            f"- {item.name}({item.code}): {item.trend_label} / {item.ma_structure} / {item.risk_label}"
            for item in weakest
        ],
        "",
        "## 技术摘要表",
        table,
    ]
    return AgentToolResult(
        tool_name="watchlist_snapshot",
        title=f"{group_name} 技术快照",
        summary=f"已分析 {len(summaries)} 只分组标的",
        content="\n".join(content_lines),
        metadata={"group_name": group_name, "sample_size": len(summaries)},
    )


def _position_snapshot_tool(execution_context: AgentToolExecutionContext) -> AgentToolResult:
    cached = _get_cached_tool_result(execution_context, "position_snapshot")
    if cached is not None:
        return cached

    broker = execution_context.runtime_context.broker
    if not broker.connected:
        result = AgentToolResult(
            tool_name="position_snapshot",
            title="账户持仓摘要",
            summary="当前未连接券商账户",
            content="- 当前账户未连接，无法获取持仓信息。",
        )
        return _store_cached_tool_result(execution_context, "position_snapshot", result)

    lines = [
        f"- 账户ID: {broker.account_id or '-'}",
        f"- 可用资金: {broker.available_cash:,.0f}",
        f"- 总资产: {broker.total_asset:,.0f}",
        f"- 持仓数量: {broker.position_count}",
        "",
    ]
    if broker.top_positions:
        lines.extend([
            "| 代码 | 持仓数量 | 成本价 | 市值 |",
            "| --- | --- | --- | --- |",
        ])
        for pos in broker.top_positions:
            lines.append(
                "| {code} | {volume} | {cost:.3f} | {mv:,.0f} |".format(
                    code=pos.get("code", "-"),
                    volume=int(pos.get("volume", 0) or 0),
                    cost=float(pos.get("cost_price", 0.0) or 0.0),
                    mv=float(pos.get("market_value", 0.0) or 0.0),
                )
            )
    else:
        lines.append("- 当前无持仓明细。")

    result = AgentToolResult(
        tool_name="position_snapshot",
        title="账户持仓摘要",
        summary=f"账户已连接，持仓 {broker.position_count} 只",
        content="\n".join(lines),
        metadata={"position_count": broker.position_count},
    )
    return _store_cached_tool_result(execution_context, "position_snapshot", result)


def _compare_symbols_tool(
    execution_context: AgentToolExecutionContext,
    codes: List[str] | None = None,
) -> AgentToolResult:
    target_codes = codes or extract_symbol_codes(execution_context.user_text)
    unique_codes: List[str] = []
    for code in target_codes:
        if code not in unique_codes:
            unique_codes.append(code)
    if len(unique_codes) < 2:
        return AgentToolResult(
            tool_name="compare_symbols",
            title="多标的对比摘要",
            summary="缺少足够的标的代码",
            content="- 至少需要 2 个六位代码，才能生成对比摘要。",
        )

    rows = []
    for code in unique_codes[:4]:
        df = _load_symbol_df(code, execution_context.raw_context)
        if df is None or df.empty:
            continue
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        volume = pd.to_numeric(df.get("volume"), errors="coerce").dropna() if "volume" in df.columns else pd.Series(dtype=float)
        latest = float(close.iloc[-1])
        pct20 = _period_pct(close, 20)
        pct60 = _period_pct(close, 60)
        ma20 = float(close.tail(20).mean())
        ma60 = float(close.tail(min(60, len(close))).mean())
        vol_ratio = 0.0
        if not volume.empty:
            vol20 = float(volume.tail(min(20, len(volume))).mean())
            vol5 = float(volume.tail(min(5, len(volume))).mean())
            vol_ratio = vol5 / vol20 if vol20 > 0 else 0.0
        rows.append({
            "code": code,
            "latest": latest,
            "pct20": pct20,
            "pct60": pct60,
            "ma_structure": "偏强" if latest > ma20 > ma60 else "偏弱" if latest < ma20 < ma60 else "震荡",
            "vol_ratio": vol_ratio,
        })

    if len(rows) < 2:
        return AgentToolResult(
            tool_name="compare_symbols",
            title="多标的对比摘要",
            summary="可用行情数据不足",
            content="- 目标标的中可用数据不足，无法生成可靠对比。",
        )

    sorted_rows = sorted(rows, key=lambda item: (item["pct60"], item["pct20"]), reverse=True)
    content_lines = [
        "| 代码 | 最新价 | 20日% | 60日% | 均线结构 | 量比5/20 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in rows:
        content_lines.append(
            "| {code} | {latest:.2f} | {pct20:.1f} | {pct60:.1f} | {ma_structure} | {vol_ratio:.2f} |".format(**item)
        )
    content_lines.extend([
        "",
        f"- 相对最强: {sorted_rows[0]['code']}",
        f"- 相对最弱: {sorted_rows[-1]['code']}",
    ])
    return AgentToolResult(
        tool_name="compare_symbols",
        title="多标的对比摘要",
        summary=f"已对比 {len(rows)} 个标的，最强 {sorted_rows[0]['code']}",
        content="\n".join(content_lines),
        metadata={"codes": [item["code"] for item in rows]},
    )


def _load_symbol_df(
    code: str,
    raw_context: Dict[str, Any],
    asset_type: str | None = None,
) -> pd.DataFrame | None:
    data_dir = raw_context.get("data_dir", "")
    inferred_asset_type = asset_type or _infer_asset_type(code, raw_context)
    run_context = _get_run_context(raw_context)
    df, _ = load_symbol_view(
        code,
        asset_type=inferred_asset_type,
        data_dir=data_dir,
        run_context=run_context,
        use_cache=True,
    )
    if df is not None and not df.empty:
        return df
    if _try_fetch_missing_symbol_history(code, raw_context, inferred_asset_type):
        df, _ = load_symbol_view(
            code,
            asset_type=inferred_asset_type,
            data_dir=data_dir,
            run_context=run_context,
            use_cache=False,
        )
        if df is not None and not df.empty:
            return df
    return df


def _get_current_symbol_dataframe(
    execution_context: AgentToolExecutionContext,
    code: str,
    asset_type: str,
) -> pd.DataFrame | None:
    df_hook = _get_tool_hook(execution_context.raw_context, "get_current_symbol_df")
    if callable(df_hook):
        try:
            df = df_hook(code, asset_type)
            if df is not None and not df.empty:
                return df.copy()
        except TypeError:
            try:
                df = df_hook()
                if df is not None and not df.empty:
                    return df.copy()
            except Exception:
                pass
        except Exception:
            pass
    return _load_symbol_df(code, execution_context.raw_context, asset_type=asset_type)


def _try_fetch_missing_symbol_history(
    code: str,
    raw_context: Dict[str, Any],
    asset_type: str,
) -> bool:
    market_data = raw_context.get("market_data", {}) or {}
    data_source = str(market_data.get("data_source", "xtquant") or "xtquant").strip().lower()
    data_dir = str(raw_context.get("data_dir", "") or "").strip()
    plain_code = str(code or "").split(".", 1)[0].strip().upper()
    if not plain_code or not data_dir or data_source != "xtquant":
        return False
    out_dir = Path(data_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    try:
        from scripts import fetch_kline_xtquant
    except Exception as exc:
        logger.warning("按需补拉 K 线失败，无法导入 xtquant 抓取模块: %s", exc)
        return False
    try:
        if not fetch_kline_xtquant.check_xtquant_available():
            return False
        connected, msg = fetch_kline_xtquant.check_connection()
        if not connected:
            logger.warning("按需补拉 K 线失败，miniQMT 未连接: %s", msg)
            return False
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=540)).strftime("%Y%m%d")
        if asset_type == "ETF":
            fetch_kline_xtquant.fetch_etf_one(plain_code, start, end, out_dir, "1d")
        else:
            fetch_kline_xtquant.fetch_one(plain_code, start, end, out_dir, "1d")
        logger.info("已按需补拉 %s 的日线数据，用于 AI 技术分析", plain_code)
        return True
    except Exception as exc:
        logger.warning("按需补拉 %s 的日线数据失败: %s", plain_code, exc)
        return False


def _get_tool_hook(raw_context: Dict[str, Any], hook_name: str):
    hooks = raw_context.get("_agent_tool_hooks", {}) or {}
    return hooks.get(hook_name)


def _infer_asset_type(code: str, raw_context: Dict[str, Any]) -> str:
    symbol = raw_context.get("symbol", {}) or {}
    if symbol.get("code") == code and symbol.get("asset_type") == "ETF":
        return "ETF"
    watchlist = raw_context.get("watchlist", {}) or {}
    for item in list(watchlist.get("visible_items", []) or []):
        if item.get("code") == code:
            return item.get("asset_type", "股票")
    if code.startswith(("51", "52", "56", "58", "15", "16", "18")):
        return "ETF"
    return "股票"


def _get_run_context(raw_context: Dict[str, Any]) -> DecisionRunContext:
    payload = raw_context.get("decision_run_context", {}) or {}
    if payload:
        return DecisionRunContext.from_dict(payload)
    return build_decision_run_context(prefer_realtime=True)


def _build_symbol_summary(
    df: pd.DataFrame,
    symbol_code: str,
    symbol_name: str,
    asset_type: str,
) -> Dict[str, str]:
    df = df.copy().sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        return {
            "summary": "收盘价数据为空",
            "content": "- 该标的缺少有效收盘价数据。",
        }

    volume = pd.to_numeric(df.get("volume"), errors="coerce").dropna() if "volume" in df.columns else pd.Series(dtype=float)
    latest_close = float(close.iloc[-1])
    pct5 = _period_pct(close, 5)
    pct20 = _period_pct(close, 20)
    pct60 = _period_pct(close, 60)
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(min(60, len(close))).mean())
    ma120 = float(close.tail(min(120, len(close))).mean())
    high60 = float(pd.to_numeric(df["high"], errors="coerce").tail(min(60, len(df))).max())
    low60 = float(pd.to_numeric(df["low"], errors="coerce").tail(min(60, len(df))).min())
    vol_ratio = 0.0
    if not volume.empty:
        vol20 = float(volume.tail(min(20, len(volume))).mean())
        vol5 = float(volume.tail(min(5, len(volume))).mean())
        vol_ratio = vol5 / vol20 if vol20 > 0 else 0.0
    dif, dea, hist = _calc_macd(close)
    k_val, d_val, j_val = _calc_kdj(df)
    trend = "偏强" if latest_close > ma20 > ma60 else "偏弱" if latest_close < ma20 < ma60 else "震荡"
    content_lines = [
        f"- 标的: {symbol_name}({symbol_code}) / {asset_type}",
        f"- 最新价格: {latest_close:.2f}",
        f"- 近5日涨跌幅: {pct5:.2f}%",
        f"- 近20日涨跌幅: {pct20:.2f}%",
        f"- 近60日涨跌幅: {pct60:.2f}%",
        f"- 均线结构: MA20={ma20:.2f}, MA60={ma60:.2f}, MA120={ma120:.2f}",
        f"- 60日区间: {low60:.2f} ~ {high60:.2f}",
        f"- 量比(5/20): {vol_ratio:.2f}",
        f"- MACD: DIF={dif:.3f}, DEA={dea:.3f}, HIST={hist:.3f}",
        f"- KDJ: K={k_val:.1f}, D={d_val:.1f}, J={j_val:.1f}",
        f"- 趋势判断: {trend}",
    ]
    return {
        "summary": f"{symbol_code} 当前 {trend}，20日涨跌 {pct20:.1f}%，量比 {vol_ratio:.2f}",
        "content": "\n".join(content_lines),
    }


def _period_pct(close: pd.Series, period: int) -> float:
    if close.empty:
        return 0.0
    base_index = 0 if len(close) <= period else -period - 1
    base = float(close.iloc[base_index])
    latest = float(close.iloc[-1])
    return (latest - base) / base * 100 if base > 0 else 0.0


def _calc_macd(close: pd.Series) -> tuple[float, float, float]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    return float(dif.iloc[-1]), float(dea.iloc[-1]), float(hist.iloc[-1])


def _calc_kdj(df: pd.DataFrame) -> tuple[float, float, float]:
    low_n = pd.to_numeric(df["low"], errors="coerce").rolling(window=9, min_periods=1).min()
    high_n = pd.to_numeric(df["high"], errors="coerce").rolling(window=9, min_periods=1).max()
    close = pd.to_numeric(df["close"], errors="coerce")
    rsv = (close - low_n) / (high_n - low_n).replace(0, pd.NA) * 100
    rsv = rsv.fillna(50.0)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return float(k.iloc[-1]), float(d.iloc[-1]), float(j.iloc[-1])


def _market_context_snapshot_tool(
    execution_context: AgentToolExecutionContext,
) -> AgentToolResult:
    cached = _get_cached_tool_result(execution_context, "market_context_snapshot")
    if cached is not None:
        return cached

    svc = MarketContextService()
    snap = svc.build_snapshot(run_context=_get_run_context(execution_context.raw_context))
    lines = snap.to_prompt_lines()
    content = "\n".join(lines) if lines else "- 暂无可用的大盘环境数据"
    idx_count = len(snap.indices)
    result = AgentToolResult(
        tool_name="market_context_snapshot",
        title="大盘环境快照",
        summary=f"已采集 {idx_count} 个指数, 情绪: {snap.sentiment_tag or '未知'}",
        content=content,
        metadata={
            "date": snap.date,
            "sentiment": snap.sentiment_tag,
            "breadth_up": snap.breadth_up,
            "breadth_down": snap.breadth_down,
            "index_count": idx_count,
        },
    )
    return _store_cached_tool_result(execution_context, "market_context_snapshot", result)


def _portfolio_industry_exposure_tool(
    execution_context: AgentToolExecutionContext,
) -> AgentToolResult:
    cached = _get_cached_tool_result(execution_context, "portfolio_industry_exposure")
    if cached is not None:
        return cached

    broker = execution_context.runtime_context.broker
    svc = PortfolioRiskService()
    summary = svc.get_portfolio_industry_summary(broker)

    if not summary:
        result = AgentToolResult(
            tool_name="portfolio_industry_exposure",
            title="持仓行业暴露",
            summary="账户未连接或无持仓",
            content="- 当前无法获取持仓行业分布信息。",
        )
        return _store_cached_tool_result(execution_context, "portfolio_industry_exposure", result)

    lines = [f"持仓行业暴露 (总市值 ¥{summary['total_market_value']:,.0f}):"]
    industries = summary.get("industries", {})
    for ind, info in industries.items():
        lines.append(
            f"  {ind}: ¥{info['market_value']:,.0f} ({info['pct']:.1f}%) "
            f"[{info['count']}只: {', '.join(info['codes'][:4])}]"
        )

    result = AgentToolResult(
        tool_name="portfolio_industry_exposure",
        title="持仓行业暴露",
        summary=f"{len(industries)} 个行业, 总市值 ¥{summary['total_market_value']:,.0f}",
        content="\n".join(lines),
        metadata={"industry_count": len(industries)},
    )
    return _store_cached_tool_result(execution_context, "portfolio_industry_exposure", result)
