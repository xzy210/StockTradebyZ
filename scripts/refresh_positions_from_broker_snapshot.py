"""根据 miniqmt 实盘持仓截图刷新主账本 avg_cost。

背景：截图（2026-04-17 盘后）显示的成本价是券商侧"动态摊销成本"，和我们
主账本里的加权均价在两种情况下不一致：
  1. 费用未并入：我们的 avg_cost 只是成交价的加权均价，不含佣金/印花税，
     而券商端把费用摊进成本。对于近几笔买入的股票，差异 ≤ 0.05 元/股。
  2. 有卖出后再持有：券商端会把已实现盈亏从剩余成本里扣除（动态摊销），
     而我们用加权均价——"成本"保留为"实际买入均价"。

决策：
  - 对单纯"差几分钱"的股票（仅费用未并入），按截图刷新，让浮盈跟券商对齐。
  - 对有卖出历史导致的算法差异（目前只有 坤彩科技 603826），维持我们的
    加权均价，不跟随 miniqmt 的动态摊销口径。原因：
      * 我们的"已实现盈亏 + 浮盈"汇总值和 miniqmt 的"浮盈"是一致的，
        只是切分方式不同；
      * 加权均价更直观（看到的成本就是当前持仓的买入均价）。

执行：先关掉交易主程序（软件运行期间 state 可能被内存里的值覆盖），
然后在 conda `stock` 环境下运行本脚本。
"""
from __future__ import annotations

import sys
from pathlib import Path


# 截图成本价（None = 保持不变）
# 格式: { strategy_id: { stock_code: new_avg_cost or None } }
UPDATES: dict[str, dict[str, float | None]] = {
    "ai_trade_decision_center": {
        "000155": 16.13,  # 川能动力
        "000703": 15.46,  # 恒逸石化
        "600867": 9.94,   # 通化东宝
        "603826": None,   # 坤彩科技：维持 28.75（加权均价），不跟随 miniqmt 动态成本
    },
    "etf_rotation": {
        "159949": 1.641,  # 创业板50ETF
    },
    # unmanaged 的 avg_cost 每次对账都会被 reconcile_unmanaged_with_broker
    # 用券商 open_price 覆盖（和截图一致），所以这里不用显式改
}


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    from trading_app.services.strategy.strategy_budget_service import get_strategy_budget_service

    svc = get_strategy_budget_service()

    changes: list[tuple[str, str, float, float]] = []
    for strategy_id, code_map in UPDATES.items():
        state = svc.get_strategy_state_record(strategy_id)
        if state is None:
            print(f"[skip] 策略 {strategy_id} 不存在")
            continue
        raw_positions = state.positions or {}
        for code, new_cost in code_map.items():
            if new_cost is None:
                continue
            pos_dict = raw_positions.get(code)
            if not pos_dict or int(pos_dict.get("quantity", 0) or 0) <= 0:
                print(f"[skip] 策略 {strategy_id} 未持有 {code}")
                continue
            old_cost = float(pos_dict.get("avg_cost", 0.0) or 0.0)
            if abs(old_cost - float(new_cost)) < 1e-6:
                print(f"[same] {strategy_id} {code} 成本已一致 {old_cost}")
                continue
            pos_dict["avg_cost"] = float(new_cost)
            changes.append((strategy_id, code, old_cost, float(new_cost)))
        svc.save_strategy_state_record(state)

    if not changes:
        print("无需更新的持仓成本")
        return 0

    print("\n已更新持仓成本：")
    for strategy_id, code, old_cost, new_cost in changes:
        diff = new_cost - old_cost
        print(f"  {strategy_id:<32s} {code}  {old_cost:.4f} -> {new_cost:.4f}  (Δ {diff:+.4f})")
    print(f"\n共更新 {len(changes)} 条记录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
