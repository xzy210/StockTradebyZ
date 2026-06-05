"""Headless smoke test for declarative risk-policy schema + auto-rendered panel.

覆盖:
1. ``is_configurable`` 正确识别声明式 policy / 排除普通 policy
2. ``ETFRotationRiskPolicy.config_schema / get_config`` 返回与
   ``RotationConfig`` 一致的字段 + 默认值
3. ``StrategyRiskSettingsPanel`` 基于 schema 自动渲染所需控件
4. 修改控件 -> 保存 -> 通过 ``config_saver`` 回写 policy 读到的 config
5. depends_on 联动（关闭 ``enable_risk_check`` 时其它控件变灰）
6. 恢复默认 -> 控件值重置

Run::
    conda run -n stock --no-capture-output python scripts/strategy_risk_schema_smoketest.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtCore import QTime
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QTimeEdit,
)

from live_rotation.config import RotationConfig
from live_rotation.rotation_risk_policy import ETFRotationRiskPolicy
from trading_app.services.strategy.strategy_risk import (
    RiskConfigField,
    is_configurable,
)
from trading_app.widgets.strategy_risk_settings_panel import (
    StrategyRiskSettingsPanel,
)


class _PlainPolicy:
    strategy_id = "plain"

    def evaluate(self, request, context):  # pragma: no cover - not invoked
        raise NotImplementedError


def _make_etf_policy() -> tuple[ETFRotationRiskPolicy, RotationConfig, list[dict]]:
    cfg = RotationConfig()
    saved: list[dict] = []

    def saver(values: dict) -> None:
        for k, v in values.items():
            setattr(cfg, k, v)
        saved.append(dict(values))

    policy = ETFRotationRiskPolicy(
        strategy_id="smoketest_etf_schema",
        config_provider=lambda: cfg,
        state_provider=lambda: SimpleNamespace(
            get_trades_today=lambda: 0,
            buy_date="",
            buy_price=0.0,
        ),
        config_saver=saver,
    )
    return policy, cfg, saved


def case_is_configurable_detection() -> None:
    plain = _PlainPolicy()
    assert is_configurable(plain) is False, "普通 policy 不应被识别为 configurable"

    policy, _, _ = _make_etf_policy()
    assert is_configurable(policy) is True, "ETF policy 应为 configurable"
    print("[is_configurable_detection] OK")


def case_schema_matches_rotation_config() -> None:
    policy, cfg, _ = _make_etf_policy()
    schema = policy.config_schema()
    names = {f.name for f in schema}
    expected = {
        "enable_risk_check",
        "trading_start",
        "trading_end",
        "max_trades_per_day",
        "min_hold_days",
        "max_single_loss_pct",
    }
    assert names == expected, f"schema 字段与预期不一致: {names}"

    current = policy.get_config()
    assert current["enable_risk_check"] == cfg.enable_risk_check
    assert current["trading_start"] == cfg.trading_start
    assert current["trading_end"] == cfg.trading_end
    assert current["max_trades_per_day"] == cfg.max_trades_per_day
    assert current["min_hold_days"] == cfg.min_hold_days
    assert abs(current["max_single_loss_pct"] - cfg.max_single_loss_pct) < 1e-6
    print("[schema_matches_rotation_config] OK")


def case_panel_renders_widgets_for_every_field() -> None:
    policy, _, _ = _make_etf_policy()
    panel = StrategyRiskSettingsPanel(policy=policy)

    expected_widget_types = {
        "enable_risk_check": QCheckBox,
        "trading_start": QTimeEdit,
        "trading_end": QTimeEdit,
        "max_trades_per_day": QSpinBox,
        "min_hold_days": QSpinBox,
        "max_single_loss_pct": QDoubleSpinBox,
    }
    for name, wtype in expected_widget_types.items():
        widget = panel._widgets.get(name)  # type: ignore[attr-defined]
        assert isinstance(widget, wtype), (
            f"字段 {name} 渲染的控件类型应为 {wtype.__name__}，实为 {type(widget).__name__}"
        )
    print("[panel_renders_widgets] OK")


def case_save_roundtrips_through_config_saver() -> None:
    policy, cfg, saved = _make_etf_policy()
    panel = StrategyRiskSettingsPanel(policy=policy)

    panel._widgets["enable_risk_check"].setChecked(True)  # type: ignore[attr-defined]
    panel._widgets["trading_start"].setTime(QTime(9, 45))  # type: ignore[attr-defined]
    panel._widgets["trading_end"].setTime(QTime(14, 55))  # type: ignore[attr-defined]
    panel._widgets["max_trades_per_day"].setValue(3)  # type: ignore[attr-defined]
    panel._widgets["min_hold_days"].setValue(4)  # type: ignore[attr-defined]
    panel._widgets["max_single_loss_pct"].setValue(7.5)  # type: ignore[attr-defined]

    panel._on_save()  # type: ignore[attr-defined]

    assert saved, "config_saver 未被调用"
    last = saved[-1]
    assert last["trading_start"] == "09:45"
    assert last["trading_end"] == "14:55"
    assert last["max_trades_per_day"] == 3
    assert last["min_hold_days"] == 4
    assert abs(last["max_single_loss_pct"] - 7.5) < 1e-6

    # The config_saver should have mutated the RotationConfig in place
    assert cfg.trading_start == "09:45"
    assert cfg.max_trades_per_day == 3
    assert abs(cfg.max_single_loss_pct - 7.5) < 1e-6

    # reload 应读回新值
    panel.reload()
    assert panel._widgets["max_trades_per_day"].value() == 3  # type: ignore[attr-defined]
    print("[save_roundtrips_through_config_saver] OK")


def case_depends_on_disables_dependent_fields() -> None:
    policy, _, _ = _make_etf_policy()
    panel = StrategyRiskSettingsPanel(policy=policy)

    chk = panel._widgets["enable_risk_check"]  # type: ignore[attr-defined]
    dep = panel._widgets["max_trades_per_day"]  # type: ignore[attr-defined]
    assert dep.isEnabled() is True

    chk.setChecked(False)
    # Qt 信号在 headless 模式下同步触发 stateChanged
    assert dep.isEnabled() is False, "关闭 enable_risk_check 后依赖字段应禁用"
    chk.setChecked(True)
    assert dep.isEnabled() is True
    print("[depends_on_disables_dependent_fields] OK")


def case_restore_defaults_resets_widgets() -> None:
    policy, _, _ = _make_etf_policy()
    panel = StrategyRiskSettingsPanel(policy=policy)
    panel._widgets["max_trades_per_day"].setValue(9)  # type: ignore[attr-defined]
    panel._widgets["max_single_loss_pct"].setValue(42.5)  # type: ignore[attr-defined]
    panel._on_restore_defaults()  # type: ignore[attr-defined]

    # schema 默认值
    schema = {f.name: f for f in policy.config_schema()}
    assert panel._widgets["max_trades_per_day"].value() == schema["max_trades_per_day"].default  # type: ignore[attr-defined]
    assert abs(
        panel._widgets["max_single_loss_pct"].value()  # type: ignore[attr-defined]
        - schema["max_single_loss_pct"].default
    ) < 1e-6
    print("[restore_defaults_resets_widgets] OK")


def case_non_configurable_policy_shows_placeholder() -> None:
    panel = StrategyRiskSettingsPanel(policy=_PlainPolicy())
    # 非 configurable 时字段应为空，没有控件但不抛异常
    assert panel._fields == []  # type: ignore[attr-defined]
    assert panel._widgets == {}  # type: ignore[attr-defined]
    print("[non_configurable_policy_shows_placeholder] OK")


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    _ = app
    case_is_configurable_detection()
    case_schema_matches_rotation_config()
    case_panel_renders_widgets_for_every_field()
    case_save_roundtrips_through_config_saver()
    case_depends_on_disables_dependent_fields()
    case_restore_defaults_resets_widgets()
    case_non_configurable_policy_shows_placeholder()
    print("ALL_PASSED")


if __name__ == "__main__":
    main()
