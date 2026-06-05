"""Smoketest for Step C: AI stock policy declarative schema + UI panel reuse.

Covers:
  1. ``AIStockRiskPolicy.config_schema()`` exposes all fields declared on
     :data:`RiskGuardService.DEFAULT_CONFIG` (no silent drift).
  2. ``get_config()`` initial snapshot equals defaults.
  3. ``apply_config()`` persists updates to the backing JSON file and the
     in-memory ``RiskGuardService.config``, including display-scale unit
     conversion (percent <-> ratio).
  4. After persistence, a freshly constructed ``RiskGuardService`` on the same
     path reads the saved values back.
  5. ``RiskGuardService.evaluate`` honors the new ``min_confidence`` threshold
     (blocks an order that previously would have passed).
  6. ``StrategyRiskSettingsPanel`` renders the AI policy and round-trips a save.

Run via::

    conda run -n stock --no-capture-output python scripts/ai_risk_schema_smoketest.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# trading_app/widgets/__init__.py eager-loads widgets that use bare `widgets.*`
# imports (see timeshare_widget.py), so expose the package dir on sys.path too.

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QCheckBox  # noqa: E402

from common.agent.agent_context_service import BrokerContext  # noqa: E402
from trading_app.services.ai.ai_stock_risk_policy import AIStockRiskPolicy  # noqa: E402
from trading_app.services.execution.risk_guard_service import (  # noqa: E402
    DEFAULT_CONFIG as RG_DEFAULTS,
    RiskGuardService,
)
from trading_app.services.strategy.strategy_risk import (  # noqa: E402
    RiskConfigField,
    is_configurable,
)
from trading_app.services.ai.trade_decision_models import (  # noqa: E402
    TradeAction,
    TradeDecision,
)
from trading_app.widgets.strategy_risk_settings_panel import (  # noqa: E402
    StrategyRiskSettingsPanel,
)


def _make_decision(symbol: str = "600519.SH", confidence: float = 0.8) -> TradeDecision:
    return TradeDecision(
        action=TradeAction.BUY.value,
        symbol_code=symbol,
        symbol_name="贵州茅台",
        confidence=confidence,
        current_price=100.0,
        stop_loss_price=95.0,
        target_price=120.0,
        position_pct=0.05,
        risk_score=0.30,
        reasoning="smoketest",
    )


def test_schema_covers_all_defaults() -> None:
    policy = AIStockRiskPolicy(risk_guard=RiskGuardService(config_path=Path(tempfile.mktemp(suffix=".json"))))
    schema = policy.config_schema()
    assert schema, "schema should not be empty"
    assert all(isinstance(f, RiskConfigField) for f in schema)

    schema_names = {f.name for f in schema}
    default_names = set(RG_DEFAULTS.keys())
    missing = default_names - schema_names
    assert not missing, f"schema missing fields from RiskGuardService defaults: {missing}"
    extra = schema_names - default_names
    assert not extra, f"schema declares unknown fields: {extra}"
    print(f"[PASS] schema covers all {len(default_names)} risk-guard fields")


def test_get_config_returns_defaults() -> None:
    tmp_path = Path(tempfile.mktemp(suffix=".json"))
    policy = AIStockRiskPolicy(risk_guard=RiskGuardService(config_path=tmp_path))
    snap = policy.get_config()
    for key, default in RG_DEFAULTS.items():
        assert snap[key] == default, f"{key}: expected {default}, got {snap[key]}"
    print("[PASS] get_config() initial snapshot matches defaults")


def test_apply_config_persists_and_reloads() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "risk_guard_config.json"
        policy = AIStockRiskPolicy(risk_guard=RiskGuardService(config_path=cfg_path))

        # display units: percents for *_pct / min_confidence fields
        policy.apply_config(
            {
                "min_confidence": 75.0,
                "max_stop_loss_pct": 8.0,
                "block_st_stocks": True,
                "warn_st_stocks": False,
            }
        )

        # In-memory values updated
        cfg = policy.risk_guard.config
        assert abs(cfg["min_confidence"] - 0.75) < 1e-9
        assert abs(cfg["max_stop_loss_pct"] - 0.08) < 1e-9
        assert cfg["block_st_stocks"] is True
        assert cfg["warn_st_stocks"] is False

        # File persisted
        assert cfg_path.exists(), "config file should be created"
        with open(cfg_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert abs(saved["min_confidence"] - 0.75) < 1e-9
        assert saved["block_st_stocks"] is True

        # Fresh service reads it back
        reborn = RiskGuardService(config_path=cfg_path)
        assert abs(reborn.config["min_confidence"] - 0.75) < 1e-9
        assert reborn.config["block_st_stocks"] is True
    print("[PASS] apply_config() persists + reload works")


def test_evaluate_honors_updated_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "risk_guard_config.json"
        policy = AIStockRiskPolicy(risk_guard=RiskGuardService(config_path=cfg_path))

        # Default threshold 0.6: confidence 0.65 should pass
        decision = _make_decision(confidence=0.65)
        before = policy.risk_guard.evaluate(decision, broker=None)
        assert before.passed, "expected pass before threshold change"

        # Raise threshold to 80% (display units) -> same order should be blocked
        policy.apply_config({"min_confidence": 80.0})
        after = policy.risk_guard.evaluate(decision, broker=None)
        assert not after.passed, "expected block after threshold raised"
        assert any("置信度" in msg for msg in after.blocked_reasons)
    print("[PASS] evaluate() picks up updated min_confidence in real time")


def test_display_scale_roundtrip() -> None:
    """UI 提交的 '10.0 %' 应转回 0.10 存入。"""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "risk_guard_config.json"
        policy = AIStockRiskPolicy(risk_guard=RiskGuardService(config_path=cfg_path))

        schema = {f.name: f for f in policy.config_schema()}
        field = schema["max_stop_loss_pct"]
        assert field.display_scale == 100.0

        # What the UI would have (display units: percent)
        display_value = field.to_display(policy.get_config()["max_stop_loss_pct"])
        assert abs(display_value - 10.0) < 1e-9

        # User bumps it to 12.5% in the UI -> we pass the raw display number to apply_config
        policy.apply_config({"max_stop_loss_pct": 12.5})
        assert abs(policy.risk_guard.config["max_stop_loss_pct"] - 0.125) < 1e-9
    print("[PASS] display_scale conversion preserved through apply_config()")


def test_panel_roundtrip(app: QApplication) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "risk_guard_config.json"
        policy = AIStockRiskPolicy(risk_guard=RiskGuardService(config_path=cfg_path))
        assert is_configurable(policy), "AIStockRiskPolicy must be configurable"

        panel = StrategyRiskSettingsPanel(policy=policy, title="AI 风控 (测试)")
        panel.reload()

        # Panel should render one widget per schema field
        schema = policy.config_schema()
        assert len(panel._widgets) == len(schema), (
            f"panel rendered {len(panel._widgets)} widgets for {len(schema)} fields"
        )

        # Flip block_st_stocks via the checkbox widget, then trigger save
        cb = panel._widgets["block_st_stocks"]
        assert isinstance(cb, QCheckBox)
        cb.setChecked(True)
        panel._on_save()
        assert policy.risk_guard.config["block_st_stocks"] is True
    print("[PASS] StrategyRiskSettingsPanel round-trips AI policy config")


def test_panel_preserves_display_scale_on_save(app: QApplication) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "risk_guard_config.json"
        policy = AIStockRiskPolicy(risk_guard=RiskGuardService(config_path=cfg_path))
        panel = StrategyRiskSettingsPanel(policy=policy, title="AI 风控 (百分比测试)")
        panel.reload()

        min_conf = panel._widgets["min_confidence"]
        stop_loss = panel._widgets["max_stop_loss_pct"]
        assert hasattr(min_conf, "setValue")
        assert hasattr(stop_loss, "setValue")

        min_conf.setValue(60.0)
        stop_loss.setValue(10.0)
        panel._on_save()

        assert abs(policy.risk_guard.config["min_confidence"] - 0.60) < 1e-9
        assert abs(policy.risk_guard.config["max_stop_loss_pct"] - 0.10) < 1e-9

        panel.reload()
        assert abs(min_conf.value() - 60.0) < 1e-9
        assert abs(stop_loss.value() - 10.0) < 1e-9
    print("[PASS] StrategyRiskSettingsPanel preserves percentage display_scale on save")


def test_unknown_keys_are_ignored() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "risk_guard_config.json"
        policy = AIStockRiskPolicy(risk_guard=RiskGuardService(config_path=cfg_path))
        # Unknown key + a known key -> only known should persist
        policy.apply_config({"nonexistent_field": 123, "min_confidence": 77.0})
        assert abs(policy.risk_guard.config["min_confidence"] - 0.77) < 1e-9
        assert "nonexistent_field" not in policy.risk_guard.config
    print("[PASS] update_config ignores unknown fields")


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    test_schema_covers_all_defaults()
    test_get_config_returns_defaults()
    test_apply_config_persists_and_reloads()
    test_evaluate_honors_updated_threshold()
    test_display_scale_roundtrip()
    test_panel_roundtrip(app)
    test_panel_preserves_display_scale_on_save(app)
    test_unknown_keys_are_ignored()
    print("\nAll AI risk schema smoketests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
