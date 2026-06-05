"""Headless smoke test for :class:`LiveStrategyAccountSettingsDialog`.

不弹出窗口，只验证:
1. 对话框与 AutoTradeConfigService 的字段映射正确（加载 -> 显示）
2. 保存后 AutoTradeConfig JSON 被正确写回
3. "恢复默认" 按钮确实把控件值重置为 ``AutoTradeConfig()`` 的默认

Run::
    python scripts/account_settings_dialog_smoketest.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication

from trading_app.services.ai.auto_trade_config_service import (
    AutoTradeConfig,
    AutoTradeConfigService,
)
from trading_app.widgets.live_strategy_account_settings_dialog import (
    LiveStrategyAccountSettingsDialog,
)


def _make_service(tmp_path: Path) -> AutoTradeConfigService:
    cfg_path = tmp_path / "auto_trade_config.json"
    cfg_path.write_text(
        json.dumps(AutoTradeConfig().to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    return AutoTradeConfigService(config_path=cfg_path)


def case_load_reflects_service_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _make_service(Path(tmp))
        svc.update_config(
            manual_orders_enabled=False,
            require_trading_time=False,
            duplicate_window_seconds=7,
            status_poll_seconds=4.5,
            status_poll_interval_seconds=0.7,
        )
        dialog = LiveStrategyAccountSettingsDialog(service=svc)
        assert dialog.chk_manual.isChecked() is False
        assert dialog.chk_trading_time.isChecked() is False
        assert dialog.spin_dup_window.value() == 7
        assert abs(dialog.spin_poll_total.value() - 4.5) < 1e-6
        assert abs(dialog.spin_poll_step.value() - 0.7) < 1e-6
        print("[load_reflects_service_state] OK")


def case_accept_writes_back() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _make_service(Path(tmp))
        dialog = LiveStrategyAccountSettingsDialog(service=svc)
        dialog.chk_manual.setChecked(False)
        dialog.chk_trading_time.setChecked(False)
        dialog.spin_dup_window.setValue(60)
        dialog.spin_poll_total.setValue(3.0)
        dialog.spin_poll_step.setValue(0.5)
        dialog._on_accept()
        cfg = svc.get_config()
        assert cfg.manual_orders_enabled is False
        assert cfg.require_trading_time is False
        assert cfg.duplicate_window_seconds == 60
        assert abs(cfg.status_poll_seconds - 3.0) < 1e-6
        assert abs(cfg.status_poll_interval_seconds - 0.5) < 1e-6
        print("[accept_writes_back] OK")


def case_restore_defaults_resets_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _make_service(Path(tmp))
        svc.update_config(
            manual_orders_enabled=False,
            duplicate_window_seconds=500,
            status_poll_seconds=90.0,
        )
        dialog = LiveStrategyAccountSettingsDialog(service=svc)
        dialog._on_restore_defaults()
        d = AutoTradeConfig()
        assert dialog.chk_manual.isChecked() == d.manual_orders_enabled
        assert dialog.chk_trading_time.isChecked() == d.require_trading_time
        assert dialog.spin_dup_window.value() == d.duplicate_window_seconds
        assert abs(dialog.spin_poll_total.value() - d.status_poll_seconds) < 1e-6
        assert abs(dialog.spin_poll_step.value() - d.status_poll_interval_seconds) < 1e-6
        print("[restore_defaults_resets_fields] OK")


def case_poll_interval_larger_than_total_blocked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _make_service(Path(tmp))
        dialog = LiveStrategyAccountSettingsDialog(service=svc)
        dialog.spin_poll_total.setValue(1.0)
        dialog.spin_poll_step.setValue(5.0)

        # 把 QMessageBox 替换掉，避免弹窗阻塞 headless 测试
        from PyQt6.QtWidgets import QMessageBox

        captured: list[str] = []

        def _capture(*args, **_kwargs):
            captured.append(str(args[2]) if len(args) >= 3 else "")
            return QMessageBox.StandardButton.Ok

        original = QMessageBox.warning
        QMessageBox.warning = staticmethod(_capture)  # type: ignore[assignment]
        try:
            dialog._on_accept()
        finally:
            QMessageBox.warning = original  # type: ignore[assignment]

        assert dialog.result() != dialog.DialogCode.Accepted
        assert captured and "间隔" in captured[0]
        print("[poll_interval_guard] OK")


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    _ = app  # keep reference alive in headless mode
    case_load_reflects_service_state()
    case_accept_writes_back()
    case_restore_defaults_resets_fields()
    case_poll_interval_larger_than_total_blocked()
    print("ALL_PASSED")


if __name__ == "__main__":
    main()
