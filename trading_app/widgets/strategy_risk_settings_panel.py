"""按 ``RiskConfigField`` schema 自动渲染的策略风控设置面板。

理想态的第一次落地：policy 只要实现 :class:`ConfigurableStrategyRiskPolicy`
协议（``config_schema / get_config / apply_config``），UI 无需再为每个字段
手写控件。ETF 轮动实盘是首个消费者，后续 AI / grid / pair 可直接复用。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QTime

from trading_app.services.strategy.strategy_risk import (
    RiskConfigField,
    is_configurable,
)

logger = logging.getLogger(__name__)


class StrategyRiskSettingsPanel(QGroupBox):
    """Auto-rendered settings form driven by a policy's declarative schema.

    Usage::

        panel = StrategyRiskSettingsPanel(policy=policy)
        layout.addWidget(panel)

    ``policy`` 需要满足 :func:`is_configurable` 判定（提供 ``config_schema /
    get_config / apply_config``）。不满足时面板会退化为一条说明文本，不抛异常，
    方便逐步推广到新策略。
    """

    config_saved = pyqtSignal(dict)

    def __init__(
        self,
        policy: Any,
        *,
        title: str = "策略风控（网关统一）",
        parent: Optional[QWidget] = None,
        reload_on_save: bool = True,
    ) -> None:
        super().__init__(title, parent)
        self._policy = policy
        self._reload_on_save = reload_on_save
        self._widgets: Dict[str, QWidget] = {}
        self._fields: List[RiskConfigField] = []
        self._build_ui()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read ``policy.get_config()`` and refresh all widgets."""
        if not self._fields:
            return
        try:
            current = dict(self._policy.get_config() or {})
        except Exception as exc:
            logger.error("reload 读取 policy 配置失败: %s", exc, exc_info=True)
            return
        for field in self._fields:
            self._write_widget(field, current.get(field.name, field.default))
        self._refresh_dependency_state()

    def collect_values(self) -> Dict[str, Any]:
        """Gather current UI values in display units.

        ``ConfigurableStrategyRiskPolicy.apply_config()`` owns the final
        display->storage conversion via ``RiskConfigField.from_display``.
        The panel therefore forwards raw widget values to avoid double scaling
        for percentage-like fields.
        """
        values: Dict[str, Any] = {}
        for field in self._fields:
            widget = self._widgets.get(field.name)
            if widget is None:
                continue
            values[field.name] = self._read_widget(field, widget)
        return values

    # ------------------------------------------------------------------
    #  UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        if not is_configurable(self._policy):
            note = QLabel("当前策略 policy 未声明 config_schema，未启用声明式风控面板。")
            note.setStyleSheet("color:#94A3B8;font-size:11px;")
            note.setWordWrap(True)
            root.addWidget(note)
            return

        try:
            self._fields = list(self._policy.config_schema() or [])
            current = dict(self._policy.get_config() or {})
        except Exception as exc:
            logger.error("加载 policy schema 失败: %s", exc, exc_info=True)
            note = QLabel(f"加载策略风控配置失败: {exc}")
            note.setStyleSheet("color:#d9534f;")
            root.addWidget(note)
            return

        if not self._fields:
            note = QLabel("本策略当前没有可配置的风控字段。")
            note.setStyleSheet("color:#94A3B8;font-size:11px;")
            root.addWidget(note)
            return

        form = QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(form)

        for field in self._fields:
            widget = self._make_widget(field, current.get(field.name, field.default))
            self._widgets[field.name] = widget
            label = QLabel(f"{field.label}:")
            if field.help:
                label.setToolTip(field.help)
                widget.setToolTip(field.help)
            form.addRow(label, widget)

        # depends_on wiring
        for field in self._fields:
            if field.depends_on and field.depends_on in self._widgets:
                driver = self._widgets[field.depends_on]
                if isinstance(driver, QCheckBox):
                    driver.stateChanged.connect(self._refresh_dependency_state)
        self._refresh_dependency_state()

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#64748B;font-size:11px;")
        button_row.addWidget(self.status_label, 1)

        self.btn_restore = QPushButton("恢复默认")
        self.btn_restore.clicked.connect(self._on_restore_defaults)
        button_row.addWidget(self.btn_restore)

        self.btn_save = QPushButton("保存策略风控")
        self.btn_save.setStyleSheet(
            "QPushButton{background:#3B82F6;color:white;padding:4px 10px;"
            "border-radius:4px;}"
            "QPushButton:hover{background:#2563EB;}"
        )
        self.btn_save.clicked.connect(self._on_save)
        button_row.addWidget(self.btn_save)
        root.addLayout(button_row)

        # separator for aesthetics
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#E2E8F0;")
        root.addWidget(sep)

    # ------------------------------------------------------------------
    #  Widget factory / IO
    # ------------------------------------------------------------------

    def _make_widget(self, field: RiskConfigField, current: Any) -> QWidget:
        t = field.type
        if t == "bool":
            w = QCheckBox()
            w.setChecked(bool(current if current is not None else field.default))
            return w
        if t == "int":
            w = QSpinBox()
            if field.min_value is not None:
                w.setMinimum(int(field.min_value))
            else:
                w.setMinimum(-1_000_000)
            if field.max_value is not None:
                w.setMaximum(int(field.max_value))
            else:
                w.setMaximum(1_000_000)
            if field.step is not None:
                w.setSingleStep(int(field.step))
            if field.suffix:
                w.setSuffix(field.suffix)
            value = current if current is not None else field.default
            w.setValue(int(field.to_display(value) or 0))
            return w
        if t == "float":
            w = QDoubleSpinBox()
            if field.min_value is not None:
                w.setMinimum(float(field.min_value))
            else:
                w.setMinimum(-1e9)
            if field.max_value is not None:
                w.setMaximum(float(field.max_value))
            else:
                w.setMaximum(1e9)
            if field.step is not None:
                w.setSingleStep(float(field.step))
            if field.decimals is not None:
                w.setDecimals(int(field.decimals))
            if field.suffix:
                w.setSuffix(field.suffix)
            value = current if current is not None else field.default
            w.setValue(float(field.to_display(value) or 0.0))
            return w
        if t == "time":
            w = QTimeEdit()
            w.setDisplayFormat("HH:mm")
            value = str(current or field.default or "00:00")
            try:
                hh, mm = [int(p) for p in value.split(":")[:2]]
            except ValueError:
                hh, mm = 9, 30
            w.setTime(QTime(hh, mm))
            return w
        if t == "choice":
            w = QComboBox()
            for val, label in (field.choices or ()):
                w.addItem(str(label), val)
            if current is not None:
                idx = w.findData(current)
                if idx >= 0:
                    w.setCurrentIndex(idx)
            return w
        # text / fallback
        w = QLineEdit()
        w.setText(str(current if current is not None else (field.default or "")))
        return w

    def _write_widget(self, field: RiskConfigField, value: Any) -> None:
        widget = self._widgets.get(field.name)
        if widget is None:
            return
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
            return
        if isinstance(widget, QSpinBox):
            widget.setValue(int(field.to_display(value or 0) or 0))
            return
        if isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(field.to_display(value or 0.0) or 0.0))
            return
        if isinstance(widget, QTimeEdit):
            text = str(value or "00:00")
            try:
                hh, mm = [int(p) for p in text.split(":")[:2]]
            except ValueError:
                hh, mm = 9, 30
            widget.setTime(QTime(hh, mm))
            return
        if isinstance(widget, QComboBox):
            idx = widget.findData(value)
            if idx >= 0:
                widget.setCurrentIndex(idx)
            return
        if isinstance(widget, QLineEdit):
            widget.setText(str(value or ""))
            return

    def _read_widget(self, field: RiskConfigField, widget: QWidget) -> Any:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QTimeEdit):
            t = widget.time()
            return f"{t.hour():02d}:{t.minute():02d}"
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return field.default

    # ------------------------------------------------------------------
    #  Dependency enabling
    # ------------------------------------------------------------------

    def _refresh_dependency_state(self) -> None:
        for field in self._fields:
            if not field.depends_on:
                continue
            driver = self._widgets.get(field.depends_on)
            dependent = self._widgets.get(field.name)
            if driver is None or dependent is None:
                continue
            enabled = bool(driver.isChecked()) if isinstance(driver, QCheckBox) else True
            dependent.setEnabled(enabled)

    # ------------------------------------------------------------------
    #  Actions
    # ------------------------------------------------------------------

    def _on_restore_defaults(self) -> None:
        for field in self._fields:
            self._write_widget(field, field.default)
        self._refresh_dependency_state()
        self.status_label.setText("已恢复为 policy 默认值，点击保存后生效")
        self.status_label.setStyleSheet("color:#F59E0B;font-size:11px;")

    def _on_save(self) -> None:
        values = self.collect_values()
        try:
            self._policy.apply_config(values)
        except Exception as exc:
            logger.error("apply_config 失败: %s", exc, exc_info=True)
            QMessageBox.critical(self, "策略风控", f"保存失败: {exc}")
            self.status_label.setText(f"保存失败: {exc}")
            self.status_label.setStyleSheet("color:#d9534f;font-size:11px;")
            return

        self.status_label.setText("已保存")
        self.status_label.setStyleSheet("color:#16A34A;font-size:11px;")
        if self._reload_on_save:
            self.reload()
        self.config_saved.emit(values)
