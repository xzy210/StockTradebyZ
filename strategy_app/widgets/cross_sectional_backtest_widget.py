import os
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, 
    QTableWidget, QTableWidgetItem, QProgressBar, QLabel, QHeaderView,
    QSplitter, QGroupBox, QDateEdit, QSpinBox, QMessageBox, QTabWidget,
    QSlider, QDialog, QTreeWidget, QTreeWidgetItem, QCheckBox, QScrollArea,
    QDoubleSpinBox, QFormLayout, QGridLayout, QFileDialog, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QDate, QTimer
from PyQt6.QtGui import QColor

# Import through the package path first so strategy modules keep their relative imports.
try:
    from strategy_app.factors import factor_registry
    from strategy_app.strategies import get_all_strategies, get_strategy
    from strategy_app.strategies.cross_sectional_strategy import CrossSectionalStrategy
    from strategy_app.backtest import BacktestConfig, UnifiedBacktestEngine
except ImportError:
    from factors import factor_registry
    from strategies import get_all_strategies, get_strategy
    from strategies.cross_sectional_strategy import CrossSectionalStrategy
    from backtest import BacktestConfig, UnifiedBacktestEngine

from common.data_portal import get_data_portal

class CrossSectionalBacktestThread(QThread):
    """截面选股回测后台线程"""
    progress_updated = pyqtSignal(int, int) # current, total
    finished_signal = pyqtSignal(dict) # result dict
    error_signal = pyqtSignal(str)
    info_signal = pyqtSignal(str) # For status updates

    def __init__(self, strategy_name, start_date, end_date, initial_cash, data_dir, stock_codes=None, selected_factors=None, xgb_params=None):
        super().__init__()
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.data_dir = data_dir
        self.stock_codes = stock_codes  # List of stock codes from selected pool file
        self.selected_factors = selected_factors  # List of selected factor names
        self.xgb_params = xgb_params  # XGBoost strategy parameters

    def run(self):
        try:
            # 1. 获取策略实例
            strategy = get_strategy(self.strategy_name)
            if not strategy:
                self.error_signal.emit("策略未找到")
                return

            if not isinstance(strategy, CrossSectionalStrategy):
                self.error_signal.emit(f"策略 {self.strategy_name} 不是截面选股策略")
                return
            
            # If custom factors are selected, override the strategy's factor_cols in params
            if self.selected_factors and len(self.selected_factors) > 0:
                strategy.params['factor_cols'] = self.selected_factors
                self.info_signal.emit(f"使用自定义因子({len(self.selected_factors)}个): {', '.join(self.selected_factors[:3])}...")
            
            # Apply XGBoost parameters if provided
            if self.xgb_params and self.strategy_name == "xgboost_cross_sectional":
                for key, value in self.xgb_params.items():
                    strategy.params[key] = value
                self.info_signal.emit(f"XGBoost参数: 持仓{self.xgb_params.get('top_k', 5)}只, 调仓周期{self.xgb_params.get('rebalance_period', 20)}日")

            self.info_signal.emit("正在扫描股票池...")
            # Use provided stock codes from the selected pool file
            if self.stock_codes:
                target_codes = self.stock_codes
            else:
                # Fallback to all stocks if no pool specified
                target_codes = get_data_portal().list_symbols(asset_type="stock", data_dir=self.data_dir)
            
            total = len(target_codes)
            data_bundle = get_data_portal().get_market_data_bundle(
                target_codes,
                data_dir=self.data_dir,
                start=self.start_date,
                end=self.end_date,
                asset_type="stock",
            )
            valid_symbols = []
            for i, code in enumerate(target_codes):
                if self.isInterruptionRequested():
                    return
                self.progress_updated.emit(i + 1, total)
                self.info_signal.emit(f"加载数据 ({i+1}/{total}): {code}")
                view = data_bundle.get(code)
                if view is not None and not view.data.empty and len(view.data) > 50:
                    valid_symbols.append(view.symbol)

            if not valid_symbols:
                self.error_signal.emit("未能加载任何有效股票数据")
                return

            data_bundle = get_data_portal().get_market_data_bundle(
                valid_symbols,
                data_dir=self.data_dir,
                start=self.start_date,
                end=self.end_date,
                asset_type="stock",
            )

            self.info_signal.emit(f"数据加载完成，共 {len(data_bundle.data)} 只股票。开始计算因子...")
            
            # 为了支持更丰富的回放（包括评分），我们可以在 on_rebalance 中捕获数据
            # 这里通过 monkey patch 的方式临时注入一个钩子到 strategy 中
            # 或者更优雅的方式是修改 Engine，但为了保持侵入性最小，我们在 strategy 上挂载一个 recorder
            
            history_scores = {} # {date_str: dataframe_with_scores}
            history_train_info = {} # {date_str: train_info} - XGBoost 等 ML 策略的训练信息
            original_on_rebalance = strategy.on_rebalance
            
            def on_rebalance_wrapper(context, valid_codes, daily_factors):
                # 执行原始调仓逻辑
                original_on_rebalance(context, valid_codes, daily_factors)
                
                # 捕获逻辑：从策略的 last_scores 属性中读取评分数据
                if hasattr(strategy, 'last_scores') and strategy.last_scores is not None:
                    # 归一化日期格式为 YYYY-MM-DD
                    dt = context.current_dt
                    if hasattr(dt, 'strftime'):
                        date_str = dt.strftime('%Y-%m-%d')
                    elif hasattr(dt, 'date'):
                        date_str = str(dt.date())
                    else:
                        date_str = str(dt).split(' ')[0].split('T')[0]
                    
                    # 只保留分数高的前 20 名以节省内存
                    top_scored = strategy.last_scores.sort_values('score', ascending=False).head(20).copy()
                    history_scores[date_str] = top_scored
                    
                    # 捕获 ML 策略的训练信息（如特征重要性）
                    if hasattr(strategy, 'last_train_info') and strategy.last_train_info is not None:
                        history_train_info[date_str] = strategy.last_train_info.copy()
                        strategy.last_train_info = None
                    
                    # 重置，避免重复记录
                    strategy.last_scores = None
            
            # 替换方法
            strategy.on_rebalance = on_rebalance_wrapper
            
            engine = UnifiedBacktestEngine(
                BacktestConfig(initial_cash=self.initial_cash, mode="cross_sectional")
            )

            def on_engine_progress(current, total, message=""):
                if total <= 0:
                    return
                self.progress_updated.emit(current, total)
                step = max(1, total // 100)
                if current == 1 or current == total or current % step == 0:
                    self.info_signal.emit(f"回测进度 ({current}/{total}): {message}")

            result = engine.run(
                strategy,
                data_bundle,
                mode="cross_sectional",
                progress_callback=on_engine_progress,
            )
            
            # 将评分历史附加到结果中
            result['history_scores'] = history_scores
            result['history_train_info'] = history_train_info  # ML策略的训练信息
            
            if 'closed_trades' not in result:
                result['closed_trades'] = []
                
            self.finished_signal.emit(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"截面选股回测出错: {str(e)}")

class ReplayWidget(QWidget):
    """回放控制面板"""
    
    stepChanged = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.prev_trade_btn = QPushButton("⏮ 上一调仓")
        self.prev_trade_btn.clicked.connect(self.prev_trade_step)
        layout.addWidget(self.prev_trade_btn)
        
        self.play_btn = QPushButton("▶ 播放")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_btn)
        
        self.next_trade_btn = QPushButton("下一调仓 ⏭")
        self.next_trade_btn.clicked.connect(self.next_trade_step)
        layout.addWidget(self.next_trade_btn)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)
        
        self.date_label = QLabel("----/--/--")
        self.date_label.setFixedWidth(90)
        layout.addWidget(self.date_label)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_step)
        self.timer.setInterval(100) # 100ms per step for smoother playback
        
        self.total_steps = 0
        self.current_step = 0
        self.trade_dates = [] # List of indices where trades happened
        
    def setup(self, total_steps, trade_indices):
        self.total_steps = total_steps
        self.trade_dates = sorted(list(set(trade_indices)))
        self.slider.setRange(0, total_steps - 1)
        self.current_step = 0
        self.slider.setValue(0)
        
    def toggle_play(self, checked):
        if checked:
            self.play_btn.setText("⏸ 暂停")
            self.timer.start()
        else:
            self.play_btn.setText("▶ 播放")
            self.timer.stop()
            
    def next_step(self):
        if self.current_step < self.total_steps - 1:
            self.slider.setValue(self.current_step + 1)
        else:
            self.toggle_play(False)
            self.play_btn.setChecked(False)
            
    def on_slider_changed(self, val):
        self.current_step = val
        self.stepChanged.emit(val)
        
    def next_trade_step(self):
        """Jump to next trade date"""
        for idx in self.trade_dates:
            if idx > self.current_step:
                self.slider.setValue(idx)
                return
        # If no next trade, go to end
        self.slider.setValue(self.total_steps - 1)
        
    def prev_trade_step(self):
        """Jump to previous trade date"""
        for idx in reversed(self.trade_dates):
            if idx < self.current_step:
                self.slider.setValue(idx)
                return
        # If no prev trade, go to start
        self.slider.setValue(0)
        
    def set_date_text(self, text):
        self.date_label.setText(text)

class CrossSectionalBacktestWidget(QWidget):
    """截面选股回测主界面"""
    
    def __init__(self, data_dir="../data"):
        super().__init__()
        self.data_dir = data_dir
        self.backtest_thread = None
        self.backtest_result = None # Store result for replay
        self.stock_name_map = {}
        self.normalized_dates = []  # 归一化后的日期列表
        self.xgb_default_params = self._load_strategy_default_params("xgboost_cross_sectional")
        
        self.setupUI()
        self.load_names()

    def _load_strategy_default_params(self, strategy_id):
        """Load strategy params once so UI defaults stay in sync with the strategy."""
        try:
            strategy = get_strategy(strategy_id)
            params = getattr(strategy, "params", {}) or {}
            defaults = dict(params)
            defaults["xgb_params"] = dict(params.get("xgb_params", {}) or {})
            defaults["factor_cols"] = list(params.get("factor_cols", []) or [])
            return defaults
        except Exception:
            return {}

    def load_names(self):
        self.stock_name_map = get_data_portal().get_name_map(asset_type="stock")

    def _get_stocklist_dir(self):
        """Get the stocklist directory path"""
# Get the project root directory.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        return os.path.join(project_root, "stocklist")
    
    def _load_stock_pools(self):
        """Load available stock pool files from stocklist folder"""
        stocklist_dir = self._get_stocklist_dir()
        
        if not os.path.exists(stocklist_dir):
            self.pool_combo.addItem("未找到股票池文件夹", None)
            return
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(stocklist_dir) if f.endswith('.csv')]
        
        if not csv_files:
            self.pool_combo.addItem("未找到股票池文件", None)
            return
        
        # Sort and add to combo box
        csv_files.sort()
        for filename in csv_files:
            # Extract display name (remove _股票列表.csv suffix)
            display_name = filename.replace('_股票列表.csv', '').replace('.csv', '')
            file_path = os.path.join(stocklist_dir, filename)
            self.pool_combo.addItem(display_name, file_path)
    
    def _on_pool_changed(self):
        """Handle stock pool selection change"""
        stock_codes = self._get_selected_pool_codes()
        count = len(stock_codes) if stock_codes else 0
        self.pool_count_label.setText(f"{count}只")
    
    def _get_selected_pool_codes(self):
        """Get stock codes from the selected pool file"""
        file_path = self.pool_combo.currentData()
        
        if not file_path or not os.path.exists(file_path):
            return []
        
        try:
            # Read CSV file (format: code,name or just code)
            codes = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Handle CSV format: code,name
                    parts = line.split(',')
                    if parts:
                        code = parts[0].strip()
                        if code:
                            # Remove exchange suffix (.SH, .SZ, etc.) if present
                            if '.' in code:
                                code = code.split('.')[0]
                            codes.append(code)
            return codes
        except Exception as e:
            print(f"Error reading stock pool file: {e}")
            return []

    def _load_factor_tree(self):
        """Load factors from factor_registry into tree widget"""
        # Category name mapping
        category_names = {
            'momentum': '动量因子',
            'volatility': '波动率因子',
            'volume': '量价因子',
            'technical': '技术指标',
            'financial': '财务因子'
        }
        
        # Get all categories
        categories = ['momentum', 'volatility', 'volume', 'technical', 'financial']
        
        for category in categories:
            factors = factor_registry.list_factors(category=category)
            if not factors:
                continue
            
            # Create category item
            cat_item = QTreeWidgetItem(self.factor_tree)
            cat_display_name = category_names.get(category, category)
            cat_item.setText(0, f"{cat_display_name} ({len(factors)})")
            cat_item.setFlags(cat_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
            cat_item.setCheckState(0, Qt.CheckState.Unchecked)
            cat_item.setExpanded(False)
            
            # Add factor items
            for factor_name in sorted(factors):
                factor_item = QTreeWidgetItem(cat_item)
                # Get factor description if available
                info = factor_registry.get_factor_info(factor_name)
                desc = info.get('description', '') if info else ''
                display_text = factor_name if not desc else f"{factor_name}"
                factor_item.setText(0, display_text)
                factor_item.setToolTip(0, desc if desc else factor_name)
                factor_item.setData(0, Qt.ItemDataRole.UserRole, factor_name)
                factor_item.setFlags(factor_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                factor_item.setCheckState(0, Qt.CheckState.Unchecked)
        
        # Connect signal for count update
        self.factor_tree.itemChanged.connect(self._on_factor_selection_changed)
    
    def _set_factor_tree_checkable(self, checkable: bool):
        """Enable or disable checkbox interaction on factor tree items.
        
        When checkable is False, users can still expand/collapse categories
        but cannot toggle checkboxes. The tree is also visually dimmed.
        """
        for i in range(self.factor_tree.topLevelItemCount()):
            cat_item = self.factor_tree.topLevelItem(i)
            if checkable:
                cat_item.setFlags(cat_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate | Qt.ItemFlag.ItemIsEnabled)
            else:
                # Remove checkable flags but keep enabled so expand/collapse works
                cat_item.setFlags((cat_item.flags() | Qt.ItemFlag.ItemIsEnabled) & ~Qt.ItemFlag.ItemIsUserCheckable & ~Qt.ItemFlag.ItemIsAutoTristate)
            for j in range(cat_item.childCount()):
                child = cat_item.child(j)
                if checkable:
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                else:
                    child.setFlags((child.flags() | Qt.ItemFlag.ItemIsEnabled) & ~Qt.ItemFlag.ItemIsUserCheckable)
        # Visual dimming
        self.factor_tree.setStyleSheet("QTreeWidget { opacity: 0.5; color: #888; }" if not checkable else "")

    def _on_factor_mode_changed(self, state):
        """Handle factor mode checkbox change"""
        use_default = state == Qt.CheckState.Checked.value
        self._set_factor_tree_checkable(not use_default)
        if use_default:
            self.factor_count_label.setText("默认")
        else:
            self._update_factor_count()
    
    def _on_factor_selection_changed(self, item, column):
        """Update selected factor count when selection changes"""
        self._update_factor_count()
    
    def _update_factor_count(self):
        """Update the selected factor count label"""
        selected = self._get_selected_factors()
        self.factor_count_label.setText(f"已选: {len(selected)}")
    
    def _get_selected_factors(self):
        """Get list of selected factor names from tree widget"""
        selected_factors = []
        
        # Iterate through all category items
        for i in range(self.factor_tree.topLevelItemCount()):
            cat_item = self.factor_tree.topLevelItem(i)
            
            # Iterate through factor items in category
            for j in range(cat_item.childCount()):
                factor_item = cat_item.child(j)
                if factor_item.checkState(0) == Qt.CheckState.Checked:
                    factor_name = factor_item.data(0, Qt.ItemDataRole.UserRole)
                    if factor_name:
                        selected_factors.append(factor_name)
        
        return selected_factors

    def _browse_factor_dir(self):
        """Select the raw factor directory used by XGBoost."""
        current_dir = self.factor_dir_edit.text().strip() if hasattr(self, "factor_dir_edit") else ""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "选择原始因子目录",
            current_dir or self.data_dir,
        )
        if selected_dir:
            self.factor_dir_edit.setText(selected_dir)
            self._update_factor_dir_status()

    def _get_factor_dir(self):
        return self.factor_dir_edit.text().strip() if hasattr(self, "factor_dir_edit") else ""

    @staticmethod
    def _clean_symbol(code):
        value = str(code or "").strip()
        return value.split(".")[0] if "." in value else value

    def _get_effective_factor_cols(self, sid):
        if sid == "xgboost_cross_sectional" and not self.use_default_factors_cb.isChecked():
            return self._get_selected_factors()
        return list(self.xgb_default_params.get("factor_cols", []) or [])

    def _inspect_factor_files(self, stock_codes, factor_cols, factors_dir):
        """Return factor file availability and a light schema check."""
        if not factors_dir or not os.path.isdir(factors_dir):
            return {
                "ok": False,
                "message": "原始因子目录不存在",
                "available": 0,
                "total": len(stock_codes),
                "missing": list(stock_codes),
                "missing_cols": [],
            }

        total = len(stock_codes)
        available_files = []
        missing = []
        for code in stock_codes:
            factor_file = os.path.join(factors_dir, f"{self._clean_symbol(code)}.csv")
            if os.path.exists(factor_file):
                available_files.append(factor_file)
            else:
                missing.append(code)

        missing_cols = []
        if available_files and factor_cols:
            try:
                sample = pd.read_csv(available_files[0], nrows=5)
                missing_cols = [col for col in factor_cols if col not in sample.columns]
            except Exception as exc:
                return {
                    "ok": False,
                    "message": f"因子文件读取失败: {exc}",
                    "available": len(available_files),
                    "total": total,
                    "missing": missing,
                    "missing_cols": [],
                }

        ok = bool(available_files) and not missing_cols
        if not available_files:
            message = "未找到任何股票的原始因子文件"
        elif missing_cols:
            message = f"原始因子文件缺少字段: {', '.join(missing_cols[:5])}"
        elif missing:
            message = f"原始因子文件 {len(available_files)}/{total} 可用"
        else:
            message = f"原始因子文件 {len(available_files)}/{total} 可用"

        return {
            "ok": ok,
            "message": message,
            "available": len(available_files),
            "total": total,
            "missing": missing,
            "missing_cols": missing_cols,
        }

    def _update_factor_dir_status(self):
        if not hasattr(self, "factor_dir_status_label"):
            return
        factors_dir = self._get_factor_dir()
        if not factors_dir:
            self.factor_dir_status_label.setText("未设置原始因子目录")
        elif os.path.isdir(factors_dir):
            csv_count = len([f for f in os.listdir(factors_dir) if f.endswith(".csv")])
            self.factor_dir_status_label.setText(f"目录可用，含 {csv_count} 个原始因子文件")
        else:
            self.factor_dir_status_label.setText("原始因子目录不存在")

    def _confirm_factor_precheck(self, sid, stock_codes, selected_factors):
        if sid != "xgboost_cross_sectional":
            return True
        factor_cols = selected_factors or self._get_effective_factor_cols(sid)
        inspection = self._inspect_factor_files(stock_codes, factor_cols, self._get_factor_dir())
        self.factor_dir_status_label.setText(inspection["message"])
        if not inspection["ok"]:
            QMessageBox.warning(self, "原始因子预检查失败", inspection["message"])
            return False
        missing_count = len(inspection["missing"])
        if missing_count <= 0:
            return True
        answer = QMessageBox.question(
            self,
            "原始因子文件不完整",
            (
                f"股票池中有 {missing_count} 只股票缺少原始因子文件，"
                f"当前可用 {inspection['available']}/{inspection['total']}。\n是否继续回测？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes
    
    def _on_strategy_changed(self):
        """Handle strategy selection change - show/hide XGBoost params"""
        sid = self.strategy_combo.currentData()
        is_xgboost = (sid == "xgboost_cross_sectional")
        
        # Show XGBoost params group only for XGBoost strategy
        if hasattr(self, 'xgb_params_group'):
            self.xgb_params_group.setVisible(is_xgboost)
        if hasattr(self, 'factor_dir_group'):
            self.factor_dir_group.setVisible(is_xgboost)
        if hasattr(self, 'cross_preprocess_group'):
            self.cross_preprocess_group.setVisible(is_xgboost)

    def setupUI(self):
        layout = QHBoxLayout(self)
        
        # --- 左侧设置面板 ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 10, 0)
        
        # 1. 策略选择
        strat_group = QGroupBox("策略")
        strat_layout = QVBoxLayout(strat_group)
        strat_layout.setSpacing(2)
        self.strategy_combo = QComboBox()
        
        # 只加载 CrossSectionalStrategy 类型的策略
        strategies = get_all_strategies()
        for sid, name in strategies.items():
            strat = get_strategy(sid)
            if isinstance(strat, CrossSectionalStrategy):
                self.strategy_combo.addItem(name, sid)
                
        strat_layout.addWidget(self.strategy_combo)
        left_layout.addWidget(strat_group)
        
        # 2. 股票池设置
        pool_group = QGroupBox("股票池")
        pool_layout = QVBoxLayout(pool_group)
        pool_layout.setSpacing(2)
        pool_h = QHBoxLayout()
        self.pool_combo = QComboBox()
        pool_h.addWidget(self.pool_combo)
        self.pool_count_label = QLabel("-")
        self.pool_count_label.setFixedWidth(50)
        pool_h.addWidget(self.pool_count_label)
        pool_layout.addLayout(pool_h)
        
        # Load available stock pool files and connect signal
        self._load_stock_pools()
        self.pool_combo.currentIndexChanged.connect(self._on_pool_changed)
        self._on_pool_changed()  # Update count for initial selection
        
        left_layout.addWidget(pool_group)
        
        # 3. 因子设置
        factor_group = QGroupBox("因子")
        factor_layout = QVBoxLayout(factor_group)
        factor_layout.setSpacing(2)
        
        # 默认因子和已选数量放一行
        factor_top = QHBoxLayout()
        self.use_default_factors_cb = QCheckBox("默认因子")
        self.use_default_factors_cb.setChecked(True)
        self.use_default_factors_cb.stateChanged.connect(self._on_factor_mode_changed)
        factor_top.addWidget(self.use_default_factors_cb)
        self.factor_count_label = QLabel("已选: 0")
        factor_top.addWidget(self.factor_count_label)
        factor_layout.addLayout(factor_top)
        
        # Factor selection tree (scrollable)
        self.factor_tree = QTreeWidget()
        self.factor_tree.setHeaderHidden(True)
        self.factor_tree.setMinimumHeight(120)
        self.factor_tree.setMaximumHeight(300)
        self._load_factor_tree()
        factor_layout.addWidget(self.factor_tree)
        
        # Initially dim the factor tree (use strategy default)
        # Don't disable the entire tree - that prevents expand/collapse.
        # Instead, we control checkbox interaction via _set_factor_tree_checkable.
        self._set_factor_tree_checkable(False)
        
        left_layout.addWidget(factor_group)

        # 3.1 XGBoost 原始因子目录
        self.factor_dir_group = QGroupBox("XGBoost原始因子目录")
        factor_dir_layout = QVBoxLayout(self.factor_dir_group)
        factor_dir_layout.setSpacing(4)
        factor_dir_row = QHBoxLayout()
        self.factor_dir_edit = QLineEdit()
        self.factor_dir_edit.setText(str(self.xgb_default_params.get("factors_dir", "")))
        self.factor_dir_edit.setToolTip("原始因子 CSV 所在目录，文件名格式如 000001.csv")
        self.factor_dir_edit.editingFinished.connect(self._update_factor_dir_status)
        factor_dir_row.addWidget(self.factor_dir_edit)
        self.factor_dir_btn = QPushButton("选择")
        self.factor_dir_btn.clicked.connect(self._browse_factor_dir)
        factor_dir_row.addWidget(self.factor_dir_btn)
        factor_dir_layout.addLayout(factor_dir_row)
        self.factor_dir_status_label = QLabel("")
        self.factor_dir_status_label.setWordWrap(True)
        factor_dir_layout.addWidget(self.factor_dir_status_label)
        self._update_factor_dir_status()
        left_layout.addWidget(self.factor_dir_group)

        # 3.2 XGBoost 截面因子预处理
        self.cross_preprocess_group = QGroupBox("截面因子预处理")
        cross_grid = QGridLayout(self.cross_preprocess_group)
        cross_grid.setSpacing(4)

        self.cross_preprocess_cb = QCheckBox("启用截面预处理")
        self.cross_preprocess_cb.setChecked(bool(self.xgb_default_params.get("enable_cross_sectional_preprocess", True)))
        self.cross_preprocess_cb.setToolTip("按交易日对股票池截面做缺失值处理、去极值和标准化")
        cross_grid.addWidget(self.cross_preprocess_cb, 0, 0, 1, 4)

        cross_grid.addWidget(QLabel("去极值:"), 1, 0)
        self.cross_winsorize_combo = QComboBox()
        self.cross_winsorize_combo.addItem("MAD法", "mad")
        self.cross_winsorize_combo.addItem("3σ法", "sigma")
        self.cross_winsorize_combo.addItem("分位数", "percentile")
        self.cross_winsorize_combo.addItem("不去极值", "none")
        default_winsorize = self.xgb_default_params.get("cross_winsorize_method", "mad")
        winsorize_idx = self.cross_winsorize_combo.findData(default_winsorize)
        self.cross_winsorize_combo.setCurrentIndex(max(0, winsorize_idx))
        cross_grid.addWidget(self.cross_winsorize_combo, 1, 1)

        cross_grid.addWidget(QLabel("阈值:"), 1, 2)
        self.cross_winsorize_n_spin = QDoubleSpinBox()
        self.cross_winsorize_n_spin.setRange(0.001, 10.0)
        self.cross_winsorize_n_spin.setSingleStep(0.1)
        self.cross_winsorize_n_spin.setDecimals(3)
        self.cross_winsorize_n_spin.setValue(float(self.xgb_default_params.get("cross_winsorize_n", 3.0)))
        self.cross_winsorize_n_spin.setToolTip("MAD/3σ为倍数；分位数法可填0.01表示1%和99%截断")
        cross_grid.addWidget(self.cross_winsorize_n_spin, 1, 3)

        cross_grid.addWidget(QLabel("标准化:"), 2, 0)
        self.cross_standardize_combo = QComboBox()
        self.cross_standardize_combo.addItem("Z-Score", "zscore")
        self.cross_standardize_combo.addItem("排名", "rank")
        self.cross_standardize_combo.addItem("Min-Max", "minmax")
        self.cross_standardize_combo.addItem("不标准化", "none")
        default_standardize = self.xgb_default_params.get("cross_standardize_method", "zscore")
        standardize_idx = self.cross_standardize_combo.findData(default_standardize)
        self.cross_standardize_combo.setCurrentIndex(max(0, standardize_idx))
        cross_grid.addWidget(self.cross_standardize_combo, 2, 1, 1, 3)

        left_layout.addWidget(self.cross_preprocess_group)
        
        # 4. XGBoost策略参数设置 (使用紧凑的网格布局)
        self.xgb_params_group = QGroupBox("XGBoost参数")
        xgb_grid = QGridLayout(self.xgb_params_group)
        xgb_grid.setSpacing(4)
        
        # Row 0: 持仓数量 | 调仓周期
        xgb_grid.addWidget(QLabel("持仓:"), 0, 0)
        self.xgb_top_k_spin = QSpinBox()
        self.xgb_top_k_spin.setRange(1, 50)
        self.xgb_top_k_spin.setValue(int(self.xgb_default_params.get("top_k", 5)))
        xgb_grid.addWidget(self.xgb_top_k_spin, 0, 1)
        
        xgb_grid.addWidget(QLabel("调仓:"), 0, 2)
        self.xgb_rebalance_spin = QSpinBox()
        self.xgb_rebalance_spin.setRange(1, 60)
        self.xgb_rebalance_spin.setValue(int(self.xgb_default_params.get("rebalance_period", 20)))
        self.xgb_rebalance_spin.setSuffix("日")
        xgb_grid.addWidget(self.xgb_rebalance_spin, 0, 3)
        
        # Row 1: 训练窗口 | 最小样本
        xgb_grid.addWidget(QLabel("窗口:"), 1, 0)
        self.xgb_train_window_spin = QSpinBox()
        self.xgb_train_window_spin.setRange(60, 500)
        self.xgb_train_window_spin.setValue(int(self.xgb_default_params.get("train_window", 252)))
        xgb_grid.addWidget(self.xgb_train_window_spin, 1, 1)
        
        xgb_grid.addWidget(QLabel("样本:"), 1, 2)
        self.xgb_min_samples_spin = QSpinBox()
        self.xgb_min_samples_spin.setRange(50, 2000)
        self.xgb_min_samples_spin.setValue(int(self.xgb_default_params.get("min_train_samples", 500)))
        xgb_grid.addWidget(self.xgb_min_samples_spin, 1, 3)
        
        # Row 2: 趋势过滤 | 趋势均线
        self.xgb_trend_filter_cb = QCheckBox("趋势过滤")
        self.xgb_trend_filter_cb.setChecked(bool(self.xgb_default_params.get("filter_downtrend", True)))
        xgb_grid.addWidget(self.xgb_trend_filter_cb, 2, 0, 1, 2)
        
        xgb_grid.addWidget(QLabel("均线:"), 2, 2)
        self.xgb_trend_ma_spin = QSpinBox()
        self.xgb_trend_ma_spin.setRange(5, 60)
        self.xgb_trend_ma_spin.setValue(int(self.xgb_default_params.get("trend_ma", 20)))
        self.xgb_trend_ma_spin.setSuffix("日")
        xgb_grid.addWidget(self.xgb_trend_ma_spin, 2, 3)
        
        # Row 3: 树深度 | 学习率 | 树数量
        xgb_grid.addWidget(QLabel("深度:"), 3, 0)
        self.xgb_max_depth_spin = QSpinBox()
        self.xgb_max_depth_spin.setRange(2, 10)
        self.xgb_max_depth_spin.setValue(int(self.xgb_default_params.get("xgb_params", {}).get("max_depth", 4)))
        xgb_grid.addWidget(self.xgb_max_depth_spin, 3, 1)
        
        xgb_grid.addWidget(QLabel("学习率:"), 3, 2)
        self.xgb_learning_rate_spin = QDoubleSpinBox()
        self.xgb_learning_rate_spin.setRange(0.01, 0.5)
        self.xgb_learning_rate_spin.setSingleStep(0.01)
        self.xgb_learning_rate_spin.setValue(float(self.xgb_default_params.get("xgb_params", {}).get("learning_rate", 0.1)))
        xgb_grid.addWidget(self.xgb_learning_rate_spin, 3, 3)
        
        # Row 4: 树数量 | 标签周期
        xgb_grid.addWidget(QLabel("树数:"), 4, 0)
        self.xgb_n_estimators_spin = QSpinBox()
        self.xgb_n_estimators_spin.setRange(10, 500)
        self.xgb_n_estimators_spin.setValue(int(self.xgb_default_params.get("xgb_params", {}).get("n_estimators", 100)))
        xgb_grid.addWidget(self.xgb_n_estimators_spin, 4, 1)
        
        xgb_grid.addWidget(QLabel("标签:"), 4, 2)
        self.xgb_label_period_spin = QSpinBox()
        self.xgb_label_period_spin.setRange(0, 60)
        label_period = self.xgb_default_params.get("label_period")
        self.xgb_label_period_spin.setValue(int(label_period or 0))
        self.xgb_label_period_spin.setSuffix("日")
        self.xgb_label_period_spin.setToolTip("标签收益率间隔天数 (0=跟随调仓周期)")
        xgb_grid.addWidget(self.xgb_label_period_spin, 4, 3)
        
        # Row 5: Clip范围
        xgb_grid.addWidget(QLabel("Clip:"), 5, 0)
        self.xgb_clip_range_spin = QDoubleSpinBox()
        self.xgb_clip_range_spin.setRange(0.05, 1.0)
        self.xgb_clip_range_spin.setSingleStep(0.05)
        self.xgb_clip_range_spin.setValue(float(self.xgb_default_params.get("clip_range", 0.2)))
        self.xgb_clip_range_spin.setDecimals(2)
        self.xgb_clip_range_spin.setToolTip("训练标签收益率clip范围 (±clip_range)")
        xgb_grid.addWidget(self.xgb_clip_range_spin, 5, 1)

        # Row 6: 止损参数
        self.xgb_stop_loss_cb = QCheckBox("个股止损")
        self.xgb_stop_loss_cb.setChecked(bool(self.xgb_default_params.get("enable_stop_loss", True)))
        xgb_grid.addWidget(self.xgb_stop_loss_cb, 6, 0, 1, 2)

        xgb_grid.addWidget(QLabel("止损:"), 6, 2)
        self.xgb_stop_loss_spin = QDoubleSpinBox()
        self.xgb_stop_loss_spin.setRange(0.01, 0.5)
        self.xgb_stop_loss_spin.setSingleStep(0.01)
        self.xgb_stop_loss_spin.setValue(float(self.xgb_default_params.get("stop_loss_pct", 0.08)))
        self.xgb_stop_loss_spin.setDecimals(2)
        self.xgb_stop_loss_spin.setToolTip("个股从成本价回撤达到该比例时卖出")
        xgb_grid.addWidget(self.xgb_stop_loss_spin, 6, 3)
        
        left_layout.addWidget(self.xgb_params_group)
        
        # 监听策略变化，显示/隐藏XGBoost参数
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        self._on_strategy_changed()  # 初始化时检查
        
        # 5. 回测参数设置 (紧凑布局)
        param_group = QGroupBox("回测参数")
        param_grid = QGridLayout(param_group)
        param_grid.setSpacing(4)
        
        # 起始/结束日期放一行
        param_grid.addWidget(QLabel("起始:"), 0, 0)
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setDate(QDate.currentDate().addYears(-1))
        param_grid.addWidget(self.start_date_edit, 0, 1)
        
        param_grid.addWidget(QLabel("结束:"), 1, 0)
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDate(QDate.currentDate())
        param_grid.addWidget(self.end_date_edit, 1, 1)
        
        param_grid.addWidget(QLabel("资金:"), 2, 0)
        self.capital_spin = QSpinBox()
        self.capital_spin.setRange(1000, 100000000)
        self.capital_spin.setSingleStep(10000)
        self.capital_spin.setValue(1000000)
        self.capital_spin.setSuffix("元")
        param_grid.addWidget(self.capital_spin, 2, 1)
        
        left_layout.addWidget(param_group)
        
        # 6. 基准指数设置
        benchmark_group = QGroupBox("基准")
        benchmark_layout = QHBoxLayout(benchmark_group)
        benchmark_layout.setSpacing(4)
        self.benchmark_combo = QComboBox()
        self.benchmark_combo.addItem("无", None)
        for idx_info in get_data_portal().list_assets(asset_type="index", include_status=False):
            self.benchmark_combo.addItem(idx_info["name"], idx_info["code"])
        # 默认选择沪深300
        for i in range(self.benchmark_combo.count()):
            if self.benchmark_combo.itemData(i) == "000300":
                self.benchmark_combo.setCurrentIndex(i)
                break
        benchmark_layout.addWidget(self.benchmark_combo)
        left_layout.addWidget(benchmark_group)
        
        # 按钮
        self.run_btn = QPushButton("开始截面选股回测")
        self.run_btn.setProperty("class", "primary")
        self.run_btn.clicked.connect(self.run_backtest)
        left_layout.addWidget(self.run_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addStretch()
        
        # 用 QScrollArea 包裹左侧面板，添加垂直滚动条
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        
        # --- 右侧结果面板 ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 回放控制条
        self.replay_widget = ReplayWidget()
        self.replay_widget.stepChanged.connect(self.update_replay_view)
        self.replay_widget.setVisible(False)
        right_layout.addWidget(self.replay_widget)
        
        # 结果选项卡
        self.result_tabs = QTabWidget()
        
        # Tab 1: 资金曲线
        self.chart_widget = pg.PlotWidget()
        self.chart_widget.setBackground('#1e1e1e')
        self.chart_widget.showGrid(x=True, y=True, alpha=0.3)
        self.chart_widget.setLabel('left', '总资产')
        self.chart_widget.setLabel('bottom', '日期')
        self.chart_widget.addLegend()
        
        # 添加一条垂直线用于回放指示
        self.replay_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=1, style=Qt.PenStyle.DashLine))
        self.chart_widget.addItem(self.replay_line)
        self.replay_line.setVisible(False)
        
        self.result_tabs.addTab(self.chart_widget, "资金曲线")
        
        # Tab 2: 评分快照 (回放用)
        scores_container = QWidget()
        scores_layout = QVBoxLayout(scores_container)
        scores_layout.setContentsMargins(0, 0, 0, 0)
        
        # 训练信息标签（用于显示 XGBoost 等 ML 策略的特征重要性）
        self.train_info_label = QLabel("")
        self.train_info_label.setStyleSheet("font-family: Consolas, monospace; padding: 5px;")
        self.train_info_label.setWordWrap(True)
        scores_layout.addWidget(self.train_info_label)

        self.feature_importance_table = QTableWidget()
        self.feature_importance_table.setColumnCount(3)
        self.feature_importance_table.setHorizontalHeaderLabels(["因子", "重要性", "占比条"])
        self.feature_importance_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.feature_importance_table.setMaximumHeight(180)
        self.feature_importance_table.setVisible(False)
        scores_layout.addWidget(self.feature_importance_table)
        
        self.scores_table = QTableWidget()
        self.scores_table.setColumnCount(4)
        self.scores_table.setHorizontalHeaderLabels(["排名", "代码", "名称", "评分"])
        self.scores_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        scores_layout.addWidget(self.scores_table)
        
        self.result_tabs.addTab(scores_container, "评分排名 (Top 20)")
        
        # Tab 3: 持仓/交易 (回放用)
        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(3)
        self.holdings_table.setHorizontalHeaderLabels(["代码", "名称", "操作/市值"])
        self.holdings_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.result_tabs.addTab(self.holdings_table, "当日交易")
        
        # Tab 4: 交易详情
        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(8)
        self.trade_table.setHorizontalHeaderLabels([
            "日期", "标的", "操作", "价格", "数量", "手续费", "原因", "剩余资金"
        ])
        self.result_tabs.addTab(self.trade_table, "所有交易记录")
        
        # Tab 5: 统计摘要
        self.stats_label = QLabel("暂无结果")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.stats_label.setStyleSheet("font-family: Consolas, monospace;")
        self.result_tabs.addTab(self.stats_label, "截面选股报告")
        
        right_layout.addWidget(self.result_tabs)
        
        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 750])
        
        layout.addWidget(splitter)

    def run_backtest(self):
        if self.backtest_thread and self.backtest_thread.isRunning():
            return

        sid = self.strategy_combo.currentData()
        if not sid:
            QMessageBox.warning(self, "提示", "请选择一个截面选股策略")
            return

        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
        initial_cash = self.capital_spin.value()
        
        # Get stock codes from selected pool file
        stock_codes = self._get_selected_pool_codes()
        if not stock_codes:
            QMessageBox.warning(self, "提示", "股票池为空，请选择有效的股票池文件")
            return
        
        # Get selected factors (None means use strategy default)
        selected_factors = None
        if not self.use_default_factors_cb.isChecked():
            selected_factors = self._get_selected_factors()
            if not selected_factors:
                QMessageBox.warning(self, "提示", "请至少选择一个因子，或勾选\"使用策略默认因子\"")
                return

        if not self._confirm_factor_precheck(sid, stock_codes, selected_factors):
            return
        
        self.run_btn.setEnabled(False)
        self.run_btn.setText("截面选股回测中...")
        self.chart_widget.clear()
        self.chart_widget.addItem(self.replay_line) # Re-add replay line
        self.trade_table.setRowCount(0)
        self.stats_label.setText("正在初始化...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.replay_widget.setVisible(False)
        
        # Get XGBoost strategy parameters if applicable
        xgb_params = None
        if sid == "xgboost_cross_sectional":
            label_period_val = self.xgb_label_period_spin.value()
            xgb_params = {
                "top_k": self.xgb_top_k_spin.value(),
                "rebalance_period": self.xgb_rebalance_spin.value(),
                "train_window": self.xgb_train_window_spin.value(),
                "min_train_samples": self.xgb_min_samples_spin.value(),
                "filter_downtrend": self.xgb_trend_filter_cb.isChecked(),
                "trend_ma": self.xgb_trend_ma_spin.value(),
                "label_period": label_period_val if label_period_val > 0 else None,
                "clip_range": self.xgb_clip_range_spin.value(),
                "enable_stop_loss": self.xgb_stop_loss_cb.isChecked(),
                "stop_loss_pct": self.xgb_stop_loss_spin.value(),
                "factors_dir": self._get_factor_dir(),
                "enable_cross_sectional_preprocess": self.cross_preprocess_cb.isChecked(),
                "cross_missing_method": "median",
                "cross_winsorize_method": self.cross_winsorize_combo.currentData(),
                "cross_winsorize_n": self.cross_winsorize_n_spin.value(),
                "cross_standardize_method": self.cross_standardize_combo.currentData(),
                "xgb_params": {
                    "objective": "reg:squarederror",
                    "max_depth": self.xgb_max_depth_spin.value(),
                    "learning_rate": self.xgb_learning_rate_spin.value(),
                    "n_estimators": self.xgb_n_estimators_spin.value(),
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": 1,
                }
            }
        
        self.backtest_thread = CrossSectionalBacktestThread(
            sid, start_date, end_date, initial_cash, self.data_dir, stock_codes, selected_factors, xgb_params
        )
        self.backtest_thread.finished_signal.connect(self.on_finished)
        self.backtest_thread.error_signal.connect(self.on_error)
        self.backtest_thread.info_signal.connect(self.on_info)
        self.backtest_thread.progress_updated.connect(self.on_progress)
        self.backtest_thread.start()
        
    def on_finished(self, result):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始截面选股回测")
        self.progress_bar.setVisible(False)
        self.stats_label.setText("截面选股回测完成")
        self.backtest_result = result
        
        # 1. 绘制曲线
        equity_df = result['equity_curve']
        if not equity_df.empty:
            # 统一日期格式为 YYYY-MM-DD 字符串
            def normalize_date(d):
                """将各种日期格式统一转换为 YYYY-MM-DD 字符串"""
                if hasattr(d, 'strftime'):
                    return d.strftime('%Y-%m-%d')
                elif hasattr(d, 'date'):
                    return str(d.date())
                else:
                    # 如果是字符串，可能带有时间部分，只取日期部分
                    return str(d).split(' ')[0].split('T')[0]
            
            dates = [normalize_date(d) for d in equity_df['date']]
            
            x = range(len(equity_df))
            y = equity_df['total_asset'].values
            
            self.chart_widget.plot(x, y, pen=pg.mkPen('b', width=2), name="策略收益")
            
            # 绘制基准收益曲线
            self._plot_benchmark_curve(equity_df, dates)
            
            ax = self.chart_widget.getAxis('bottom')
            
            ticks = []
            n = max(1, len(dates) // 10)
            for i in range(0, len(dates), n):
                ticks.append((i, dates[i]))
            ax.setTicks([ticks])
            
            # 准备回放数据
            # 从 history_scores 中获取调仓日期索引
            trade_indices = []
            date_to_idx = {d: i for i, d in enumerate(dates)}
            
            # 优先从 history_scores 获取调仓日期
            history_scores = result.get('history_scores', {})
            if history_scores:
                for date_str in history_scores.keys():
                    # history_scores 的 key 已经是 YYYY-MM-DD 格式
                    if date_str in date_to_idx:
                        trade_indices.append(date_to_idx[date_str])
                    else:
                        # 调试：打印不匹配的日期
                        print(f"[Debug] history_scores date '{date_str}' not found in equity_curve dates")
            else:
                # 如果没有 history_scores，回退到从 trades 获取
                trades = result.get('trades', [])
                for t in trades:
                    d_str = normalize_date(t.date)
                    if d_str in date_to_idx:
                        trade_indices.append(date_to_idx[d_str])
            
            # 存储归一化后的日期列表，供 replay 使用
            self.normalized_dates = dates
            
            print(f"[Debug] Total dates: {len(dates)}, Trade indices: {trade_indices[:5]}...")
            
            # Setup replay
            self.replay_widget.setup(len(equity_df), trade_indices)
            self.replay_widget.setVisible(True)
            self.replay_line.setVisible(True)
            
            # 自动切换到评分Tab
            self.result_tabs.setCurrentIndex(1)

        # 2. 填充交易列表
        trades = result.get('trades', [])
        self.trade_table.setRowCount(len(trades))
        for i, t in enumerate(trades):
            self.trade_table.setItem(i, 0, QTableWidgetItem(str(t.date)))
            self.trade_table.setItem(i, 1, QTableWidgetItem(t.symbol))
            
            action_item = QTableWidgetItem(t.action)
            if t.action == 'BUY':
                action_item.setForeground(QColor("red"))
            else:
                action_item.setForeground(QColor("green"))
            self.trade_table.setItem(i, 2, action_item)
            
            self.trade_table.setItem(i, 3, QTableWidgetItem(f"{t.price:.2f}"))
            self.trade_table.setItem(i, 4, QTableWidgetItem(str(t.quantity)))
            total_fee = getattr(t, "total_fee", t.commission)
            self.trade_table.setItem(i, 5, QTableWidgetItem(f"{total_fee:.2f}"))
            self.trade_table.setItem(i, 6, QTableWidgetItem(t.reason))
            self.trade_table.setItem(i, 7, QTableWidgetItem(f"{t.cash_after:.2f}"))

        # 3. 统计报告
        final_value = result.get('final_value', 0)
        init_cash = self.capital_spin.value()
        ret = (final_value - init_cash) / init_cash * 100 if init_cash != 0 else 0
        
        # Calculate Sharpe Ratio from equity curve
        sharpe_text = ""
        equity_df = result.get('equity_curve', pd.DataFrame())
        if not equity_df.empty and len(equity_df) > 1:
            daily_returns = equity_df['total_asset'].pct_change().dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                sharpe_text = f"\n        夏普比率: {sharpe_ratio:.3f}"
        
        closed_trades = result.get('closed_trades', [])
        pnl_text = ""
        if closed_trades:
            win_count = sum(1 for t in closed_trades if t.pnl > 0)
            total_closed = len(closed_trades)
            win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0
            pnl_text = f"\n        平仓次数: {total_closed}\n        胜率    : {win_rate:.2f}%"
        
        report = f"""
        === 截面选股报告 ===
        
        初始资金: {init_cash:,.2f}
        最终资产: {final_value:,.2f}
        收益率  : {ret:+.2f}%{sharpe_text}
        
        交易次数: {len(trades)}{pnl_text}
        """
        self.stats_label.setText(report)
    
    def _plot_benchmark_curve(self, equity_df, dates):
        """
        绘制基准指数收益曲线
        
        Args:
            equity_df: 策略资产曲线DataFrame
            dates: 归一化后的日期列表 (YYYY-MM-DD格式)
        """
        # 获取选择的基准指数
        benchmark_code = self.benchmark_combo.currentData()
        if not benchmark_code:
            return
        
        benchmark_name = self.benchmark_combo.currentText()
        
        # 获取日期范围
        start_date = dates[0] if dates else None
        end_date = dates[-1] if dates else None
        
        if not start_date or not end_date:
            return
        
        # 加载基准指数数据
        benchmark_df = get_data_portal().get_daily_bars(
            benchmark_code,
            data_dir=self.data_dir,
            start=start_date,
            end=end_date,
            asset_type="index",
            use_cache=False,
        )
        
        if benchmark_df is None or benchmark_df.empty:
            print(f"[Warning] 未能加载基准指数 {benchmark_code} 的数据，请先更新指数数据")
            return
        
        # 将基准数据日期转为字符串格式
        benchmark_df['date_str'] = benchmark_df['date'].apply(
            lambda d: d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d).split(' ')[0]
        )
        
        # 创建日期到收盘价的映射
        benchmark_prices = dict(zip(benchmark_df['date_str'], benchmark_df['close']))
        
        # 获取策略的初始资产值用于归一化
        initial_asset = equity_df['total_asset'].iloc[0]
        
        # 获取基准的初始价格
        initial_price = None
        for d in dates:
            if d in benchmark_prices:
                initial_price = benchmark_prices[d]
                break
        
        if initial_price is None or initial_price == 0:
            print(f"[Warning] 无法获取基准指数 {benchmark_code} 的初始价格")
            return
        
        # 计算基准净值曲线（与策略资产同步）
        benchmark_values = []
        last_price = initial_price
        
        for d in dates:
            if d in benchmark_prices:
                last_price = benchmark_prices[d]
            # 计算基准净值（归一化到与策略初始资产相同的起点）
            benchmark_value = (last_price / initial_price) * initial_asset
            benchmark_values.append(benchmark_value)
        
        # 绘制基准曲线
        x = range(len(dates))
        self.chart_widget.plot(
            list(x), 
            benchmark_values, 
            pen=pg.mkPen('#FFA500', width=2),  # 橙色
            name=f"基准({benchmark_name})"
        )
        
    def on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("开始截面选股回测")
        self.progress_bar.setVisible(False)
        self.stats_label.setText(f"错误: {msg}")
        QMessageBox.critical(self, "截面选股回测失败", msg)

    def on_info(self, msg):
        self.stats_label.setText(msg)
        
    def on_progress(self, current, total):
        self.progress_bar.setValue(int(current / total * 100))

    def update_replay_view(self, step):
        """Update chart and info based on replay step"""
        if not self.backtest_result or 'equity_curve' not in self.backtest_result:
            return 
            
        df = self.backtest_result['equity_curve']
        if step >= len(df):
            return 
        
        # 使用归一化后的日期（与 history_scores 的 key 格式一致）
        if hasattr(self, 'normalized_dates') and step < len(self.normalized_dates):
            date = self.normalized_dates[step]
        else:
            # 回退：手动归一化
            raw_date = df.iloc[step]['date']
            if hasattr(raw_date, 'strftime'):
                date = raw_date.strftime('%Y-%m-%d')
            elif hasattr(raw_date, 'date'):
                date = str(raw_date.date())
            else:
                date = str(raw_date).split(' ')[0].split('T')[0]
        
        # Update date label
        self.replay_widget.set_date_text(date)
        
        # Move vertical line
        self.replay_line.setValue(step)
        
        # 1. Update Scores Table and Train Info
        self.scores_table.setRowCount(0)
        self.train_info_label.setText("")  # 清空训练信息
        self.feature_importance_table.setRowCount(0)
        self.feature_importance_table.setVisible(False)
        
        history_scores = self.backtest_result.get('history_scores', {})
        history_train_info = self.backtest_result.get('history_train_info', {})
        
        if date in history_scores:
            scores_df = history_scores[date]
            self.scores_table.setRowCount(len(scores_df))
            for i, (idx, s_row) in enumerate(scores_df.iterrows()):
                code = s_row['code'] if 'code' in s_row else idx
                name = self.stock_name_map.get(code, "")
                score = s_row['score'] if 'score' in s_row else 0.0
                
                self.scores_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.scores_table.setItem(i, 1, QTableWidgetItem(str(code)))
                self.scores_table.setItem(i, 2, QTableWidgetItem(name))
                
                score_item = QTableWidgetItem(f"{score:.4f}")
                if score > 0:
                    score_item.setForeground(QColor("red"))
                elif score < 0:
                    score_item.setForeground(QColor("green"))
                self.scores_table.setItem(i, 3, score_item)
            
            # 显示训练信息（如 XGBoost 特征重要性）
            if date in history_train_info:
                train_info = history_train_info[date]
                self.train_info_label.setText(f"训练样本: {train_info.get('train_samples', 'N/A')}")
                
                if 'feature_importance' in train_info:
                    fi_items = list(train_info['feature_importance'][:10])
                    max_importance = max([float(imp) for _, imp in fi_items], default=0.0)
                    self.feature_importance_table.setRowCount(len(fi_items))
                    for row, (name, importance) in enumerate(fi_items):
                        importance = float(importance)
                        bar_len = int((importance / max_importance) * 20) if max_importance > 0 else 0
                        self.feature_importance_table.setItem(row, 0, QTableWidgetItem(str(name)))
                        self.feature_importance_table.setItem(row, 1, QTableWidgetItem(f"{importance:.4f}"))
                        self.feature_importance_table.setItem(row, 2, QTableWidgetItem("#" * bar_len))
                    self.feature_importance_table.setVisible(bool(fi_items))
        else:
            # 如果当天没有评分数据（非调仓日），显示提示
            self.train_info_label.setText("非调仓日")
        
        # 2. Update Trades Table (Day's activity)
        trades = self.backtest_result.get('trades', [])
        
        def normalize_trade_date(d):
            if hasattr(d, 'strftime'):
                return d.strftime('%Y-%m-%d')
            elif hasattr(d, 'date'):
                return str(d.date())
            else:
                return str(d).split(' ')[0].split('T')[0]
        
        todays_trades = [t for t in trades if normalize_trade_date(t.date) == date]
        
        self.holdings_table.setRowCount(len(todays_trades))
        for i, t in enumerate(todays_trades):
            name = self.stock_name_map.get(t.symbol, "")
            self.holdings_table.setItem(i, 0, QTableWidgetItem(t.symbol))
            self.holdings_table.setItem(i, 1, QTableWidgetItem(name))
            
            action_text = f"{t.action} {t.quantity}"
            item = QTableWidgetItem(action_text)
            if t.action == "BUY":
                item.setForeground(QColor("red"))
            else:
                item.setForeground(QColor("green"))
                
            self.holdings_table.setItem(i, 2, item)
            
        if not todays_trades:
            if self.holdings_table.rowCount() == 0:
                self.holdings_table.setRowCount(1)
                self.holdings_table.setItem(0, 0, QTableWidgetItem("-"))
                self.holdings_table.setItem(0, 1, QTableWidgetItem("今日无交易"))
                self.holdings_table.setItem(0, 2, QTableWidgetItem("-"))
