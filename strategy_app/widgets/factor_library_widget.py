"""
Factor Library Widget - Factor management and visualization interface
"""
import os
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QTableWidget, QTableWidgetItem, QLabel, QHeaderView,
    QSplitter, QGroupBox, QDateEdit, QMessageBox, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QCheckBox, QScrollArea,
    QFrame, QGridLayout, QLineEdit, QProgressBar, QFileDialog, QApplication,
    QSpinBox
)
from PyQt6.QtGui import QAction, QColor, QBrush, QFont
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal

import pyqtgraph as pg
import pandas as pd
import numpy as np

try:
    from strategy_app.styles import Colors
except ImportError:
    from styles import Colors

# Local modules
try:
    from factors import factor_registry
    from factors.registry import FactorRegistry
    from factors.financial_data import FinancialDataLoader
    from factors.preprocessor import FactorPreprocessor, PreprocessConfig
    from factors.analysis import FactorAnalysisService, FactorPanelBuilder
except ImportError:
    from strategy_app.factors import factor_registry
    from strategy_app.factors.registry import FactorRegistry
    from strategy_app.factors.financial_data import FinancialDataLoader
    from strategy_app.factors.preprocessor import FactorPreprocessor, PreprocessConfig
    from strategy_app.factors.analysis import FactorAnalysisService, FactorPanelBuilder

from common.data_portal import get_data_portal


class BatchFactorComputeThread(QThread):
    """Background thread for batch computing factors for multiple stocks"""
    finished_signal = pyqtSignal(str, int, int)  # output_dir, success_count, fail_count
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)  # current, total, current_stock

    def __init__(self, stock_codes, data_dir, start_date, end_date, factor_names, output_dir):
        super().__init__()
        self.stock_codes = stock_codes
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date
        self.factor_names = factor_names
        self.output_dir = output_dir

    def run(self):
        try:
            total = len(self.stock_codes)
            success_count = 0
            fail_count = 0
            
            # Ensure output directory exists
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            for i, code in enumerate(self.stock_codes):
                self.progress_signal.emit(i + 1, total, code)
                
                try:
                    # Load stock data
                    df = get_data_portal().get_daily_bars(
                        code,
                        data_dir=self.data_dir,
                        start=self.start_date,
                        end=self.end_date,
                        asset_type="stock",
                    )

                    if df is None or df.empty:
                        fail_count += 1
                        continue

                    # Add stock code column
                    df['code'] = code
                    
                    # Compute factors
                    for name in self.factor_names:
                        try:
                            df[name] = factor_registry.compute(name, df)
                        except Exception as e:
                            df[name] = np.nan

                    # Keep only necessary columns
                    cols = ['code', 'date'] + self.factor_names
                    cols = [c for c in cols if c in df.columns]
                    result_df = df[cols]
                    
                    # Save to individual file per stock
                    output_file = os.path.join(self.output_dir, f"{code}.csv")
                    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error computing factors for {code}: {e}")
                    fail_count += 1
                    continue

            if success_count == 0:
                self.error_signal.emit("未能计算任何股票的因子数据")
                return
            
            self.finished_signal.emit(self.output_dir, success_count, fail_count)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"因子批量计算错误: {str(e)}")


class FactorAnalysisThread(QThread):
    """Background thread for native factor effectiveness analysis."""

    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)

    def __init__(
        self,
        stock_codes,
        data_dir,
        factors_dir,
        factor_name,
        start_date,
        end_date,
        forward_period,
        quantiles,
    ):
        super().__init__()
        self.stock_codes = stock_codes
        self.data_dir = data_dir
        self.factors_dir = factors_dir
        self.factor_name = factor_name
        self.start_date = start_date
        self.end_date = end_date
        self.forward_period = forward_period
        self.quantiles = quantiles

    def run(self):
        try:
            builder = FactorPanelBuilder(self.data_dir, self.factors_dir)
            build_result = builder.build_panel(
                self.stock_codes,
                [self.factor_name],
                start_date=self.start_date,
                end_date=self.end_date,
                progress_callback=lambda current, total, code: self.progress_signal.emit(current, total, code),
            )
            if build_result.panel.empty:
                self.error_signal.emit("未能构建有效因子面板，请检查原始因子文件和行情数据")
                return

            service = FactorAnalysisService()
            result = service.analyze(
                build_result.panel,
                self.factor_name,
                forward_period=self.forward_period,
                quantiles=self.quantiles,
            )
            result["build"] = {
                "success_count": build_result.success_count,
                "fail_count": build_result.fail_count,
                "missing_codes": build_result.missing_codes[:20],
            }
            self.finished_signal.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_signal.emit(f"因子有效性分析失败: {str(e)}")


class FactorLibraryWidget(QWidget):
    """Factor Library main interface"""

    def __init__(self, data_dir="../data", stocklist_path=None, tushare_token=None):
        super().__init__()
        self.data_dir = data_dir
        self.stocklist_path = stocklist_path
        self.tushare_token = tushare_token
        self.batch_compute_thread = None
        self.factor_analysis_thread = None
        self.stock_list = []
        self.name_map = {}
        self.current_df = None
        self.factor_checkboxes = {}
        self.plot_curves = {}

        # Color palette for factor plots
        self.colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
            '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
            '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000',
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
        ]

        self.setupUI()
        self.load_data()

    def setupUI(self):
        layout = QVBoxLayout(self)

        # Set table style for better visibility in dark theme
        # 样式已在全局定义

        # Main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # === Left Panel: Factor List ===
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)

        # === Right Panel: Visualization and Info ===
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)

        main_splitter.setSizes([300, 900])
        layout.addWidget(main_splitter)

    def create_left_panel(self):
        """Create left panel with factor tree and controls"""
        # Use a vertical splitter so user can drag to resize factor list vs controls
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setContentsMargins(0, 0, 10, 0)

        # --- Top: Factor Tree ---
        factor_group = QGroupBox("因子研究列表")
        factor_layout = QVBoxLayout(factor_group)
        factor_layout.setContentsMargins(6, 6, 6, 6)
        factor_layout.setSpacing(4)

        # Search box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入因子名称...")
        self.search_input.textChanged.connect(self.filter_factors)
        search_layout.addWidget(self.search_input)
        factor_layout.addLayout(search_layout)

        # Factor tree
        self.factor_tree = QTreeWidget()
        self.factor_tree.setHeaderLabels(["因子/类别", "选中"])
        self.factor_tree.setColumnWidth(0, 180)
        self.factor_tree.setColumnWidth(1, 60)
        self.factor_tree.itemClicked.connect(self.on_factor_clicked)
        self.populate_factor_tree()
        factor_layout.addWidget(self.factor_tree, 1)  # stretch=1 to fill available space

        # Quick selection buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_factors)
        self.clear_all_btn = QPushButton("清除")
        self.clear_all_btn.clicked.connect(self.clear_all_factors)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.clear_all_btn)
        factor_layout.addLayout(btn_layout)

        left_splitter.addWidget(factor_group)

        # --- Bottom: Controls (scrollable) ---
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        # --- Stock Pool Batch Compute ---
        pool_group = QGroupBox("批量因子计算 (股票池)")
        pool_layout = QVBoxLayout(pool_group)
        
        pool_layout.addWidget(QLabel("选择股票池:"))
        self.pool_combo = QComboBox()
        pool_layout.addWidget(self.pool_combo)
        
        # Show stock count label (create before loading pools)
        self.pool_count_label = QLabel("股票数量: -")
        self.pool_count_label.setProperty("class", "status-pending")
        pool_layout.addWidget(self.pool_count_label)
        
        # Load pools and connect signal after label is created
        self._load_stock_pools()
        self.pool_combo.currentIndexChanged.connect(self._on_pool_changed)

        # Date range in a horizontal layout
        date_layout = QHBoxLayout()
        
        start_vbox = QVBoxLayout()
        start_vbox.addWidget(QLabel("起始日期:"))
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setDate(QDate.currentDate().addMonths(-6))
        start_vbox.addWidget(self.start_date_edit)
        date_layout.addLayout(start_vbox)

        end_vbox = QVBoxLayout()
        end_vbox.addWidget(QLabel("结束日期:"))
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setDate(QDate.currentDate())
        end_vbox.addWidget(self.end_date_edit)
        date_layout.addLayout(end_vbox)
        
        pool_layout.addLayout(date_layout)

        # Batch compute button (for stock pool)
        self.batch_compute_btn = QPushButton("批量计算原始因子")
        self.batch_compute_btn.setProperty("class", "success")
        self.batch_compute_btn.clicked.connect(self.batch_compute_factors)
        pool_layout.addWidget(self.batch_compute_btn)

        # Progress bar for batch computation
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        pool_layout.addWidget(self.batch_progress_bar)
        
        self.batch_progress_label = QLabel("")
        self.batch_progress_label.setProperty("class", "status-pending")
        self.batch_progress_label.setVisible(False)
        pool_layout.addWidget(self.batch_progress_label)
        
        controls_layout.addWidget(pool_group)

        # --- Single Stock Factor Plot ---
        plot_group = QGroupBox("因子研究可视化 (单股)")
        plot_layout = QVBoxLayout(plot_group)

        plot_layout.addWidget(QLabel("股票代码:"))
        self.stock_combo = QComboBox()
        self.stock_combo.setEditable(True)
        self.stock_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.stock_combo.completer().setCompletionMode(
            self.stock_combo.completer().CompletionMode.PopupCompletion
        )
        self.stock_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        plot_layout.addWidget(self.stock_combo)

        # Plot button inside the group
        self.plot_btn = QPushButton("加载并可视化因子")
        self.plot_btn.setProperty("class", "primary")
        self.plot_btn.clicked.connect(self.plot_factors)
        plot_layout.addWidget(self.plot_btn)

        controls_layout.addWidget(plot_group)

        # --- Other Action Buttons ---
        self.export_btn = QPushButton("导出当前可视化数据")
        self.export_btn.clicked.connect(self.export_data)
        controls_layout.addWidget(self.export_btn)
        
        # Anomaly check button
        self.anomaly_check_btn = QPushButton("检查因子数据异常")
        self.anomaly_check_btn.setProperty("class", "warning")
        self.anomaly_check_btn.clicked.connect(self.check_factor_anomalies)
        controls_layout.addWidget(self.anomaly_check_btn)

        controls_layout.addStretch()
        controls_scroll.setWidget(controls_container)
        left_splitter.addWidget(controls_scroll)

        # Give more space to factor tree (60%) vs controls (40%)
        left_splitter.setSizes([400, 250])
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 2)

        return left_splitter

    def create_right_panel(self):
        """Create right panel with visualization and info tabs"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.result_tabs = QTabWidget()

        # Tab 1: Factor Chart
        chart_tab = self.create_chart_tab()
        self.result_tabs.addTab(chart_tab, "因子图表")

        # Tab 2: Factor Info
        info_tab = self.create_info_tab()
        self.result_tabs.addTab(info_tab, "因子详情")

        # Tab 3: Data Table
        data_tab = self.create_data_tab()
        self.result_tabs.addTab(data_tab, "数据预览")

        # Tab 4: Factor Statistics
        stats_tab = self.create_stats_tab()
        self.result_tabs.addTab(stats_tab, "统计分析")

        # Tab 5: Data Preprocessing
        preprocess_tab = self.create_preprocess_tab()
        self.result_tabs.addTab(preprocess_tab, "单股时序预处理（研究用）")

        # Tab 6: Factor Effectiveness Analysis
        analysis_tab = self.create_factor_analysis_tab()
        self.result_tabs.addTab(analysis_tab, "因子有效性分析")

        right_layout.addWidget(self.result_tabs)

        return right_widget

    def create_factor_analysis_tab(self):
        """Create native factor effectiveness analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        config_group = QGroupBox("分析配置")
        config_grid = QGridLayout(config_group)

        config_grid.addWidget(QLabel("后端:"), 0, 0)
        self.analysis_backend_combo = QComboBox()
        self.analysis_backend_combo.addItem("Native（内置）", "native")
        config_grid.addWidget(self.analysis_backend_combo, 0, 1)

        config_grid.addWidget(QLabel("因子:"), 0, 2)
        self.analysis_factor_combo = QComboBox()
        self.analysis_factor_combo.setToolTip("点击刷新会读取左侧已勾选因子")
        config_grid.addWidget(self.analysis_factor_combo, 0, 3)

        self.refresh_analysis_factors_btn = QPushButton("刷新已选因子")
        self.refresh_analysis_factors_btn.clicked.connect(self.refresh_analysis_factor_combo)
        config_grid.addWidget(self.refresh_analysis_factors_btn, 0, 4)

        config_grid.addWidget(QLabel("原始因子目录:"), 1, 0)
        self.analysis_factor_dir_edit = QLineEdit(os.path.join(self.data_dir, "factors"))
        config_grid.addWidget(self.analysis_factor_dir_edit, 1, 1, 1, 3)

        self.analysis_factor_dir_btn = QPushButton("选择")
        self.analysis_factor_dir_btn.clicked.connect(self.browse_analysis_factor_dir)
        config_grid.addWidget(self.analysis_factor_dir_btn, 1, 4)

        config_grid.addWidget(QLabel("起始:"), 2, 0)
        self.analysis_start_date_edit = QDateEdit()
        self.analysis_start_date_edit.setCalendarPopup(True)
        self.analysis_start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.analysis_start_date_edit.setDate(QDate.currentDate().addYears(-1))
        config_grid.addWidget(self.analysis_start_date_edit, 2, 1)

        config_grid.addWidget(QLabel("结束:"), 2, 2)
        self.analysis_end_date_edit = QDateEdit()
        self.analysis_end_date_edit.setCalendarPopup(True)
        self.analysis_end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.analysis_end_date_edit.setDate(QDate.currentDate())
        config_grid.addWidget(self.analysis_end_date_edit, 2, 3)

        config_grid.addWidget(QLabel("未来收益:"), 3, 0)
        self.analysis_forward_spin = QSpinBox()
        self.analysis_forward_spin.setRange(1, 120)
        self.analysis_forward_spin.setValue(5)
        self.analysis_forward_spin.setSuffix("日")
        config_grid.addWidget(self.analysis_forward_spin, 3, 1)

        config_grid.addWidget(QLabel("分组数:"), 3, 2)
        self.analysis_quantile_spin = QSpinBox()
        self.analysis_quantile_spin.setRange(2, 10)
        self.analysis_quantile_spin.setValue(5)
        config_grid.addWidget(self.analysis_quantile_spin, 3, 3)

        self.run_analysis_btn = QPushButton("运行因子有效性分析")
        self.run_analysis_btn.setProperty("class", "primary")
        self.run_analysis_btn.clicked.connect(self.run_factor_analysis)
        config_grid.addWidget(self.run_analysis_btn, 3, 4)

        layout.addWidget(config_group)

        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setVisible(False)
        layout.addWidget(self.analysis_progress_bar)

        self.analysis_status_label = QLabel("选择左侧因子并点击刷新后开始分析")
        self.analysis_status_label.setWordWrap(True)
        layout.addWidget(self.analysis_status_label)

        result_splitter = QSplitter(Qt.Orientation.Vertical)

        self.analysis_summary_label = QLabel("暂无分析结果")
        self.analysis_summary_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.analysis_summary_label.setStyleSheet("font-family: Consolas, monospace; padding: 8px;")
        result_splitter.addWidget(self.analysis_summary_label)

        table_container = QWidget()
        table_layout = QHBoxLayout(table_container)

        self.analysis_ic_table = QTableWidget()
        self.analysis_ic_table.setColumnCount(4)
        self.analysis_ic_table.setHorizontalHeaderLabels(["日期", "IC", "Rank IC", "样本数"])
        self.analysis_ic_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.analysis_ic_table)

        self.analysis_quantile_table = QTableWidget()
        self.analysis_quantile_table.setColumnCount(4)
        self.analysis_quantile_table.setHorizontalHeaderLabels(["分组", "平均收益", "波动", "样本日数"])
        self.analysis_quantile_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.analysis_quantile_table)

        self.analysis_long_short_table = QTableWidget()
        self.analysis_long_short_table.setColumnCount(4)
        self.analysis_long_short_table.setHorizontalHeaderLabels(["日期", "多头", "空头", "多空"])
        self.analysis_long_short_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.analysis_long_short_table)

        result_splitter.addWidget(table_container)
        result_splitter.setSizes([160, 520])
        layout.addWidget(result_splitter)

        return widget

    def create_chart_tab(self):
        """Create chart visualization tab"""
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)

        # Top splitter: Price chart + Factor charts
        chart_splitter = QSplitter(Qt.Orientation.Vertical)

        # Price chart
        price_group = QGroupBox("股价走势")
        price_layout = QVBoxLayout(price_group)
        self.price_chart = pg.PlotWidget()
        self.price_chart.setBackground(Colors.BG_DARK)
        self.price_chart.showGrid(x=True, y=True, alpha=0.3)
        self.price_chart.setLabel('left', '价格')
        self.price_chart.addLegend()
        price_layout.addWidget(self.price_chart)
        chart_splitter.addWidget(price_group)

        # Factor chart
        factor_group = QGroupBox("因子走势")
        factor_layout = QVBoxLayout(factor_group)
        self.factor_chart = pg.PlotWidget()
        self.factor_chart.setBackground(Colors.BG_DARK)
        self.factor_chart.showGrid(x=True, y=True, alpha=0.3)
        self.factor_chart.setLabel('left', '因子值')
        self.factor_chart.addLegend()

        # Link X axis with price chart
        self.factor_chart.setXLink(self.price_chart)
        factor_layout.addWidget(self.factor_chart)
        chart_splitter.addWidget(factor_group)

        chart_splitter.setSizes([300, 400])
        chart_layout.addWidget(chart_splitter)

        return chart_widget

    def create_info_tab(self):
        """Create factor information tab"""
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        # Factor info display
        self.factor_info_label = QLabel("选择左侧因子查看详情")
        self.factor_info_label.setProperty("class", "description")
        self.factor_info_label.setStyleSheet(f"""
            QLabel {{
                font-family: Consolas, 'Microsoft YaHei';
                font-size: 14px;
                padding: 20px;
                background-color: {Colors.BG_LIGHTER};
                border-radius: 5px;
            }}
        """)
        self.factor_info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.factor_info_label.setWordWrap(True)

        info_layout.addWidget(self.factor_info_label)

        # All factors summary table
        summary_group = QGroupBox("所有因子一览")
        summary_layout = QVBoxLayout(summary_group)

        self.factor_summary_table = QTableWidget()
        self.factor_summary_table.setColumnCount(4)
        self.factor_summary_table.setHorizontalHeaderLabels(["因子名称", "类别", "默认窗口", "描述"])
        self.factor_summary_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.factor_summary_table.setAlternatingRowColors(True)
        self.populate_factor_summary()
        summary_layout.addWidget(self.factor_summary_table)

        info_layout.addWidget(summary_group)

        return info_widget

    def create_data_tab(self):
        """Create data preview tab"""
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)

        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        data_layout.addWidget(self.data_table)

        return data_widget

    def create_stats_tab(self):
        """Create statistics analysis tab"""
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)

        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(8)
        self.stats_table.setHorizontalHeaderLabels([
            "因子", "均值", "标准差", "最小值", "25%", "中位数", "75%", "最大值"
        ])
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        stats_layout.addWidget(self.stats_table)

        # Correlation matrix
        corr_group = QGroupBox("因子相关性矩阵")
        corr_layout = QVBoxLayout(corr_group)
        self.corr_table = QTableWidget()
        self.corr_table.setAlternatingRowColors(True)
        corr_layout.addWidget(self.corr_table)
        stats_layout.addWidget(corr_group)

        return stats_widget

    def create_preprocess_tab(self):
        """Create single-stock time-series preprocessing tab."""
        preprocess_widget = QWidget()
        main_layout = QHBoxLayout(preprocess_widget)

        # === Left: Preprocessing Configuration ===
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 10, 0)

        # Title
        title_label = QLabel("单股时序预处理流程")
        title_label.setProperty("class", "section-title")
        config_layout.addWidget(title_label)

        # Flow description
        flow_label = QLabel(
            "缺失值处理 → 去极值 → 标准化 → 中性化\n"
            "说明：此处按单股历史序列处理，适合单股因子研究；XGBoost截面回测会在策略内按日期做截面预处理。"
        )
        flow_label.setProperty("class", "description")
        flow_label.setWordWrap(True)
        config_layout.addWidget(flow_label)

        # --- Step 1: Missing Value Handling ---
        missing_group = QGroupBox("1. 缺失值处理")
        missing_layout = QVBoxLayout(missing_group)
        
        self.missing_method_combo = QComboBox()
        self.missing_method_combo.addItems([
            "中位数填充 (median)",
            "均值填充 (mean)",
            "前向填充 (ffill)",
            "后向填充 (bfill)",
            "删除缺失行 (drop)",
            "行业均值填充 (industry_mean)"
        ])
        self.missing_method_combo.setCurrentIndex(0)
        missing_layout.addWidget(QLabel("填充方法:"))
        missing_layout.addWidget(self.missing_method_combo)
        
        config_layout.addWidget(missing_group)

        # --- Step 2: Winsorization ---
        winsorize_group = QGroupBox("2. 去极值")
        winsorize_layout = QVBoxLayout(winsorize_group)
        
        self.winsorize_method_combo = QComboBox()
        self.winsorize_method_combo.addItems([
            "MAD法 (mad)",
            "3σ法 (sigma)",
            "分位数截断 (percentile)",
            "不去极值 (none)"
        ])
        self.winsorize_method_combo.setCurrentIndex(0)
        winsorize_layout.addWidget(QLabel("去极值方法:"))
        winsorize_layout.addWidget(self.winsorize_method_combo)
        
        # Threshold parameter
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("阈值参数:"))
        self.winsorize_n_edit = QLineEdit("3.0")
        self.winsorize_n_edit.setToolTip(
            "MAD/3σ法: 倍数 (默认3)\n分位数法: 截断比例 (如0.01表示1%和99%)"
        )
        threshold_layout.addWidget(self.winsorize_n_edit)
        winsorize_layout.addLayout(threshold_layout)
        
        config_layout.addWidget(winsorize_group)

        # --- Step 3: Standardization ---
        standardize_group = QGroupBox("3. 标准化")
        standardize_layout = QVBoxLayout(standardize_group)
        
        self.standardize_method_combo = QComboBox()
        self.standardize_method_combo.addItems([
            "Z-Score标准化 (zscore)",
            "Min-Max标准化 (minmax)",
            "排名标准化 (rank)",
            "不标准化 (none)"
        ])
        self.standardize_method_combo.setCurrentIndex(0)
        standardize_layout.addWidget(QLabel("标准化方法:"))
        standardize_layout.addWidget(self.standardize_method_combo)
        
        config_layout.addWidget(standardize_group)

        # --- Step 4: Neutralization ---
        neutralize_group = QGroupBox("4. 中性化")
        neutralize_layout = QVBoxLayout(neutralize_group)
        
        self.neutralize_method_combo = QComboBox()
        self.neutralize_method_combo.addItems([
            "不中性化 (none)",
            "市值中性化 (size)",
            "行业中性化 (industry)",
            "市值+行业中性化 (size_industry)"
        ])
        self.neutralize_method_combo.setCurrentIndex(0)
        neutralize_layout.addWidget(QLabel("中性化方法:"))
        neutralize_layout.addWidget(self.neutralize_method_combo)
        
        # Column settings
        col_layout = QGridLayout()
        col_layout.addWidget(QLabel("市值列名:"), 0, 0)
        self.size_col_edit = QLineEdit("total_mv")
        col_layout.addWidget(self.size_col_edit, 0, 1)
        col_layout.addWidget(QLabel("行业列名:"), 1, 0)
        self.industry_col_edit = QLineEdit("industry")
        col_layout.addWidget(self.industry_col_edit, 1, 1)
        neutralize_layout.addLayout(col_layout)
        
        config_layout.addWidget(neutralize_group)

        # --- Action Buttons ---
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        
        # Single stock preview button
        self.preview_preprocess_btn = QPushButton("预览单股时序预处理 (当前股票)")
        self.preview_preprocess_btn.setProperty("class", "primary")
        self.preview_preprocess_btn.clicked.connect(self.preview_preprocessing)
        action_layout.addWidget(self.preview_preprocess_btn)
        
        # Batch preprocess button
        self.batch_preprocess_btn = QPushButton("批量单股时序预处理")
        self.batch_preprocess_btn.setProperty("class", "success")
        self.batch_preprocess_btn.clicked.connect(self.batch_preprocess_factors)
        action_layout.addWidget(self.batch_preprocess_btn)
        
        # Progress
        self.preprocess_progress_bar = QProgressBar()
        self.preprocess_progress_bar.setVisible(False)
        action_layout.addWidget(self.preprocess_progress_bar)
        
        self.preprocess_progress_label = QLabel("")
        self.preprocess_progress_label.setStyleSheet("color: #888; font-size: 11px;")
        self.preprocess_progress_label.setVisible(False)
        action_layout.addWidget(self.preprocess_progress_label)
        
        config_layout.addWidget(action_group)
        config_layout.addStretch()

        # === Right: Results Preview ===
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(0, 0, 0, 0)

        # Splitter for before/after comparison
        result_splitter = QSplitter(Qt.Orientation.Vertical)

        # Before preprocessing stats
        before_group = QGroupBox("预处理前")
        before_layout = QVBoxLayout(before_group)
        self.preprocess_before_table = QTableWidget()
        self.preprocess_before_table.setColumnCount(8)
        self.preprocess_before_table.setHorizontalHeaderLabels([
            "因子", "有效值", "缺失值", "缺失比例", "均值", "标准差", "最小值", "最大值"
        ])
        self.preprocess_before_table.setAlternatingRowColors(True)
        self.preprocess_before_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        before_layout.addWidget(self.preprocess_before_table)
        result_splitter.addWidget(before_group)

        # After preprocessing stats
        after_group = QGroupBox("预处理后")
        after_layout = QVBoxLayout(after_group)
        self.preprocess_after_table = QTableWidget()
        self.preprocess_after_table.setColumnCount(8)
        self.preprocess_after_table.setHorizontalHeaderLabels([
            "因子", "有效值", "缺失值", "缺失比例", "均值", "标准差", "最小值", "最大值"
        ])
        self.preprocess_after_table.setAlternatingRowColors(True)
        self.preprocess_after_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        after_layout.addWidget(self.preprocess_after_table)
        result_splitter.addWidget(after_group)

        # Distribution comparison chart
        chart_group = QGroupBox("分布对比")
        chart_layout = QVBoxLayout(chart_group)
        self.preprocess_chart = pg.PlotWidget()
        self.preprocess_chart.setBackground('w')
        self.preprocess_chart.showGrid(x=True, y=True)
        self.preprocess_chart.setLabel('left', '频数')
        self.preprocess_chart.setLabel('bottom', '因子值')
        self.preprocess_chart.addLegend()
        chart_layout.addWidget(self.preprocess_chart)
        
        # Factor selector for chart
        chart_selector_layout = QHBoxLayout()
        chart_selector_layout.addWidget(QLabel("选择因子:"))
        self.preprocess_factor_combo = QComboBox()
        self.preprocess_factor_combo.currentIndexChanged.connect(
            self.update_preprocess_chart
        )
        chart_selector_layout.addWidget(self.preprocess_factor_combo)
        chart_selector_layout.addStretch()
        chart_layout.addLayout(chart_selector_layout)
        
        result_splitter.addWidget(chart_group)
        result_splitter.setSizes([200, 200, 300])

        result_layout.addWidget(result_splitter)

        # Add to main layout
        main_layout.addWidget(config_widget, 1)
        main_layout.addWidget(result_widget, 2)

        return preprocess_widget

    def _create_tree_checkbox(self):
        """Create a styled checkbox for factor tree with enhanced visibility"""
        checkbox = QCheckBox()
        checkbox.setStyleSheet("""
            QCheckBox {
                padding: 2px 4px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #8a8a8a;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #0078d4;
                background-color: #2a2d2e;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 2px solid #0078d4;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #1a8cdb;
                border: 2px solid #1a8cdb;
            }
        """)
        return checkbox

    def populate_factor_tree(self):
        """Populate factor tree with categories"""
        self.factor_tree.clear()
        self.factor_checkboxes.clear()

        categories = factor_registry.list_categories()

        for category in sorted(categories):
            # Category item
            cat_item = QTreeWidgetItem(self.factor_tree)
            cat_item.setText(0, category.upper())
            cat_item.setFont(0, QFont("Arial", 10, QFont.Weight.Bold))
            cat_item.setExpanded(True)

            # Add category checkbox
            cat_checkbox = self._create_tree_checkbox()
            cat_checkbox.stateChanged.connect(
                lambda state, c=category: self.on_category_checked(c, state)
            )
            self.factor_tree.setItemWidget(cat_item, 1, cat_checkbox)

            # Factor items
            factors = factor_registry.list_factors(category=category)
            for factor_name in sorted(factors):
                factor_item = QTreeWidgetItem(cat_item)
                factor_item.setText(0, factor_name)
                factor_item.setData(0, Qt.ItemDataRole.UserRole, factor_name)

                checkbox = self._create_tree_checkbox()
                self.factor_tree.setItemWidget(factor_item, 1, checkbox)
                self.factor_checkboxes[factor_name] = checkbox

    def populate_factor_summary(self):
        """Populate factor summary table"""
        all_info = factor_registry.get_all_factor_info()
        self.factor_summary_table.setRowCount(len(all_info))

        for i, info in enumerate(all_info):
            self.factor_summary_table.setItem(i, 0, QTableWidgetItem(info['name']))
            self.factor_summary_table.setItem(i, 1, QTableWidgetItem(info['category']))
            self.factor_summary_table.setItem(i, 2, QTableWidgetItem(str(info['default_window'])))
            self.factor_summary_table.setItem(i, 3, QTableWidgetItem(info['description']))

    def load_data(self):
        """Load stock list"""
        portal = get_data_portal()
        self.stock_list = portal.list_symbols(asset_type="stock", data_dir=self.data_dir)
        self.name_map = portal.get_name_map(asset_type="stock", stocklist_path=self.stocklist_path)

        self.stock_combo.clear()
        for code in self.stock_list:
            name = self.name_map.get(code, "")
            self.stock_combo.addItem(f"{code} {name}", code)

    def filter_factors(self, text):
        """Filter factor tree based on search text"""
        text = text.lower()

        for i in range(self.factor_tree.topLevelItemCount()):
            cat_item = self.factor_tree.topLevelItem(i)
            cat_visible = False

            for j in range(cat_item.childCount()):
                factor_item = cat_item.child(j)
                factor_name = factor_item.text(0).lower()
                visible = text in factor_name or not text
                factor_item.setHidden(not visible)
                if visible:
                    cat_visible = True

            cat_item.setHidden(not cat_visible)

    def on_category_checked(self, category, state):
        """Handle category checkbox state change"""
        checked = state == Qt.CheckState.Checked.value
        factors = factor_registry.list_factors(category=category)
        for factor_name in factors:
            if factor_name in self.factor_checkboxes:
                self.factor_checkboxes[factor_name].setChecked(checked)

    def on_factor_clicked(self, item, column):
        """Handle factor tree item click"""
        factor_name = item.data(0, Qt.ItemDataRole.UserRole)
        if factor_name:
            info = factor_registry.get_factor_info(factor_name)
            if info:
                info_text = f"""
<h2>{info['name']}</h2>
<hr>
<p><b>类别:</b> {info['category']}</p>
<p><b>默认窗口:</b> {info['default_window']}</p>
<p><b>描述:</b> {info['description']}</p>
<hr>
<h3>使用方法</h3>
<pre>
from factors import factor_registry

# 计算单个因子
result = factor_registry.compute('{info['name']}', df)

# 自定义窗口
result = factor_registry.compute('{info['name']}', df, window=30)
</pre>
"""
                self.factor_info_label.setText(info_text)

    def select_all_factors(self):
        """Select all factors"""
        for checkbox in self.factor_checkboxes.values():
            checkbox.setChecked(True)

    def clear_all_factors(self):
        """Clear all factor selections"""
        for checkbox in self.factor_checkboxes.values():
            checkbox.setChecked(False)

    def get_selected_factors(self):
        """Get list of selected factor names"""
        selected = []
        for name, checkbox in self.factor_checkboxes.items():
            if checkbox.isChecked():
                selected.append(name)
        return selected

    def refresh_analysis_factor_combo(self):
        """Refresh factor choices from the left factor tree selection."""
        current = self.analysis_factor_combo.currentText() if hasattr(self, "analysis_factor_combo") else ""
        selected = self.get_selected_factors()
        self.analysis_factor_combo.clear()
        self.analysis_factor_combo.addItems(selected)
        if current in selected:
            self.analysis_factor_combo.setCurrentText(current)
        if selected:
            self.analysis_status_label.setText(f"已加载 {len(selected)} 个候选因子")
        else:
            self.analysis_status_label.setText("请先在左侧勾选至少一个因子")

    def browse_analysis_factor_dir(self):
        """Select raw factor directory for factor analysis."""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "选择原始因子目录",
            self.analysis_factor_dir_edit.text().strip() or os.path.join(self.data_dir, "factors"),
        )
        if selected_dir:
            self.analysis_factor_dir_edit.setText(selected_dir)

    def run_factor_analysis(self):
        """Run native factor effectiveness analysis."""
        if self.factor_analysis_thread and self.factor_analysis_thread.isRunning():
            return

        if self.analysis_factor_combo.count() == 0:
            self.refresh_analysis_factor_combo()

        factor_name = self.analysis_factor_combo.currentText().strip()
        if not factor_name:
            QMessageBox.warning(self, "提示", "请先在左侧选择因子，并点击刷新已选因子")
            return

        stock_codes = self._get_selected_pool_codes()
        if not stock_codes:
            QMessageBox.warning(self, "提示", "请选择有效的股票池")
            return

        factors_dir = self.analysis_factor_dir_edit.text().strip()
        if not factors_dir or not os.path.isdir(factors_dir):
            QMessageBox.warning(self, "提示", "原始因子目录不存在")
            return

        self.run_analysis_btn.setEnabled(False)
        self.analysis_progress_bar.setVisible(True)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.setMaximum(len(stock_codes))
        self.analysis_status_label.setText("正在构建统一因子面板...")
        self.analysis_summary_label.setText("分析中...")
        self.analysis_ic_table.setRowCount(0)
        self.analysis_quantile_table.setRowCount(0)
        self.analysis_long_short_table.setRowCount(0)

        self.factor_analysis_thread = FactorAnalysisThread(
            stock_codes,
            self.data_dir,
            factors_dir,
            factor_name,
            self.analysis_start_date_edit.date().toString("yyyy-MM-dd"),
            self.analysis_end_date_edit.date().toString("yyyy-MM-dd"),
            self.analysis_forward_spin.value(),
            self.analysis_quantile_spin.value(),
        )
        self.factor_analysis_thread.progress_signal.connect(self.on_factor_analysis_progress)
        self.factor_analysis_thread.finished_signal.connect(self.on_factor_analysis_finished)
        self.factor_analysis_thread.error_signal.connect(self.on_factor_analysis_error)
        self.factor_analysis_thread.start()

    def on_factor_analysis_progress(self, current, total, code):
        self.analysis_progress_bar.setMaximum(total)
        self.analysis_progress_bar.setValue(current)
        self.analysis_status_label.setText(f"构建因子面板: {code} ({current}/{total})")

    @staticmethod
    def _fmt_number(value, digits=4):
        try:
            if pd.isna(value):
                return "N/A"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "N/A"

    @staticmethod
    def _fmt_pct(value, digits=2):
        try:
            if pd.isna(value):
                return "N/A"
            return f"{float(value) * 100:.{digits}f}%"
        except Exception:
            return "N/A"

    def on_factor_analysis_finished(self, result):
        self.run_analysis_btn.setEnabled(True)
        self.analysis_progress_bar.setVisible(False)

        summary = result.get("summary", {})
        build = result.get("build", {})
        self.analysis_status_label.setText(
            f"分析完成: 成功 {build.get('success_count', 0)} 只，失败 {build.get('fail_count', 0)} 只"
        )
        self.analysis_summary_label.setText(
            "\n".join([
                "=== 因子有效性分析（Native） ===",
                f"因子: {result.get('factor_name', '')}",
                f"未来收益: {result.get('forward_period', '')}日",
                f"样本行数: {summary.get('rows', 0):,}",
                f"交易日数: {summary.get('dates', 0):,}",
                f"股票数: {summary.get('symbols', 0):,}",
                f"IC均值: {self._fmt_number(summary.get('ic_mean'))}",
                f"ICIR: {self._fmt_number(summary.get('ic_ir'))}",
                f"Rank IC均值: {self._fmt_number(summary.get('rank_ic_mean'))}",
                f"Rank ICIR: {self._fmt_number(summary.get('rank_ic_ir'))}",
                f"多空平均收益: {self._fmt_pct(summary.get('long_short_mean'))}",
                f"多空胜率: {self._fmt_pct(summary.get('long_short_win_rate'))}",
                f"分组收益差: {self._fmt_pct(summary.get('quantile_return_spread'))}",
            ])
        )

        self._populate_ic_table(result.get("ic_by_date", pd.DataFrame()))
        self._populate_quantile_table(result.get("quantile_returns", pd.DataFrame()))
        self._populate_long_short_table(result.get("long_short_by_date", pd.DataFrame()))

    def _populate_ic_table(self, df):
        rows = min(len(df), 300) if df is not None else 0
        self.analysis_ic_table.setRowCount(rows)
        if rows == 0:
            return
        view = df.tail(rows).reset_index(drop=True)
        for i, row in view.iterrows():
            date = row.get("date")
            date_text = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            self.analysis_ic_table.setItem(i, 0, QTableWidgetItem(date_text))
            self.analysis_ic_table.setItem(i, 1, QTableWidgetItem(self._fmt_number(row.get("ic"))))
            self.analysis_ic_table.setItem(i, 2, QTableWidgetItem(self._fmt_number(row.get("rank_ic"))))
            self.analysis_ic_table.setItem(i, 3, QTableWidgetItem(str(int(row.get("count", 0)))))

    def _populate_quantile_table(self, df):
        rows = len(df) if df is not None else 0
        self.analysis_quantile_table.setRowCount(rows)
        if rows == 0:
            return
        for i, row in df.iterrows():
            self.analysis_quantile_table.setItem(i, 0, QTableWidgetItem(str(int(row.get("quantile", 0)))))
            self.analysis_quantile_table.setItem(i, 1, QTableWidgetItem(self._fmt_pct(row.get("mean"))))
            self.analysis_quantile_table.setItem(i, 2, QTableWidgetItem(self._fmt_pct(row.get("std"))))
            self.analysis_quantile_table.setItem(i, 3, QTableWidgetItem(str(int(row.get("count", 0)))))

    def _populate_long_short_table(self, df):
        rows = min(len(df), 300) if df is not None else 0
        self.analysis_long_short_table.setRowCount(rows)
        if rows == 0:
            return
        view = df.tail(rows).reset_index(drop=True)
        for i, row in view.iterrows():
            date = row.get("date")
            date_text = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            self.analysis_long_short_table.setItem(i, 0, QTableWidgetItem(date_text))
            self.analysis_long_short_table.setItem(i, 1, QTableWidgetItem(self._fmt_pct(row.get("long_return"))))
            self.analysis_long_short_table.setItem(i, 2, QTableWidgetItem(self._fmt_pct(row.get("short_return"))))
            self.analysis_long_short_table.setItem(i, 3, QTableWidgetItem(self._fmt_pct(row.get("long_short_return"))))

    def on_factor_analysis_error(self, msg):
        self.run_analysis_btn.setEnabled(True)
        self.analysis_progress_bar.setVisible(False)
        self.analysis_status_label.setText(msg)
        QMessageBox.critical(self, "因子有效性分析失败", msg)

    def plot_factors(self):
        """Load and plot factors from saved factor data files"""
        selected = self.get_selected_factors()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个因子")
            return

        # Get stock code
        import re
        text = self.stock_combo.currentText()
        match = re.search(r'\d{6}', text)
        if match:
            code = match.group(0)
        else:
            code = self.stock_combo.currentData()

        if not code:
            QMessageBox.warning(self, "提示", "请输入有效的6位股票代码")
            return

        # Look for factor data file in factors directory
        factors_dir = os.path.join(self.data_dir, "factors")
        factor_file = os.path.join(factors_dir, f"{code}.csv")
        
        if not os.path.exists(factor_file):
            # Let user select the factors directory
            selected_dir = QFileDialog.getExistingDirectory(
                self, "选择因子数据文件夹", factors_dir
            )
            if not selected_dir:
                return
            factor_file = os.path.join(selected_dir, f"{code}.csv")
            
            if not os.path.exists(factor_file):
                QMessageBox.warning(self, "提示", f"未找到股票 {code} 的因子数据文件\n请先使用批量因子计算功能生成原始因子数据")
                return

        try:
            # Load factor data
            df = pd.read_csv(factor_file)
            
            if df.empty:
                QMessageBox.warning(self, "提示", f"股票 {code} 的因子数据文件为空")
                return
            
            # Format date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # Check if selected factors exist in the data
            missing_factors = [f for f in selected if f not in df.columns]
            if missing_factors:
                QMessageBox.warning(self, "提示", 
                    f"以下因子在数据中不存在:\n{', '.join(missing_factors)}\n\n请确保因子已被计算")
                # Filter to only existing factors
                selected = [f for f in selected if f in df.columns]
                if not selected:
                    return
            
            # Get date range from factor data for loading stock price
            if 'date' in df.columns and len(df) > 0:
                start_date = df['date'].min()
                end_date = df['date'].max()
            else:
                start_date = None
                end_date = None
            
            # Load stock price data for the price chart
            stock_df = get_data_portal().get_daily_bars(
                code,
                data_dir=self.data_dir,
                start=start_date,
                end=end_date,
                asset_type="stock",
            )
            if stock_df is not None and not stock_df.empty:
                # Merge stock data with factor data
                stock_df['date'] = pd.to_datetime(stock_df['date']).dt.strftime('%Y-%m-%d')
                # Ensure both dataframes have the same date column type
                df = pd.merge(df, stock_df[['date', 'open', 'high', 'low', 'close', 'volume']], 
                             on='date', how='left')
            
            self.current_df = df
            
            # Update charts
            self.update_charts(df, selected)
            
            # Update data table
            self.update_data_table(df, selected)
            
            # Update statistics
            self.update_statistics(df, selected)
            
            # Switch to chart tab
            self.result_tabs.setCurrentIndex(0)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "加载失败", f"加载因子数据失败:\n{str(e)}")

    def update_charts(self, df, factor_names):
        """Update price and factor charts"""
        self.price_chart.clear()
        self.factor_chart.clear()
        self.plot_curves.clear()

        if df.empty:
            return

        x = np.arange(len(df))

        # Plot price
        if 'close' in df.columns:
            close_vals = df['close'].values
            # Handle NaN in price data
            valid_mask = ~np.isnan(close_vals.astype(float))
            if valid_mask.any():
                self.price_chart.plot(
                    x, close_vals,
                    pen=pg.mkPen('#0078d4', width=2),
                    name="收盘价",
                    connect='finite'  # Skip NaN values when connecting points
                )

        # Plot factors with different colors
        legend = self.factor_chart.addLegend()
        plotted_count = 0
        for i, name in enumerate(factor_names):
            if name in df.columns:
                color = self.colors[i % len(self.colors)]
                vals = df[name].values
                
                # Check if factor has any valid (non-NaN) values
                try:
                    float_vals = vals.astype(float)
                    valid_mask = ~np.isnan(float_vals)
                    if not valid_mask.any():
                        print(f"Warning: Factor '{name}' has no valid values (all NaN)")
                        continue
                except (ValueError, TypeError):
                    print(f"Warning: Factor '{name}' cannot be converted to float")
                    continue
                
                curve = self.factor_chart.plot(
                    x, float_vals,
                    pen=pg.mkPen(color, width=1.5),
                    name=name,
                    connect='finite'  # Skip NaN values when connecting points
                )
                self.plot_curves[name] = curve
                plotted_count += 1
        
        if plotted_count == 0:
            print("Warning: No factors were plotted (all selected factors have no valid data)")

        # Set X axis ticks
        self.setup_date_axis(df)

    def setup_date_axis(self, df):
        """Setup date axis ticks"""
        if 'date' not in df.columns:
            return

        dates = df['date'].astype(str).tolist()
        n = max(1, len(dates) // 10)
        ticks = [(i, dates[i]) for i in range(0, len(dates), n)]

        for chart in [self.price_chart, self.factor_chart]:
            ax = chart.getAxis('bottom')
            ax.setTicks([ticks])

    def update_data_table(self, df, factor_names):
        """Update data preview table"""
        display_cols = ['date', 'open', 'high', 'low', 'close', 'volume'] + factor_names
        display_cols = [c for c in display_cols if c in df.columns]

        self.data_table.clear()
        self.data_table.setColumnCount(len(display_cols))
        self.data_table.setHorizontalHeaderLabels(display_cols)
        self.data_table.setRowCount(min(100, len(df)))  # Show max 100 rows

        for i in range(min(100, len(df))):
            for j, col in enumerate(display_cols):
                val = df.iloc[i][col]
                if pd.isna(val):
                    text = ""
                elif isinstance(val, float):
                    text = f"{val:.4f}"
                else:
                    text = str(val)
                self.data_table.setItem(i, j, QTableWidgetItem(text))

    def update_statistics(self, df, factor_names):
        """Update statistics tables"""
        # Basic statistics
        self.stats_table.setRowCount(len(factor_names))

        for i, name in enumerate(factor_names):
            if name not in df.columns:
                continue

            series = df[name].dropna()
            if len(series) == 0:
                continue

            stats = series.describe()

            self.stats_table.setItem(i, 0, QTableWidgetItem(name))
            self.stats_table.setItem(i, 1, QTableWidgetItem(f"{stats['mean']:.4f}"))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{stats['std']:.4f}"))
            self.stats_table.setItem(i, 3, QTableWidgetItem(f"{stats['min']:.4f}"))
            self.stats_table.setItem(i, 4, QTableWidgetItem(f"{stats['25%']:.4f}"))
            self.stats_table.setItem(i, 5, QTableWidgetItem(f"{stats['50%']:.4f}"))
            self.stats_table.setItem(i, 6, QTableWidgetItem(f"{stats['75%']:.4f}"))
            self.stats_table.setItem(i, 7, QTableWidgetItem(f"{stats['max']:.4f}"))

        # Correlation matrix
        if len(factor_names) > 1:
            factor_df = df[factor_names].dropna()
            if len(factor_df) > 0:
                corr_matrix = factor_df.corr()

                self.corr_table.setRowCount(len(factor_names))
                self.corr_table.setColumnCount(len(factor_names))
                self.corr_table.setHorizontalHeaderLabels(factor_names)
                self.corr_table.setVerticalHeaderLabels(factor_names)

                for i, row_name in enumerate(factor_names):
                    for j, col_name in enumerate(factor_names):
                        val = corr_matrix.loc[row_name, col_name]
                        item = QTableWidgetItem(f"{val:.3f}")

                        # Color code correlation
                        if i != j:
                            if val > 0.7:
                                item.setBackground(QColor("#ff6b6b"))  # High positive
                            elif val > 0.3:
                                item.setBackground(QColor("#ffd93d"))  # Medium positive
                            elif val < -0.7:
                                item.setBackground(QColor("#6bcb77"))  # High negative
                            elif val < -0.3:
                                item.setBackground(QColor("#a8e6cf"))  # Medium negative

                        self.corr_table.setItem(i, j, item)

    def export_data(self):
        """Export computed factor data to CSV"""
        if self.current_df is None or self.current_df.empty:
            QMessageBox.warning(self, "提示", "请先计算因子数据")
            return

        from PyQt6.QtWidgets import QFileDialog

        # Default save to data/factors directory
        factors_dir = os.path.join(self.data_dir, "factors")
        if not os.path.exists(factors_dir):
            os.makedirs(factors_dir)
        default_path = os.path.join(factors_dir, f"factors_{QDate.currentDate().toString('yyyyMMdd')}.csv")

        filename, _ = QFileDialog.getSaveFileName(
            self, "导出数据", default_path, "CSV Files (*.csv);;All Files (*)"
        )

        if filename:
            try:
                self.current_df.to_csv(filename, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", f"数据已导出到:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", str(e))

    def _get_stocklist_dir(self):
        """Get the stocklist directory path"""
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
            display_name = filename.replace('_股票列表.csv', '').replace('.csv', '')
            file_path = os.path.join(stocklist_dir, filename)
            self.pool_combo.addItem(display_name, file_path)
        
        # Trigger initial count update
        self._on_pool_changed()
    
    def _on_pool_changed(self):
        """Handle stock pool selection change"""
        stock_codes = self._get_selected_pool_codes()
        count = len(stock_codes) if stock_codes else 0
        self.pool_count_label.setText(f"股票数量: {count}")
    
    def _get_selected_pool_codes(self):
        """Get stock codes from the selected pool file"""
        file_path = self.pool_combo.currentData()
        
        if not file_path or not os.path.exists(file_path):
            return []
        
        try:
            codes = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if parts:
                        code = parts[0].strip()
                        if code:
                            # Remove market suffix (e.g., .SH, .SZ)
                            if '.' in code:
                                code = code.split('.')[0]
                            codes.append(code)
            return codes
        except Exception as e:
            print(f"Error reading stock pool file: {e}")
            return []

    def _get_financial_factors(self) -> list:
        """Get list of financial factor names"""
        # Financial factors from financial_factors.py
        return [
            'pe', 'pe_ttm', 'pb', 'ps_ttm', 'dv_ttm', 'total_mv', 'circ_mv',
            'roe', 'roe_dt', 'roa', 'roic', 'gross_margin', 'eps',
            'netprofit_yoy', 'dt_netprofit_yoy', 'tr_yoy', 'or_yoy', 'op_yoy', 'basic_eps_yoy',
            'current_ratio', 'quick_ratio', 'debt_to_assets',
            'turnover_rate_daily', 'turnover_rate_f', 'volume_ratio_daily'
        ]
    
    def _has_financial_factors(self, selected_factors: list) -> bool:
        """Check if any financial factors are selected"""
        financial_factors = self._get_financial_factors()
        return any(f in financial_factors for f in selected_factors)
    
    def _get_tushare_token(self) -> str:
        """
        Get Tushare token from multiple sources (in priority order):
        1. Instance variable (self.tushare_token)
        2. TuShareToken.txt file in strategy_app directory
        3. Environment variable TUSHARE_TOKEN
        4. QSettings configuration
        """
        # 1. Already have token
        if self.tushare_token:
            return self.tushare_token
        
        # 2. Try to read from TuShareToken.txt
        try:
            # Get strategy_app directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            strategy_app_dir = os.path.dirname(current_dir)
            token_file = os.path.join(strategy_app_dir, "TuShareToken.txt")
            
            if os.path.exists(token_file):
                with open(token_file, 'r', encoding='utf-8') as f:
                    token = f.read().strip()
                    if token:
                        self.tushare_token = token
                        return token
        except Exception as e:
            print(f"Error reading TuShareToken.txt: {e}")
        
        # 3. Try environment variable
        token = os.environ.get("TUSHARE_TOKEN", "")
        if token:
            self.tushare_token = token
            return token
        
        # 4. Try QSettings
        try:
            from PyQt6.QtCore import QSettings
            settings = QSettings("StockTradebyZ", "StockApp")
            token = settings.value("tushare_token", "")
            if token:
                self.tushare_token = token
                return token
        except:
            pass
        
        return ""
    
    def _download_financial_data_sync(self, stock_codes: list, start_date: str = None, end_date: str = None) -> tuple:
        """Download financial data synchronously (for use before batch compute)
        
        Args:
            stock_codes: List of stock codes to download
            start_date: Start date in yyyy-MM-dd format (from UI date picker)
            end_date: End date in yyyy-MM-dd format (from UI date picker)
        """
        token = self._get_tushare_token()
        
        if not token:
            return 0, len(stock_codes), "Tushare Token 未配置\n请在 strategy_app 目录创建 TuShareToken.txt 文件并填入 Token"
        
        # Convert date format from yyyy-MM-dd to YYYYMMDD for Tushare API
        ts_start_date = start_date.replace('-', '') if start_date else None
        ts_end_date = end_date.replace('-', '') if end_date else None
        
        try:
            loader = FinancialDataLoader(
                data_dir=os.path.join(self.data_dir, "financial"),
                tushare_token=token
            )
            
            success_count = 0
            fail_count = 0
            total = len(stock_codes)
            
            for i, code in enumerate(stock_codes):
                self.batch_progress_label.setText(f"下载财务数据: {code} ({i+1}/{total})")
                self.batch_progress_bar.setValue(i + 1)
                QApplication.processEvents()  # Keep UI responsive
                
                try:
                    # Download daily basic data with user-specified date range
                    result1 = loader.download_daily_basic(code, start_date=ts_start_date, end_date=ts_end_date)
                    # Download financial indicators
                    result2 = loader.download_fina_indicator(code)
                    
                    if (result1 is not None and not result1.empty) or \
                       (result2 is not None and not result2.empty):
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    print(f"Error downloading financial data for {code}: {e}")
                    fail_count += 1
                
                # Rate limiting for Tushare API
                import time
                time.sleep(0.15)
            
            return success_count, fail_count, None
            
        except Exception as e:
            return 0, len(stock_codes), str(e)

    def batch_compute_factors(self):
        """Batch compute factors for all stocks in selected pool and save"""
        if self.batch_compute_thread and self.batch_compute_thread.isRunning():
            return

        selected = self.get_selected_factors()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个因子")
            return

        stock_codes = self._get_selected_pool_codes()
        if not stock_codes:
            QMessageBox.warning(self, "提示", "请选择有效的股票池")
            return

        # Check if financial factors are selected
        has_financial = self._has_financial_factors(selected)
        
        if has_financial:
            # Try to get tushare token from multiple sources
            token = self._get_tushare_token()
            
            if not token:
                QMessageBox.warning(self, "提示", 
                    "您选择了财务因子，但 Tushare Token 未配置\n\n"
                    "请在 strategy_app 目录创建 TuShareToken.txt 文件并填入 Token\n"
                    "或设置环境变量 TUSHARE_TOKEN\n"
                    "Token 可以从 https://tushare.pro 获取")
                return
            
            # Confirm download
            reply = QMessageBox.question(
                self, "下载财务数据",
                f"您选择了财务因子，需要从 Tushare 下载 {len(stock_codes)} 只股票的财务数据\n\n"
                f"这可能需要几分钟时间，是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Default save to data/factors directory
        factors_dir = os.path.join(self.data_dir, "factors")
        if not os.path.exists(factors_dir):
            os.makedirs(factors_dir)
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "选择保存目录", factors_dir
        )
        
        if not output_dir:
            return

        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")

        # Setup UI for progress
        self.batch_compute_btn.setEnabled(False)
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setMaximum(len(stock_codes))
        self.batch_progress_label.setVisible(True)
        
        # Download financial data first if needed
        if has_financial:
            self.batch_progress_label.setText("正在下载财务数据...")
            QApplication.processEvents()
            
            success, fail, error = self._download_financial_data_sync(stock_codes, start_date, end_date)
            
            if error:
                self.batch_compute_btn.setEnabled(True)
                self.batch_progress_bar.setVisible(False)
                self.batch_progress_label.setVisible(False)
                QMessageBox.critical(self, "下载失败", f"财务数据下载失败: {error}")
                return
            
            if fail > 0:
                # Some failed, but continue with computation
                self.batch_progress_label.setText(f"财务数据下载完成 (成功:{success}, 失败:{fail})，开始计算...")
            else:
                self.batch_progress_label.setText("财务数据下载完成，开始计算...")
            
            QApplication.processEvents()
        
        self.batch_progress_label.setText("正在准备计算...")
        self.batch_progress_bar.setValue(0)

        self.batch_compute_thread = BatchFactorComputeThread(
            stock_codes, self.data_dir, start_date, end_date, selected, output_dir
        )
        self.batch_compute_thread.finished_signal.connect(self.on_batch_compute_finished)
        self.batch_compute_thread.error_signal.connect(self.on_batch_compute_error)
        self.batch_compute_thread.progress_signal.connect(self.on_batch_progress)
        self.batch_compute_thread.start()

    def on_batch_progress(self, current, total, stock_code):
        """Handle batch computation progress update"""
        self.batch_progress_bar.setValue(current)
        self.batch_progress_label.setText(f"正在计算: {stock_code} ({current}/{total})")

    def on_batch_compute_finished(self, output_dir, success_count, fail_count):
        """Handle batch computation completion"""
        self.batch_compute_btn.setEnabled(True)
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_label.setVisible(False)
        
        msg = f"因子数据已保存到:\n{output_dir}\n\n"
        msg += f"成功: {success_count} 只股票\n"
        if fail_count > 0:
            msg += f"失败: {fail_count} 只股票"
        
        QMessageBox.information(self, "完成", msg)

    def on_batch_compute_error(self, msg):
        """Handle batch computation error"""
        self.batch_compute_btn.setEnabled(True)
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_label.setVisible(False)
        QMessageBox.critical(self, "因子批量计算失败", msg)

    def check_factor_anomalies(self):
        """Check for anomalies in all factor files in factors folder"""
        # Default factors directory
        factors_dir = os.path.join(self.data_dir, "factors")
        
        # Let user select the factors directory
        selected_dir = QFileDialog.getExistingDirectory(
            self, "选择因子数据文件夹", factors_dir
        )
        
        if not selected_dir:
            return
        
        # Find all CSV files in the directory
        csv_files = [f for f in os.listdir(selected_dir) if f.endswith('.csv')]
        
        if not csv_files:
            QMessageBox.warning(self, "提示", "所选文件夹中没有CSV文件")
            return
        
        # Store all anomaly information
        all_anomalies = []  # List of dict with anomaly summary per factor
        inf_details = []    # List of dict with detailed inf locations
        total_files = len(csv_files)
        processed_files = 0
        failed_files = []
        total_rows = 0
        all_factor_cols = set()
        
        # Process each file
        for csv_file in csv_files:
            file_path = os.path.join(selected_dir, csv_file)
            stock_code = csv_file.replace('.csv', '')
            
            try:
                df = pd.read_csv(file_path)
                processed_files += 1
                total_rows += len(df)
            except Exception as e:
                failed_files.append((csv_file, str(e)))
                continue
            
            # Get factor columns (exclude code, date, OHLCV)
            exclude_cols = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
            factor_cols = [col for col in df.columns if col not in exclude_cols]
            all_factor_cols.update(factor_cols)
            
            if not factor_cols:
                continue
            
            # Check each factor column for anomalies
            for col in factor_cols:
                col_data = df[col]
                
                # Count statistics
                total_count = len(col_data)
                nan_count = col_data.isna().sum()
                
                # Check for inf values and get their locations
                inf_mask = np.isinf(col_data.replace([np.nan], 0))
                inf_count = inf_mask.sum()
                
                # If there are inf values, record detailed locations
                if inf_count > 0:
                    inf_rows = df[inf_mask]
                    for idx, row in inf_rows.iterrows():
                        date_val = row.get('date', 'N/A')
                        inf_val = row[col]
                        inf_details.append({
                            'stock': stock_code,
                            'factor': col,
                            'date': date_val,
                            'value': inf_val
                        })
                
                # Check for extreme values (beyond 5 std)
                valid_data = col_data.replace([np.inf, -np.inf], np.nan).dropna()
                if len(valid_data) > 0:
                    mean = valid_data.mean()
                    std = valid_data.std()
                    if std > 0:
                        extreme_count = ((valid_data < mean - 5*std) | (valid_data > mean + 5*std)).sum()
                    else:
                        extreme_count = 0
                else:
                    extreme_count = 0
                
                # Aggregate anomalies by factor
                if nan_count > 0 or inf_count > 0 or extreme_count > 0:
                    # Check if this factor already exists in the report
                    existing = next((x for x in all_anomalies if x['factor'] == col), None)
                    if existing:
                        existing['nan_count'] += nan_count
                        existing['inf_count'] += inf_count
                        existing['extreme_count'] += extreme_count
                        existing['total'] += total_count
                        if inf_count > 0:
                            existing['stocks_with_inf'].add(stock_code)
                    else:
                        all_anomalies.append({
                            'factor': col,
                            'total': total_count,
                            'nan_count': nan_count,
                            'inf_count': inf_count,
                            'extreme_count': extreme_count,
                            'stocks_with_inf': {stock_code} if inf_count > 0 else set()
                        })
        
        # Calculate nan percentage
        for item in all_anomalies:
            item['nan_pct'] = item['nan_count'] / item['total'] * 100 if item['total'] > 0 else 0
            item['stocks_with_inf'] = list(item['stocks_with_inf'])  # Convert set to list for display
        
        # Show report
        self._show_anomaly_report_batch(
            selected_dir, all_anomalies, inf_details, 
            total_rows, list(all_factor_cols), 
            total_files, processed_files, failed_files
        )

    def _show_anomaly_report_batch(self, folder_path, anomaly_report, inf_details, 
                                     total_rows, factor_cols, total_files, 
                                     processed_files, failed_files):
        """Show batch anomaly report in a dialog"""
        from PyQt6.QtWidgets import QDialog, QTextEdit, QVBoxLayout, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("因子数据异常检查报告")
        dialog.setMinimumSize(900, 700)
        
        layout = QVBoxLayout(dialog)
        
        report_text = QTextEdit()
        report_text.setReadOnly(True)
        
        # Build report
        report = f"""<h2>因子数据异常检查报告</h2>
<hr>
<p><b>文件夹:</b> {folder_path}</p>
<p><b>文件总数:</b> {total_files} (成功读取: {processed_files}, 失败: {len(failed_files)})</p>
<p><b>总数据行数:</b> {total_rows}</p>
<p><b>检查因子数:</b> {len(factor_cols)}</p>
<hr>
"""
        
        # Show failed files if any
        if failed_files:
            report += "<h3 style='color: #d83b01;'>读取失败的文件:</h3>"
            report += "<ul>"
            for fname, error in failed_files[:10]:  # Show max 10
                report += f"<li>{fname}: {error}</li>"
            if len(failed_files) > 10:
                report += f"<li>... 及其他 {len(failed_files) - 10} 个文件</li>"
            report += "</ul><hr>"
        
        # Show summary of anomalies
        if not anomaly_report:
            report += "<p style='color: green;'><b>✓ 未发现异常值，所有因子数据正常！</b></p>"
        else:
            report += f"<h3 style='color: #d83b01;'>发现 {len(anomaly_report)} 个因子存在异常值:</h3>"
            report += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
            report += "<tr style='background-color: #404040;'>"
            report += "<th>因子名称</th><th>缺失值(NaN)</th><th>缺失比例</th><th>无穷值(Inf)</th><th>极端值(>5σ)</th>"
            report += "</tr>"
            
            for item in anomaly_report:
                nan_color = "#ff6b6b" if item['nan_pct'] > 10 else "#ffd93d" if item['nan_pct'] > 1 else ""
                inf_color = "#ff6b6b" if item['inf_count'] > 0 else ""
                extreme_color = "#ffd93d" if item['extreme_count'] > 0 else ""
                
                report += f"<tr>"
                report += f"<td>{item['factor']}</td>"
                report += f"<td style='background-color: {nan_color};'>{item['nan_count']}</td>"
                report += f"<td style='background-color: {nan_color};'>{item['nan_pct']:.2f}%</td>"
                report += f"<td style='background-color: {inf_color};'>{item['inf_count']}</td>"
                report += f"<td style='background-color: {extreme_color};'>{item['extreme_count']}</td>"
                report += "</tr>"
            
            report += "</table>"
        
        # Show detailed Inf locations
        if inf_details:
            report += f"<hr><h3 style='color: #ff6b6b;'>无穷值(Inf)详细位置 (共 {len(inf_details)} 处):</h3>"
            report += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
            report += "<tr style='background-color: #404040;'>"
            report += "<th>股票代码</th><th>因子名称</th><th>日期</th><th>异常值</th>"
            report += "</tr>"
            
            # Show max 100 records
            for item in inf_details[:100]:
                val_str = "+Inf" if item['value'] == np.inf else "-Inf" if item['value'] == -np.inf else str(item['value'])
                report += f"<tr>"
                report += f"<td>{item['stock']}</td>"
                report += f"<td>{item['factor']}</td>"
                report += f"<td>{item['date']}</td>"
                report += f"<td style='color: #ff6b6b;'>{val_str}</td>"
                report += "</tr>"
            
            if len(inf_details) > 100:
                report += f"<tr><td colspan='4'>... 及其他 {len(inf_details) - 100} 处异常值</td></tr>"
            
            report += "</table>"
            
            # Show which stocks have inf values for each factor
            report += "<h4>各因子的Inf异常股票列表:</h4>"
            for item in anomaly_report:
                if item['inf_count'] > 0 and item['stocks_with_inf']:
                    stocks_str = ", ".join(item['stocks_with_inf'][:20])
                    if len(item['stocks_with_inf']) > 20:
                        stocks_str += f" ... 及其他 {len(item['stocks_with_inf']) - 20} 只"
                    report += f"<p><b>{item['factor']}:</b> {stocks_str}</p>"
        
        report += """
<hr>
<h3>说明</h3>
<ul>
<li><b>缺失值(NaN)</b>: 因子计算结果为空，通常是因为数据不足或计算窗口期未满</li>
<li><b>无穷值(Inf)</b>: 因子计算出现除零或溢出，需要检查计算逻辑</li>
<li><b>极端值(>5σ)</b>: 超过5个标准差的异常值，可能需要截断处理</li>
</ul>
<p><i>注: 本报告仅检测异常值，不进行自动修复</i></p>
"""
        
        report_text.setHtml(report)
        layout.addWidget(report_text)
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

    # ==================== Data Preprocessing Methods ====================
    
    def _get_preprocess_config(self) -> dict:
        """Get preprocessing configuration from UI controls"""
        # Missing value method
        missing_map = {
            0: 'median',
            1: 'mean',
            2: 'ffill',
            3: 'bfill',
            4: 'drop',
            5: 'industry_mean'
        }
        missing_method = missing_map.get(self.missing_method_combo.currentIndex(), 'median')
        
        # Winsorize method
        winsorize_map = {
            0: 'mad',
            1: 'sigma',
            2: 'percentile',
            3: 'none'
        }
        winsorize_method = winsorize_map.get(self.winsorize_method_combo.currentIndex(), 'mad')
        
        # Winsorize threshold
        try:
            winsorize_n = float(self.winsorize_n_edit.text())
        except ValueError:
            winsorize_n = 3.0
        
        # Standardize method
        standardize_map = {
            0: 'zscore',
            1: 'minmax',
            2: 'rank',
            3: 'none'
        }
        standardize_method = standardize_map.get(self.standardize_method_combo.currentIndex(), 'zscore')
        
        # Neutralize method
        neutralize_map = {
            0: 'none',
            1: 'size',
            2: 'industry',
            3: 'size_industry'
        }
        neutralize_method = neutralize_map.get(self.neutralize_method_combo.currentIndex(), 'none')
        
        return {
            'missing_method': missing_method,
            'winsorize_method': winsorize_method,
            'winsorize_n': winsorize_n,
            'standardize_method': standardize_method,
            'neutralize_method': neutralize_method,
            'size_col': self.size_col_edit.text() or 'total_mv',
            'industry_col': self.industry_col_edit.text() or 'industry'
        }

    def _compute_factor_stats(self, df: pd.DataFrame, factor_columns: list) -> list:
        """Compute statistics for factors"""
        stats = []
        for col in factor_columns:
            if col not in df.columns:
                continue
            series = df[col]
            valid_count = series.notna().sum()
            missing_count = series.isna().sum()
            total = len(series)
            missing_pct = (missing_count / total * 100) if total > 0 else 0
            
            valid_data = series.dropna()
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                min_val = valid_data.min()
                max_val = valid_data.max()
            else:
                mean_val = std_val = min_val = max_val = np.nan
            
            stats.append({
                'factor': col,
                'valid_count': valid_count,
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            })
        return stats

    def _update_preprocess_stats_table(self, table: QTableWidget, stats: list):
        """Update preprocessing statistics table"""
        table.setRowCount(len(stats))
        
        for i, s in enumerate(stats):
            table.setItem(i, 0, QTableWidgetItem(s['factor']))
            table.setItem(i, 1, QTableWidgetItem(str(s['valid_count'])))
            table.setItem(i, 2, QTableWidgetItem(str(s['missing_count'])))
            table.setItem(i, 3, QTableWidgetItem(f"{s['missing_pct']:.2f}%"))
            table.setItem(i, 4, QTableWidgetItem(f"{s['mean']:.4f}" if not np.isnan(s['mean']) else "N/A"))
            table.setItem(i, 5, QTableWidgetItem(f"{s['std']:.4f}" if not np.isnan(s['std']) else "N/A"))
            table.setItem(i, 6, QTableWidgetItem(f"{s['min']:.4f}" if not np.isnan(s['min']) else "N/A"))
            table.setItem(i, 7, QTableWidgetItem(f"{s['max']:.4f}" if not np.isnan(s['max']) else "N/A"))
            
            # Highlight rows with high missing rate
            if s['missing_pct'] > 10:
                for j in range(8):
                    item = table.item(i, j)
                    if item:
                        item.setBackground(QColor("#ff6b6b"))

    def preview_preprocessing(self):
        """Preview preprocessing effect on current stock data"""
        selected = self.get_selected_factors()
        if not selected:
            QMessageBox.warning(self, "提示", "请至少选择一个因子")
            return

        # Get stock code
        import re
        text = self.stock_combo.currentText()
        match = re.search(r'\d{6}', text)
        if match:
            code = match.group(0)
        else:
            code = self.stock_combo.currentData()

        if not code:
            QMessageBox.warning(self, "提示", "请输入有效的6位股票代码")
            return

        # Look for factor data file
        factors_dir = os.path.join(self.data_dir, "factors")
        factor_file = os.path.join(factors_dir, f"{code}.csv")
        
        if not os.path.exists(factor_file):
            # Let user select the factors directory
            selected_dir = QFileDialog.getExistingDirectory(
                self, "选择因子数据文件夹", factors_dir
            )
            if not selected_dir:
                return
            factor_file = os.path.join(selected_dir, f"{code}.csv")
            
            if not os.path.exists(factor_file):
                QMessageBox.warning(self, "提示", f"未找到股票 {code} 的因子数据文件\n请先使用批量因子计算功能生成原始因子数据")
                return

        try:
            # Load factor data
            df = pd.read_csv(factor_file)
            
            if df.empty:
                QMessageBox.warning(self, "提示", f"股票 {code} 的因子数据文件为空")
                return

            # Check if selected factors exist
            available_factors = [f for f in selected if f in df.columns]
            if not available_factors:
                QMessageBox.warning(self, "提示", "所选因子在数据中不存在")
                return

            # Compute before stats
            before_stats = self._compute_factor_stats(df, available_factors)
            self._update_preprocess_stats_table(self.preprocess_before_table, before_stats)

            # Get preprocessing config
            config = self._get_preprocess_config()
            
            # Apply preprocessing
            preprocessor = FactorPreprocessor()
            processed_df = preprocessor.process_dataframe(
                df.copy(),
                available_factors,
                missing_method=config['missing_method'],
                winsorize_method=config['winsorize_method'],
                winsorize_n=config['winsorize_n'],
                standardize_method=config['standardize_method'],
                neutralize_method=config['neutralize_method'],
                size_col=config['size_col'] if config['neutralize_method'] in ['size', 'size_industry'] else None,
                industry_col=config['industry_col'] if config['neutralize_method'] in ['industry', 'size_industry'] else None
            )

            # Compute after stats
            after_stats = self._compute_factor_stats(processed_df, available_factors)
            self._update_preprocess_stats_table(self.preprocess_after_table, after_stats)

            # Store data for chart
            self._preprocess_before_df = df[available_factors].copy()
            self._preprocess_after_df = processed_df[available_factors].copy()
            
            # Update factor combo for chart
            self.preprocess_factor_combo.clear()
            self.preprocess_factor_combo.addItems(available_factors)
            
            # Update chart
            self.update_preprocess_chart()
            
            # Switch to preprocessing tab
            self.result_tabs.setCurrentIndex(4)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "预处理失败", f"预处理预览失败:\n{str(e)}")

    def update_preprocess_chart(self):
        """Update preprocessing distribution comparison chart"""
        self.preprocess_chart.clear()
        
        if not hasattr(self, '_preprocess_before_df') or not hasattr(self, '_preprocess_after_df'):
            return
        
        factor_name = self.preprocess_factor_combo.currentText()
        if not factor_name:
            return
        
        if factor_name not in self._preprocess_before_df.columns:
            return
        
        before_data = self._preprocess_before_df[factor_name].dropna().values
        after_data = self._preprocess_after_df[factor_name].dropna().values
        
        if len(before_data) == 0 and len(after_data) == 0:
            return
        
        # Compute histograms
        bins = 30
        
        # Before histogram
        if len(before_data) > 0:
            before_hist, before_edges = np.histogram(before_data, bins=bins)
            before_centers = (before_edges[:-1] + before_edges[1:]) / 2
            self.preprocess_chart.plot(
                before_centers, before_hist,
                pen=pg.mkPen('#e6194b', width=2),
                fillLevel=0,
                fillBrush=pg.mkBrush(230, 25, 75, 80),
                name="预处理前"
            )
        
        # After histogram
        if len(after_data) > 0:
            after_hist, after_edges = np.histogram(after_data, bins=bins)
            after_centers = (after_edges[:-1] + after_edges[1:]) / 2
            self.preprocess_chart.plot(
                after_centers, after_hist,
                pen=pg.mkPen('#3cb44b', width=2),
                fillLevel=0,
                fillBrush=pg.mkBrush(60, 180, 75, 80),
                name="预处理后"
            )

    def batch_preprocess_factors(self):
        """Batch preprocess factor data files"""
        # Select input directory
        factors_dir = os.path.join(self.data_dir, "factors")
        input_dir = QFileDialog.getExistingDirectory(
            self, "选择因子数据输入文件夹", factors_dir
        )
        
        if not input_dir:
            return
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        
        if not csv_files:
            QMessageBox.warning(self, "提示", "所选文件夹中没有CSV文件")
            return
        
        # Select output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "选择预处理结果保存文件夹", 
            os.path.join(self.data_dir, "factors_timeseries_preprocessed")
        )
        
        if not output_dir:
            return
        
        if input_dir == output_dir:
            reply = QMessageBox.question(
                self, "确认",
                "输入和输出文件夹相同，原文件将被覆盖。是否继续？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Get preprocessing config
        config = self._get_preprocess_config()
        
        # Get selected factors (or detect from first file)
        selected = self.get_selected_factors()
        if not selected:
            # Try to detect factors from first file
            try:
                first_df = pd.read_csv(os.path.join(input_dir, csv_files[0]))
                exclude_cols = ['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount',
                               'total_mv', 'circ_mv', 'industry']
                selected = [c for c in first_df.columns if c not in exclude_cols]
            except:
                pass
        
        if not selected:
            QMessageBox.warning(self, "提示", "请选择要预处理的因子或确保因子数据文件格式正确")
            return
        
        # Confirm
        reply = QMessageBox.question(
            self, "确认批量单股时序预处理",
            f"将对 {len(csv_files)} 个文件进行单股时序预处理\n\n"
            f"处理流程:\n"
            f"1. 缺失值处理: {config['missing_method']}\n"
            f"2. 去极值: {config['winsorize_method']} (n={config['winsorize_n']})\n"
            f"3. 标准化: {config['standardize_method']}\n"
            f"4. 中性化: {config['neutralize_method']}\n\n"
            f"处理因子: {', '.join(selected[:5])}{'...' if len(selected) > 5 else ''}\n\n"
            f"是否继续?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Start batch processing
        self.batch_preprocess_btn.setEnabled(False)
        self.preprocess_progress_bar.setVisible(True)
        self.preprocess_progress_bar.setValue(0)
        self.preprocess_progress_bar.setMaximum(len(csv_files))
        self.preprocess_progress_label.setVisible(True)
        
        preprocessor = FactorPreprocessor()
        success_count = 0
        fail_count = 0
        total = len(csv_files)
        
        try:
            for i, csv_file in enumerate(csv_files):
                stock_code = csv_file.replace('.csv', '')
                self.preprocess_progress_bar.setValue(i + 1)
                self.preprocess_progress_label.setText(f"处理: {stock_code} ({i+1}/{total})")
                QApplication.processEvents()
                
                try:
                    input_path = os.path.join(input_dir, csv_file)
                    output_path = os.path.join(output_dir, csv_file)
                    
                    # Load data
                    df = pd.read_csv(input_path)
                    if df.empty:
                        fail_count += 1
                        continue
                    
                    # Filter to existing factors
                    available_factors = [f for f in selected if f in df.columns]
                    if not available_factors:
                        fail_count += 1
                        continue
                    
                    # Apply preprocessing
                    processed_df = preprocessor.process_dataframe(
                        df,
                        available_factors,
                        missing_method=config['missing_method'],
                        winsorize_method=config['winsorize_method'],
                        winsorize_n=config['winsorize_n'],
                        standardize_method=config['standardize_method'],
                        neutralize_method=config['neutralize_method'],
                        size_col=config['size_col'] if config['neutralize_method'] in ['size', 'size_industry'] else None,
                        industry_col=config['industry_col'] if config['neutralize_method'] in ['industry', 'size_industry'] else None
                    )
                    
                    # Save processed data
                    processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    success_count += 1
                    
                except Exception as e:
                    print(f"Error processing {csv_file}: {e}")
                    fail_count += 1
                    continue
            
            # Show completion message
            msg = f"批量单股时序预处理完成!\n\n"
            msg += f"成功: {success_count} 个文件\n"
            if fail_count > 0:
                msg += f"失败: {fail_count} 个文件\n"
            msg += f"\n结果已保存到:\n{output_dir}"
            
            QMessageBox.information(self, "完成", msg)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "批量单股时序预处理失败", f"错误: {str(e)}")
        
        finally:
            self.batch_preprocess_btn.setEnabled(True)
            self.preprocess_progress_bar.setVisible(False)
            self.preprocess_progress_label.setVisible(False)
