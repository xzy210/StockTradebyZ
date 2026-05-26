"""
XGBoost 截面选股策略

该策略面向股票池横截面排序：每个调仓日对全市场候选股票计算因子，使用 XGBoost 学习因子与未来收益之间的关系，再选择预测收益最高的若干股票持仓。
"""

from .cross_sectional_strategy import CrossSectionalStrategy
import pandas as pd
import numpy as np
from typing import ClassVar, Dict, List, Any
from pathlib import Path

from common.strategy_spec import StrategySpec

try:
    from strategy_app.factors.preprocessor import FactorPreprocessor
except ImportError:
    from factors.preprocessor import FactorPreprocessor

# 尝试导入 XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: xgboost not installed. Please run: pip install xgboost")


class XGBoostCrossSectionalStrategy(CrossSectionalStrategy):
    """
    XGBoost 机器学习截面选股策略
    
    核心思想：
    1. 使用历史因子数据训练 XGBoost 模型预测未来收益
    2. 每次调仓时用滚动窗口重新训练模型（适应市场变化）
    3. 用训练好的模型对当期所有股票打分，选择预测收益最高的股票
    
    与传统多因子策略的区别：
    - 传统多因子：线性加权 score = Σ(factor × weight)
    - XGBoost：非线性模型 score = model.predict(factors)，能捕捉因子间的交互效应
    """

    spec: ClassVar[StrategySpec] = StrategySpec(
        strategy_id="xgboost_cross_sectional",
        strategy_name="XGBoost截面选股策略",
        owner_type="research",
        asset_class="stock",
        frequency="daily",
        metadata={"source": "strategy_app", "strategy_family": "xgboost_cross_sectional"},
    )
    
    def __init__(self):
        super().__init__()
        self.name = "XGBoost截面选股策略"
        self.description = "使用XGBoost模型学习因子与收益的非线性关系，进行截面选股"
        
        # Get default factors directory
        project_root = Path(__file__).parent.parent.parent
        default_factors_dir = str(project_root / "data" / "factors")
        
        self.params = {
            "top_k": 5,                 # 持仓数量
            "rebalance_period": 20,     # 调仓周期 (交易日)
            "train_window": 252,        # 训练窗口 (约1年的交易日)
            "min_train_samples": 500,   # 最小训练样本数
            "filter_downtrend": True,   # 是否开启趋势过滤
            "trend_ma": 20,             # 趋势判断均线
            "label_period": None,       # 标签收益率间隔天数, None则使用rebalance_period
            "clip_range": 0.2,          # 训练标签收益率clip范围 (±clip_range)
            
            # Stop loss settings
            "enable_stop_loss": True,   # 是否开启个股止损
            "stop_loss_pct": 0.08,      # 止损阈值 (8%)
            
            # Factor input and cross-sectional preprocessing settings
            "factors_dir": default_factors_dir,  # 原始因子文件目录
            "enable_cross_sectional_preprocess": True,  # 是否按交易日做截面预处理
            "cross_missing_method": "median",            # 截面缺失值处理
            "cross_winsorize_method": "mad",             # 截面去极值方法
            "cross_winsorize_n": 3.0,                    # 截面去极值阈值
            "cross_standardize_method": "zscore",        # 截面标准化方法
            
            # XGBoost 超参数
            "xgb_params": {
                "objective": "reg:squarederror",  # 回归任务：预测收益率
                "max_depth": 4,                   # 树深度（防止过拟合）
                "learning_rate": 0.1,             # 学习率
                "n_estimators": 100,              # 树的数量
                "subsample": 0.8,                 # 行采样比例
                "colsample_bytree": 0.8,          # 列采样比例
                "random_state": 42,
                "n_jobs": 1,                      # 单线程避免UI卡顿
            },
            
            # 因子列表
            "factor_cols": [
                # Technical factors
                "momentum_20d",     # 20日动量
                "momentum_60d",     # 60日动量
                "reversal_5d",      # 5日反转
                "volatility_20d",   # 20日波动率
                "turnover_20d",     # 20日平均换手率
                "volume_ratio",     # 量比
                "bias_20",          # 20日乖离率
                
                # Financial factors - Valuation
                "pe_ttm",           # 市盈率(TTM)
                "pb",               # 市净率
                "ps_ttm",           # 市销率(TTM)
                "dv_ttm",           # 股息率(TTM)
                
                # Financial factors - Profitability
                "roe",              # 净资产收益率
                "roa",              # 总资产收益率
                "gross_margin",     # 销售毛利率
                
                # Financial factors - Growth
                "netprofit_yoy",    # 净利润同比增长率
                "tr_yoy",           # 营业总收入同比增长率
                "basic_eps_yoy",    # 基本每股收益同比增长率
                
                # Financial factors - Solvency
                "current_ratio",    # 流动比率
                "debt_to_assets",   # 资产负债率
            ]
        }
        
        self.rebalance_counter = 0
        self.model = None
        self.all_data = None  # 存储所有历史数据用于训练
        self.last_scores = None  # 存储评分供回放功能使用
        self.feature_importance = {}  # 特征重要性
        self.last_train_info = None  # 存储最近一次训练的信息（供UI显示）
        self.cost_prices = {}  # 持仓成本价跟踪
        
    def prepare_factors(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        从原始因子文件读取因子数据，并按交易日做截面预处理。
        
        注意：这里不训练模型，模型在 on_rebalance 中滚动训练，
        以避免使用未来数据（数据泄露）
        """
        if not HAS_XGBOOST:
            raise ImportError("xgboost 未安装，请运行: pip install xgboost")
        all_factors = []
        factor_cols = self.params['factor_cols']
        factors_dir = Path(self.params.get('factors_dir', ''))
        
        if not factors_dir.exists():
            raise FileNotFoundError(f"Factors directory does not exist: {factors_dir}")
        
        for code, df in data_dict.items():
            if len(df) < 60:
                continue
            
            # Build factor file path (remove suffix like .SH, .SZ if exists)
            code_clean = code.split('.')[0] if '.' in code else code
            factor_file = factors_dir / f"{code_clean}.csv"
            
            if not factor_file.exists():
                print(f"Warning: Factor file not found for {code}: {factor_file}")
                continue
            
            try:
                # Read factor file
                factor_df = pd.read_csv(factor_file)
                factor_df['date'] = pd.to_datetime(factor_df['date'])
                factor_df = factor_df.set_index('date').sort_index()
                
                # Prepare original price data for trend filtering and target calculation
                df = df.copy()
                df = df.set_index('date').sort_index()
                
                # Merge factor data with price data
                merged = factor_df.join(df[['close']], how='inner')
                
                if len(merged) < 60:
                    continue
                
                # Filter to only include required factor columns that exist
                available_factor_cols = [c for c in factor_cols if c in merged.columns]
                if not available_factor_cols:
                    print(f"Warning: No required factors found in file for {code}")
                    continue
                
                # === Trend filtering ===
                if self.params.get('filter_downtrend', False):
                    ma_window = self.params.get('trend_ma', 20)
                    merged['trend_ma'] = merged['close'].rolling(ma_window).mean()
                    merged['is_uptrend'] = merged['close'] > merged['trend_ma']
                
                # === Target variable ===
                # Forward return over label_period days for model training
                # label_period defaults to rebalance_period if not specified
                label_k = self.params.get('label_period') or self.params['rebalance_period']
                merged['next_ret'] = merged['close'].shift(-label_k) / merged['close'] - 1.0
                
                # Select output columns
                cols = available_factor_cols.copy()
                cols.extend(['next_ret', 'close'])
                if 'is_uptrend' in merged.columns:
                    cols.append('is_uptrend')
                
                # Keep only valid columns
                valid_cols = [c for c in cols if c in merged.columns]
                factors = merged[valid_cols].dropna(subset=['next_ret', 'close'])
                factors.index.name = 'date'
                factors['code'] = code
                all_factors.append(factors)
                
            except Exception as e:
                print(f"Error reading factor file for {code}: {e}")
                continue
        
        if not all_factors:
            print("Warning: No valid factor data loaded from files")
            return pd.DataFrame()
        
        # Combine data, then apply one cross-sectional preprocessing pass shared
        # by both rolling training and prediction slices.
        combined = pd.concat(all_factors)
        combined = self._apply_cross_sectional_preprocess(combined, factor_cols)
        self.all_data = combined.copy()  # Save for training
        
        return combined.reset_index().set_index(['date', 'code']).sort_index()

    def _apply_cross_sectional_preprocess(
        self,
        combined: pd.DataFrame,
        factor_cols: List[str],
    ) -> pd.DataFrame:
        """Apply factor preprocessing by date so train and predict use one scale."""
        if combined.empty or not self.params.get("enable_cross_sectional_preprocess", True):
            return combined

        panel = combined.reset_index()
        if "date" not in panel.columns and "index" in panel.columns:
            panel = panel.rename(columns={"index": "date"})

        available_factor_cols = [col for col in factor_cols if col in panel.columns]
        if not available_factor_cols:
            return combined

        preprocessor = FactorPreprocessor()
        processed = preprocessor.process_cross_sectional(
            panel,
            date_col="date",
            factor_columns=available_factor_cols,
            missing_method=self.params.get("cross_missing_method", "median"),
            winsorize_method=self.params.get("cross_winsorize_method", "mad"),
            winsorize_n=float(self.params.get("cross_winsorize_n", 3.0)),
            standardize_method=self.params.get("cross_standardize_method", "zscore"),
            neutralize_method="none",
        )
        processed["date"] = pd.to_datetime(processed["date"])
        return processed.set_index("date").sort_index()
    
    def _train_model(self, train_data: pd.DataFrame):
        """
        训练 XGBoost 模型
        
        :param train_data: 训练数据，包含因子列和 next_ret 列
        """
        feature_cols = [c for c in self.params['factor_cols'] if c in train_data.columns]
        
        if not feature_cols:
            return
        
        X = train_data[feature_cols].values
        y = train_data['next_ret'].values
        
        # 处理异常值：限制收益率范围，防止极端值影响模型
        clip_range = self.params.get('clip_range', 0.2)
        y = np.clip(y, -clip_range, clip_range)
        
        # 处理 NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < self.params['min_train_samples']:
            return
        
        # 创建并训练模型
        self.model = xgb.XGBRegressor(**self.params['xgb_params'])
        self.model.fit(X, y, verbose=False)
        
        # 记录特征重要性
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        sorted_importance = sorted(self.feature_importance.items(), key=lambda x: -x[1])
        
        # 保存训练信息供 UI 显示
        self.last_train_info = {
            'train_samples': len(X),
            'feature_importance': sorted_importance
        }
        
    def _check_stop_loss(self, context, daily_factors: pd.DataFrame):
        """
        检查止损条件，对达到止损阈值的个股执行卖出
        
        :param context: 回测上下文
        :param daily_factors: 当日因子数据 (包含 close 价格)
        :return: 被止损卖出的股票列表
        """
        stopped_codes = []
        
        if not self.params.get('enable_stop_loss', False):
            return stopped_codes
        
        stop_loss_pct = self.params.get('stop_loss_pct', 0.08)
        
        for code, position in list(context.positions.items()):
            # Get cost price
            cost_price = self.cost_prices.get(code)
            if cost_price is None or cost_price <= 0:
                continue
            
            # Get current price from daily_factors or position
            current_price = None
            if daily_factors is not None and code in daily_factors.index:
                if 'close' in daily_factors.columns:
                    current_price = daily_factors.loc[code, 'close']
            
            # Fallback: try to get from position's market value
            if current_price is None and hasattr(position, 'price'):
                current_price = position.price
            
            if current_price is None or current_price <= 0:
                continue
            
            # Calculate return
            ret = (current_price - cost_price) / cost_price
            
            # Check stop loss
            if ret <= -stop_loss_pct:
                context.order_target_percent(code, 0.0, reason=f"止损卖出(亏损{ret*100:.1f}%)")
                stopped_codes.append(code)
                # Remove from cost tracking
                if code in self.cost_prices:
                    del self.cost_prices[code]
        
        return stopped_codes
    
    def _update_cost_prices(self, context, bought_codes: List[str], daily_factors: pd.DataFrame):
        """
        更新持仓成本价
        
        :param context: 回测上下文
        :param bought_codes: 本次买入的股票列表
        :param daily_factors: 当日因子数据
        """
        for code in bought_codes:
            if daily_factors is not None and code in daily_factors.index:
                if 'close' in daily_factors.columns:
                    self.cost_prices[code] = daily_factors.loc[code, 'close']
        
        # Clean up cost prices for positions no longer held
        held_codes = set(context.positions.keys())
        codes_to_remove = [c for c in self.cost_prices if c not in held_codes]
        for code in codes_to_remove:
            del self.cost_prices[code]
    
    def on_rebalance(self, context, valid_codes: List[str], daily_factors: pd.DataFrame):
        """
        调仓日逻辑：滚动训练模型 + 预测选股
        """
        self.rebalance_counter += 1
        
        # === Step 0: Check stop loss every day (not just on rebalance days) ===
        stopped_codes = self._check_stop_loss(context, daily_factors)
        if stopped_codes:
            print(f"[{context.current_dt}] Stop loss triggered for: {stopped_codes}")
        
        # 检查调仓周期
        if self.rebalance_counter % self.params['rebalance_period'] != 0:
            return
            
        if daily_factors is None or daily_factors.empty:
            return
        
        current_date = context.current_dt
        
        # 1. 准备训练数据 (只能用历史数据，避免未来数据泄露)
        if self.all_data is None or self.all_data.empty:
            return
        
        # IMPORTANT: next_ret uses T+label_k forward return, so a sample at date T
        # requires close price at T+label_k. To avoid look-ahead bias, we must
        # exclude the last label_k trading days before current_date, because their
        # next_ret labels depend on prices at or after current_date.
        label_k = self.params.get('label_period') or self.params['rebalance_period']
        k = label_k
        historical_data = self.all_data[self.all_data.index < current_date].copy()
        
        if len(historical_data) == 0:
            return
        
        # Remove last k dates from training to prevent future data leakage
        unique_hist_dates = sorted(historical_data.index.unique())
        if len(unique_hist_dates) > k:
            safe_cutoff = unique_hist_dates[-k]
            historical_data = historical_data[historical_data.index < safe_cutoff]
        else:
            # Not enough history to form valid labels
            return
        
        if len(historical_data) == 0:
            return
        
        # 滚动窗口：只用最近 train_window 天的数据
        train_window = self.params['train_window']
        unique_dates = sorted(historical_data.index.unique())
        
        if len(unique_dates) > train_window:
            cutoff_date = unique_dates[-train_window]
            train_data = historical_data[historical_data.index >= cutoff_date]
        else:
            train_data = historical_data
        
        if len(train_data) < self.params['min_train_samples']:
            return
        
        # 2. 训练模型
        self._train_model(train_data)
        
        if self.model is None:
            return
        
        # 3. 预测当期股票
        available_factors = daily_factors.loc[daily_factors.index.intersection(valid_codes)].copy()
        
        if len(available_factors) < self.params['top_k']:
            return
        
        # 因子已在 prepare_factors 中按交易日截面预处理，预测阶段不再重复标准化。
        feature_cols = [c for c in self.params['factor_cols'] if c in available_factors.columns]
        if not feature_cols:
            return
        
        # 模型预测
        X_pred = available_factors[feature_cols].fillna(0).values
        predictions = self.model.predict(X_pred)
        available_factors['score'] = predictions
        
        # 保存评分供回放功能使用
        self.last_scores = available_factors[['score']].copy()
        self.last_scores['code'] = self.last_scores.index
        
        # 4. 排序选股
        ranked = available_factors.sort_values('score', ascending=False)
        candidates = ranked.head(self.params['top_k'])
        
        # 5. 趋势过滤
        final_targets = []
        if self.params.get('filter_downtrend', False) and 'is_uptrend' in candidates.columns:
            uptrend_stocks = candidates[candidates['is_uptrend'] == True]
            final_targets = uptrend_stocks.index.tolist()
        else:
            final_targets = candidates.index.tolist()
        
        # 6. 执行调仓
        # 6.1 卖出不在目标列表中的持仓
        for code in list(context.positions.keys()):
            if code not in final_targets:
                context.order_target_percent(code, 0.0, reason="调仓卖出")
        
        # 6.2 买入目标股票
        new_buys = []  # Track newly bought stocks
        if final_targets:
            single_pos_weight = 0.95 / self.params['top_k']
            for code in final_targets:
                is_new_position = code not in context.positions
                context.order_target_percent(code, single_pos_weight, reason="调仓买入")
                if is_new_position:
                    new_buys.append(code)
        
        # 6.3 Update cost prices for new positions
        self._update_cost_prices(context, new_buys, daily_factors)
        
        # 如果 final_targets 为空或少于 top_k，剩余资金留作现金
