"""
Factor Library Module

This module provides a centralized factor registry for computing stock factors.
All factors are automatically registered when the module is imported.

Usage:
    from strategy_app.factors import factor_registry
    
    # Compute single factor
    momentum = factor_registry.compute('momentum_20d', df)
    
    # Batch compute multiple factors
    df_with_factors = factor_registry.compute_batch(
        ['momentum_20d', 'volatility_20d', 'volume_ratio'], 
        df
    )
    
    # List all available factors
    all_factors = factor_registry.list_factors()
    
    # List factors by category
    momentum_factors = factor_registry.list_factors(category='momentum')
    
    # Get factor information
    info = factor_registry.get_factor_info('momentum_20d')
    
    # Check if factor exists
    if 'momentum_20d' in factor_registry:
        ...

Available Categories:
    - momentum: Price momentum factors (momentum_20d, momentum_60d, reversal_5d, etc.)
    - volatility: Volatility factors (volatility_20d, bias_20, atr_20, etc.)
    - volume: Volume-related factors (turnover_20d, volume_ratio, vwap_20d, etc.)
    - technical: Technical indicators (rsi_14, macd, kdj_k, bollinger_position, etc.)
    - financial: Financial factors from Tushare (pe, pb, roe, netprofit_yoy, etc.)

Data Preprocessing (数据预处理):
    from strategy_app.factors import FactorPreprocessor, preprocess_factors
    
    # Quick preprocessing
    processed_df = preprocess_factors(
        df, 
        factor_columns=['momentum_20d', 'volatility_20d'],
        missing='median',      # 缺失值处理
        winsorize='mad',       # 去极值
        standardize='zscore',  # 标准化
        neutralize='size'      # 中性化
    )
    
    # Or use preprocessor class
    preprocessor = FactorPreprocessor()
    processed_df = preprocessor.process_dataframe(df, factor_columns)
"""

from .registry import FactorRegistry, factor_registry
from .base_factor import BaseFactor

# Import all factor modules to trigger registration
from . import momentum_factors
from . import volatility_factors
from . import volume_factors
from . import technical_factors
from . import financial_factors
from . import etf_momentum_factors_optimized  # ETF三因子动量因子（优化版）

# Import financial data loader
from .financial_data import FinancialDataLoader, get_financial_data_loader

# Import preprocessor
from .preprocessor import (
    FactorPreprocessor,
    PreprocessPipeline,
    PreprocessConfig,
    MissingValueHandler,
    Winsorizer,
    Standardizer,
    Neutralizer,
    preprocess_factor,
    preprocess_factors,
    MissingMethod,
    WinsorizeMethod,
    StandardizeMethod,
    NeutralizeMethod,
)

from .analysis import FactorAnalysisService, FactorPanelBuilder, FactorPanelBuildResult

__all__ = [
    # Registry
    'FactorRegistry',
    'factor_registry',
    'BaseFactor',
    # Financial data
    'FinancialDataLoader',
    'get_financial_data_loader',
    # Preprocessor
    'FactorPreprocessor',
    'PreprocessPipeline',
    'PreprocessConfig',
    'MissingValueHandler',
    'Winsorizer',
    'Standardizer',
    'Neutralizer',
    'preprocess_factor',
    'preprocess_factors',
    'MissingMethod',
    'WinsorizeMethod',
    'StandardizeMethod',
    'NeutralizeMethod',
    # Analysis
    'FactorAnalysisService',
    'FactorPanelBuilder',
    'FactorPanelBuildResult',
]
