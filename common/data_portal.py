"""
Unified data portal MVP.

The MVP intentionally focuses on local daily bars and freshness metadata. It
provides a stable access seam for live strategies and future backtest code while
reusing existing loaders and update logic.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

SymbolInput = Union[str, Iterable[str]]

_DAILY_FREQUENCIES = {"1d", "d", "day", "daily"}
_STANDARD_BAR_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _normalize_symbol_code(code: str) -> str:
    value = str(code or "").strip().upper()
    return value.split(".", 1)[0] if "." in value else value


class StockDataCache:
    """In-memory stock daily bars cache owned by the unified data portal."""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._data_dir: str = ""
        self._is_loaded: bool = False

    def preload_all(
        self,
        data_dir: str,
        stock_codes: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_workers: int = 8,
    ) -> int:
        self._data_dir = data_dir
        self._cache.clear()

        total = len(stock_codes)
        loaded_count = 0

        def load_one(code: str) -> Tuple[str, Optional[pd.DataFrame]]:
            df = _load_stock_data_from_parquet(code, data_dir)
            return code, df

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_one, code): code for code in stock_codes}
            for i, future in enumerate(as_completed(futures)):
                code, df = future.result()
                if df is not None:
                    self._cache[_normalize_symbol_code(code)] = df
                    loaded_count += 1
                if progress_callback:
                    progress_callback(i + 1, total, code)

        self._is_loaded = True
        return loaded_count

    def get(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        normalized = _normalize_symbol_code(code)
        if normalized not in self._cache:
            return None

        df = self._cache[normalized].copy()
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        return df.reset_index(drop=True) if not df.empty else None

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_cached_codes(self) -> List[str]:
        return list(self._cache.keys())

    def clear(self):
        self._cache.clear()
        self._is_loaded = False

    def reload_stock(self, code: str, data_dir: str = None) -> bool:
        dir_path = data_dir or self._data_dir
        df = _load_stock_data_from_parquet(code, dir_path)
        if df is not None:
            self._cache[_normalize_symbol_code(code)] = df
            return True
        return False

    def reload_all(
        self,
        data_dir: str = None,
        stock_codes: List[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_workers: int = 8,
    ) -> int:
        dir_path = data_dir or self._data_dir
        codes = stock_codes if stock_codes is not None else list(self._cache.keys())
        if not codes:
            return 0

        self._cache.clear()
        total = len(codes)
        loaded_count = 0

        def load_one(code: str) -> Tuple[str, Optional[pd.DataFrame]]:
            df = _load_stock_data_from_parquet(code, dir_path)
            return code, df

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_one, code): code for code in codes}
            for i, future in enumerate(as_completed(futures)):
                code, df = future.result()
                if df is not None:
                    self._cache[_normalize_symbol_code(code)] = df
                    loaded_count += 1
                if progress_callback:
                    progress_callback(i + 1, total, code)

        self._is_loaded = True
        self._data_dir = dir_path
        return loaded_count


_stock_cache = StockDataCache()


def get_stock_cache() -> StockDataCache:
    """Return the process-wide stock daily bars cache."""
    return _stock_cache


def _load_stock_data_from_parquet(
    code: str,
    data_dir: str = "../data",
    adj: str = "qfq",
) -> Optional[pd.DataFrame]:
    normalized_code = _normalize_symbol_code(code)
    parquet_path = Path(data_dir) / f"{normalized_code}.parquet"
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        print(f"读取 Parquet 失败 {parquet_path}: {exc}")
        return None

    if df.empty:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return None

    if "adj_factor" in df.columns:
        df["adj_factor"] = pd.to_numeric(df["adj_factor"], errors="coerce")
        if adj in ("qfq", "hfq") and df["adj_factor"].notna().any():
            if adj == "qfq":
                latest_factor = df["adj_factor"].iloc[-1]
                if pd.notna(latest_factor) and latest_factor != 0:
                    ratio = df["adj_factor"] / latest_factor
                    for column in ["open", "high", "low", "close"]:
                        df[column] = df[column] * ratio
            elif adj == "hfq":
                earliest_factor = df["adj_factor"].iloc[0]
                if pd.notna(earliest_factor) and earliest_factor != 0:
                    ratio = df["adj_factor"] / earliest_factor
                    for column in ["open", "high", "low", "close"]:
                        df[column] = df[column] * ratio
        df = df.drop(columns=["adj_factor"])

    return df


# Backward-compatible private name used by cache code in older integrations.
_load_stock_data_from_csv = _load_stock_data_from_parquet


def load_stock_data(
    code: str,
    data_dir: str = "../data",
    adj: str = "qfq",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """Load stock daily bars through the unified data portal backend."""
    code = _normalize_symbol_code(code)
    if use_cache and _stock_cache.is_loaded():
        df = _stock_cache.get(code, start_date, end_date)
        if df is not None:
            return df

    df = _load_stock_data_from_parquet(code, data_dir, adj)
    if df is None:
        return None
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df.reset_index(drop=True) if not df.empty else None


def get_stock_list(data_dir: str = "../data") -> List[str]:
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    parquet_files = sorted(data_path.glob("*.parquet"))
    return [path.stem for path in parquet_files]


def load_stock_name_map(stocklist_path: str = "../stocklist/stocklist.csv") -> Dict[str, str]:
    try:
        file_path = Path(stocklist_path)
        if not file_path.exists():
            return {}
        df = pd.read_csv(file_path, dtype={"symbol": str})
        if "symbol" not in df.columns or "name" not in df.columns:
            return {}
        return {str(code).strip(): name for code, name in zip(df["symbol"], df["name"])}
    except Exception:
        return {}


def get_date_range(code: str, data_dir: str = "../data") -> Optional[Tuple[str, str]]:
    df = load_stock_data(code, data_dir, adj=None)
    if df is None or df.empty:
        return None
    return df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")


class ETFDataCache:
    """In-memory ETF daily bars cache owned by the unified data portal."""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._data_dir: str = ""
        self._is_loaded: bool = False

    def preload_all(
        self,
        data_dir: str,
        etf_codes: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_workers: int = 8,
    ) -> int:
        self._data_dir = data_dir
        self._cache.clear()

        total = len(etf_codes)
        loaded_count = 0

        def load_one(code: str) -> Tuple[str, Optional[pd.DataFrame]]:
            df = _load_etf_data_from_parquet(code, data_dir)
            return code, df

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_one, code): code for code in etf_codes}
            for i, future in enumerate(as_completed(futures)):
                code, df = future.result()
                if df is not None:
                    self._cache[_normalize_symbol_code(code)] = df
                    loaded_count += 1
                if progress_callback:
                    progress_callback(i + 1, total, code)

        self._is_loaded = True
        return loaded_count

    def get(self, code: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        normalized = _normalize_symbol_code(code)
        if normalized not in self._cache:
            return None

        df = self._cache[normalized].copy()
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        return df.reset_index(drop=True) if not df.empty else None

    def is_loaded(self) -> bool:
        return self._is_loaded

    def get_cached_codes(self) -> List[str]:
        return list(self._cache.keys())

    def clear(self):
        self._cache.clear()
        self._is_loaded = False

    def reload_etf(self, code: str, data_dir: str = None) -> bool:
        dir_path = data_dir or self._data_dir
        df = _load_etf_data_from_parquet(code, dir_path)
        if df is not None:
            self._cache[_normalize_symbol_code(code)] = df
            return True
        return False

    def reload_all(
        self,
        data_dir: str = None,
        etf_codes: List[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_workers: int = 8,
    ) -> int:
        dir_path = data_dir or self._data_dir
        codes = etf_codes if etf_codes is not None else list(self._cache.keys())
        if not codes:
            return 0

        self._cache.clear()
        total = len(codes)
        loaded_count = 0

        def load_one(code: str) -> Tuple[str, Optional[pd.DataFrame]]:
            df = _load_etf_data_from_parquet(code, dir_path)
            return code, df

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_one, code): code for code in codes}
            for i, future in enumerate(as_completed(futures)):
                code, df = future.result()
                if df is not None:
                    self._cache[_normalize_symbol_code(code)] = df
                    loaded_count += 1
                if progress_callback:
                    progress_callback(i + 1, total, code)

        self._is_loaded = True
        self._data_dir = dir_path
        return loaded_count


_etf_cache = ETFDataCache()


def get_etf_cache() -> ETFDataCache:
    """Return the process-wide ETF daily bars cache."""
    return _etf_cache


def _load_etf_data_from_parquet(
    code: str,
    data_dir: str = "../data",
) -> Optional[pd.DataFrame]:
    normalized_code = _normalize_symbol_code(code)
    parquet_path = Path(data_dir) / "etf" / f"{normalized_code}.parquet"
    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        print(f"读取 ETF Parquet 失败 {parquet_path}: {exc}")
        return None

    if df.empty:
        return None

    df = df.sort_values("date").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df if not df.empty else None


def load_etf_data(
    code: str,
    data_dir: str = "../data",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> Optional[pd.DataFrame]:
    """Load ETF daily bars through the unified data portal backend."""
    code = _normalize_symbol_code(code)
    if use_cache and _etf_cache.is_loaded():
        df = _etf_cache.get(code, start_date, end_date)
        if df is not None:
            return df

    df = _load_etf_data_from_parquet(code, data_dir)
    if df is None:
        return None
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    return df.reset_index(drop=True) if not df.empty else None


def get_etf_list(data_dir: str = "../data") -> List[str]:
    etf_path = Path(data_dir) / "etf"
    if not etf_path.exists():
        return []
    parquet_files = sorted(etf_path.glob("*.parquet"))
    return [path.stem for path in parquet_files]


def _default_etf_config_paths() -> list[Path]:
    current_file_dir = Path(__file__).parent
    return [
        current_file_dir / ".." / "trading_app" / "config" / "etf_list.json",
        current_file_dir / ".." / "strategy_app" / "config" / "etf_list.json",
        current_file_dir / ".." / "config" / "etf_list.json",
    ]


def load_etf_name_map(config_path: str = None) -> Dict[str, str]:
    import json

    possible_paths = [Path(config_path)] if config_path else _default_etf_config_paths()
    for path in possible_paths:
        try:
            resolved_path = path.resolve()
            if not resolved_path.exists():
                continue
            with open(resolved_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            name_map = {}
            for category in config.get("categories", []):
                for etf in category.get("etfs", []):
                    code = etf.get("code", "")
                    name = etf.get("name", "")
                    if code and name:
                        name_map[code] = name
            return name_map
        except Exception:
            continue
    return {}


def load_etf_categories(config_path: str = None) -> List[Dict]:
    import json

    possible_paths = [Path(config_path)] if config_path else _default_etf_config_paths()
    for path in possible_paths:
        try:
            resolved_path = path.resolve()
            if not resolved_path.exists():
                continue
            with open(resolved_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            return config.get("categories", [])
        except Exception:
            continue
    return []


def get_etf_date_range(code: str, data_dir: str = "../data") -> Optional[Tuple[str, str]]:
    df = load_etf_data(code, data_dir)
    if df is None or df.empty:
        return None
    return df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")


@dataclass(frozen=True)
class BarsMetadata:
    """Metadata attached to a local bars response."""

    symbol: str
    asset_type: str
    frequency: str
    adjust: str
    data_dir: str
    start: Optional[str]
    end: Optional[str]
    rows: int
    latest_date: Optional[str]
    schema_version: str = "daily_bars.v1"
    first_date: Optional[str] = None
    expected_date: Optional[str] = None
    is_fresh: Optional[bool] = None
    data_path: Optional[str] = None
    data_source: str = "unknown"
    data_hash: str = ""
    data_version: str = ""
    sidecar_path: Optional[str] = None
    sidecar_exists: bool = False


@dataclass(frozen=True)
class FreshnessStatus:
    """Daily bars freshness status for one symbol."""

    symbol: str
    is_fresh: bool
    latest_date: str
    expected_date: str
    reason: str = ""


@dataclass(frozen=True)
class DailyDataStatus:
    """Unified local daily parquet metadata and freshness status."""

    symbol: str
    asset_type: str
    data_dir: str
    data_path: str
    exists: bool
    rows: int
    first_date: Optional[str]
    latest_date: Optional[str]
    expected_date: str
    is_fresh: bool
    reason: str
    schema_version: str = "daily_bars.v1"
    data_source: str = "unknown"
    data_hash: str = ""
    data_version: str = ""
    sidecar_path: Optional[str] = None
    sidecar_exists: bool = False
    sidecar_schema_version: str = ""


@dataclass(frozen=True)
class AssetMetadata:
    """Unified symbol metadata for stocks, ETFs, and indices."""

    symbol: str
    asset_type: str
    name: str = ""
    market: str = ""
    category: str = ""
    first_date: Optional[str] = None
    latest_date: Optional[str] = None
    rows: int = 0
    data_path: Optional[str] = None
    is_fresh: Optional[bool] = None
    data_source: str = "unknown"
    data_version: str = ""


@dataclass(frozen=True)
class ParquetSidecarMetadata:
    """Auditable sidecar metadata stored next to one parquet file."""

    symbol: str
    asset_type: str
    frequency: str
    data_source: str
    data_path: str
    sidecar_path: str
    rows: int
    columns: list[str]
    first_date: Optional[str]
    latest_date: Optional[str]
    data_hash: str
    params_hash: str
    data_version: str
    created_at: str
    updated_at: str
    schema_version: str = "parquet_sidecar.v1"
    provider_symbol: str = ""
    update_mode: str = ""
    fetch_start: Optional[str] = None
    fetch_end: Optional[str] = None
    source_start: Optional[str] = None
    source_end: Optional[str] = None
    extra: Optional[dict] = None

    def to_dict(self) -> dict:
        result = self.__dict__.copy()
        result["extra"] = dict(self.extra or {})
        return result


@dataclass(frozen=True)
class DataVersionAudit:
    """Aggregated data version audit for one data snapshot."""

    data_version: str
    generated_at: str
    scope: str
    symbols: list[str]
    assets: list[dict]
    sources: list[str]
    hashes: list[str]
    missing_sidecars: list[str]
    schema_version: str = "data_version_audit.v1"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class TradingCalendarDay:
    """One trading-calendar day returned by DataPortal."""

    date: str
    is_trading_day: bool
    reason: str = ""
    schema_version: str = "trading_calendar_day.v1"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class CorporateActionRecord:
    """Local corporate-action proxy derived from adjustment-factor changes."""

    symbol: str
    date: str
    action_type: str
    adj_factor: float
    previous_adj_factor: Optional[float] = None
    ratio: Optional[float] = None
    source: str = "local_adj_factor"
    schema_version: str = "corporate_action.v1"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class CacheRefreshResult:
    """Result of refreshing already-loaded local data caches."""

    stock_count: int = 0
    etf_count: int = 0
    stock_cache_loaded: bool = False
    etf_cache_loaded: bool = False

    @property
    def refreshed(self) -> bool:
        return bool(self.stock_count or self.etf_count)


@dataclass(frozen=True)
class StrategyDataView:
    """One symbol's normalized bars and metadata for strategy/backtest inputs."""

    symbol: str
    asset_type: str
    data: pd.DataFrame
    metadata: BarsMetadata

    def to_frame(self) -> pd.DataFrame:
        """Return a defensive copy of the underlying bars DataFrame."""
        return self.data.copy()


@dataclass(frozen=True)
class MarketDataBundle:
    """Unified strategy/backtest market data input contract."""

    data: Dict[str, StrategyDataView]
    primary_symbol: Optional[str] = None
    benchmark: Optional[StrategyDataView] = None
    schema_version: str = "market_data_bundle.v1"
    data_audit: Optional[dict] = None

    @property
    def symbols(self) -> list[str]:
        return list(self.data.keys())

    @property
    def benchmark_symbol(self) -> Optional[str]:
        return self.benchmark.symbol if self.benchmark is not None else None

    def get(self, symbol: str) -> Optional[StrategyDataView]:
        return self.data.get(DataPortal.normalize_symbol(symbol))

    def require(self, symbol: str) -> StrategyDataView:
        """Return one symbol view or raise a clear error for strategy code."""
        normalized = DataPortal.normalize_symbol(symbol)
        view = self.data.get(normalized)
        if view is None:
            raise KeyError(f"MarketDataBundle does not contain symbol: {normalized}")
        return view

    def iter_views(self):
        """Yield normalized symbol and StrategyDataView pairs in bundle order."""
        return iter(self.data.items())

    def to_frame(self, symbol: str) -> pd.DataFrame:
        """Return one symbol's legacy DataFrame copy from the unified contract."""
        return self.require(symbol).to_frame()

    def require_single_frame(self) -> tuple[str, pd.DataFrame]:
        """Return the single/primary frame for legacy single-symbol engines."""
        symbol = self.primary_symbol or (self.symbols[0] if self.symbols else None)
        if not symbol or symbol not in self.data:
            raise ValueError("MarketDataBundle does not contain a primary symbol")
        return symbol, self.data[symbol].to_frame()

    def to_data_dict(self) -> Dict[str, pd.DataFrame]:
        """Return legacy {symbol: DataFrame} data for current strategy implementations."""
        return {symbol: view.to_frame() for symbol, view in self.data.items()}


@dataclass(frozen=True)
class BarsResult:
    """Bars dataframe plus MVP metadata."""

    data: pd.DataFrame
    metadata: BarsMetadata


class DataPortal:
    """Unified local data access seam for research, backtest, and live services."""

    def __init__(self, *, default_data_dir: Optional[Path] = None) -> None:
        self.default_data_dir = Path(default_data_dir) if default_data_dir is not None else self._project_root() / "data"

    def get_bars(
        self,
        symbols: SymbolInput,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        frequency: str = "1d",
        adjust: str = "qfq",
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        use_cache: bool = True,
    ) -> Dict[str, BarsResult]:
        """Return normalized local daily bars for one or more symbols."""
        self._ensure_daily_frequency(frequency)
        result: Dict[str, BarsResult] = {}
        for symbol in self._normalize_symbols(symbols):
            bars = self.get_daily_bars(
                symbol,
                start=start,
                end=end,
                adjust=adjust,
                asset_type=asset_type,
                data_dir=data_dir,
                use_cache=use_cache,
            )
            if bars is not None:
                resolved_type = self.resolve_asset_type(symbol, asset_type)
                effective_dir = self._effective_data_dir(resolved_type, data_dir)
                daily_status = self.get_daily_metadata(
                    symbol,
                    asset_type=resolved_type,
                    data_dir=effective_dir,
                    now=None,
                )
                result[symbol] = BarsResult(
                    data=bars,
                    metadata=BarsMetadata(
                        symbol=symbol,
                        asset_type=resolved_type,
                        frequency="1d",
                        adjust=adjust,
                        data_dir=str(effective_dir),
                        start=start,
                        end=end,
                        rows=len(bars),
                        latest_date=self._latest_date_str(bars),
                        first_date=self._first_date_str(bars),
                        expected_date=daily_status.expected_date,
                        is_fresh=daily_status.is_fresh,
                        data_path=daily_status.data_path,
                        data_source=daily_status.data_source,
                        data_hash=daily_status.data_hash,
                        data_version=daily_status.data_version,
                        sidecar_path=daily_status.sidecar_path,
                        sidecar_exists=daily_status.sidecar_exists,
                    ),
                )
        return result

    def get_market_data_bundle(
        self,
        symbols: SymbolInput,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        frequency: str = "1d",
        adjust: str = "qfq",
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        use_cache: bool = True,
        primary_symbol: Optional[str] = None,
        benchmark_symbol: Optional[str] = None,
        benchmark_asset_type: str = "index",
    ) -> MarketDataBundle:
        """Build the unified market data contract for strategy/backtest code."""
        bars_map = self.get_bars(
            symbols,
            start=start,
            end=end,
            frequency=frequency,
            adjust=adjust,
            asset_type=asset_type,
            data_dir=data_dir,
            use_cache=use_cache,
        )
        views = {
            symbol: StrategyDataView(
                symbol=symbol,
                asset_type=result.metadata.asset_type,
                data=result.data,
                metadata=result.metadata,
            )
            for symbol, result in bars_map.items()
        }
        normalized_primary = self.normalize_symbol(primary_symbol) if primary_symbol else None
        if normalized_primary is None and len(views) == 1:
            normalized_primary = next(iter(views.keys()))

        benchmark_view: Optional[StrategyDataView] = None
        if benchmark_symbol:
            benchmark_map = self.get_bars(
                benchmark_symbol,
                start=start,
                end=end,
                frequency=frequency,
                adjust=adjust,
                asset_type=benchmark_asset_type,
                data_dir=data_dir,
                use_cache=False,
            )
            normalized_benchmark = self.normalize_symbol(benchmark_symbol)
            benchmark_result = benchmark_map.get(normalized_benchmark)
            if benchmark_result is not None:
                benchmark_view = StrategyDataView(
                    symbol=normalized_benchmark,
                    asset_type=benchmark_result.metadata.asset_type,
                    data=benchmark_result.data,
                    metadata=benchmark_result.metadata,
                )

        data_audit = self._build_bundle_data_audit(views, benchmark_view)

        return MarketDataBundle(
            data=views,
            primary_symbol=normalized_primary,
            benchmark=benchmark_view,
            data_audit=data_audit,
        )

    def get_daily_bars(
        self,
        symbol: str,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        adjust: str = "qfq",
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Return one symbol's normalized daily bars, or None when unavailable."""
        normalized = self.normalize_symbol(symbol)
        resolved_type = self.resolve_asset_type(normalized, asset_type)
        effective_dir = self._effective_data_dir(resolved_type, data_dir)

        if resolved_type == "etf":
            df = load_etf_data(
                normalized,
                str(effective_dir),
                start_date=start,
                end_date=end,
                use_cache=use_cache,
            )
            if df is None:
                df = self._load_etf_daily_bars_from_local_dir(
                    normalized,
                    effective_dir,
                    start=start,
                    end=end,
                )
        elif resolved_type == "index":
            df = self._load_index_daily_bars_from_local_dir(
                normalized,
                effective_dir,
                start=start,
                end=end,
            )
        else:
            df = load_stock_data(
                normalized,
                str(effective_dir),
                adj=adjust,
                start_date=start,
                end_date=end,
                use_cache=use_cache,
            )
        return self._normalize_daily_bars(df)

    def latest_close(
        self,
        symbol: str,
        *,
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
    ) -> float:
        """Return the latest close from local daily bars, or 0 when unavailable."""
        df = self.get_daily_bars(
            symbol,
            asset_type=asset_type,
            data_dir=data_dir,
            use_cache=False,
        )
        if df is None or df.empty or "close" not in df.columns:
            return 0.0
        try:
            return float(df["close"].iloc[-1] or 0.0)
        except Exception:
            return 0.0

    def check_daily_freshness(
        self,
        symbol: str,
        *,
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> FreshnessStatus:
        """Check whether local daily bars include the expected latest trading day."""
        status = self.get_daily_metadata(
            symbol,
            asset_type=asset_type,
            data_dir=data_dir,
            now=now,
        )
        return FreshnessStatus(
            status.symbol,
            status.is_fresh,
            status.latest_date or "",
            status.expected_date,
            status.reason,
        )

    def get_daily_metadata(
        self,
        symbol: str,
        *,
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> DailyDataStatus:
        """Return unified metadata and freshness status for one local daily parquet file."""
        normalized = self.normalize_symbol(symbol)
        resolved_type = self.resolve_asset_type(normalized, asset_type)
        effective_dir = self._effective_data_dir(resolved_type, data_dir)
        parquet_path = self._resolve_daily_parquet_path(normalized, resolved_type, effective_dir)
        return self.get_daily_file_metadata(
            parquet_path,
            symbol=normalized,
            asset_type=resolved_type,
            data_dir=effective_dir,
            now=now,
        )

    def get_daily_metadata_map(
        self,
        symbols: Iterable[str],
        *,
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, DailyDataStatus]:
        """Return daily parquet metadata for multiple symbols."""
        result: Dict[str, DailyDataStatus] = {}
        for symbol in self._normalize_symbols(symbols):
            result[symbol] = self.get_daily_metadata(
                symbol,
                asset_type=asset_type,
                data_dir=data_dir,
                now=now,
            )
        return result

    def get_trading_calendar(
        self,
        start: Union[str, date, datetime],
        end: Union[str, date, datetime],
        *,
        include_non_trading: bool = True,
    ) -> list[dict]:
        """Return A-share trading-calendar records for an inclusive date range."""
        from live_rotation.holiday_calendar import get_non_trading_reason, is_trading_day

        start_date = self._to_date(start)
        end_date = self._to_date(end)
        if end_date < start_date:
            raise ValueError("end must be greater than or equal to start")

        days: list[dict] = []
        current = start_date
        while current <= end_date:
            trading = is_trading_day(current)
            if trading or include_non_trading:
                days.append(
                    TradingCalendarDay(
                        date=current.isoformat(),
                        is_trading_day=trading,
                        reason="" if trading else get_non_trading_reason(current),
                    ).to_dict()
                )
            current += timedelta(days=1)
        return days

    def get_instruments(
        self,
        *,
        asset_type: str = "stock",
        data_dir: Optional[Path] = None,
        stocklist_path: Optional[Path] = None,
        etf_config_path: Optional[Path] = None,
        include_status: bool = True,
    ) -> list[dict]:
        """Return unified instrument records backed by local metadata."""
        return self.list_assets(
            asset_type=asset_type,
            data_dir=data_dir,
            stocklist_path=stocklist_path,
            etf_config_path=etf_config_path,
            include_status=include_status,
        )

    def get_corporate_actions(
        self,
        symbol: str,
        *,
        asset_type: str = "stock",
        data_dir: Optional[Path] = None,
    ) -> list[dict]:
        """Return local corporate-action proxies derived from adj_factor changes."""
        normalized = self.normalize_symbol(symbol)
        resolved_type = self.resolve_asset_type(normalized, asset_type)
        effective_dir = self._effective_data_dir(resolved_type, data_dir)
        parquet_path = self._resolve_daily_parquet_path(normalized, resolved_type, effective_dir)
        if not parquet_path.exists():
            return []
        try:
            df = pd.read_parquet(parquet_path, columns=["date", "adj_factor"])
        except Exception:
            return []
        if df.empty or "adj_factor" not in df.columns or "date" not in df.columns:
            return []
        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["adj_factor"] = pd.to_numeric(frame["adj_factor"], errors="coerce")
        frame = frame.dropna(subset=["date", "adj_factor"]).sort_values("date").reset_index(drop=True)
        if frame.empty:
            return []

        records: list[dict] = []
        previous = None
        for row in frame.itertuples(index=False):
            current = float(row.adj_factor)
            if previous is not None and current != previous:
                ratio = current / previous if previous else None
                records.append(
                    CorporateActionRecord(
                        symbol=normalized,
                        date=pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                        action_type="adj_factor_change",
                        adj_factor=current,
                        previous_adj_factor=previous,
                        ratio=ratio,
                    ).to_dict()
                )
            previous = current
        return records

    def get_data_version(
        self,
        symbols: Optional[Iterable[str]] = None,
        *,
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        scope: str = "daily_bars",
        now: Optional[datetime] = None,
    ) -> DataVersionAudit:
        """Return an auditable aggregate data version for local parquet snapshots."""
        if symbols is None:
            list_type = "stock" if (asset_type or "auto").strip().lower() == "auto" else asset_type
            symbols_list = self.list_symbols(asset_type=list_type, data_dir=data_dir)
        else:
            symbols_list = list(self._normalize_symbols(symbols))

        assets: list[dict] = []
        missing_sidecars: list[str] = []
        for symbol in symbols_list:
            status = self.get_daily_metadata(symbol, asset_type=asset_type, data_dir=data_dir, now=now)
            if status.exists and not status.sidecar_exists:
                missing_sidecars.append(status.symbol)
            assets.append(self._status_to_version_asset(status))

        assets = sorted(assets, key=lambda item: (item.get("asset_type", ""), item.get("symbol", "")))
        payload = {
            "scope": scope,
            "assets": assets,
            "schema_version": "data_version_audit.v1",
        }
        return DataVersionAudit(
            data_version=self._stable_hash(payload),
            generated_at=self._utc_now_iso(),
            scope=scope,
            symbols=[item.get("symbol", "") for item in assets],
            assets=assets,
            sources=sorted({str(item.get("data_source", "unknown") or "unknown") for item in assets}),
            hashes=sorted({str(item.get("data_hash", "") or "") for item in assets if item.get("data_hash")}),
            missing_sidecars=missing_sidecars,
        )

    def write_parquet_sidecar(
        self,
        parquet_path: Path,
        *,
        symbol: str = "",
        asset_type: str = "auto",
        frequency: str = "1d",
        data_source: str = "unknown",
        provider_symbol: str = "",
        update_mode: str = "",
        fetch_start: Optional[str] = None,
        fetch_end: Optional[str] = None,
        source_start: Optional[str] = None,
        source_end: Optional[str] = None,
        params: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> ParquetSidecarMetadata:
        """Write a .meta.json sidecar next to a parquet file and return it."""
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        normalized = self.normalize_symbol(symbol or path.stem)
        resolved_type = self._infer_asset_type_from_path(path, normalized, asset_type)
        sidecar_path = self.sidecar_path_for(path)
        existing = self._read_sidecar_dict(sidecar_path)
        created_at = str(existing.get("created_at", "") or self._utc_now_iso()) if existing else self._utc_now_iso()
        updated_at = self._utc_now_iso()

        frame_info = self._parquet_frame_info(path)
        data_hash = self._file_sha256(path)
        params_payload = {
            "symbol": normalized,
            "asset_type": resolved_type,
            "frequency": frequency,
            "data_source": data_source or "unknown",
            "provider_symbol": provider_symbol,
            "update_mode": update_mode,
            "fetch_start": fetch_start,
            "fetch_end": fetch_end,
            "source_start": source_start,
            "source_end": source_end,
            "params": dict(params or {}),
            "extra": dict(extra or {}),
        }
        params_hash = self._stable_hash(params_payload)
        version_payload = {
            "schema_version": "parquet_sidecar.v1",
            "symbol": normalized,
            "asset_type": resolved_type,
            "frequency": frequency,
            "data_source": data_source or "unknown",
            "rows": frame_info["rows"],
            "columns": frame_info["columns"],
            "first_date": frame_info["first_date"],
            "latest_date": frame_info["latest_date"],
            "data_hash": data_hash,
            "params_hash": params_hash,
        }
        metadata = ParquetSidecarMetadata(
            symbol=normalized,
            asset_type=resolved_type,
            frequency=frequency,
            data_source=data_source or "unknown",
            data_path=str(path),
            sidecar_path=str(sidecar_path),
            rows=frame_info["rows"],
            columns=frame_info["columns"],
            first_date=frame_info["first_date"],
            latest_date=frame_info["latest_date"],
            data_hash=data_hash,
            params_hash=params_hash,
            data_version=self._stable_hash(version_payload),
            created_at=created_at,
            updated_at=updated_at,
            provider_symbol=provider_symbol,
            update_mode=update_mode,
            fetch_start=fetch_start,
            fetch_end=fetch_end,
            source_start=source_start,
            source_end=source_end,
            extra=dict(extra or {}),
        )
        self._write_json(sidecar_path, metadata.to_dict())
        return metadata

    def read_parquet_sidecar(self, parquet_path: Path) -> Optional[ParquetSidecarMetadata]:
        """Read one parquet sidecar if it exists and is valid enough."""
        sidecar_path = self.sidecar_path_for(parquet_path)
        payload = self._read_sidecar_dict(sidecar_path)
        if not payload:
            return None
        try:
            return ParquetSidecarMetadata(
                symbol=str(payload.get("symbol", "") or Path(parquet_path).stem),
                asset_type=str(payload.get("asset_type", "auto") or "auto"),
                frequency=str(payload.get("frequency", "1d") or "1d"),
                data_source=str(payload.get("data_source", "unknown") or "unknown"),
                data_path=str(payload.get("data_path", "") or parquet_path),
                sidecar_path=str(payload.get("sidecar_path", "") or sidecar_path),
                rows=int(payload.get("rows", 0) or 0),
                columns=list(payload.get("columns", []) or []),
                first_date=payload.get("first_date"),
                latest_date=payload.get("latest_date"),
                data_hash=str(payload.get("data_hash", "") or ""),
                params_hash=str(payload.get("params_hash", "") or ""),
                data_version=str(payload.get("data_version", "") or ""),
                created_at=str(payload.get("created_at", "") or ""),
                updated_at=str(payload.get("updated_at", "") or ""),
                schema_version=str(payload.get("schema_version", "parquet_sidecar.v1") or "parquet_sidecar.v1"),
                provider_symbol=str(payload.get("provider_symbol", "") or ""),
                update_mode=str(payload.get("update_mode", "") or ""),
                fetch_start=payload.get("fetch_start"),
                fetch_end=payload.get("fetch_end"),
                source_start=payload.get("source_start"),
                source_end=payload.get("source_end"),
                extra=dict(payload.get("extra", {}) or {}),
            )
        except Exception:
            return None

    @staticmethod
    def sidecar_path_for(parquet_path: Path) -> Path:
        """Return the sidecar path for one parquet file."""
        return Path(f"{Path(parquet_path)}.meta.json")

    def check_daily_freshness_map(
        self,
        symbols: Iterable[str],
        *,
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, FreshnessStatus]:
        """Return daily freshness statuses for multiple symbols."""
        return {
            symbol: FreshnessStatus(
                status.symbol,
                status.is_fresh,
                status.latest_date or "",
                status.expected_date,
                status.reason,
            )
            for symbol, status in self.get_daily_metadata_map(
                symbols,
                asset_type=asset_type,
                data_dir=data_dir,
                now=now,
            ).items()
        }

    def get_daily_file_metadata(
        self,
        parquet_path: Path,
        *,
        symbol: str = "",
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> DailyDataStatus:
        """Return metadata and freshness status for an explicit daily parquet path."""
        path = Path(parquet_path)
        normalized = self.normalize_symbol(symbol or path.stem)
        resolved_type = self._infer_asset_type_from_path(path, normalized, asset_type)
        effective_dir = Path(data_dir) if data_dir is not None else path.parent
        expected = self.latest_expected_trading_day(now)
        expected_str = expected.strftime("%Y-%m-%d")
        sidecar_path = self.sidecar_path_for(path)
        sidecar = self.read_parquet_sidecar(path)

        if not path.exists():
            return DailyDataStatus(
                normalized,
                resolved_type,
                str(effective_dir),
                str(path),
                False,
                0,
                None,
                None,
                expected_str,
                False,
                "missing_file",
                sidecar_path=str(sidecar_path),
                sidecar_exists=sidecar is not None,
                sidecar_schema_version=sidecar.schema_version if sidecar else "",
            )
        try:
            date_df = pd.read_parquet(path, columns=["date"])
        except Exception as exc:
            return DailyDataStatus(
                normalized,
                resolved_type,
                str(effective_dir),
                str(path),
                True,
                0,
                None,
                None,
                expected_str,
                False,
                f"read_error: {exc}",
                data_source=sidecar.data_source if sidecar else "unknown",
                data_hash=sidecar.data_hash if sidecar else "",
                data_version=sidecar.data_version if sidecar else "",
                sidecar_path=str(sidecar_path),
                sidecar_exists=sidecar is not None,
                sidecar_schema_version=sidecar.schema_version if sidecar else "",
            )
        if date_df.empty:
            fallback_hash = sidecar.data_hash if sidecar else self._file_sha256(path)
            fallback_version = sidecar.data_version if sidecar else self._stable_hash({"path": str(path), "data_hash": fallback_hash, "rows": 0})
            return DailyDataStatus(
                normalized,
                resolved_type,
                str(effective_dir),
                str(path),
                True,
                0,
                None,
                None,
                expected_str,
                False,
                "empty_file",
                data_source=sidecar.data_source if sidecar else "unknown",
                data_hash=fallback_hash,
                data_version=fallback_version,
                sidecar_path=str(sidecar_path),
                sidecar_exists=sidecar is not None,
                sidecar_schema_version=sidecar.schema_version if sidecar else "",
            )
        dates = pd.to_datetime(date_df["date"], errors="coerce").dropna()
        if dates.empty:
            fallback_hash = sidecar.data_hash if sidecar else self._file_sha256(path)
            fallback_version = sidecar.data_version if sidecar else self._stable_hash({"path": str(path), "data_hash": fallback_hash, "rows": len(date_df)})
            return DailyDataStatus(
                normalized,
                resolved_type,
                str(effective_dir),
                str(path),
                True,
                len(date_df),
                None,
                None,
                expected_str,
                False,
                "missing_date",
                data_source=sidecar.data_source if sidecar else "unknown",
                data_hash=fallback_hash,
                data_version=fallback_version,
                sidecar_path=str(sidecar_path),
                sidecar_exists=sidecar is not None,
                sidecar_schema_version=sidecar.schema_version if sidecar else "",
            )
        first_str = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
        latest_str = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
        is_fresh = latest_str >= expected_str
        data_hash = sidecar.data_hash if sidecar else self._file_sha256(path)
        data_version = sidecar.data_version if sidecar else self._stable_hash({
            "symbol": normalized,
            "asset_type": resolved_type,
            "path": str(path),
            "rows": len(date_df),
            "first_date": first_str,
            "latest_date": latest_str,
            "data_hash": data_hash,
        })
        return DailyDataStatus(
            normalized,
            resolved_type,
            str(effective_dir),
            str(path),
            True,
            len(date_df),
            first_str,
            latest_str,
            expected_str,
            is_fresh,
            "fresh" if is_fresh else "stale",
            data_source=sidecar.data_source if sidecar else "unknown",
            data_hash=data_hash,
            data_version=data_version,
            sidecar_path=str(sidecar_path),
            sidecar_exists=sidecar is not None,
            sidecar_schema_version=sidecar.schema_version if sidecar else "",
        )

    @staticmethod
    def format_daily_status_message(status: DailyDataStatus) -> str:
        """Format a daily metadata status as a legacy freshness message."""
        if status.reason == "missing_file":
            return "文件不存在"
        if status.reason == "empty_file":
            return "空文件"
        if status.reason == "missing_date":
            return "缺少date列"
        if status.reason.startswith("read_error: "):
            return f"读取失败: {status.reason.split(': ', 1)[1]}"
        return status.latest_date or status.reason

    def is_pool_fresh(
        self,
        symbols: Iterable[str],
        *,
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
    ) -> bool:
        """Return True when every symbol has fresh daily bars."""
        return all(
            status.is_fresh
            for status in self.get_daily_metadata_map(
                symbols,
                asset_type=asset_type,
                data_dir=data_dir,
            ).values()
        )

    def list_symbols(
        self,
        *,
        asset_type: str = "stock",
        data_dir: Optional[Path] = None,
    ) -> list[str]:
        """List symbols with local daily-bar parquet files."""
        resolved_type = self.resolve_asset_type("", asset_type)
        effective_dir = self._effective_data_dir(resolved_type, data_dir)
        if not effective_dir.exists():
            return []
        if resolved_type == "etf":
            etf_dir = effective_dir / "etf"
            if etf_dir.exists():
                return sorted({path.stem for path in etf_dir.glob("*.parquet")})
            return sorted({path.stem for path in effective_dir.glob("*.parquet")})
        if resolved_type == "index":
            index_dir = effective_dir / "index"
            if index_dir.exists():
                return sorted({path.stem for path in index_dir.glob("*.parquet")})
            return []
        return sorted({path.stem for path in effective_dir.glob("*.parquet")})

    def get_name_map(
        self,
        *,
        asset_type: str = "stock",
        stocklist_path: Optional[Path] = None,
        etf_config_path: Optional[Path] = None,
    ) -> Dict[str, str]:
        """Return a unified code-to-name map for one asset type."""
        resolved_type = self.resolve_asset_type("", asset_type)
        if resolved_type == "stock":
            return load_stock_name_map(str(stocklist_path)) if stocklist_path else load_stock_name_map()
        if resolved_type == "etf":
            return load_etf_name_map(str(etf_config_path)) if etf_config_path else load_etf_name_map()
        if resolved_type == "index":
            return {item["symbol"]: item["name"] for item in self.list_assets(asset_type="index", include_status=False)}
        return {}

    def get_categories(
        self,
        *,
        asset_type: str = "etf",
        config_path: Optional[Path] = None,
    ) -> list[Dict]:
        """Return classification metadata for asset types that have categories."""
        resolved_type = self.resolve_asset_type("", asset_type)
        if resolved_type == "etf":
            return load_etf_categories(str(config_path)) if config_path else load_etf_categories()
        return []

    def get_date_range(
        self,
        symbol: str,
        *,
        asset_type: str = "auto",
        data_dir: Optional[Path] = None,
    ) -> Optional[tuple[str, str]]:
        """Return the first/latest local daily-bar date for one symbol."""
        status = self.get_daily_metadata(symbol, asset_type=asset_type, data_dir=data_dir)
        if not status.first_date or not status.latest_date:
            return None
        return status.first_date, status.latest_date

    def list_assets(
        self,
        *,
        asset_type: str = "stock",
        data_dir: Optional[Path] = None,
        stocklist_path: Optional[Path] = None,
        etf_config_path: Optional[Path] = None,
        include_status: bool = True,
    ) -> list[dict]:
        """Return unified asset metadata records for stocks, ETFs, or indices."""
        resolved_type = self.resolve_asset_type("", asset_type)
        effective_dir = self._effective_data_dir(resolved_type, data_dir)
        if resolved_type == "index":
            assets = self._list_index_assets()
            if include_status:
                assets = [self._attach_asset_status(asset, effective_dir) for asset in assets]
            return [self._asset_to_dict(asset) for asset in assets]

        symbols = self.list_symbols(asset_type=resolved_type, data_dir=effective_dir)
        name_map = self.get_name_map(
            asset_type=resolved_type,
            stocklist_path=stocklist_path,
            etf_config_path=etf_config_path,
        )
        category_map = self._load_etf_category_map(etf_config_path) if resolved_type == "etf" else {}
        assets: list[AssetMetadata] = []
        for symbol in symbols:
            asset = AssetMetadata(
                symbol=symbol,
                asset_type=resolved_type,
                name=name_map.get(symbol, ""),
                category=category_map.get(symbol, ""),
            )
            assets.append(self._attach_asset_status(asset, effective_dir) if include_status else asset)
        return [self._asset_to_dict(asset) for asset in assets]

    def refresh_loaded_caches(
        self,
        *,
        data_dir: Optional[Path] = None,
        stock_codes: Optional[list[str]] = None,
        etf_codes: Optional[list[str]] = None,
        max_workers: int = 8,
    ) -> CacheRefreshResult:
        """Reload already-loaded stock/ETF memory caches after parquet updates."""
        effective_dir = self._effective_data_dir("stock", data_dir)
        stock_count = 0
        etf_count = 0
        stock_loaded = False
        etf_loaded = False

        stock_cache = get_stock_cache()
        stock_loaded = stock_cache.is_loaded()
        if stock_loaded:
            codes = stock_codes if stock_codes is not None else get_stock_list(str(effective_dir))
            stock_count = stock_cache.reload_all(
                data_dir=str(effective_dir),
                stock_codes=codes,
                max_workers=max_workers,
            )

        etf_cache = get_etf_cache()
        etf_loaded = etf_cache.is_loaded()
        if etf_loaded:
            codes = etf_codes if etf_codes is not None else get_etf_list(str(effective_dir))
            etf_count = etf_cache.reload_all(
                data_dir=str(effective_dir),
                etf_codes=codes,
                max_workers=max_workers,
            )

        return CacheRefreshResult(
            stock_count=stock_count,
            etf_count=etf_count,
            stock_cache_loaded=stock_loaded,
            etf_cache_loaded=etf_loaded,
        )

    def resolve_asset_type(self, symbol: str, asset_type: str = "auto") -> str:
        """Resolve asset type for MVP loaders."""
        value = (asset_type or "auto").strip().lower()
        if value in ("stock", "equity"):
            return "stock"
        if value in ("etf", "fund"):
            return "etf"
        if value in ("index", "idx"):
            return "index"
        if value != "auto":
            raise ValueError(f"Unsupported asset_type: {asset_type}")
        return "etf" if self._is_etf_like_code(symbol) else "stock"

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        value = str(symbol or "").strip().upper()
        return value.split(".", 1)[0] if "." in value else value

    @staticmethod
    def latest_expected_trading_day(now: Optional[datetime] = None) -> date:
        from common.market_data_policy import latest_expected_trading_day

        return latest_expected_trading_day(now)

    @classmethod
    def _project_root(cls) -> Path:
        return Path(__file__).resolve().parent.parent

    def _effective_data_dir(self, asset_type: str, data_dir: Optional[Path]) -> Path:
        if data_dir is not None:
            return Path(data_dir)
        return self.default_data_dir

    def _resolve_daily_parquet_path(self, symbol: str, asset_type: str, data_dir: Path) -> Path:
        normalized = self.normalize_symbol(symbol)
        if asset_type == "index":
            return Path(data_dir) / "index" / f"{normalized}.parquet"
        if asset_type == "etf":
            etf_path = Path(data_dir) / "etf" / f"{normalized}.parquet"
            if etf_path.exists():
                return etf_path
            return Path(data_dir) / f"{normalized}.parquet"
        return Path(data_dir) / f"{normalized}.parquet"

    def _infer_asset_type_from_path(self, path: Path, symbol: str, asset_type: str = "auto") -> str:
        value = (asset_type or "auto").strip().lower()
        if value != "auto":
            return self.resolve_asset_type(symbol, value)
        parent_name = path.parent.name.lower()
        if parent_name == "index":
            return "index"
        if parent_name == "etf":
            return "etf"
        return self.resolve_asset_type(symbol, "auto")

    @staticmethod
    def _normalize_daily_bars(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        bars = df.copy()
        if "date" not in bars.columns:
            return None
        bars["date"] = pd.to_datetime(bars["date"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            if column in bars.columns:
                bars[column] = pd.to_numeric(bars[column], errors="coerce")
        required = ["date", "open", "high", "low", "close"]
        if not all(column in bars.columns for column in required):
            return None
        bars = bars.dropna(subset=required)
        if bars.empty:
            return None
        existing_columns = [column for column in _STANDARD_BAR_COLUMNS if column in bars.columns]
        extra_columns = [column for column in bars.columns if column not in existing_columns]
        bars = bars[existing_columns + extra_columns]
        bars = bars.sort_values("date").reset_index(drop=True)
        return bars

    @staticmethod
    def _load_etf_daily_bars_from_local_dir(
        symbol: str,
        data_dir: Path,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Load ETF bars from a direct-code parquet directory such as live_rotation/data."""
        parquet_path = Path(data_dir) / f"{symbol}.parquet"
        if not parquet_path.exists():
            return None
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            return None
        if start and "date" in df.columns:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end and "date" in df.columns:
            df = df[df["date"] <= pd.to_datetime(end)]
        return df.reset_index(drop=True) if not df.empty else None

    @staticmethod
    def _load_index_daily_bars_from_local_dir(
        symbol: str,
        data_dir: Path,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Load index bars from data/index/{symbol}.parquet."""
        parquet_path = Path(data_dir) / "index" / f"{symbol}.parquet"
        if not parquet_path.exists():
            return None
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            return None
        if start and "date" in df.columns:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end and "date" in df.columns:
            df = df[df["date"] <= pd.to_datetime(end)]
        return df.reset_index(drop=True) if not df.empty else None

    def _list_index_assets(self) -> list[AssetMetadata]:
        from trading_app.services.market_data.index_service import get_index_list

        return [
            AssetMetadata(
                symbol=item.get("code", ""),
                asset_type="index",
                name=item.get("name", ""),
                market=item.get("market", item.get("exchange", "")),
            )
            for item in get_index_list()
            if item.get("code")
        ]

    def _attach_asset_status(self, asset: AssetMetadata, data_dir: Path) -> AssetMetadata:
        status = self.get_daily_metadata(asset.symbol, asset_type=asset.asset_type, data_dir=data_dir)
        return AssetMetadata(
            symbol=asset.symbol,
            asset_type=asset.asset_type,
            name=asset.name,
            market=asset.market,
            category=asset.category,
            first_date=status.first_date,
            latest_date=status.latest_date,
            rows=status.rows,
            data_path=status.data_path,
            is_fresh=status.is_fresh,
            data_source=status.data_source,
            data_version=status.data_version,
        )

    def _build_bundle_data_audit(
        self,
        views: Dict[str, StrategyDataView],
        benchmark_view: Optional[StrategyDataView] = None,
    ) -> dict:
        assets: list[dict] = []
        missing_sidecars: list[str] = []
        for symbol, view in views.items():
            asset = self._metadata_to_version_asset(view.metadata)
            assets.append(asset)
            if not view.metadata.sidecar_exists:
                missing_sidecars.append(symbol)
        if benchmark_view is not None:
            asset = self._metadata_to_version_asset(benchmark_view.metadata)
            assets.append(asset)
            if not benchmark_view.metadata.sidecar_exists:
                missing_sidecars.append(benchmark_view.symbol)
        assets = sorted(assets, key=lambda item: (item.get("asset_type", ""), item.get("symbol", "")))
        payload = {
            "scope": "market_data_bundle",
            "assets": assets,
            "schema_version": "data_version_audit.v1",
        }
        return DataVersionAudit(
            data_version=self._stable_hash(payload),
            generated_at=self._utc_now_iso(),
            scope="market_data_bundle",
            symbols=[item.get("symbol", "") for item in assets],
            assets=assets,
            sources=sorted({str(item.get("data_source", "unknown") or "unknown") for item in assets}),
            hashes=sorted({str(item.get("data_hash", "") or "") for item in assets if item.get("data_hash")}),
            missing_sidecars=missing_sidecars,
        ).to_dict()

    @staticmethod
    def _metadata_to_version_asset(metadata: BarsMetadata) -> dict:
        return {
            "symbol": metadata.symbol,
            "asset_type": metadata.asset_type,
            "frequency": metadata.frequency,
            "rows": int(metadata.rows or 0),
            "first_date": metadata.first_date,
            "latest_date": metadata.latest_date,
            "data_path": metadata.data_path,
            "data_source": metadata.data_source or "unknown",
            "data_hash": metadata.data_hash or "",
            "data_version": metadata.data_version or "",
            "sidecar_path": metadata.sidecar_path,
            "sidecar_exists": metadata.sidecar_exists,
        }

    @staticmethod
    def _status_to_version_asset(status: DailyDataStatus) -> dict:
        return {
            "symbol": status.symbol,
            "asset_type": status.asset_type,
            "frequency": "1d",
            "exists": status.exists,
            "rows": int(status.rows or 0),
            "first_date": status.first_date,
            "latest_date": status.latest_date,
            "data_path": status.data_path,
            "data_source": status.data_source or "unknown",
            "data_hash": status.data_hash or "",
            "data_version": status.data_version or "",
            "sidecar_path": status.sidecar_path,
            "sidecar_exists": status.sidecar_exists,
        }

    @staticmethod
    def _parquet_frame_info(path: Path) -> dict:
        try:
            df = pd.read_parquet(path)
        except Exception:
            return {"rows": 0, "columns": [], "first_date": None, "latest_date": None}
        columns = list(df.columns)
        if df.empty:
            return {"rows": 0, "columns": columns, "first_date": None, "latest_date": None}
        first_date = None
        latest_date = None
        time_column = "date" if "date" in df.columns else ("time" if "time" in df.columns else "")
        if time_column:
            dates = pd.to_datetime(df[time_column], errors="coerce").dropna()
            if not dates.empty:
                first_date = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
                latest_date = pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
        return {
            "rows": int(len(df)),
            "columns": columns,
            "first_date": first_date,
            "latest_date": latest_date,
        }

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _stable_hash(payload) -> str:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _read_sidecar_dict(path: Path) -> dict:
        try:
            resolved = Path(path)
            if not resolved.exists():
                return {}
            with open(resolved, "r", encoding="utf-8") as file:
                payload = json.load(file)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True, default=str)

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    @staticmethod
    def _to_date(value: Union[str, date, datetime]) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value or "").strip()
        if not text:
            raise ValueError("date value is required")
        if len(text) == 8 and text.isdigit():
            return datetime.strptime(text, "%Y%m%d").date()
        return date.fromisoformat(text)

    @staticmethod
    def _load_etf_category_map(config_path: Optional[Path] = None) -> Dict[str, str]:
        categories = load_etf_categories(str(config_path)) if config_path else load_etf_categories()
        result: Dict[str, str] = {}
        for category in categories:
            category_name = category.get("name", "")
            for etf in category.get("etfs", []):
                code = etf.get("code", "")
                if code:
                    result[code] = category_name
        return result

    @staticmethod
    def _asset_to_dict(asset: AssetMetadata) -> dict:
        result = asset.__dict__.copy()
        result["code"] = asset.symbol
        return result

    @staticmethod
    def _latest_date_str(df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty or "date" not in df.columns:
            return None
        latest = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.isna(latest):
            return None
        return pd.Timestamp(latest).strftime("%Y-%m-%d")

    @staticmethod
    def _first_date_str(df: pd.DataFrame) -> Optional[str]:
        if df is None or df.empty or "date" not in df.columns:
            return None
        first = pd.to_datetime(df["date"], errors="coerce").min()
        if pd.isna(first):
            return None
        return pd.Timestamp(first).strftime("%Y-%m-%d")

    @staticmethod
    def _normalize_symbols(symbols: SymbolInput) -> Iterable[str]:
        if isinstance(symbols, str):
            return [DataPortal.normalize_symbol(symbols)]
        return [DataPortal.normalize_symbol(symbol) for symbol in symbols]

    @staticmethod
    def _ensure_daily_frequency(frequency: str) -> None:
        if str(frequency or "").strip().lower() not in _DAILY_FREQUENCIES:
            raise ValueError("DataPortal MVP only supports daily bars")

    @staticmethod
    def _is_etf_like_code(symbol: str) -> bool:
        from common.market_data_policy import is_etf_like_code

        return is_etf_like_code(symbol)


_data_portal: Optional[DataPortal] = None


def get_data_portal() -> DataPortal:
    """Return the process-wide DataPortal instance."""
    global _data_portal
    if _data_portal is None:
        _data_portal = DataPortal()
    return _data_portal


def set_data_portal(portal: Optional[DataPortal]) -> None:
    """Replace the process-wide DataPortal instance, mainly for tests."""
    global _data_portal
    _data_portal = portal


__all__ = [
    "AssetMetadata",
    "BarsMetadata",
    "BarsResult",
    "CacheRefreshResult",
    "CorporateActionRecord",
    "DailyDataStatus",
    "DataPortal",
    "DataVersionAudit",
    "ETFDataCache",
    "FreshnessStatus",
    "_load_etf_data_from_parquet",
    "_load_stock_data_from_csv",
    "_load_stock_data_from_parquet",
    "_normalize_symbol_code",
    "MarketDataBundle",
    "ParquetSidecarMetadata",
    "StockDataCache",
    "StrategyDataView",
    "TradingCalendarDay",
    "get_data_portal",
    "get_date_range",
    "get_etf_cache",
    "get_etf_date_range",
    "get_etf_list",
    "get_stock_cache",
    "get_stock_list",
    "load_etf_categories",
    "load_etf_data",
    "load_etf_name_map",
    "load_stock_data",
    "load_stock_name_map",
    "set_data_portal",
]
