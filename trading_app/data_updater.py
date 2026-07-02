import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from PyQt6.QtCore import QThread, pyqtSignal
from common.daily_update_policy import get_daily_update_policy
from common.kline_update_engine import (
    check_xtquant_ready,
    format_failed_update_message,
    run_batched_updates,
    run_xtquant_daily_history_precheck,
)
from trading_app.services.market_data.data_update_result import DataUpdateResult
# Import from scripts directory
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from scripts import fetch_kline
    from scripts import fetch_kline_xtquant
except ImportError:
    # Direct import if scripts is in path
    sys.path.insert(0, str(project_root / "scripts"))
    import fetch_kline
    import fetch_kline_xtquant

# Tushare 延迟导入（可能未安装）
try:
    import tushare as ts
    HAS_TUSHARE = True
except ImportError:
    HAS_TUSHARE = False
    ts = None


_STOCK_FRESHNESS_PROBE_CODES = ("000001", "600000", "000333", "300750", "600519")


def _asset_type_from_subdir(subdir: str) -> str:
    if subdir == "index":
        return "index"
    if subdir == "etf":
        return "etf"
    return "auto"


def _check_update_output_freshness(data_dir: Path, code: str, subdir: str = "") -> tuple[bool, str]:
    data_path = data_dir / subdir / f"{str(code or '').strip().upper().split('.', 1)[0]}.parquet" if subdir else None
    return get_daily_update_policy().check_daily_freshness(
        code,
        asset_type=_asset_type_from_subdir(subdir),
        data_dir=data_dir / subdir if subdir else data_dir,
        data_path=data_path,
    )


def _run_xtquant_daily_history_precheck() -> tuple[bool, str]:
    """Verify miniQMT can really fetch the latest daily K-line, not only connect."""
    return run_xtquant_daily_history_precheck(
        action_hint="请确认 miniQMT 已正常运行并登录后再更新/执行策略。",
    )


class DataUpdateThread(QThread):
    """
    Background thread for updating stock data.
    
    支持多数据源：
    - tushare: 使用 Tushare API（需要 token）
    - xtquant: 使用 xtquant/miniQMT（需要本地运行 miniQMT）
    
    支持多周期：
    - 1d: 日线
    - 1m/5m/15m/30m/60m: 分钟线
    """
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # success, message
    result_signal = pyqtSignal(object)  # DataUpdateResult

    def __init__(
        self,
        data_dir: str,
        stocklist_path: str,
        tushare_token: str = "",
        full_update: bool = False,
        exclude_boards: List[str] = None,
        max_workers: int = 6,
        codes: List[str] = None,
        start_date: str = None,
        data_source: str = "tushare",  # "tushare" or "xtquant"
        period: str = "1d",  # "1d", "1m", "5m", "15m", "30m", "60m"
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.stocklist_path = Path(stocklist_path) if stocklist_path else None
        self.tushare_token = tushare_token
        self.full_update = full_update
        self.exclude_boards = set(exclude_boards) if exclude_boards else set()
        self.max_workers = max_workers
        self.codes = codes
        self.start_date = start_date
        self.data_source = data_source
        self.period = period
        self._is_running = True
        self.last_result = DataUpdateResult(ok=False, message="尚未执行")

    def _finish(self, result: DataUpdateResult):
        self.last_result = result
        self.result_signal.emit(result)
        self.finished_signal.emit(*result.to_legacy_tuple())

    def run(self):
        try:
            if self.data_source == "tushare":
                self._run_tushare()
            elif self.data_source == "xtquant":
                self._run_xtquant()
            else:
                self._finish(DataUpdateResult(ok=False, message=f"未知数据源: {self.data_source}"))
        except Exception as e:
            self._finish(DataUpdateResult(ok=False, message=f"发生错误: {str(e)}"))

    def _run_tushare(self):
        """使用 Tushare 数据源更新"""
        if not HAS_TUSHARE:
            self._finish(DataUpdateResult(ok=False, message="Tushare 未安装，请执行: pip install tushare"))
            return
        
        self.log_message.emit("正在初始化 Tushare API...")
        if not self.tushare_token:
            self._finish(DataUpdateResult(ok=False, message="Tushare Token 未提供"))
            return

        ts.set_token(self.tushare_token)
        pro = ts.pro_api()
        fetch_kline.set_api(pro)

        # 获取股票代码列表
        codes = self._get_stock_codes(fetch_kline.load_codes_from_stocklist)
        if codes is None:
            return

        total_stocks = len(codes)
        if not self.codes:
            self.log_message.emit(
                f"找到 {total_stocks} 只股票（排除板块: {', '.join(self.exclude_boards) or '无'}）。开始更新..."
            )

        # 确保输出目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 确定获取函数（Tushare 只支持日线）
        if self.period != "1d":
            self.log_message.emit("注意: Tushare 数据源仅支持日线，已自动切换为日线模式")
        
        fetch_func = fetch_kline.fetch_one_full if self.full_update else fetch_kline.fetch_one

        # 日期范围
        window = get_daily_update_policy().resolve_fetch_window(
            asset_type="stock",
            explicit_start=self.start_date,
            full_update=self.full_update,
        )

        # 执行更新
        self._execute_update(codes, fetch_func, window.start_date, window.end_date)

    def _run_xtquant(self):
        """使用 xtquant/miniQMT 数据源更新"""
        self.log_message.emit("正在检查 miniQMT 连接状态...")
        ready, msg = check_xtquant_ready()
        if not ready:
            self._finish(DataUpdateResult(ok=False, message=msg))
            return
        
        self.log_message.emit("miniQMT 连接正常")

        self.log_message.emit("正在验证 miniQMT 历史K线是否更新到最新交易日...")
        history_ok, history_msg = _run_xtquant_daily_history_precheck()
        if not history_ok:
            self._finish(DataUpdateResult(ok=False, message=history_msg))
            return
        self.log_message.emit(history_msg)

        # 获取股票代码列表
        codes = self._get_stock_codes(fetch_kline_xtquant.load_codes_from_stocklist)
        if codes is None:
            return

        total_stocks = len(codes)
        period_name = self._get_period_name()
        if not self.codes:
            self.log_message.emit(
                f"找到 {total_stocks} 只股票（排除板块: {', '.join(self.exclude_boards) or '无'}）。"
                f"开始更新 {period_name} 数据..."
            )

        # 确保输出目录存在
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 确定获取函数
        if self.full_update:
            fetch_func = lambda code, start, end, out_dir: \
                fetch_kline_xtquant.fetch_one_full(code, start, end, out_dir, self.period)
        else:
            fetch_func = lambda code, start, end, out_dir: \
                fetch_kline_xtquant.fetch_one(code, start, end, out_dir, self.period)

        # 日期范围
        window = get_daily_update_policy().resolve_fetch_window(
            asset_type="stock",
            explicit_start=self.start_date,
            full_update=self.full_update,
        )

        # 执行更新
        self._execute_update(codes, fetch_func, window.start_date, window.end_date)

    def _get_stock_codes(self, load_func) -> Optional[List[str]]:
        """获取股票代码列表"""
        if self.codes:
            self.log_message.emit(f"正在更新 {len(self.codes)} 只指定股票...")
            return self.codes
        elif self.stocklist_path:
            self.log_message.emit(f"正在从 {self.stocklist_path} 加载股票列表...")
            codes = load_func(self.stocklist_path, self.exclude_boards)
            if not codes:
                self._finish(DataUpdateResult(ok=False, message="未找到股票代码"))
                return None
            return codes
        else:
            self._finish(DataUpdateResult(ok=False, message="未提供股票代码或股票列表路径"))
            return None

    def _get_period_name(self) -> str:
        """获取周期的中文名称"""
        period_names = {
            "1d": "日线",
            "1m": "1分钟",
            "5m": "5分钟",
            "15m": "15分钟",
            "30m": "30分钟",
            "60m": "60分钟",
        }
        return period_names.get(self.period, self.period)

    def _execute_update(self, codes: List[str], fetch_func, start_date: str, end_date: str):
        """执行数据更新"""
        summary = run_batched_updates(
            list(codes),
            lambda code: fetch_func(code, start_date, end_date, self.data_dir),
            max_workers=self.max_workers,
            batch_size=50,
            should_stop=lambda: not self._is_running,
            progress_cb=lambda current, total, _code, msg: self.progress_updated.emit(current, total, msg),
            success_message=lambda code, current, total: f"已更新 {code} ({current}/{total})",
            failure_message=lambda code, exc: f"更新 {code} 失败: {exc}",
        )

        if not self._is_running:
            self._finish(DataUpdateResult(ok=False, message="更新已停止"))
        elif summary.failed_items:
            self._finish(DataUpdateResult(
                ok=False,
                failed_codes=summary.failed_codes,
                message=format_failed_update_message("股票", summary.failed_items, "只"),
            ))
        elif self.data_source == "xtquant" and self.period == "1d":
            check_codes = list(self.codes or [])
            if not check_codes:
                code_set = {str(code).strip().upper().split(".", 1)[0] for code in codes}
                check_codes = [code for code in _STOCK_FRESHNESS_PROBE_CODES if code in code_set]
            stale_items = []
            for code in check_codes:
                fresh, info = _check_update_output_freshness(self.data_dir, code)
                if not fresh:
                    stale_items.append(f"{code}: {info}")
            if stale_items:
                self._finish(DataUpdateResult(
                    ok=False,
                    stale_codes=[item.split(":", 1)[0] for item in stale_items],
                    message=get_daily_update_policy().format_stale_items("股票数据", stale_items, "只"),
                ))
            else:
                self._finish(DataUpdateResult(ok=True, updated_stocks=summary.success, message="数据更新完成"))
        else:
            self._finish(DataUpdateResult(ok=True, updated_stocks=summary.success, message="数据更新完成"))

    def stop(self):
        self._is_running = False
        self.wait()


class ETFUpdateThread(QThread):
    """
    Background thread for updating ETF data.
    使用 xtquant/miniQMT 获取ETF数据
    """
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # success, message
    result_signal = pyqtSignal(object)  # DataUpdateResult

    def __init__(
        self,
        data_dir: str,
        etf_config_path: str = None,
        full_update: bool = False,
        max_workers: int = 4,
        codes: List[str] = None,
        start_date: str = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.etf_config_path = Path(etf_config_path) if etf_config_path else None
        self.full_update = full_update
        self.max_workers = max_workers
        self.codes = codes
        self.start_date = start_date
        self._is_running = True
        self.last_result = DataUpdateResult(ok=False, message="尚未执行")

    def _finish(self, result: DataUpdateResult):
        self.last_result = result
        self.result_signal.emit(result)
        self.finished_signal.emit(*result.to_legacy_tuple())

    def run(self):
        try:
            self._run_etf_update()
        except Exception as e:
            self._finish(DataUpdateResult(ok=False, message=f"发生错误: {str(e)}"))

    def _run_etf_update(self):
        """使用 xtquant 更新ETF数据"""
        self.log_message.emit("正在检查 miniQMT 连接状态...")
        ready, msg = check_xtquant_ready()
        if not ready:
            self._finish(DataUpdateResult(ok=False, message=msg))
            return
        
        self.log_message.emit("miniQMT 连接正常")

        self.log_message.emit("正在验证 miniQMT 历史K线是否更新到最新交易日...")
        history_ok, history_msg = _run_xtquant_daily_history_precheck()
        if not history_ok:
            self._finish(DataUpdateResult(ok=False, message=history_msg))
            return
        self.log_message.emit(history_msg)

        # 获取ETF代码列表
        strict_freshness_check = bool(self.codes)
        if self.codes:
            codes = self.codes
            self.log_message.emit(f"正在更新 {len(codes)} 只指定ETF...")
        elif self.etf_config_path and self.etf_config_path.exists():
            codes = fetch_kline_xtquant.load_etf_codes_from_config(self.etf_config_path)
            if not codes:
                self._finish(DataUpdateResult(ok=False, message="未找到ETF代码"))
                return
            self.log_message.emit(f"从配置文件找到 {len(codes)} 只ETF，开始更新...")
        else:
            self._finish(DataUpdateResult(ok=False, message="未提供ETF代码或配置文件路径"))
            return

        # 确保输出目录存在
        etf_dir = self.data_dir / "etf"
        etf_dir.mkdir(parents=True, exist_ok=True)

        # 确定获取函数
        if self.full_update:
            fetch_func = fetch_kline_xtquant.fetch_etf_one_full
        else:
            fetch_func = fetch_kline_xtquant.fetch_etf_one

        # 日期范围
        window = get_daily_update_policy().resolve_fetch_window(
            asset_type="etf",
            explicit_start=self.start_date,
            full_update=self.full_update,
        )
        start_date, end_date = window.start_date, window.end_date

        summary = run_batched_updates(
            list(codes),
            lambda code: fetch_func(code, start_date, end_date, self.data_dir),
            max_workers=self.max_workers,
            batch_size=20,
            should_stop=lambda: not self._is_running,
            progress_cb=lambda current, total, _code, msg: self.progress_updated.emit(current, total, msg),
            success_message=lambda code, current, total: f"已更新ETF {code} ({current}/{total})",
            failure_message=lambda code, exc: f"更新ETF {code} 失败: {exc}",
        )

        if not self._is_running:
            self._finish(DataUpdateResult(ok=False, message="ETF更新已停止"))
        elif summary.failed_items:
            self._finish(DataUpdateResult(
                ok=False,
                failed_codes=summary.failed_codes,
                message=format_failed_update_message("ETF", summary.failed_items, "只"),
            ))
        else:
            stale_items = []
            for code in codes:
                fresh, info = _check_update_output_freshness(self.data_dir, code, subdir="etf")
                if not fresh:
                    stale_items.append(f"{code}: {info}")
            if stale_items:
                stale_message = get_daily_update_policy().format_stale_items("ETF数据", stale_items, "只")
                stale_codes = [item.split(":", 1)[0] for item in stale_items]
                if strict_freshness_check:
                    self._finish(DataUpdateResult(
                        ok=False,
                        stale_codes=stale_codes,
                        message=stale_message,
                    ))
                else:
                    warning = f"ETF数据更新完成，共更新 {summary.success} 只；{len(stale_items)} 只ETF无最新行情或为空文件，已跳过: {stale_message}"
                    self.log_message.emit(f"⚠ {warning}")
                    self._finish(DataUpdateResult(
                        ok=True,
                        updated_etfs=summary.success,
                        message=warning,
                        details={"stale_etfs": "；".join(stale_codes[:50])},
                    ))
            else:
                self._finish(DataUpdateResult(ok=True, updated_etfs=summary.success, message=f"ETF数据更新完成，共更新 {summary.success} 只"))

    def stop(self):
        self._is_running = False
        self.wait()


class IndexUpdateThread(QThread):
    """
    Background thread for updating index data.
    Uses xtquant/miniQMT to get index data
    """
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    log_message = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # success, message
    result_signal = pyqtSignal(object)  # DataUpdateResult

    def __init__(
        self,
        data_dir: str,
        index_config_path: str = None,
        index_codes: Optional[List[Dict[str, Any]]] = None,
        full_update: bool = False,
        max_workers: int = 4,
        start_date: str = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.index_config_path = Path(index_config_path) if index_config_path else None
        self.index_codes = list(index_codes or [])
        self.full_update = full_update
        self.max_workers = max_workers
        self.start_date = start_date
        self._is_running = True
        self.last_result = DataUpdateResult(ok=False, message="尚未执行")

    def _finish(self, result: DataUpdateResult):
        self.last_result = result
        self.result_signal.emit(result)
        self.finished_signal.emit(*result.to_legacy_tuple())

    def run(self):
        try:
            self._run_index_update()
        except Exception as e:
            self._finish(DataUpdateResult(ok=False, message=f"Error occurred: {str(e)}"))

    def _run_index_update(self):
        """Update index data using xtquant"""
        self.log_message.emit("Checking miniQMT connection status...")
        ready, msg = check_xtquant_ready()
        if not ready:
            self._finish(DataUpdateResult(ok=False, message=msg))
            return
        
        self.log_message.emit("miniQMT connected successfully")

        self.log_message.emit("正在验证 miniQMT 历史K线是否更新到最新交易日...")
        history_ok, history_msg = _run_xtquant_daily_history_precheck()
        if not history_ok:
            self._finish(DataUpdateResult(ok=False, message=history_msg))
            return
        self.log_message.emit(history_msg)

        # Get index list
        indices = self.index_codes or fetch_kline_xtquant.load_index_codes_from_config(self.index_config_path)
        if not indices:
            self._finish(DataUpdateResult(ok=False, message="No index codes found"))
            return
        
        self.log_message.emit(f"Found {len(indices)} indices to update...")

        # Ensure output directory exists
        index_dir = self.data_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)

        # Determine fetch function
        if self.full_update:
            fetch_func = fetch_kline_xtquant.fetch_index_one_full
        else:
            fetch_func = fetch_kline_xtquant.fetch_index_one

        # Date range
        window = get_daily_update_policy().resolve_fetch_window(
            asset_type="index",
            explicit_start=self.start_date,
            full_update=self.full_update,
        )
        start_date, end_date = window.start_date, window.end_date

        def update_one(idx: Dict[str, Any]) -> None:
            fetch_func(
                idx.get("code"),
                start_date,
                end_date,
                self.data_dir,
                "1d",
                idx.get("exchange"),
            )

        summary = run_batched_updates(
            list(indices),
            update_one,
            max_workers=self.max_workers,
            batch_size=10,
            should_stop=lambda: not self._is_running,
            progress_cb=lambda current, total, _idx, msg: self.progress_updated.emit(current, total, msg),
            success_message=lambda idx, current, total: f"Updated index {idx.get('code')} ({idx.get('name', '')}) ({current}/{total})",
            failure_message=lambda idx, exc: f"Update index {idx.get('code')} failed: {exc}",
            item_label=lambda idx: str(idx.get("code", "") or ""),
        )

        if not self._is_running:
            self._finish(DataUpdateResult(ok=False, message="Index update stopped"))
        elif summary.failed_items:
            self._finish(DataUpdateResult(
                ok=False,
                failed_codes=summary.failed_codes,
                message=format_failed_update_message("指数", summary.failed_items, "个"),
            ))
        else:
            stale_items = []
            for idx in indices:
                code = str(idx.get("code", "") or "")
                fresh, info = _check_update_output_freshness(self.data_dir, code, subdir="index")
                if not fresh:
                    stale_items.append(f"{code}: {info}")
            if stale_items:
                self._finish(DataUpdateResult(
                    ok=False,
                    stale_codes=[item.split(":", 1)[0] for item in stale_items],
                    message=get_daily_update_policy().format_stale_items("指数数据", stale_items, "个"),
                ))
            else:
                self._finish(DataUpdateResult(ok=True, updated_indices=summary.success, message=f"Index data update completed, updated {summary.success} indices"))

    def stop(self):
        self._is_running = False
        self.wait()
