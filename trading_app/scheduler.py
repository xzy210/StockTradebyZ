# scheduler.py - 定时任务管理器
"""
负责管理定时执行的数据更新和通知任务

注意：选股功能已迁移到 strategy_app，此文件不再包含选股相关的定时任务
"""
import json
import logging
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import Optional, Dict, List

from PyQt6.QtCore import QObject, QTimer, pyqtSignal, QThread

try:
    from trading_app.data_updater import DataUpdateThread, ETFUpdateThread, IndexUpdateThread
    from trading_app.notifier import get_notification_manager
    from trading_app.services.market_data.data_update_result import DataUpdateResult
except ImportError:
    from .data_updater import DataUpdateThread, ETFUpdateThread, IndexUpdateThread
    from .notifier import get_notification_manager
    from .services.data_update_result import DataUpdateResult


class ScheduledTaskWorker(QObject):
    """
    定时任务执行器，负责按顺序运行更新和通知
    """
    finished = pyqtSignal(bool, str)
    log_message = pyqtSignal(str)
    
    def __init__(self, config: dict, data_dir: str, stocklist_path: str):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.stocklist_path = stocklist_path
        self.update_thread = None
        self.etf_update_thread = None
        self.index_update_thread = None
        self._last_stock_update_result = None
        self._last_etf_update_result = None
        self._last_index_update_result = None

    def stop(self):
        """停止当前执行的流水线"""
        if self.update_thread and self.update_thread.isRunning():
            self.update_thread.stop()
        if self.etf_update_thread and self.etf_update_thread.isRunning():
            self.etf_update_thread.stop()
        if self.index_update_thread and self.index_update_thread.isRunning():
            self.index_update_thread.stop()

    def _store_update_result(self, attr_name: str, result: object):
        if isinstance(result, DataUpdateResult):
            setattr(self, attr_name, result)

    def wait_all(self, timeout_ms: int = 5000):
        """
        等待所有内部线程完全退出
        
        Args:
            timeout_ms: 每个线程的最大等待时间（毫秒）
        """
        if self.update_thread:
            if self.update_thread.isRunning():
                self.update_thread.wait(timeout_ms)
        if self.etf_update_thread:
            if self.etf_update_thread.isRunning():
                self.etf_update_thread.wait(timeout_ms)
        if self.index_update_thread:
            if self.index_update_thread.isRunning():
                self.index_update_thread.wait(timeout_ms)

    def start(self):
        """开始执行任务流水线"""
        task_name = self.config.get("name", "未命名任务")
        self.log_message.emit(f"🚀 开始执行定时任务 [{task_name}]: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.config.get("step_update", True):
            self._step1_update_data()
        elif self.config.get("step_notify", True):
            self._step3_send_notification()
        else:
            self.log_message.emit("ℹ️ 未勾选任何执行步骤，任务提前结束")
            self.finished.emit(True, "任务结束")

    def _step1_update_data(self):
        """步骤1: 更新数据"""
        is_full = self.config.get("full_update", False)
        start_date = self.config.get("start_date", None)
        
        mode_str = f"全量更新 (从 {start_date})" if is_full else "增量更新"
        self.log_message.emit(f"🔄 步骤1: 正在进行数据{mode_str}...")
        
        data_source = self.config.get("data_source", "xtquant")
        token = self.config.get("tushare_token", "")
        
        self.update_thread = DataUpdateThread(
            data_dir=self.data_dir,
            stocklist_path=self.stocklist_path,
            tushare_token=token,
            full_update=is_full,
            start_date=start_date,
            data_source=data_source,
            period="1d"
        )
        
        self.update_thread.result_signal.connect(lambda result: self._store_update_result("_last_stock_update_result", result))
        self.update_thread.finished_signal.connect(self._on_update_finished)
        self.update_thread.start()

    def _on_update_finished(self, success, message):
        result = self._last_stock_update_result
        display_message = result.to_ui_message() if isinstance(result, DataUpdateResult) else message
        if not success:
            self.log_message.emit(f"❌ 股票数据更新失败: {display_message}")
            self.finished.emit(False, f"股票数据更新失败: {display_message}")
            return
            
        self.log_message.emit(f"✅ 股票数据更新完成: {display_message}")
        
        # 如果是全量更新，继续更新ETF数据
        if self.config.get("full_update", False):
            self._step1b_update_etf_data()
        elif self.config.get("step_notify", True):
            self._step3_send_notification()
        else:
            self.finished.emit(True, "全部任务已完成")

    def _step1b_update_etf_data(self):
        """步骤1b: 更新ETF数据（全量更新时执行）"""
        start_date = self.config.get("start_date", None)
        self.log_message.emit(f"🔄 步骤1b: 正在进行ETF数据全量更新 (从 {start_date})...")
        
        # ETF配置文件路径
        from pathlib import Path
        etf_config_path = Path(self.data_dir).parent / "trading_app" / "config" / "etf_list.json"
        
        self.etf_update_thread = ETFUpdateThread(
            data_dir=self.data_dir,
            etf_config_path=str(etf_config_path),
            full_update=True,
            start_date=start_date,
        )
        
        self.etf_update_thread.result_signal.connect(lambda result: self._store_update_result("_last_etf_update_result", result))
        self.etf_update_thread.finished_signal.connect(self._on_etf_update_finished)
        self.etf_update_thread.start()

    def _on_etf_update_finished(self, success, message):
        result = self._last_etf_update_result
        display_message = result.to_ui_message() if isinstance(result, DataUpdateResult) else message
        if not success:
            self.log_message.emit(f"❌ ETF数据更新失败: {display_message}")
            self.finished.emit(False, f"ETF数据更新失败: {display_message}")
            return
            
        self.log_message.emit(f"✅ ETF数据更新完成: {display_message}")
        
        # Continue to update index data
        self._step1c_update_index_data()

    def _step1c_update_index_data(self):
        """步骤1c: 更新指数数据（全量更新时执行）"""
        start_date = self.config.get("start_date", None)
        self.log_message.emit(f"🔄 步骤1c: 正在进行指数数据全量更新 (从 {start_date})...")
        
        self.index_update_thread = IndexUpdateThread(
            data_dir=self.data_dir,
            index_config_path=None,  # Use default index list
            full_update=True,
            start_date=start_date,
        )
        
        self.index_update_thread.result_signal.connect(lambda result: self._store_update_result("_last_index_update_result", result))
        self.index_update_thread.finished_signal.connect(self._on_index_update_finished)
        self.index_update_thread.start()

    def _on_index_update_finished(self, success, message):
        result = self._last_index_update_result
        display_message = result.to_ui_message() if isinstance(result, DataUpdateResult) else message
        if not success:
            self.log_message.emit(f"❌ 指数数据更新失败: {display_message}")
            self.finished.emit(False, f"指数数据更新失败: {display_message}")
            return
            
        self.log_message.emit(f"✅ 指数数据更新完成: {display_message}")
        
        if self.config.get("step_notify", True):
            self._step3_send_notification()
        else:
            self.finished.emit(True, "全部任务已完成")

    def _step3_send_notification(self):
        """步骤3: 发送企微通知（选股功能已移除）"""
        self.log_message.emit("ℹ️ 选股功能已迁移到策略应用，定时任务仅执行数据更新")
        self.finished.emit(True, "数据更新任务已完成")


class ScheduledTaskManager(QObject):
    """
    定时任务管理器，负责管理多个定时任务的调度和执行
    """
    task_finished = pyqtSignal(str, bool, str)  # task_id, success, message
    task_log = pyqtSignal(str)  # log message
    
    def __init__(self, data_dir: str, stocklist_path: str):
        super().__init__()
        self.data_dir = data_dir
        self.stocklist_path = stocklist_path
        self.config_path = Path(__file__).parent / "config" / "scheduler_config.json"
        self.tasks: Dict[str, dict] = {}
        self.timers: Dict[str, QTimer] = {}
        self.current_worker: Optional[ScheduledTaskWorker] = None
        self.config: dict = {}  # 当前配置
        
        self._load_config()
        self._setup_timers()
    
    @property
    def config(self) -> dict:
        """获取当前配置（兼容旧代码）"""
        return self._config
    
    @config.setter
    def config(self, value: dict):
        self._config = value
    
    def save_config(self, config: dict):
        """保存配置"""
        self.config = config
        # 转换为任务格式保存
        self.tasks = {
            "update": {
                "name": "每日数据更新",
                "enabled": config.get("update_enabled", False),
                "time": config.get("update_time", "14:30"),
                "step_update": config.get("step_update", True),
                "data_source": config.get("data_source", "xtquant"),
                "tushare_token": config.get("tushare_token", ""),
            },
            "maintenance": {
                "name": "全量数据更新",
                "enabled": config.get("maint_enabled", False),
                "time": config.get("maint_time", "18:00"),
                "full_update": True,
                "start_date": config.get("maint_start_date", "20080101"),
                "data_source": config.get("data_source", "xtquant"),
                "tushare_token": config.get("tushare_token", ""),
            }
        }
        self._save_config()
        self._setup_timers()
    
    def run_now(self, task_id: str) -> tuple:
        """立即执行任务"""
        if task_id in self.tasks:
            self.execute_task(task_id)
            return True, "任务已启动"
        return False, f"任务 {task_id} 不存在"
    
    def _load_config(self):
        """加载定时任务配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tasks = data.get("tasks", {})
            except Exception as e:
                logging.error(f"加载定时任务配置失败: {e}")
                self.tasks = {}
        
        # 设置默认配置
        update_task = self.tasks.get("update", {})
        maint_task = self.tasks.get("maintenance", {})
        
        self.config = {
            "update_enabled": update_task.get("enabled", False),
            "update_time": update_task.get("time", "14:30"),
            "step_update": update_task.get("step_update", True),
            
            "maint_enabled": maint_task.get("enabled", False),
            "maint_time": maint_task.get("time", "18:00"),
            "maint_start_date": maint_task.get("start_date", "20080101"),
            
            "data_source": update_task.get("data_source", "xtquant"),
            "tushare_token": update_task.get("tushare_token", ""),
        }
    
    def _save_config(self):
        """保存定时任务配置"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump({"tasks": self.tasks}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存定时任务配置失败: {e}")
    
    def _setup_timers(self):
        """设置定时器"""
        for task_id, task_config in self.tasks.items():
            if task_config.get("enabled", False):
                self._setup_timer(task_id, task_config)
    
    def _setup_timer(self, task_id: str, task_config: dict):
        """为单个任务设置定时器"""
        scheduled_time = task_config.get("time", "09:00")
        try:
            hour, minute = map(int, scheduled_time.split(":"))
        except ValueError:
            logging.error(f"任务 {task_id} 的时间格式错误: {scheduled_time}")
            return
        
        # 计算下次执行时间
        now = datetime.now()
        target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        if target_time <= now:
            # 如果今天的时间已过，设置为明天
            target_time += timedelta(days=1)
        
        # 计算毫秒数
        ms_until = int((target_time - now).total_seconds() * 1000)
        
        # 创建单次定时器
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._on_timer_triggered(task_id))
        timer.start(ms_until)
        
        self.timers[task_id] = timer
        logging.info(f"任务 {task_id} 将在 {target_time} 执行")
    
    def _on_timer_triggered(self, task_id: str):
        """定时器触发"""
        if task_id in self.tasks:
            self.execute_task(task_id)
            
            # 重新设置明天的定时器
            self._setup_timer(task_id, self.tasks[task_id])
    
    def execute_task(self, task_id: str):
        """立即执行任务"""
        if task_id not in self.tasks:
            logging.error(f"任务 {task_id} 不存在")
            return
        
        if self.current_worker and self.current_worker.isRunning():
            logging.warning("已有任务正在执行，请等待完成")
            return
        
        task_config = self.tasks[task_id]
        
        # 创建并启动工作线程
        self.current_worker = ScheduledTaskWorker(
            task_config,
            self.data_dir,
            self.stocklist_path
        )
        
        self.current_worker.finished.connect(
            lambda success, msg: self.task_finished.emit(task_id, success, msg)
        )
        self.current_worker.start()
    
    def stop_task(self):
        """停止当前执行的任务"""
        if self.current_worker:
            self.current_worker.stop()
            self.current_worker.wait(5000)
    
    def stop(self):
        """停止所有定时器"""
        for timer in self.timers.values():
            timer.stop()
        self.stop_task()
    
    def get_all_tasks(self) -> Dict[str, dict]:
        """获取所有任务配置"""
        return self.tasks.copy()
    
    def add_task(self, task_id: str, task_config: dict):
        """添加新任务"""
        self.tasks[task_id] = task_config
        self._save_config()
        
        if task_config.get("enabled", False):
            self._setup_timer(task_id, task_config)
    
    def update_task(self, task_id: str, task_config: dict):
        """更新任务配置"""
        # 停止现有定时器
        if task_id in self.timers:
            self.timers[task_id].stop()
            del self.timers[task_id]
        
        self.tasks[task_id] = task_config
        self._save_config()
        
        if task_config.get("enabled", False):
            self._setup_timer(task_id, task_config)
    
    def delete_task(self, task_id: str):
        """删除任务"""
        if task_id in self.timers:
            self.timers[task_id].stop()
            del self.timers[task_id]
        
        if task_id in self.tasks:
            del self.tasks[task_id]
            self._save_config()
    
    def toggle_task(self, task_id: str, enabled: bool):
        """启用/禁用任务"""
        if task_id in self.tasks:
            self.tasks[task_id]["enabled"] = enabled
            self._save_config()
            
            if enabled:
                self._setup_timer(task_id, self.tasks[task_id])
            elif task_id in self.timers:
                self.timers[task_id].stop()
                del self.timers[task_id]


class FullDataSyncWorker(QThread):
    """
    全量数据同步工作器

    按顺序执行：股票全量前复权 -> ETF全量前复权 -> 指数数据
    """
    progress_signal = pyqtSignal(str, int, int)  # phase_name, current, total
    log_signal = pyqtSignal(str)  # log message
    finished_signal = pyqtSignal(bool, str)  # success, message

    def __init__(self, data_dir: str, stocklist_path: str, start_date: str = "20080101"):
        super().__init__()
        self.data_dir = data_dir
        self.stocklist_path = stocklist_path
        self.start_date = start_date
        self._stop_requested = False

        # 子线程引用
        self._stock_thread = None
        self._etf_thread = None
        self._index_thread = None

        # 同步状态
        self._current_phase = ""
        self._phases_completed = 0
        self._total_phases = 3
        self._last_phase_message = ""

    def stop(self):
        """请求停止同步"""
        self._stop_requested = True
        # 停止当前运行的子线程
        if self._stock_thread and self._stock_thread.isRunning():
            self._stock_thread.stop()
        if self._etf_thread and self._etf_thread.isRunning():
            self._etf_thread.stop()
        if self._index_thread and self._index_thread.isRunning():
            self._index_thread.stop()

    def run(self):
        """执行全量同步流程"""
        try:
            self.log_signal.emit("🚀 开始全量数据同步...")
            self.log_signal.emit(f"📅 起始日期: {self.start_date}")

            # 阶段1: 股票数据
            if not self._stop_requested:
                success = self._sync_stocks()
                if not success and not self._stop_requested:
                    self.finished_signal.emit(False, f"股票数据同步失败: {self._last_phase_message}".rstrip())
                    return

            # 阶段2: ETF数据
            if not self._stop_requested:
                success = self._sync_etf()
                if not success and not self._stop_requested:
                    self.finished_signal.emit(False, f"ETF数据同步失败: {self._last_phase_message}".rstrip())
                    return

            # 阶段3: 指数数据
            if not self._stop_requested:
                success = self._sync_index()
                if not success and not self._stop_requested:
                    self.finished_signal.emit(False, f"指数数据同步失败: {self._last_phase_message}".rstrip())
                    return

            if self._stop_requested:
                self.finished_signal.emit(False, "同步已取消")
            else:
                self.finished_signal.emit(True, "全量数据同步完成")

        except Exception as e:
            self.log_signal.emit(f"❌ 同步异常: {e}")
            self.finished_signal.emit(False, f"同步异常: {e}")

    def _sync_stocks(self) -> bool:
        """阶段1: 同步股票数据"""
        self._current_phase = "股票"
        self.log_signal.emit("\n" + "="*50)
        self.log_signal.emit("📊 阶段1: 正在同步股票数据（全量前复权）...")
        self.progress_signal.emit("股票数据", 0, 100)

        # 使用事件循环等待线程完成
        from PyQt6.QtCore import QEventLoop
        loop = QEventLoop()
        result = {"success": False, "message": ""}

        self._stock_thread = DataUpdateThread(
            data_dir=self.data_dir,
            stocklist_path=self.stocklist_path,
            tushare_token="",
            full_update=True,
            start_date=self.start_date,
            data_source="xtquant",
            period="1d",
            max_workers=1,
        )

        def on_progress(current, total, msg):
            self.progress_signal.emit("股票数据", current, total)
            self.log_signal.emit(f"  [股票] {msg}")

        def on_result(update_result):
            if isinstance(update_result, DataUpdateResult):
                result["message"] = update_result.to_ui_message()

        def on_finished(success, msg):
            result["success"] = success
            if not result.get("message"):
                result["message"] = msg
            loop.quit()

        self._stock_thread.progress_updated.connect(on_progress)
        self._stock_thread.result_signal.connect(on_result)
        self._stock_thread.finished_signal.connect(on_finished)
        self._stock_thread.start()

        loop.exec()

        self._last_phase_message = str(result.get("message", "") or "")
        if result["success"]:
            self._phases_completed += 1
            self.log_signal.emit(f"✅ 股票数据同步完成: {result['message']}")
        else:
            self.log_signal.emit(f"❌ 股票数据同步失败: {result['message']}")

        return result["success"]

    def _sync_etf(self) -> bool:
        """阶段2: 同步ETF数据"""
        self._current_phase = "ETF"
        self.log_signal.emit("\n" + "="*50)
        self.log_signal.emit("📈 阶段2: 正在同步ETF数据（全量前复权）...")
        self.progress_signal.emit("ETF数据", 0, 100)

        from PyQt6.QtCore import QEventLoop
        from pathlib import Path

        loop = QEventLoop()
        result = {"success": False, "message": ""}

        etf_config_path = Path(self.data_dir).parent / "trading_app" / "config" / "etf_list.json"

        self._etf_thread = ETFUpdateThread(
            data_dir=self.data_dir,
            etf_config_path=str(etf_config_path),
            full_update=True,
            start_date=self.start_date,
            max_workers=1,
        )

        def on_progress(current, total, msg):
            self.progress_signal.emit("ETF数据", current, total)
            self.log_signal.emit(f"  [ETF] {msg}")

        def on_result(update_result):
            if isinstance(update_result, DataUpdateResult):
                result["message"] = update_result.to_ui_message()

        def on_finished(success, msg):
            result["success"] = success
            if not result.get("message"):
                result["message"] = msg
            loop.quit()

        self._etf_thread.progress_updated.connect(on_progress)
        self._etf_thread.result_signal.connect(on_result)
        self._etf_thread.finished_signal.connect(on_finished)
        self._etf_thread.start()

        loop.exec()

        self._last_phase_message = str(result.get("message", "") or "")
        if result["success"]:
            self._phases_completed += 1
            self.log_signal.emit(f"✅ ETF数据同步完成: {result['message']}")
        else:
            self.log_signal.emit(f"❌ ETF数据同步失败: {result['message']}")

        return result["success"]

    def _sync_index(self) -> bool:
        """阶段3: 同步指数数据"""
        self._current_phase = "指数"
        self.log_signal.emit("\n" + "="*50)
        self.log_signal.emit("📉 阶段3: 正在同步指数数据...")
        self.progress_signal.emit("指数数据", 0, 100)

        from PyQt6.QtCore import QEventLoop

        loop = QEventLoop()
        result = {"success": False, "message": ""}

        self._index_thread = IndexUpdateThread(
            data_dir=self.data_dir,
            index_config_path=None,
            full_update=True,
            start_date=self.start_date,
            max_workers=1,
        )

        def on_progress(current, total, msg):
            self.progress_signal.emit("指数数据", current, total)
            self.log_signal.emit(f"  [指数] {msg}")

        def on_result(update_result):
            if isinstance(update_result, DataUpdateResult):
                result["message"] = update_result.to_ui_message()

        def on_finished(success, msg):
            result["success"] = success
            if not result.get("message"):
                result["message"] = msg
            loop.quit()

        self._index_thread.progress_updated.connect(on_progress)
        self._index_thread.result_signal.connect(on_result)
        self._index_thread.finished_signal.connect(on_finished)
        self._index_thread.start()

        loop.exec()

        self._last_phase_message = str(result.get("message", "") or "")
        if result["success"]:
            self._phases_completed += 1
            self.log_signal.emit(f"✅ 指数数据同步完成: {result['message']}")
        else:
            self.log_signal.emit(f"❌ 指数数据同步失败: {result['message']}")

        return result["success"]


# 全局管理器实例
_scheduler_manager: Optional[ScheduledTaskManager] = None


def get_scheduler_manager(data_dir: str = None, stocklist_path: str = None) -> ScheduledTaskManager:
    """获取全局定时任务管理器"""
    global _scheduler_manager
    if _scheduler_manager is None:
        if data_dir is None or stocklist_path is None:
            raise ValueError("首次初始化需要提供 data_dir 和 stocklist_path")
        _scheduler_manager = ScheduledTaskManager(data_dir, stocklist_path)
    return _scheduler_manager
