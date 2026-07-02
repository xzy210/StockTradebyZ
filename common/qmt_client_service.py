"""
Lightweight miniQMT status helpers.

实盘启动默认 miniQMT 已由用户正常开启并登录。本模块只做非侵入式状态展示。
"""
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class QmtClientConfig:
    qmt_path: str = ""
    account: str = ""
    process_name: str = ""

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, object]]) -> "QmtClientConfig":
        source = dict(data or {})
        return cls(
            qmt_path=str(source.get("qmt_path", "") or "").strip(),
            account=str(source.get("account", "") or "").strip(),
            process_name=str(source.get("process_name", "") or "").strip(),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class QmtClientStatus:
    running: bool = False
    login_window_visible: bool = False
    main_window_visible: bool = False
    ready: bool = False
    process_ids: List[int] | None = None
    matched_titles: List[str] | None = None
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class QmtClientService:
    """Non-invasive miniQMT process status helper."""

    DEFAULT_PROCESS_NAMES = (
        "miniqmt.exe",
        "xtminiqmt.exe",
        "qmt.exe",
        "thinktrader.exe",
        "xtitclient.exe",
    )

    def __init__(self, config: Optional[Dict[str, object]] = None) -> None:
        self.config = QmtClientConfig.from_dict(config)

    def get_status(self) -> QmtClientStatus:
        process_ids = self._find_process_ids()
        running = bool(process_ids)
        return QmtClientStatus(
            running=running,
            login_window_visible=False,
            main_window_visible=running,
            ready=False,
            process_ids=process_ids,
            matched_titles=[],
            message="miniQMT 进程运行中" if running else "miniQMT 未检测到进程",
        )

    def _find_process_ids(self) -> List[int]:
        names = {name.lower() for name in self._candidate_process_names()}
        process_ids: List[int] = []
        try:
            import psutil

            for process in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    name = str(process.info.get("name") or "").lower()
                    exe_name = Path(str(process.info.get("exe") or "")).name.lower()
                    if name in names or exe_name in names:
                        process_ids.append(int(process.info["pid"]))
                except Exception:
                    continue
            return process_ids
        except Exception:
            pass

        try:
            output = subprocess.check_output(
                ["tasklist", "/fo", "csv", "/nh"],
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            return process_ids

        for raw_line in output.splitlines():
            if not raw_line.strip():
                continue
            parts = [part.strip('"') for part in raw_line.split('","')]
            if len(parts) < 2:
                continue
            image_name = parts[0].strip('"').lower()
            pid_text = parts[1].strip('"')
            if image_name in names:
                try:
                    process_ids.append(int(pid_text))
                except ValueError:
                    pass
        return process_ids

    def _candidate_process_names(self) -> Iterable[str]:
        if self.config.process_name:
            yield self.config.process_name
        for name in self.DEFAULT_PROCESS_NAMES:
            yield name
