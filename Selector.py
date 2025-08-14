from typing import Dict, List, Optional, Any

from scipy.signal import find_peaks
import numpy as np
import pandas as pd


# --------------------------- 通用指标 --------------------------- #

def compute_kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    if df.empty:
        return df.assign(K=np.nan, D=np.nan, J=np.nan)

    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_n = df["high"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100

    K = np.zeros_like(rsv, dtype=float)
    D = np.zeros_like(rsv, dtype=float)
    for i in range(len(df)):
        if i == 0:
            K[i] = D[i] = 50.0
        else:
            K[i] = 2 / 3 * K[i - 1] + 1 / 3 * rsv.iloc[i]
            D[i] = 2 / 3 * D[i - 1] + 1 / 3 * K[i]
    J = 3 * K - 2 * D
    return df.assign(K=K, D=D, J=J)


def compute_bbi(df: pd.DataFrame) -> pd.Series:
    ma3 = df["close"].rolling(3).mean()
    ma6 = df["close"].rolling(6).mean()
    ma12 = df["close"].rolling(12).mean()
    ma24 = df["close"].rolling(24).mean()
    return (ma3 + ma6 + ma12 + ma24) / 4


def compute_rsv(
    df: pd.DataFrame,
    n: int,
) -> pd.Series:
    """
    按公式：RSV(N) = 100 × (C - LLV(L,N)) ÷ (HHV(C,N) - LLV(L,N))
    - C 用收盘价最高值 (HHV of close)
    - L 用最低价最低值 (LLV of low)
    """
    low_n = df["low"].rolling(window=n, min_periods=1).min()
    high_close_n = df["close"].rolling(window=n, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_close_n - low_n + 1e-9) * 100.0
    return rsv


def compute_dif(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    """计算 MACD 指标中的 DIF (EMA fast - EMA slow)。"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = "close",
    factor: float = 2.0,      # MACD 柱是否乘2（A股常用口径=2.0）
    adjust: bool = False,     # 与 compute_dif 一致默认不调整
) -> pd.DataFrame:
    """
    计算 MACD 全量指标：
    - DIF = EMA(fast) - EMA(slow)
    - DEA = EMA(DIF, signal)
    - MACD = factor * (DIF - DEA)  # factor=2 与券商口径一致
    返回：在原 df 基础上新增 'DIF','DEA','MACD' 三列
    """
    price = pd.to_numeric(df[price_col], errors="coerce")
    ema_fast = price.ewm(span=fast, adjust=adjust).mean()
    ema_slow = price.ewm(span=slow, adjust=adjust).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=adjust).mean()
    macd = factor * (dif - dea)
    return df.assign(DIF=dif, DEA=dea, MACD=macd)

def bbi_deriv_uptrend(
    bbi: pd.Series,
    *,
    min_window: int,
    max_window: int | None = None,
    q_threshold: float = 0.0,
) -> bool:
    """
    判断 BBI 是否“整体上升”。

    令最新交易日为 T，在区间 [T-w+1, T]（w 自适应，w ≥ min_window 且 ≤ max_window）
    内，先将 BBI 归一化：BBI_norm(t) = BBI(t) / BBI(T-w+1)。

    再计算一阶差分 Δ(t) = BBI_norm(t) - BBI_norm(t-1)。  
    若 Δ(t) 的前 q_threshold 分位数 ≥ 0，则认为该窗口通过；只要存在
    **最长** 满足条件的窗口即可返回 True。q_threshold=0 时退化为
    “全程单调不降”（旧版行为）。

    Parameters
    ----------
    bbi : pd.Series
        BBI 序列（最新值在最后一位）。
    min_window : int
        检测窗口的最小长度。
    max_window : int | None
        检测窗口的最大长度；None 表示不设上限。
    q_threshold : float, default 0.0
        允许一阶差分为负的比例（0 ≤ q_threshold ≤ 1）。
    """
    if not 0.0 <= q_threshold <= 1.0:
        raise ValueError("q_threshold 必须位于 [0, 1] 区间内")

    bbi = bbi.dropna()
    if len(bbi) < min_window:
        return False

    longest = min(len(bbi), max_window or len(bbi))

    # 自最长窗口向下搜索，找到任一满足条件的区间即通过
    for w in range(longest, min_window - 1, -1):
        seg = bbi.iloc[-w:]                # 区间 [T-w+1, T]
        norm = seg / seg.iloc[0]           # 归一化
        diffs = np.diff(norm.values)       # 一阶差分
        if np.quantile(diffs, q_threshold) >= 0:
            return True
    return False


def _find_peaks(
    df: pd.DataFrame,
    *,
    column: str = "high",
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
    height: Optional[float] = None,
    width: Optional[float] = None,
    rel_height: float = 0.5,
    **kwargs: Any,
) -> pd.DataFrame:
    
    if column not in df.columns:
        raise KeyError(f"'{column}' not found in DataFrame columns: {list(df.columns)}")

    y = df[column].to_numpy()

    indices, props = find_peaks(
        y,
        distance=distance,
        prominence=prominence,
        height=height,
        width=width,
        rel_height=rel_height,
        **kwargs,
    )

    peaks_df = df.iloc[indices].copy()
    peaks_df["is_peak"] = True

    # Flatten SciPy arrays into columns (only those with same length as indices)
    for key, arr in props.items():
        if isinstance(arr, (list, np.ndarray)) and len(arr) == len(indices):
            peaks_df[f"peak_{key}"] = arr

    return peaks_df


# --------------------------- Selector 类 --------------------------- #
class BBIKDJSelector:
    """
    自适应 *BBI(导数)* + *KDJ* 选股器
        • BBI: 允许 bbi_q_threshold 比例的回撤
        • KDJ: J < threshold ；或位于历史 J 的 j_q_threshold 分位及以下
        • MACD: DIF > 0
        • 收盘价波动幅度 ≤ price_range_pct
    """

    def __init__(
        self,
        j_threshold: float = -5,
        bbi_min_window: int = 90,
        max_window: int = 90,
        price_range_pct: float = 100.0,
        bbi_q_threshold: float = 0.05,
        j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.bbi_q_threshold = bbi_q_threshold  # ← 原 q_threshold
        self.j_q_threshold = j_q_threshold      # ← 新增

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)

        # 0. 收盘价波动幅度约束（最近 max_window 根 K 线）
        win = hist.tail(self.max_window)
        high, low = win["close"].max(), win["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:           
            return False

        # 1. BBI 上升（允许部分回撤）
        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        ):            
            return False

        # 2. KDJ 过滤 —— 双重条件
        kdj = compute_kdj(hist)
        j_today = float(kdj.iloc[-1]["J"])

        # 最近 max_window 根 K 线的 J 分位
        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        if not (j_today < self.j_threshold or j_today <= j_quantile):
            
            return False

        # 3. MACD：DIF > 0
        hist["DIF"] = compute_dif(hist)
        return hist["DIF"].iloc[-1] > 0

    # ---------- 多股票批量 ---------- #
    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 额外预留 20 根 K 线缓冲
            hist = hist.tail(self.max_window + 20)
            if self._passes_filters(hist):
                picks.append(code)
        return picks
    
    def explain_selection(self, code: str, date: pd.Timestamp, df: pd.DataFrame) -> str:
        """返回该股票被选中的详细原因"""
        hist = df[df["date"] <= date].tail(self.max_window + 20)
        if not self._passes_filters(hist):
            return "未通过筛选（不应出现）"
        
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)
        reasons = []
        
        # 1. 收盘价波动幅度
        win = hist.tail(self.max_window)
        high, low = win["close"].max(), win["close"].min()
        price_range = (high / low - 1) * 100
        reasons.append(f"价格波动率: {price_range:.2f}% (≤{self.price_range_pct*100:.1f}%)")
        
        # 2. BBI 上升趋势
        bbi_pass = bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        )
        reasons.append(f"BBI上升趋势: {'通过' if bbi_pass else '未通过'} (允许{self.bbi_q_threshold*100:.1f}%回撤)")
        
        # 3. KDJ 分析
        kdj = compute_kdj(hist)
        j_today = float(kdj.iloc[-1]["J"])
        j_window = kdj["J"].tail(self.max_window).dropna()
        j_quantile = float(j_window.quantile(self.j_q_threshold))
        
        condition1 = j_today < self.j_threshold
        condition2 = j_today <= j_quantile
        kdj_reason = f"J值: {j_today:.2f}"
        if condition1:
            kdj_reason += f" < {self.j_threshold} ✓"
        if condition2:
            kdj_reason += f" ≤ {j_quantile:.2f}({self.j_q_threshold*100:.0f}%分位) ✓"
        reasons.append(kdj_reason)
        
        # 4. MACD DIF
        hist["DIF"] = compute_dif(hist)
        dif_today = hist["DIF"].iloc[-1]
        reasons.append(f"DIF: {dif_today:.4f} {'> 0 ✓' if dif_today > 0 else '≤ 0 ✗'}")
        
        return "; ".join(reasons)


class SuperB1Selector:
    """SuperB1 选股器

    过滤逻辑概览
    ----------------
    1. **历史匹配 (t_m)** — 在 *lookback_n* 个交易日窗口内，至少存在一日
       满足 :class:`BBIKDJSelector`。

    2. **盘整区间** — 区间 ``[t_m, date-1]`` 收盘价波动率不超过 ``close_vol_pct``。

    3. **当日下跌** — ``(close_{date-1} - close_date) / close_{date-1}``
       ≥ ``price_drop_pct``。

    4. **J 值极低** — ``J < j_threshold`` *或* 位于历史 ``j_q_threshold`` 分位。
    """

    # ---------------------------------------------------------------------
    # 构造函数
    # ---------------------------------------------------------------------
    def __init__(
        self,
        *,
        lookback_n: int = 60,
        close_vol_pct: float = 0.05,
        price_drop_pct: float = 0.03,
        j_threshold: float = -5,
        j_q_threshold: float = 0.10,
        # ↓↓↓ 新增：嵌套 BBIKDJSelector 配置
        B1_params: Optional[Dict[str, Any]] = None        
    ) -> None:        
        # ---------- 参数合法性检查 ----------
        if lookback_n < 2:
            raise ValueError("lookback_n 应 ≥ 2")
        if not (0 < close_vol_pct < 1):
            raise ValueError("close_vol_pct 应位于 (0, 1) 区间")
        if not (0 < price_drop_pct < 1):
            raise ValueError("price_drop_pct 应位于 (0, 1) 区间")
        if not (0 <= j_q_threshold <= 1):
            raise ValueError("j_q_threshold 应位于 [0, 1] 区间")
        if B1_params is None:
            raise ValueError("bbi_params没有给出")

        # ---------- 基本参数 ----------
        self.lookback_n = lookback_n
        self.close_vol_pct = close_vol_pct
        self.price_drop_pct = price_drop_pct
        self.j_threshold = j_threshold
        self.j_q_threshold = j_q_threshold

        # ---------- 内部 BBIKDJSelector ----------
        self.bbi_selector = BBIKDJSelector(**(B1_params or {}))

        # 为保证给 BBIKDJSelector 提供足够历史，预留额外缓冲
        self._extra_for_bbi = self.bbi_selector.max_window + 20

    # 单支股票过滤核心
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        """*hist* 必须按日期升序，且最后一行为目标 *date*。"""
        if len(hist) < 2:
            return False

        # ---------- Step-0: 数据量判断 ----------
        if len(hist) < self.lookback_n + self._extra_for_bbi:
            return False

        # ---------- Step-1: 搜索满足 BBIKDJ 的 t_m ----------
        lb_hist = hist.tail(self.lookback_n + 1)  # +1 以排除自身
        tm_idx: int | None = None
        # 遍历回溯窗口
        for idx in lb_hist.index[:-1]:            
            if self.bbi_selector._passes_filters(hist.loc[:idx]):
                tm_idx = idx
                stable_seg = hist.loc[tm_idx : hist.index[-2], "close"]
                if len(stable_seg) < 3:
                    tm_idx = None
                    break
                high, low = stable_seg.max(), stable_seg.min()
                if low <= 0 or (high / low - 1) > self.close_vol_pct:                                      
                    tm_idx = None
                    continue
                else:
                    break
        if tm_idx is None:            
            return False        
        

        # ---------- Step-3: 当日相对前一日跌幅 ----------
        close_today, close_prev = hist["close"].iloc[-1], hist["close"].iloc[-2]
        if close_prev <= 0 or (close_prev - close_today) / close_prev < self.price_drop_pct:            
            return False

        # ---------- Step-4: J 值极低 ----------
        kdj = compute_kdj(hist)
        j_today = float(kdj["J"].iloc[-1])
        j_window = kdj["J"].iloc[-self.lookback_n:].dropna()
        j_q_val = float(j_window.quantile(self.j_q_threshold)) if not j_window.empty else np.nan
        if not (j_today < self.j_threshold or j_today <= j_q_val):            
            return False

        return True

    # 批量选股接口
    def select(self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]) -> List[str]:        
        picks: List[str] = []
        min_len = self.lookback_n + self._extra_for_bbi

        for code, df in data.items():
            hist = df[df["date"] <= date].tail(min_len)
            if len(hist) < min_len:
                continue
            if self._passes_filters(hist):
                picks.append(code)

        return picks

    def explain_selection(self, code: str, date: pd.Timestamp, df: pd.DataFrame) -> str:
        """返回该股票被选中的详细原因"""
        min_len = self.lookback_n + self._extra_for_bbi
        hist = df[df["date"] <= date].tail(min_len)
        
        if not self._passes_filters(hist):
            return "未通过筛选（不应出现）"
        
        reasons = []
        
        # Step-1: 搜索满足 BBIKDJ 的 t_m
        lb_hist = hist.tail(self.lookback_n + 1)
        tm_date = None
        tm_idx = None
        
        for idx in lb_hist.index[:-1]:
            if self.bbi_selector._passes_filters(hist.loc[:idx]):
                tm_idx = idx
                tm_date = hist.loc[idx, "date"].strftime("%Y-%m-%d")
                # 验证盘整条件
                stable_seg = hist.loc[tm_idx:hist.index[-2], "close"]
                if len(stable_seg) >= 3:
                    high, low = stable_seg.max(), stable_seg.min()
                    if low > 0 and (high / low - 1) <= self.close_vol_pct:
                        break
                else:
                    tm_idx = None
                    tm_date = None
        
        if tm_date:
            reasons.append(f"B1形态日期: {tm_date}")
            
            # 盘整区间分析
            stable_seg = hist.loc[tm_idx:hist.index[-2], "close"]
            vol_pct = (stable_seg.max() / stable_seg.min() - 1) * 100
            reasons.append(f"盘整波动率: {vol_pct:.2f}% (≤{self.close_vol_pct*100:.1f}%)")
        else:
            reasons.append("B1形态: 未找到")
        
        # Step-2: 当日跌幅
        close_today = hist["close"].iloc[-1]
        close_prev = hist["close"].iloc[-2]
        drop_pct = (close_prev - close_today) / close_prev * 100
        reasons.append(f"当日跌幅: {drop_pct:.2f}% (≥{self.price_drop_pct*100:.1f}%)")
        
        # Step-3: J 值分析
        kdj = compute_kdj(hist)
        j_today = float(kdj["J"].iloc[-1])
        j_window = kdj["J"].iloc[-self.lookback_n:].dropna()
        j_q_val = float(j_window.quantile(self.j_q_threshold)) if not j_window.empty else float('nan')
        
        condition1 = j_today < self.j_threshold
        condition2 = j_today <= j_q_val if not pd.isna(j_q_val) else False
        
        j_reason = f"J值: {j_today:.2f}"
        if condition1:
            j_reason += f" < {self.j_threshold} ✓"
        if condition2:
            j_reason += f" ≤ {j_q_val:.2f}({self.j_q_threshold*100:.0f}%分位) ✓"
        reasons.append(j_reason)
        
        return "; ".join(reasons)


class PeakKDJSelector:
    """
    Peaks + KDJ 选股器    
    """

    def __init__(
        self,
        j_threshold: float = -5,
        max_window: int = 90,
        fluc_threshold: float = 0.03,
        gap_threshold: float = 0.02,
        j_q_threshold: float = 0.10,
    ) -> None:
        self.j_threshold = j_threshold
        self.max_window = max_window
        self.fluc_threshold = fluc_threshold  # 当日↔peak_(t-n) 波动率上限
        self.gap_threshold = gap_threshold    # oc_prev 必须高于区间最低收盘价的比例
        self.j_q_threshold = j_q_threshold

    # ---------- 单支股票过滤 ---------- #
        # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if hist.empty:
            return False

        hist = hist.copy().sort_values("date")
        hist["oc_max"] = hist[["open", "close"]].max(axis=1)

        # 1. 提取 peaks
        peaks_df = _find_peaks(
            hist,
            column="oc_max",
            distance=6,
            prominence=0.5,
        )
        
        # 至少两个峰      
        date_today = hist.iloc[-1]["date"]
        peaks_df = peaks_df[peaks_df["date"] < date_today]
        if len(peaks_df) < 2:               
            return False

        peak_t = peaks_df.iloc[-1]          # 最新一个峰
        peaks_list = peaks_df.reset_index(drop=True)
        oc_t = peak_t.oc_max
        total_peaks = len(peaks_list)

        # 2. 回溯寻找 peak_(t-n)
        target_peak = None        
        for idx in range(total_peaks - 2, -1, -1):
            peak_prev = peaks_list.loc[idx]
            oc_prev = peak_prev.oc_max
            if oc_t <= oc_prev:             # 要求 peak_t > peak_(t-n)
                continue

            # 只有当“总峰数 ≥ 3”时才检查区间内其他峰 oc_max
            if total_peaks >= 3 and idx < total_peaks - 2:
                inter_oc = peaks_list.loc[idx + 1 : total_peaks - 2, "oc_max"]
                if not (inter_oc < oc_prev).all():
                    continue

            # 新增： oc_prev 高于区间最低收盘价 gap_threshold
            date_prev = peak_prev.date
            mask = (hist["date"] > date_prev) & (hist["date"] < peak_t.date)
            min_close = hist.loc[mask, "close"].min()
            if pd.isna(min_close):
                continue                    # 区间无数据
            if oc_prev <= min_close * (1 + self.gap_threshold):
                continue

            target_peak = peak_prev
            
            break

        if target_peak is None:
            return False

        # 3. 当日收盘价波动率
        close_today = hist.iloc[-1]["close"]
        fluc_pct = abs(close_today - target_peak.close) / target_peak.close
        if fluc_pct > self.fluc_threshold:
            return False

        # 4. KDJ 过滤
        kdj = compute_kdj(hist)
        j_today = float(kdj.iloc[-1]["J"])
        j_window = kdj["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))
        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False

        return True

    # ---------- 多股票批量 ---------- #
    def select(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            hist = hist.tail(self.max_window + 20)  # 额外缓冲
            if self._passes_filters(hist):
                picks.append(code)
        return picks
    
    def explain_selection(self, code: str, date: pd.Timestamp, df: pd.DataFrame) -> str:
        """返回该股票被选中的详细原因"""
        hist = df[df["date"] <= date].tail(self.max_window + 20)
        
        if not self._passes_filters(hist):
            return "未通过筛选（不应出现）"
        
        hist = hist.copy().sort_values("date")
        hist["oc_max"] = hist[["open", "close"]].max(axis=1)
        reasons = []
        
        # 1. 峰值分析
        peaks_df = _find_peaks(hist, column="oc_max", distance=6, prominence=0.5)
        date_today = hist.iloc[-1]["date"]
        peaks_df = peaks_df[peaks_df["date"] < date_today]
        reasons.append(f"识别峰值数: {len(peaks_df)}个")
        
        if len(peaks_df) >= 2:
            peak_t = peaks_df.iloc[-1]
            peaks_list = peaks_df.reset_index(drop=True)
            oc_t = peak_t.oc_max
            total_peaks = len(peaks_list)
            
            # 2. 寻找目标峰值
            target_peak = None
            for idx in range(total_peaks - 2, -1, -1):
                peak_prev = peaks_list.loc[idx]
                oc_prev = peak_prev.oc_max
                if oc_t <= oc_prev:
                    continue
                
                # 检查区间内其他峰的条件
                if total_peaks >= 3 and idx < total_peaks - 2:
                    inter_oc = peaks_list.loc[idx + 1 : total_peaks - 2, "oc_max"]
                    if not (inter_oc < oc_prev).all():
                        continue
                
                # 检查gap条件
                date_prev = peak_prev.date
                mask = (hist["date"] > date_prev) & (hist["date"] < peak_t.date)
                min_close = hist.loc[mask, "close"].min()
                if pd.isna(min_close) or oc_prev <= min_close * (1 + self.gap_threshold):
                    continue
                
                target_peak = peak_prev
                break
            
            if target_peak is not None:
                target_date = target_peak.date.strftime("%Y-%m-%d")
                reasons.append(f"目标峰值日期: {target_date} (oc_max: {target_peak.oc_max:.2f})")
                
                # 3. 价格波动分析
                close_today = hist.iloc[-1]["close"]
                fluc_pct = abs(close_today - target_peak.close) / target_peak.close * 100
                reasons.append(f"价格波动率: {fluc_pct:.2f}% (≤{self.fluc_threshold*100:.1f}%)")
            else:
                reasons.append("目标峰值: 未找到符合条件的峰值")
        
        # 4. KDJ 分析
        kdj = compute_kdj(hist)
        j_today = float(kdj.iloc[-1]["J"])
        j_window = kdj["J"].tail(self.max_window).dropna()
        j_quantile = float(j_window.quantile(self.j_q_threshold))
        
        condition1 = j_today < self.j_threshold
        condition2 = j_today <= j_quantile
        kdj_reason = f"J值: {j_today:.2f}"
        if condition1:
            kdj_reason += f" < {self.j_threshold} ✓"
        if condition2:
            kdj_reason += f" ≤ {j_quantile:.2f}({self.j_q_threshold*100:.0f}%分位) ✓"
        reasons.append(kdj_reason)
        
        return "; ".join(reasons)


class BBIShortLongSelector:
    """
    BBI 上升 + 短/长期 RSV 条件 + DIF > 0 选股器
    """
    def __init__(
        self,
        n_short: int = 3,
        n_long: int = 21,
        m: int = 3,
        bbi_min_window: int = 90,
        max_window: int = 150,
        bbi_q_threshold: float = 0.05,
    ) -> None:
        if m < 2:
            raise ValueError("m 必须 ≥ 2")
        self.n_short = n_short
        self.n_long = n_long
        self.m = m
        self.bbi_min_window = bbi_min_window
        self.max_window = max_window
        self.bbi_q_threshold = bbi_q_threshold   # 新增参数

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)

        # 1. BBI 上升（允许部分回撤）
        if not bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        ):
            return False

        # 2. 计算短/长期 RSV -----------------
        hist["RSV_short"] = compute_rsv(hist, self.n_short)
        hist["RSV_long"] = compute_rsv(hist, self.n_long)

        if len(hist) < self.m:
            return False                        # 数据不足

        win = hist.iloc[-self.m :]              # 最近 m 天
        long_ok = (win["RSV_long"] >= 80).all() # 长期 RSV 全 ≥ 80

        short_series = win["RSV_short"]
        short_start_end_ok = (
            short_series.iloc[0] >= 80 and short_series.iloc[-1] >= 80
        )
        short_has_below_20 = (short_series < 20).any()

        if not (long_ok and short_start_end_ok and short_has_below_20):
            return False

        # 3. MACD：DIF > 0 -------------------
        hist["DIF"] = compute_dif(hist)
        return hist["DIF"].iloc[-1] > 0

    # ---------- 多股票批量 ---------- #
    def select(
        self,
        date: pd.Timestamp,
        data: Dict[str, pd.DataFrame],
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            # 预留足够长度：RSV 计算窗口 + BBI 检测窗口 + m
            need_len = (
                max(self.n_short, self.n_long)
                + self.bbi_min_window
                + self.m
            )
            hist = hist.tail(max(need_len, self.max_window))
            if self._passes_filters(hist):
                picks.append(code)
        return picks
    
    def explain_selection(self, code: str, date: pd.Timestamp, df: pd.DataFrame) -> str:
        """返回该股票被选中的详细原因"""
        hist = df[df["date"] <= date]
        if hist.empty:
            return "无历史数据"
        
        # 预留足够长度
        need_len = (
            max(self.n_short, self.n_long)
            + self.bbi_min_window
            + self.m
        )
        hist = hist.tail(max(need_len, self.max_window))
        
        if not self._passes_filters(hist):
            return "未通过筛选（不应出现）"
        
        hist = hist.copy()
        hist["BBI"] = compute_bbi(hist)
        hist["RSV_short"] = compute_rsv(hist, self.n_short)
        hist["RSV_long"] = compute_rsv(hist, self.n_long)
        hist["DIF"] = compute_dif(hist)
        
        reasons = []
        
        # 1. BBI 上升趋势分析
        bbi_pass = bbi_deriv_uptrend(
            hist["BBI"],
            min_window=self.bbi_min_window,
            max_window=self.max_window,
            q_threshold=self.bbi_q_threshold,
        )
        reasons.append(f"BBI上升趋势: {'通过' if bbi_pass else '未通过'} (允许{self.bbi_q_threshold*100:.1f}%回撤)")
        
        # 2. 长期RSV分析
        win = hist.iloc[-self.m:]
        long_rsv_values = win["RSV_long"].values
        long_ok = (long_rsv_values >= 80).all()
        long_min = long_rsv_values.min()
        long_max = long_rsv_values.max()
        reasons.append(f"长期RSV({self.n_long}日): 全部≥80 {'✓' if long_ok else '✗'} (范围: {long_min:.1f}-{long_max:.1f})")
        
        # 3. 短期RSV"补票"模式分析
        short_rsv_values = win["RSV_short"].values
        short_start_end_ok = (short_rsv_values[0] >= 80 and short_rsv_values[-1] >= 80)
        short_has_below_20 = (short_rsv_values < 20).any()
        
        short_reason = f"短期RSV({self.n_short}日)补票模式: "
        if short_start_end_ok:
            short_reason += f"首尾≥80 ✓ ({short_rsv_values[0]:.1f}→{short_rsv_values[-1]:.1f})"
        else:
            short_reason += f"首尾≥80 ✗ ({short_rsv_values[0]:.1f}→{short_rsv_values[-1]:.1f})"
        
        if short_has_below_20:
            below_20_indices = [i for i, v in enumerate(short_rsv_values) if v < 20]
            short_reason += f"; 中间有<20调整 ✓ (第{below_20_indices}天)"
        else:
            short_reason += f"; 中间有<20调整 ✗ (最低{short_rsv_values.min():.1f})"
        
        reasons.append(short_reason)
        
        # 4. MACD DIF分析
        dif_today = hist["DIF"].iloc[-1]
        reasons.append(f"DIF: {dif_today:.4f} {'> 0 ✓' if dif_today > 0 else '≤ 0 ✗'}")
        
        return "; ".join(reasons)

class BreakoutVolumeKDJSelector:
    """
    放量突破 + KDJ + DIF>0 + 收盘价波动幅度 选股器   
    """

    def __init__(
        self,
        j_threshold: float = 0.0,
        up_threshold: float = 3.0,
        volume_threshold: float = 2.0 / 3,
        offset: int = 15,
        max_window: int = 120,
        price_range_pct: float = 10.0,
        j_q_threshold: float = 0.10,        # ← 新增
    ) -> None:
        self.j_threshold = j_threshold
        self.up_threshold = up_threshold
        self.volume_threshold = volume_threshold
        self.offset = offset
        self.max_window = max_window
        self.price_range_pct = price_range_pct
        self.j_q_threshold = j_q_threshold  # ← 新增

    # ---------- 单支股票过滤 ---------- #
    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        if len(hist) < self.offset + 2:
            return False

        hist = hist.tail(self.max_window).copy()

        # ---- 收盘价波动幅度约束 ----
        high, low = hist["close"].max(), hist["close"].min()
        if low <= 0 or (high / low - 1) > self.price_range_pct:
            return False

        # ---- 技术指标 ----
        hist = compute_kdj(hist)
        hist["pct_chg"] = hist["close"].pct_change() * 100
        hist["DIF"] = compute_dif(hist)

        # 0) 指定日约束：J < j_threshold 或位于历史分位；且 DIF > 0
        j_today = float(hist["J"].iloc[-1])

        j_window = hist["J"].tail(self.max_window).dropna()
        if j_window.empty:
            return False
        j_quantile = float(j_window.quantile(self.j_q_threshold))

        # 若不满足任一 J 条件，则淘汰
        if not (j_today < self.j_threshold or j_today <= j_quantile):
            return False
        if hist["DIF"].iloc[-1] <= 0:
            return False

        # ---- 放量突破条件 ----
        n = len(hist)
        wnd_start = max(0, n - self.offset - 1)
        last_idx = n - 1

        for t_idx in range(wnd_start, last_idx):  # 探索突破日 T
            row = hist.iloc[t_idx]

            # 1) 单日涨幅
            if row["pct_chg"] < self.up_threshold:
                continue

            # 2) 相对放量
            vol_T = row["volume"]
            if vol_T <= 0:
                continue
            vols_except_T = hist["volume"].drop(index=hist.index[t_idx])
            if not (vols_except_T <= self.volume_threshold * vol_T).all():
                continue

            # 3) 创新高
            if row["close"] <= hist["close"].iloc[:t_idx].max():
                continue

            # 4) T 之后 J 值维持高位
            if not (hist["J"].iloc[t_idx:last_idx] > hist["J"].iloc[-1] - 10).all():
                continue

            return True  # 满足所有条件

        return False

    # ---------- 多股票批量 ---------- #
    def select(
        self, date: pd.Timestamp, data: Dict[str, pd.DataFrame]
    ) -> List[str]:
        picks: List[str] = []
        for code, df in data.items():
            hist = df[df["date"] <= date]
            if hist.empty:
                continue
            if self._passes_filters(hist):
                picks.append(code)
        return picks

    def explain_selection(self, code: str, date: pd.Timestamp, df: pd.DataFrame) -> str:
        """返回该股票被选中的详细原因"""
        hist = df[df["date"] <= date]
        
        if not self._passes_filters(hist):
            return "未通过筛选（不应出现）"
        
        hist = hist.tail(self.max_window).copy()
        reasons = []
        
        # 1. 收盘价波动幅度
        high, low = hist["close"].max(), hist["close"].min()
        price_range = (high / low - 1) * 100
        reasons.append(f"价格波动率: {price_range:.2f}% (≤{self.price_range_pct*100:.1f}%)")
        
        # 2. 计算技术指标
        hist = compute_kdj(hist)
        hist["pct_chg"] = hist["close"].pct_change() * 100
        hist["DIF"] = compute_dif(hist)
        
        # 3. KDJ 分析
        j_today = float(hist["J"].iloc[-1])
        j_window = hist["J"].tail(self.max_window).dropna()
        j_quantile = float(j_window.quantile(self.j_q_threshold))
        
        condition1 = j_today < self.j_threshold
        condition2 = j_today <= j_quantile
        kdj_reason = f"J值: {j_today:.2f}"
        if condition1:
            kdj_reason += f" < {self.j_threshold} ✓"
        if condition2:
            kdj_reason += f" ≤ {j_quantile:.2f}({self.j_q_threshold*100:.0f}%分位) ✓"
        reasons.append(kdj_reason)
        
        # 4. DIF 分析
        dif_today = hist["DIF"].iloc[-1]
        reasons.append(f"DIF: {dif_today:.4f} {'> 0 ✓' if dif_today > 0 else '≤ 0 ✗'}")
        
        # 5. 放量突破分析
        n = len(hist)
        wnd_start = max(0, n - self.offset - 1)
        last_idx = n - 1
        
        breakthrough_dates = []
        for t_idx in range(wnd_start, last_idx):
            row = hist.iloc[t_idx]
            
            # 检查所有突破条件
            if (row["pct_chg"] >= self.up_threshold and 
                row["volume"] > 0):
                
                vols_except_T = hist["volume"].drop(index=hist.index[t_idx])
                vol_condition = (vols_except_T <= self.volume_threshold * row["volume"]).all()
                
                price_condition = row["close"] > hist["close"].iloc[:t_idx].max()
                
                j_condition = (hist["J"].iloc[t_idx:last_idx] > hist["J"].iloc[-1] - 10).all()
                
                if vol_condition and price_condition and j_condition:
                    breakthrough_date = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], 'strftime') else str(row["date"])[:10]
                    breakthrough_dates.append(f"{breakthrough_date}(涨幅{row['pct_chg']:.1f}%)")
        
        if breakthrough_dates:
            reasons.append(f"突破日期: {', '.join(breakthrough_dates)}")
        else:
            reasons.append("突破条件: 未找到符合条件的突破日")
        
        return "; ".join(reasons)
