# plot_stock.py (Plotly only + BBI/MACD/KDJ)
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Optional

import pandas as pd

# 仅使用 Plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

import plotly.io as pio
from pathlib import Path
import webbrowser

# 指标计算来自 Selector.py
from Selector import compute_kdj, compute_bbi, compute_macd


def load_data(code: str, data_dir: str) -> pd.DataFrame:
    csv_path = Path(data_dir) / f"{code}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到数据文件：{csv_path}，请先用 fetch_kline.py 抓取。")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"{csv_path} 为空。")

    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df = df.sort_values("Date").set_index("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def add_ma(df: pd.DataFrame, wins: List[int], col: str = "Close") -> pd.DataFrame:
    for w in wins or []:
        if w > 1 and len(df) >= w:
            df[f"MA{w}({col})"] = df[col].rolling(w).mean()
    return df


def _attach_indicators(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    输入：索引为日期，列包含 Open/High/Low/Close/Volume
    输出：新增 BBI、DIF、DEA、MACD、K、D、J 列
    """
    ind = df_ohlc[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns=str.lower
    )  # 转小写以适配 Selector 的函数

    # BBI
    bbi = compute_bbi(ind)

    # MACD（返回带 DIF/DEA/MACD 的 df）
    ind = compute_macd(ind)  # fast=12, slow=26, signal=9, 柱体*2

    # KDJ
    ind = compute_kdj(ind)   # n=9

    out = df_ohlc.copy()
    out["BBI"] = bbi.values
    for col in ["DIF", "DEA", "MACD", "K", "D", "J"]:
        out[col] = ind[col].values
    return out


def plot_with_plotly(
    df: pd.DataFrame,
    code: str,
    ma_close: List[int],
    ma_vol: Optional[int],
    up_color: str,
    down_color: str,
):
    # 颜色
    MA_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
    BBI_COLOR = "#ff7f0e"
    DIF_COLOR = "#e377c2"
    DEA_COLOR = "#1f77b4"
    K_COLOR, D_COLOR, J_COLOR = "#f1c40f", "#3498db", "#8e44ad"
    VOL_MA_COLOR = "#555"

    ma_tags = " / ".join(
        [f"<span style='color:{MA_COLORS[i%len(MA_COLORS)]}'>MA{w}</span>"
         for i, w in enumerate(ma_close or [])]
    ) or "MA"

    price_title = f"价格：K线 + {ma_tags} + <span style='color:{BBI_COLOR}'>BBI</span>"
    macd_title  = (
        f"MACD： <span style='color:{up_color}'>■</span>/"
        f"<span style='color:{down_color}'>■</span> + "
        f"<span style='color:{DIF_COLOR}'>— DIF</span> + "
        f"<span style='color:{DEA_COLOR}'>— DEA</span>"
    )
    kdj_title   = (
        f"KDJ： <span style='color:{K_COLOR}'>— K</span> + "
        f"<span style='color:{D_COLOR}'>— D</span> + "
        f"<span style='color:{J_COLOR}'>— J</span>"
    )

    # 关键：先构造成交量均线的彩色标签，再拼进标题
    vol_ma_tag = (
        f" + <span style='color:{VOL_MA_COLOR}'>— MA{ma_vol}(Volume)</span>"
        if (ma_vol and ma_vol > 1) else ""
    )
    vol_title = (
        f"成交量： <span style='color:{up_color}'>■</span>/"
        f"<span style='color:{down_color}'>■</span>{vol_ma_tag}"
    )

    # 均线
    df = add_ma(df.copy(), ma_close, "Close")
    # 指标
    df = _attach_indicators(df)

    # 四行子图：价格+BBI | MACD | KDJ | 成交量
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.50, 0.20, 0.17, 0.13],
        subplot_titles=[price_title, macd_title, kdj_title, vol_title],  # ← 用带颜色标题
    )

    # 价格主图：蜡烛 + MA + BBI
    candle = go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color=up_color,
        decreasing_line_color=down_color,
        increasing_fillcolor=up_color,
        decreasing_fillcolor=down_color,
        name="K线",
        showlegend=False,
    )
    fig.add_trace(candle, row=1, col=1)

    for i, w in enumerate(ma_close or []):
        name = f"MA{w}(Close)"
        color = MA_COLORS[i % len(MA_COLORS)]
        fig.add_trace(
            go.Scatter(x=df.index, y=df[name], mode="lines",
                       name=name, line=dict(width=1.2, color=color)),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["BBI"], mode="lines",
                   name="BBI", line=dict(width=1.3, color=BBI_COLOR)),
        row=1, col=1
    )

    # MACD 面板：柱 + DIF/DEA
    macd_colors = [up_color if v >= 0 else down_color for v in df["MACD"].fillna(0)]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD"], marker_color=macd_colors, name="MACD"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["DIF"], mode="lines",
                   name="DIF", line=dict(width=1.2, color=DIF_COLOR)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["DEA"], mode="lines",
                   name="DEA", line=dict(width=1.2, color=DEA_COLOR)),
        row=2, col=1
    )

    # KDJ 面板：K/D/J
    fig.add_trace(go.Scatter(x=df.index, y=df["K"], mode="lines",
               name="K", line=dict(width=1.1, color=K_COLOR)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["D"], mode="lines",
               name="D", line=dict(width=1.1, color=D_COLOR)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["J"], mode="lines",
               name="J", line=dict(width=1.1, color=J_COLOR)), row=3, col=1)

    # 成交量面板
    vol_colors = [up_color if c >= o else down_color for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume"),
        row=4, col=1
    )
    if ma_vol and ma_vol > 1:
        vma = df["Volume"].rolling(ma_vol).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=vma, mode="lines",
                       name=f"MA{ma_vol}(Volume)",
                       line=dict(width=1.0, color=VOL_MA_COLOR)),
            row=4, col=1
        )

    # 布局与交互
    # 1) 布局：顶部轴只保留区间按钮；rangeslider 绑在底部轴
    fig.update_layout(
        title=f"{code} 蜡烛图 + BBI / MACD / KDJ",
        dragmode="pan",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            ),
            type="date",
        ),
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=40, r=20, t=80, b=40),
    )
    fig.update_xaxes(rangeslider_visible=False)  # 先禁用所有
    fig.update_layout(xaxis4=dict(rangeslider=dict(visible=True, thickness=0.08), type="date"))

    # 2) 各栏纵轴：开启按可见区间自动重算
    fig.update_yaxes(title_text="价格",  autorange=True, row=1, col=1)
    fig.update_yaxes(title_text="MACD", autorange=True, row=2, col=1, zeroline=True)
    fig.update_yaxes(title_text="KDJ",  autorange=True, row=3, col=1)        # 不再限制到 0~100
    fig.update_yaxes(title_text="成交量", autorange=True, row=4, col=1)

    post_js = """
window.addEventListener('load', function(){
  var gd = document.querySelector('.plotly-graph-div');
  if(!gd) return;
  function resetY(){
    Plotly.relayout(gd, {
      'yaxis.autorange':  true,
      'yaxis2.autorange': true,
      'yaxis3.autorange': true,
      'yaxis4.autorange': true
    });
  }
  gd.on('plotly_relayout', function(e){
    if(!e) return;
    var keys = Object.keys(e || {});
    // 只要任何 x 轴范围变化，就触发四个 y 轴自适应
    if(keys.some(function(k){ return k.startsWith('xaxis'); })){
      resetY();
    }
  });
});
"""

    html = pio.to_html(
        fig,
        include_plotlyjs="inline",     # ← 本地离线可打开
        full_html=True,
        config=dict(scrollZoom=True),
        post_script=post_js,
    )
    out = Path("plot_stock_view.html")
    out.write_text(html, encoding="utf-8")
    webbrowser.open_new_tab(out.resolve().as_uri())


def main():
    parser = argparse.ArgumentParser(description="按股票代码绘制 蜡烛图 + BBI/MACD/KDJ/成交量（Plotly 交互）")
    parser.add_argument("code", nargs="?", help="股票代码（如 600519 或 000001）")
    parser.add_argument("--data-dir", default="./data", help="CSV 数据目录（与 fetch_kline.py 的 --out 对齐）")
    parser.add_argument("--start", default=None, help="起始日期，YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期，YYYYMMDD 或 YYYY-MM-DD")
    parser.add_argument("--ma-close", nargs="*", type=int, default=[5, 10, 20], help="收盘价均线，默认 5 10 20")
    parser.add_argument("--ma-vol", type=int, default=5, help="成交量均线窗口，默认 5（0 关闭）")
    parser.add_argument("--up-color", default="#ec0000", help="上涨颜色")
    parser.add_argument("--down-color", default="#00da3c", help="下跌颜色")
    args = parser.parse_args()

    code = args.code or input("请输入股票代码（如 600519）: ").strip()
    if not code:
        print("必须提供股票代码。")
        sys.exit(2)

    if not HAS_PLOTLY:
        print("未安装 plotly，请执行：pip install plotly")
        sys.exit(4)

    try:
        df = load_data(code, args.data_dir)
        if args.start:
            df = df[df.index >= pd.to_datetime(args.start, errors="coerce")]
        if args.end:
            df = df[df.index <= pd.to_datetime(args.end, errors="coerce")]
        if df.empty:
            print("筛选后的数据为空。请调整日期范围。")
            sys.exit(3)

        plot_with_plotly(df, code, args.ma_close, (args.ma_vol or None), args.up_color, args.down_color)

    except Exception as e:
        print(f"绘图失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()