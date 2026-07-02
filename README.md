# Xquant

`Xquant` 是一个以 **Python + PyQt6** 为核心的本地量化交易与研究软件，包含 **行情终端、策略研究、ETF 轮动、实盘策略中心、数据脚本、强化学习实验** 等模块。

本仓库以 **桌面端单体应用** 为主要形态，不包含 Web 前后端拆分架构：

- 没有独立的 HTTP 服务或前端工程。
- 主要通过 `PyQt6` 界面、`QTimer/QThread` 后台任务、共享服务对象和本地数据文件协同工作。
- 实时行情与实盘能力依赖 `xtquant / miniQMT`。
- 历史数据拉取与部分研究能力可基于 `Tushare / AkShare / 本地数据文件` 运行。

## 应用入口

仓库包含 3 个职责清晰的入口：

1. `run_app.py`
策略研究台（自动化研究 + 回测）：ETF 轮动研究、ETF 网格回测、截面回测、因子库、AI 训练和实验记录。

2. `run_live_strategy_center.py`
实盘策略中枢（自动化实盘）：AI 策略、ETF 轮动、账户、风控、任务、告警、收益、运行日志、券商连接自检及日终流程。

3. `main.py`（+ `run.bat`）
人工看盘 + 人工下单终端：K 线、分时、自选股、模拟交易、热门板块、筹码分布等。与自动化策略并存，**保持独立**，不合入研究台。

AI 实盘决策已统一从 `run_live_strategy_center.py` 的 AI Tab 进入，旧独立入口 `launch_ai_trade.py` 已下线。

## 软件架构

### 1. 桌面 UI 层

- `app/main.py`
策略研究台主窗口，以 `app/perspectives/` 直接加载研究模块。
- `trading_app/widgets/live_strategy_hub_widget.py`
实盘策略中心容器，整合 AI 策略、ETF 轮动与运行日志页面。
- `trading_app/main_window.py`
人工看盘 + 人工下单终端窗口，承载 K 线、分时、自选股、模拟交易、板块和筹码等能力。保持独立，不计划合入研究台。

### 2. 业务服务层

- `trading_app/controllers/`
负责连接主窗口、实时行情与交易相关服务。
- `trading_app/services/`
封装行情、风控、条件单、交易执行、数据新鲜度、AI 决策、资讯、组合风险、策略注册等能力。
- `live_rotation/rotation_engine.py`
将 ETF 轮动的信号计算、风控检查、交易执行、状态管理与通知整合为完整执行引擎。

### 3. 共享基础设施层

- `common/`
提供跨应用共用能力，例如券商会话、miniQMT 状态展示、通用策略面板等。
- `common/qmt_client_service.py`
负责 miniQMT 进程状态展示；实际交易连接由券商会话服务直接通过 `xtquant.connect()` 完成。

### 4. 数据与研究层

- `strategy_app/strategies/`
策略实现与研究侧抽象定义。
- `strategy_app/factors/`
因子定义与计算。
- `strategy_app/backtest/`
回测上下文与模型。
- `scripts/`
用于日线、分钟线、xtquant 数据抓取及调试辅助。
- `rl_trading/`
强化学习环境、训练与预测脚本。

## 目录说明

```text
Xquant/
├── run_app.py                      # 策略研究台入口（自动化研究 + 回测）
├── run_live_strategy_center.py     # 实盘策略中心入口（自动化实盘）
├── main.py                         # 人工看盘 + 人工下单终端入口
├── run.bat                         # 人工看盘终端 Windows 启动脚本
├── app/                            # 研究台 shell 与 perspectives
├── trading_app/                    # 人工交易终端 + 实盘运营层
│   ├── controllers/                # UI 与服务编排
│   ├── services/                   # 行情/交易/风控/AI/定时任务
│   ├── widgets/                    # 各类桌面组件
│   └── config/                     # 主程序配置
├── strategy_app/                   # 策略研究平台
│   ├── widgets/                    # 选股/回测/因子/AI 训练等界面
│   ├── strategies/                 # 策略实现
│   ├── factors/                    # 因子库
│   └── backtest/                   # 回测模型与上下文
├── live_rotation/                  # ETF 轮动实盘模块
├── common/                         # 券商、凭据、共享组件
├── scripts/                        # 数据抓取与调试脚本
├── rl_trading/                     # 强化学习实验
├── stocklist/                      # 股票池与指数池 CSV
└── data/                           # 本地行情数据目录（运行后生成/更新）
```

## 运行结构

### 策略研究台

- `run_app.py` 启动 `app.main.XquantMainWindow`。
- 研究模块以标签页形式组织，主要包括 ETF 轮动研究、ETF 网格回测、截面回测、因子库和 AI 训练。

### ETF 轮动实盘

- `RotationEngine` 负责配置与状态加载、因子得分计算、风控检查、交易执行、对账通知以及定时触发。

### 实盘策略中心

- `run_live_strategy_center.py` 启动统一实盘界面。
- `LiveStrategyHubWidget` 包含 `AI策略`、`ETF轮动` 与 `运行日志` 三个主要页面。
- 启动后结合 `QmtStartupOrchestrator` 执行 QMT 自检，并接入统一日终流程。
- 策略启动资金在 `收益中心 -> 策略资金管理` 统一维护；交易费用规则在 `交易费用设置` 中统一维护，AI 与 ETF 共用同一套手续费口径。

### 人工看盘 + 人工下单终端

- `main.py` 创建 `QApplication` 并启动 `trading_app.main_window.MainWindow`。
- `MainWindow` 初始化 `TradingBridge`、`RealtimeController`、`SyncController`。
- 该入口承载人工看盘、人工下单、条件单、自动止损等场景，与自动化策略并存。
- 保持独立，不合入研究台；新研究 / 实盘功能请挂到 `app/perspectives/` 或 `LiveStrategyHubWidget`。

## 环境准备

推荐环境：

- Python `3.11` 或 `3.12`
- Windows 本机运行（实盘相关功能依赖 Windows + miniQMT）
- 已安装 `pip`

安装依赖：

```bash
pip install -r requirements.txt
```

说明：

- `xtquant` 不包含在 `pip install -r requirements.txt` 的自动安装范围内，需要按迅投官方方式单独安装。
- 若仅使用研究、回测或数据处理功能，可不安装 `xtquant`。
- 若使用 AI 相关功能，还需要准备对应的模型服务配置。

## 外部依赖

### 1. Tushare

部分数据脚本依赖 `TUSHARE_TOKEN`，例如 `scripts/fetch_kline.py`。

Windows PowerShell:

```powershell
setx TUSHARE_TOKEN "你的token"
```

macOS / Linux:

```bash
export TUSHARE_TOKEN=你的token
```

### 2. miniQMT / xtquant

实时行情、交易、条件单、部分实盘策略能力依赖：

- miniQMT 客户端
- `xtquant`
- 本机已登录的交易账户

未安装或未连接时，主程序中的实盘相关功能将受到限制。

## 快速启动

### 1. 启动策略研究台

```bash
python run_app.py
```

### 2. 启动实盘策略中心

```bash
python run_live_strategy_center.py
```

### 3. 启动人工看盘 + 人工下单终端

```bash
python main.py
```

Windows 环境下也可直接运行：

```bash
run.bat
```

## 数据目录与输入约定

- `data/`
默认本地行情数据目录，多个模块会读取其中的日线与分钟线数据。
- `stocklist/stocklist.csv`
默认股票池文件，研究模块与抓数脚本通常会使用该文件。
- `stocklist/*.csv`
仓库内还包含若干预置股票池，例如沪深 A 股、沪深 300、中证 500 以及指数相关清单。

## 常用脚本

### 抓取日线数据

```bash
python scripts/fetch_kline.py --stocklist ./stocklist/stocklist.csv --out ./data
```

脚本特性：

- 基于 `Tushare` 抓取日线
- 从 `stocklist` 读取股票池
- 排除创业板 / 科创板 / 北交所
- 多线程抓取
- 限流后的冷却重试

### 其他脚本

- `scripts/fetch_kline_xtquant.py`：使用 `xtquant` 抓取行情数据。
- `scripts/fetch_minute.py`：抓取分钟级数据。
- `scripts/debug_qmt_window.py`：用于调试 QMT 窗口交互。

## 可选实验模块

- `rl_trading/`
强化学习相关实验，包含环境、训练脚本、预测脚本。

这些模块不是主程序运行的前置条件，可作为研究扩展使用。

## 代码阅读入口

如需继续完善文档、拆分模块或接入新策略，建议优先从以下入口阅读代码：

- `main.py`
- `trading_app/main_window.py`
- `app/main.py`
- `app/perspectives/`
- `live_rotation/rotation_engine.py`
- `run_live_strategy_center.py`

## 免责声明

- 本仓库仅供学习、研究与工程实践使用，不构成任何投资建议。
- 实盘交易、自动化下单与外部数据接口均存在风险，请自行评估并谨慎使用。
- 数据源、券商接口与第三方模型服务可能随时间变化而失效或变更。

## 附录：
https://www.youtube.com/watch?v=kLJEXE5_7rI