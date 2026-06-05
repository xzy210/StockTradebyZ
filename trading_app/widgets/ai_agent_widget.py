import os
import json
import logging
import base64
import mimetypes
import time
import httpx
import re
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, 
    QPushButton, QLabel, QComboBox, QGroupBox, QSplitter,
    QMessageBox, QScrollArea, QFrame, QSizePolicy, QFormLayout,
    QDialog, QFileDialog, QToolButton, QMenu, QCheckBox,
    QListWidget, QListWidgetItem, QPlainTextEdit, QSpinBox,
    QTextBrowser
)

# Try to import markdown for rich text rendering
try:
    import markdown
    from markdown.extensions import fenced_code, tables, nl2br
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize, QUrl
from PyQt6.QtGui import QIcon, QFont, QColor, QTextCursor, QAction, QPixmap, QDesktopServices
from openai import OpenAI
from google import genai as google_genai
from google.genai import types as genai_types
import pandas as pd

# Import stock analyzer service
try:
    from trading_app.services.ai.stock_analyzer import get_analyzer, StockAnalyzer
    from trading_app.services.ai.agent_watchlist_scan_service import AgentWatchlistScanService
    from common.agent.agent_context_service import (
        AgentContextService,
        AgentRuntimeContext,
        TASK_MODE_GENERAL,
        TASK_MODE_LABELS,
        TASK_MODE_POSITION_DIAGNOSIS,
        TASK_MODE_SYMBOL_ANALYSIS,
        TASK_MODE_TRADE_DECISION,
        TASK_MODE_WATCHLIST_SCAN,
    )
    from trading_app.services.ai.agent_prompt_builder import AgentPromptBuilder
    from trading_app.services.ai.agent_runtime import StockAgentRuntime
    from common.agent.agent_evidence_service import TEMP_PASTED_PREFIX
    from trading_app.services.ai.agent_action_service import AgentActionService
    from trading_app.services.ai.trade_decision_extractor import TradeDecisionExtractor
    from trading_app.services.ai.trade_decision_models import (
        DecisionOutcome,
        TradeDecision,
        TRADE_ACTION_LABELS,
    )
    from trading_app.services.execution.risk_guard_service import RiskGuardService
    from trading_app.services.ai.decision_tracker_service import DecisionTrackerService
    from common.broker_session_service import get_broker_session_service
except ImportError:
    from trading_app.services.ai.stock_analyzer import get_analyzer, StockAnalyzer
    from trading_app.services.ai.agent_watchlist_scan_service import AgentWatchlistScanService
    from common.agent.agent_context_service import (
        AgentContextService,
        AgentRuntimeContext,
        TASK_MODE_GENERAL,
        TASK_MODE_LABELS,
        TASK_MODE_POSITION_DIAGNOSIS,
        TASK_MODE_SYMBOL_ANALYSIS,
        TASK_MODE_TRADE_DECISION,
        TASK_MODE_WATCHLIST_SCAN,
    )
    from trading_app.services.ai.agent_prompt_builder import AgentPromptBuilder
    from trading_app.services.ai.agent_runtime import StockAgentRuntime
    from common.agent.agent_evidence_service import TEMP_PASTED_PREFIX
    from trading_app.services.ai.agent_action_service import AgentActionService
    from trading_app.services.ai.trade_decision_extractor import TradeDecisionExtractor
    from trading_app.services.ai.trade_decision_models import (
        DecisionOutcome,
        TradeDecision,
        TRADE_ACTION_LABELS,
    )
    from trading_app.services.execution.risk_guard_service import RiskGuardService
    from trading_app.services.ai.decision_tracker_service import DecisionTrackerService
    from common.broker_session_service import get_broker_session_service

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatDisplayWidget(QTextBrowser):
    """统一的聊天显示组件 - 类似 ChatGPT/Claude 风格"""
    
    def __init__(self, parent=None, theme="light"):
        super().__init__(parent)
        self.theme = theme
        self.raw_contents = {}  # 存储每条消息的原内容 {msg_id: content}
        self.msg_counter = 0    # 消息计数器
        self.current_msg_id = None
        self.expanded_tool_cards = set()
        
        self.setup_ui()
        self.update_style()
    
    def setup_ui(self):
        """初始化UI设置"""
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setReadOnly(True)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | 
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.anchorClicked.connect(self.on_anchor_clicked)
        # 使用垂直滚动条
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 注意：HTML文档初始化在 update_style() 之后进行，因为需要 base_style
        
    def update_style(self):
        """更新样式"""
        if self.theme == "light":
            bg_color = "#ffffff"
            text_color = "#24292f"
            link_color = "#0969da"
            code_bg = "#f6f8fa"
            code_border = "#d0d7de"
            user_msg_bg = "#f7f7f8"
            assistant_msg_bg = "#ffffff"
            border_color = "#e0e0e0"
        else:
            bg_color = "#1e1e1e"
            text_color = "#c9d1d9"
            link_color = "#58a6ff"
            code_bg = "#161b22"
            code_border = "#30363d"
            user_msg_bg = "#2d2d2d"
            assistant_msg_bg = "#1e1e1e"
            border_color = "#3c3c3c"
        
        self.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {bg_color};
                border: none;
                padding: 5px;
            }}
            QScrollBar:vertical {{
                background-color: {bg_color};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {border_color};
                min-height: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {link_color};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)
        
        # 基础的 HTML 样式
        self.base_style = f"""
        <style>
            body {{
                font-family: 'Segoe UI', 'Microsoft YaHei', 'PingFang SC', 'Hiragino Sans GB', sans-serif;
                font-size: 14px;
                line-height: 1.6;
                color: {text_color};
                margin: 0;
                padding: 0;
                background-color: {bg_color};
            }}
            .message-container {{
                display: block;
                width: 100%;
                padding: 10px 0;
            }}
            .message-wrapper {{
                max-width: 100%;
                margin: 0;
                padding: 0 20px;
            }}
            .message-header {{
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 6px;
                display: flex;
                align-items: center;
            }}
            .user-header {{
                color: #10a37f;
            }}
            .assistant-header {{
                color: {link_color};
            }}
            .system-header {{
                color: #6a737d;
            }}
            .tool-trace-card {{
                border: 1px solid {code_border};
                border-radius: 10px;
                background-color: {code_bg};
                overflow: hidden;
            }}
            .tool-trace-toggle {{
                display: block;
                color: inherit;
                text-decoration: none;
                padding: 10px 14px;
                font-weight: 600;
            }}
            .tool-trace-toggle:hover {{
                text-decoration: none;
            }}
            .tool-trace-summary {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
            }}
            .tool-trace-arrow {{
                margin-right: 8px;
                color: {link_color};
            }}
            .tool-trace-title {{
                color: {text_color};
            }}
            .tool-trace-badge {{
                display: inline-block;
                padding: 2px 8px;
                border-radius: 999px;
                border: 1px solid {code_border};
                font-size: 12px;
                color: {link_color};
                background-color: {bg_color};
                white-space: nowrap;
            }}
            .tool-trace-body {{
                padding: 0 14px 12px 14px;
                border-top: 1px solid {code_border};
            }}
            .tool-trace-list {{
                margin: 10px 0 0 0;
                padding-left: 18px;
            }}
            .tool-trace-list li {{
                margin: 8px 0;
            }}
            .tool-trace-item-title {{
                font-weight: 600;
            }}
            .tool-trace-item-summary {{
                margin-top: 2px;
                color: {text_color};
                opacity: 0.85;
            }}
            .tool-trace-path {{
                margin-top: 10px;
                font-size: 12px;
                opacity: 0.8;
                word-break: break-all;
            }}
            .message-content {{
                color: {text_color};
                font-size: 14px;
            }}
            .message-content p {{
                margin: 0 0 10px 0;
            }}
            .message-content p:last-child {{
                margin-bottom: 0;
            }}
            pre {{
                background-color: {code_bg};
                padding: 12px;
                border-radius: 6px;
                overflow-x: auto;
                margin: 10px 0;
                border: 1px solid {code_border};
                font-family: 'SF Mono', Monaco, Consolas, 'Liberation Mono', monospace;
                font-size: 13px;
                line-height: 1.5;
            }}
            code {{
                background-color: {code_bg};
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'SF Mono', Monaco, Consolas, 'Liberation Mono', monospace;
                font-size: 90%;
                border: 1px solid {code_border};
            }}
            pre code {{
                background: none;
                padding: 0;
                border: none;
                font-size: inherit;
            }}
            a {{
                color: {link_color};
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            h1, h2, h3, h4, h5, h6 {{
                margin-top: 16px;
                margin-bottom: 10px;
                font-weight: 600;
                line-height: 1.3;
            }}
            h1 {{ font-size: 1.6em; border-bottom: 1px solid {code_border}; padding-bottom: 0.3em; margin-top: 0; }}
            h2 {{ font-size: 1.4em; border-bottom: 1px solid {code_border}; padding-bottom: 0.3em; }}
            h3 {{ font-size: 1.2em; }}
            h4 {{ font-size: 1.1em; }}
            ul, ol {{
                padding-left: 1.8em;
                margin: 0 0 10px 0;
            }}
            li {{
                margin: 0.3em 0;
            }}
            li + li {{
                margin-top: 0.3em;
            }}
            table {{
                border-collapse: collapse;
                width: auto;
                min-width: 400px;
                max-width: 100%;
                margin: 12px 0;
                font-size: 13px;
            }}
            th, td {{
                border: 1px solid {code_border};
                padding: 8px 12px;
                text-align: left;
            }}
            th {{
                background-color: {code_bg};
                font-weight: 600;
            }}
            tr:nth-child(2n) {{
                background-color: {code_bg};
            }}
            blockquote {{
                border-left: 4px solid {code_border};
                padding-left: 16px;
                margin: 0 0 10px 0;
                opacity: 0.8;
                font-style: italic;
            }}
            hr {{
                border: none;
                border-top: 1px solid {code_border};
                margin: 15px 0;
            }}
        </style>
        """
        
        # 初始化或刷新HTML文档结构（仅在没有消息时，避免主题切换时丢失内容）
        if not self.raw_contents:
            self._init_html_document()
    
    def _init_html_document(self):
        """初始化HTML文档结构，包含CSS样式"""
        self.setHtml(f"<!DOCTYPE html><html><head>{self.base_style}</head><body></body></html>")
    
    def _markdown_to_html(self, text):
        """将 Markdown 文本转换为 HTML"""
        if not MARKDOWN_AVAILABLE:
            return self._simple_format_to_html(text)
        
        try:
            # 暂时保存代码块，避免被 markdown 库或之前的转义逻辑破坏
            code_blocks = []
            def save_code_block(match):
                # 获取捕获组 1 的内容
                code = match.group(1)
                code_blocks.append(code)
                # 使用一个不会被 Markdown 误解析的占位符（避免使用 ___ 或 ***）
                return f"CODEBLOCKPLACEHOLDER{len(code_blocks)-1}"
            
            # 匹配 ```lang\n code ``` 格式，使用一个捕获组捕获代码内容
            processed_text = re.sub(r'```(?:\w+)?\n?(.*?)```', save_code_block, text, flags=re.DOTALL)
            
            # 转义 HTML 特殊字符（仅对非代码块部分）
            processed_text = processed_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            md = markdown.Markdown(extensions=[
                'fenced_code',
                'tables',
                'nl2br',
            ])
            html_content = md.convert(processed_text)
            
            # 恢复代码块，并对代码内容进行转义
            for i, code in enumerate(code_blocks):
                escaped_code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # 替换占位符为实际的代码 HTML
                placeholder = f"CODEBLOCKPLACEHOLDER{i}"
                html_content = html_content.replace(placeholder,
                    f'<pre><code>{escaped_code}</code></pre>')
            return html_content
        except Exception as e:
            logging.warning(f"Markdown conversion failed: {e}")
            return self._simple_format_to_html(text)
    
    def _simple_format_to_html(self, text):
        """简单的文本格式化"""
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # 代码块
        def replace_code_block(match):
            code = match.group(1)
            return f'<pre><code>{code}</code></pre>'
        text = re.sub(r'```(?:\w+)?\n?(.*?)```', replace_code_block, text, flags=re.DOTALL)
        
        # 行内代码
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # 粗体
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        # 斜体
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        # 换行
        text = text.replace('\n', '<br>')
        
        return text
    
    def get_header_html(self, role):
        """获取消息头部 HTML"""
        if role == "user":
            return '<div class="message-header user-header">你</div>'
        if role == "system":
            return '<div class="message-header system-header">工具</div>'
        else:
            return '<div class="message-header assistant-header">AI</div>'
    
    def add_message(self, role, content):
        """添加一条新消息"""
        self.msg_counter += 1
        msg_id = f"msg_{self.msg_counter}"
        self.current_msg_id = msg_id
        # 存储角色信息以便渲染时使用
        self.raw_contents[msg_id] = {"role": role, "content": content}
        
        # 重新渲染所有消息（确保CSS正确应用）
        self.render_all_messages()
        
        return msg_id
    
    def update_last_message(self, content):
        """更新最后一条消息的内容（用于流式输出）"""
        if self.current_msg_id and self.current_msg_id in self.raw_contents:
            # 保留角色信息，只更新内容
            self.raw_contents[self.current_msg_id]["content"] = content
            
            # 重新渲染整个文档
            self.render_all_messages()
    
    def render_all_messages(self):
        """重新渲染所有消息（主题切换或流式更新时使用）"""
        # 保存当前滚动位置
        scrollbar = self.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 50
        
        # 构建所有消息的 HTML
        messages_html = ""
        for msg_id, msg_data in self.raw_contents.items():
            # 兼容旧格式（纯字符串）和新格式（字典）
            if isinstance(msg_data, dict):
                role = msg_data.get("role", "user")
                content = msg_data.get("content", "")
            else:
                # 旧格式：根据 msg_id 推断角色
                msg_idx = int(msg_id.split('_')[1])
                role = "user" if msg_idx % 2 == 1 else "assistant"
                content = msg_data
            
            header = self.get_header_html(role)
            if role == "system" and isinstance(content, dict) and content.get("kind") == "tool_trace":
                content_html = self._render_tool_trace_card(msg_id, content)
            else:
                content_html = self._markdown_to_html(content) if role in {"assistant", "system"} else \
                    self._escape_html(content).replace('\n', '<br>')
            
            messages_html += f'''
            <div class="message-container" id="{msg_id}">
                <div class="message-wrapper">
                    {header}
                    <div class="message-content">{content_html}</div>
                </div>
            </div>
            <hr>
            '''
        
        # 一次性设置完整的 HTML 文档（包含样式和所有消息）
        full_html = f"<!DOCTYPE html><html><head>{self.base_style}</head><body>{messages_html}</body></html>"
        self.setHtml(full_html)
        
        # 恢复滚动位置到底部
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())
    
    def _escape_html(self, text):
        """转义 HTML 特殊字符"""
        if not text:
            return ""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def _render_tool_trace_card(self, msg_id, payload):
        """将工具调用痕迹渲染为折叠卡片"""
        title = self._escape_html(str(payload.get("title", "本轮已自动调用工具并补充证据")))
        items = payload.get("items", []) or []
        badge = f"{len(items)} 个工具"
        is_expanded = msg_id in self.expanded_tool_cards
        arrow = "▼" if is_expanded else "▶"
        item_html = []
        for item in items:
            item_title = self._escape_html(str(item.get("title", "")))
            item_summary = self._escape_html(str(item.get("summary", "")))
            summary_html = f'<div class="tool-trace-item-summary">{item_summary}</div>' if item_summary else ""
            item_html.append(
                "<li>"
                f'<div class="tool-trace-item-title">{item_title}</div>'
                f"{summary_html}"
                "</li>"
            )

        report_path = self._escape_html(str(payload.get("report_path", "")))
        report_html = ""
        if report_path:
            report_html = (
                '<div class="tool-trace-path">'
                f'证据记录: <code>{report_path}</code>'
                "</div>"
            )

        body_html = ""
        if is_expanded:
            body_html = (
                '<div class="tool-trace-body">'
                f'<ol class="tool-trace-list">{"".join(item_html)}</ol>'
                f"{report_html}"
                "</div>"
            )

        return (
            '<div class="tool-trace-card">'
            f'<a class="tool-trace-toggle" href="cursor://tool-trace-toggle/{msg_id}">'
            '<div class="tool-trace-summary">'
            f'<span class="tool-trace-title"><span class="tool-trace-arrow">{arrow}</span>{title}</span>'
            f'<span class="tool-trace-badge">{badge}</span>'
            "</div>"
            "</a>"
            f"{body_html}"
            "</div>"
        )

    def on_anchor_clicked(self, url: QUrl):
        url_text = url.toString()
        prefix = "cursor://tool-trace-toggle/"
        if url_text.startswith(prefix):
            msg_id = url_text[len(prefix):]
            if msg_id in self.expanded_tool_cards:
                self.expanded_tool_cards.remove(msg_id)
            else:
                self.expanded_tool_cards.add(msg_id)
            self.render_all_messages()
            return
        QDesktopServices.openUrl(url)
    
    def clear_messages(self):
        """清除所有消息"""
        self.raw_contents.clear()
        self.msg_counter = 0
        self.current_msg_id = None
        self.expanded_tool_cards.clear()
        # 清除并重新初始化文档
        super().clear()
        # 重新设置基础HTML结构
        self.setHtml(f"<!DOCTYPE html><html><head>{self.base_style}</head><body></body></html>")
        
    def set_theme(self, theme):
        """设置主题"""
        self.theme = theme
        self.update_style()
        self.render_all_messages()

class AISettingsDialog(QDialog):
    """模型设置对话框 - 支持每个模型独立配置"""
    def __init__(self, models, model_configs, current_system_prompt, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI 模型设置")
        self.setMinimumWidth(450)
        self.models = models
        self.model_configs = model_configs.copy()
        self.system_prompt = current_system_prompt
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 模型选择器（在设置对话框内切换要配置的模型）
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("配置模型:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems(self.models)
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        model_select_layout.addWidget(self.model_selector)
        layout.addLayout(model_select_layout)
        
        # API 配置分组
        self.form_group = QGroupBox("模型特定配置")
        form_layout = QFormLayout(self.form_group)
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.textChanged.connect(self.on_input_changed)
        form_layout.addRow("API Key:", self.api_key_input)
        
        self.base_url_input = QLineEdit()
        self.base_url_input.textChanged.connect(self.on_input_changed)
        form_layout.addRow("Base URL:", self.base_url_input)
        
        layout.addWidget(self.form_group)
        
        # 全局配置
        global_group = QGroupBox("全局配置")
        global_layout = QFormLayout(global_group)
        self.system_prompt_input = QTextEdit(self.system_prompt)
        self.system_prompt_input.setMaximumHeight(100)
        global_layout.addRow("系统提示词:", self.system_prompt_input)
        layout.addWidget(global_group)
        
        # 初始化显示第一个模型的配置
        self.on_model_changed(self.model_selector.currentText())
        
        # 底部按钮
        btn_box = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        
        btn_box.addStretch()
        btn_box.addWidget(save_btn)
        btn_box.addWidget(cancel_btn)
        layout.addLayout(btn_box)

    def on_model_changed(self, model_name):
        """当设置对话框中的模型切换时，加载该模型的配置"""
        config = self.model_configs.get(model_name, {"api_key": "", "base_url": ""})
        self.api_key_input.setText(config.get("api_key", ""))
        self.base_url_input.setText(config.get("base_url", ""))
        self.form_group.setTitle(f"配置模型: {model_name}")

    def on_input_changed(self):
        """当输入框内容变化时，实时保存到内存中的 model_configs"""
        model_name = self.model_selector.currentText()
        self.model_configs[model_name] = {
            "api_key": self.api_key_input.text().strip(),
            "base_url": self.base_url_input.text().strip()
        }

    def get_config(self):
        return {
            "model_configs": self.model_configs,
            "system_prompt": self.system_prompt_input.toPlainText().strip()
        }

class ChatThread(QThread):
    """聊天后台线程，用于异步调用大模型 API"""
    message_received = pyqtSignal(str, bool)  # content, is_error
    finished_signal = pyqtSignal()

    def __init__(
        self,
        api_key,
        base_url,
        model,
        system_prompt,
        messages,
        use_web_search=False,
        *,
        stream=True,
        request_timeout_seconds=120.0,
        log_context="",
    ):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.messages = messages
        self.use_web_search = use_web_search
        self.stream = bool(stream)
        self.request_timeout_seconds = float(request_timeout_seconds or 120.0)
        self.log_context = str(log_context or self.model)

    def run(self):
        try:
            model_lower = self.model.lower()
            # 如果是 Gemini 模型且开启了联网搜索，优先使用 Google 官方 SDK
            if "gemini" in model_lower and self.use_web_search:
                self.run_gemini_native()
            # 如果是 Kimi 模型，使用 Kimi API（支持 $web_search 联网功能）
            elif "kimi" in model_lower:
                if not self.use_web_search and not self.stream:
                    self.run_openai_compatible()
                else:
                    self.run_kimi_api()
            else:
                self.run_openai_compatible()
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            self.message_received.emit(f"\n错误: {str(e)}", True)
        finally:
            self.finished_signal.emit()

    def run_gemini_native(self):
        """使用 Google Generative AI SDK 调用 Gemini (支持联网搜索)"""
        client = google_genai.Client(api_key=self.api_key)
        
        model_name = self.model.lower()

        # 定义工具：Google Search (内置工具)
        google_search_tool = genai_types.Tool(
            google_search=genai_types.GoogleSearch()
        )
        
        config = genai_types.GenerateContentConfig(
            tools=[google_search_tool],
            system_instruction=self.system_prompt.strip() if self.system_prompt else None
        )

        # 构建历史记录和新消息
        contents = []
        
        # 转换历史记录
        for m in self.messages[:-1]:
            role = m["role"]
            content = m["content"]
            
            parts = []
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        text_val = part.get("text", "").strip()
                        if text_val:
                            parts.append(genai_types.Part(text=text_val))
            elif isinstance(content, str):
                if content.strip():
                    parts.append(genai_types.Part(text=content))
            
            if parts:
                # 使用 Content 对象
                contents.append(genai_types.Content(role=role, parts=parts))

        # 添加当前用户消息
        last_msg = self.messages[-1]["content"]
        
        if isinstance(last_msg, str):
            if last_msg.strip():
                contents.append(genai_types.Content(
                    role="user", 
                    parts=[genai_types.Part(text=last_msg.strip())]
                ))
        elif isinstance(last_msg, list):
            parts = []
            for part in last_msg:
                if part.get("type") == "text":
                    text_val = part.get("text", "").strip()
                    if text_val:
                        parts.append(genai_types.Part(text=text_val))
                elif part.get("type") == "image_url":
                    try:
                        url = part.get("image_url", {}).get("url", "")
                        if "," in url:
                            img_data = base64.b64decode(url.split(",")[1])
                            parts.append(genai_types.Part(
                                inline_data=genai_types.Blob(
                                    mime_type="image/jpeg",
                                    data=img_data
                                )
                            ))
                    except Exception as e:
                        logger.error(f"Failed to decode image: {e}")
            
            if parts:
                contents.append(genai_types.Content(role="user", parts=parts))

        # 确保至少有一个消息
        if not contents:
            contents.append(genai_types.Content(
                role="user", 
                parts=[genai_types.Part(text=" ")]
            ))

        try:
            full_content = ""
            start_ts = time.time()
            logger.info(
                "[%s] Gemini 请求开始: stream=%s timeout=%.1fs",
                self.log_context,
                self.stream,
                self.request_timeout_seconds,
            )
            if self.stream:
                response = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                first_chunk_ts = None
                for chunk in response:
                    if chunk.text:
                        if first_chunk_ts is None:
                            first_chunk_ts = time.time()
                            logger.info("[%s] Gemini 首包耗时 %.2fs", self.log_context, first_chunk_ts - start_ts)
                        self.message_received.emit(chunk.text, False)
                        full_content += chunk.text
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                full_content = getattr(response, "text", "") or ""
                if full_content:
                    self.message_received.emit(full_content, False)
            logger.info("[%s] Gemini 响应完成: 长度=%d 耗时=%.2fs", self.log_context, len(full_content), time.time() - start_ts)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini Error: {error_msg}")
            self.message_received.emit(f"\nAPI 调用错误: {error_msg}", True)

    def run_kimi_api(self):
        """使用 Kimi K2.5 API 调用（支持多模态图片、联网搜索和 thinking 模式）
        
        根据官方文档: https://platform.moonshot.ai/docs/guide/kimi-k2-5-quickstart
        - 模型名称: kimi-k2.5
        - 支持多模态: 图片使用 image_url 类型，base64 编码
        - 支持联网搜索: 使用 $web_search 内置工具
        - 支持 thinking 模式: 默认启用，需要在 assistant 消息中保留 reasoning_content
        
        注意：使用 httpx 直接发送请求，确保 reasoning_content 字段能正确传递
        （OpenAI SDK 会过滤掉不认识的字段）
        """
        base_url = self.base_url if self.base_url else "https://api.moonshot.cn/v1"
        api_endpoint = f"{base_url}/chat/completions"
        
        logger.info(f"[Kimi K2.5] Starting API call, model={self.model}, use_web_search={self.use_web_search}")
        
        # 构建完整的对话消息
        full_messages = []
        
        # 构建系统提示词（Kimi K2.5 要求 system 消息内容不能为空）
        system_content = self.system_prompt.strip() if self.system_prompt else "你是 Kimi，由 Moonshot AI 提供的智能助手。"
        
        # 如果开启联网搜索，注入引导语让模型主动使用联网搜索工具
        if self.use_web_search:
            web_search_guide = (
                "【重要】用户已开启联网搜索功能。请在回答问题时主动使用联网搜索工具获取最新、最准确的信息。"
                "尤其对于涉及时效性内容（如新闻、股票行情、天气、体育赛事、最新政策等）的问题，"
                "必须先进行联网搜索再回答，不要依赖训练数据中的旧信息。"
            )
            system_content = f"{web_search_guide}\n\n{system_content}"
        
        full_messages.append({"role": "system", "content": system_content})
        
        # 处理消息历史，确保多模态消息格式正确（Kimi K2.5 使用 image_url 类型）
        for msg in self.messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                # 多模态消息：转换为 Kimi K2.5 格式
                kimi_content = []
                for part in msg["content"]:
                    if part.get("type") == "text":
                        kimi_content.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") == "image_url":
                        kimi_content.append({
                            "type": "image_url",
                            "image_url": {"url": part.get("image_url", {}).get("url", "")}
                        })
                full_messages.append({"role": "user", "content": kimi_content})
            else:
                full_messages.append(msg)
        
        # 构建工具列表
        tools = None
        if self.use_web_search:
            tools = [{
                "type": "builtin_function",
                "function": {"name": "$web_search"}
            }]
            logger.info(f"[Kimi K2.5] Web search enabled, tools={tools}")
        
        # HTTP 请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # 循环处理工具调用（联网搜索需要多轮交互）
            max_tool_rounds = 5
            current_round = 0
            has_reasoning_content = False  # 跟踪是否收到过 reasoning_content
            
            while current_round < max_tool_rounds:
                current_round += 1
                
                # 构建请求体
                request_body = {
                    "model": self.model,
                    "messages": full_messages,
                    "stream": True
                }
                if tools:
                    request_body["tools"] = tools
                    request_body["tool_choice"] = "auto"
                    # 如果之前没有收到 reasoning_content，禁用 thinking 模式
                    # 根据官方文档：使用工具时如果模型没有产生 reasoning_content，后续请求需要禁用 thinking
                    if current_round > 1 and not has_reasoning_content:
                        request_body["thinking"] = {"type": "disabled"}
                        logger.info(f"[Kimi K2.5] Thinking disabled for round {current_round} (no reasoning_content received)")
                
                logger.info(f"[Kimi K2.5] Round {current_round}, messages count={len(full_messages)}")
                
                full_content = ""
                tool_calls_data = {}
                finish_reason = None
                reasoning_content = ""
                
                # 使用 httpx 发送流式请求
                with httpx.Client(timeout=120.0) as client:
                    with client.stream("POST", api_endpoint, headers=headers, json=request_body) as response:
                        if response.status_code != 200:
                            error_text = response.read().decode('utf-8')
                            raise Exception(f"HTTP {response.status_code}: {error_text}")
                        
                        # 处理 SSE 流式响应
                        for line in response.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            
                            data_str = line[6:]  # 去掉 "data: " 前缀
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            
                            if not chunk.get("choices"):
                                continue
                            
                            choice = chunk["choices"][0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")
                            
                            # 处理正常内容
                            if delta.get("content"):
                                content = delta["content"]
                                full_content += content
                                self.message_received.emit(content, False)
                            
                            # 收集 reasoning_content (thinking 模式)
                            if delta.get("reasoning_content"):
                                reasoning_content += delta["reasoning_content"]
                                has_reasoning_content = True  # 标记收到了 reasoning_content
                            
                            # 收集工具调用信息
                            if delta.get("tool_calls"):
                                for tool_call in delta["tool_calls"]:
                                    idx = tool_call.get("index", 0)
                                    if idx not in tool_calls_data:
                                        tool_calls_data[idx] = {
                                            "id": "",
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        }
                                    if tool_call.get("id"):
                                        tool_calls_data[idx]["id"] = tool_call["id"]
                                    if tool_call.get("type"):
                                        tool_calls_data[idx]["type"] = tool_call["type"]
                                    if tool_call.get("function"):
                                        func = tool_call["function"]
                                        if func.get("name"):
                                            tool_calls_data[idx]["function"]["name"] = func["name"]
                                        if func.get("arguments"):
                                            tool_calls_data[idx]["function"]["arguments"] += func["arguments"]
                
                logger.info(f"[Kimi K2.5] Round {current_round} done, finish_reason={finish_reason}, "
                           f"tool_calls={bool(tool_calls_data)}, reasoning_content_len={len(reasoning_content)}")
                
                # 检查是否有工具调用需要处理
                if finish_reason == "tool_calls" and tool_calls_data:
                    logger.info(f"[Kimi K2.5] Processing tool_calls...")
                    
                    # 构建 assistant 消息（包含 tool_calls）
                    tool_calls_list = [tool_calls_data[i] for i in sorted(tool_calls_data.keys())]
                    assistant_msg = {
                        "role": "assistant",
                        "content": full_content if full_content else "",
                        "tool_calls": tool_calls_list
                    }
                    # 只有当收到 reasoning_content 时才包含该字段
                    # 如果模型没有产生 reasoning_content，不应该包含空字符串（会导致 API 报错）
                    if reasoning_content:
                        assistant_msg["reasoning_content"] = reasoning_content
                    full_messages.append(assistant_msg)
                    
                    # 为每个工具调用添加 tool 消息
                    for tool_call in tool_calls_list:
                        tool_call_name = tool_call["function"]["name"]
                        tool_call_arguments = tool_call["function"]["arguments"]
                        
                        try:
                            arguments_dict = json.loads(tool_call_arguments) if tool_call_arguments else {}
                        except json.JSONDecodeError:
                            arguments_dict = {}
                        
                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": tool_call_name,
                            "content": json.dumps(arguments_dict)
                        }
                        full_messages.append(tool_msg)
                        logger.info(f"[Kimi K2.5] Tool call: {tool_call_name}")
                    
                    continue
                else:
                    break
            
            logger.info(f"[Kimi K2.5] Response finished. Length: {len(full_content)}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Kimi K2.5 API Error: {error_msg}")
            self.message_received.emit(f"\nKimi K2.5 API 调用错误: {error_msg}", True)

    def run_openai_compatible(self):
        """传统的 OpenAI 兼容模式调用"""
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_seconds,
        )
        
        # 构建完整的对话消息
        full_messages = [{"role": "system", "content": self.system_prompt}]
        full_messages.extend(self.messages)
        start_ts = time.time()
        logger.info(
            "[%s] OpenAI兼容请求开始: stream=%s timeout=%.1fs",
            self.log_context,
            self.stream,
            self.request_timeout_seconds,
        )
        full_content = ""
        if self.stream:
            response = client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                stream=True
            )
            first_chunk_ts = None
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if first_chunk_ts is None:
                        first_chunk_ts = time.time()
                        logger.info("[%s] OpenAI兼容首包耗时 %.2fs", self.log_context, first_chunk_ts - start_ts)
                    full_content += content
                    self.message_received.emit(content, False)
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                stream=False,
            )
            full_content = ((response.choices[0].message.content or "") if response.choices else "") or ""
            if full_content:
                self.message_received.emit(full_content, False)

        logger.info("[%s] OpenAI兼容响应完成: 长度=%d 耗时=%.2fs", self.log_context, len(full_content), time.time() - start_ts)

class AttachmentThumbnail(QFrame):
    """附件缩略图组件"""
    deleted = pyqtSignal(str)  # 发送要删除的文件路径

    def __init__(self, file_path, parent=None, theme="light"):
        super().__init__(parent)
        self.file_path = file_path
        self.theme = theme
        self.setup_ui()

    def setup_ui(self):
        self.setFixedSize(80, 80)
        self.update_style()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # 预览图或图标
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        mime_type, _ = mimetypes.guess_type(self.file_path)
        if mime_type and mime_type.startswith('image/'):
            pixmap = QPixmap(self.file_path)
            self.icon_label.setPixmap(pixmap.scaled(70, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            # 文本文件显示图标
            self.icon_label.setText("📄")
            self.icon_label.setStyleSheet("font-size: 24px;")
            
        layout.addWidget(self.icon_label)

        # 文件名
        self.name_label = QLabel(os.path.basename(self.file_path))
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label)
        
        self.update_style() # 再次调用以设置子组件样式

        # 删除按钮 (X)
        self.del_btn = QPushButton("×", self)
        self.del_btn.setFixedSize(16, 16)
        self.del_btn.move(62, 2)
        self.del_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 0, 0, 150);
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #f44336;
            }
        """)
        self.del_btn.clicked.connect(lambda: self.deleted.emit(self.file_path))

    def update_style(self):
        if self.theme == "light":
            bg_color = "#eeeeee"
            border_color = "#cccccc"
            text_color = "#666666"
        else:
            bg_color = "#3c3c3c"
            border_color = "#555555"
            text_color = "#bbbbbb"
            
        self.setStyleSheet(f"""
            AttachmentThumbnail {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 4px;
            }}
            AttachmentThumbnail:hover {{
                border-color: #0078d4;
            }}
            QLabel {{
                color: {text_color};
                font-size: 9px;
            }}
        """)

    def mousePressEvent(self, event):
        """点击打开大图或内容查看"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.preview_content()

    def preview_content(self):
        mime_type, _ = mimetypes.guess_type(self.file_path)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"预览: {os.path.basename(self.file_path)}")
        dialog.setMinimumSize(600, 400)
        layout = QVBoxLayout(dialog)
        
        if mime_type and mime_type.startswith('image/'):
            label = QLabel()
            pixmap = QPixmap(self.file_path)
            label.setPixmap(pixmap.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            scroll = QScrollArea()
            scroll.setWidget(label)
            layout.addWidget(scroll)
        else:
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            try:
                with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_edit.setText(f.read())
            except Exception as e:
                text_edit.setText(f"无法读取文件: {e}")
            layout.addWidget(text_edit)
            
        dialog.exec()

class MessageInput(QTextEdit):
    """支持拖放和粘贴图片的自定义输入框"""
    files_dropped = pyqtSignal(list)
    image_pasted = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            if file_paths:
                self.files_dropped.emit(file_paths)
                event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def insertFromMimeData(self, source):
        """处理粘贴事件，支持图片粘贴"""
        if source.hasImage():
            image = source.imageData()
            if isinstance(image, QPixmap):
                self.image_pasted.emit(image)
            else:
                # 某些情况下 imageData() 返回 QImage
                from PyQt6.QtGui import QImage
                if isinstance(image, QImage):
                    self.image_pasted.emit(QPixmap.fromImage(image))
            return
        
        # 处理文件路径粘贴 (例如在文件管理器中复制文件后在此处粘贴)
        if source.hasUrls():
            file_paths = [url.toLocalFile() for url in source.urls() if url.isLocalFile()]
            if file_paths:
                self.files_dropped.emit(file_paths)
                return

        super().insertFromMimeData(source)

class AIAgentWidget(QWidget):
    """智能体版块组件 - Cursor 风格优化版"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "config", "ai_config.json"
        )
        self.chat_sessions = {TASK_MODE_GENERAL: []}
        self.active_session_key = TASK_MODE_GENERAL
        self.active_task_mode = TASK_MODE_GENERAL
        self.current_context = AgentRuntimeContext()
        self.context_provider = None
        self.agent_runtime = StockAgentRuntime()
        self.risk_guard = RiskGuardService()
        self.decision_tracker = DecisionTrackerService()
        self._pending_decision_response = ""
        self.attached_files = []  # 存储当前待发送的文件路径
        self.model_configs = {}  # 存储每个模型的 api_key 和 base_url
        self.system_prompt = "你是一个专业的股票投资顾问。"
        self.theme = "light" # 强制设为浅色
        self.setup_ui()
        self.load_config()

    @property
    def chat_history(self):
        return self.chat_sessions.setdefault(self.active_session_key, [])

    @chat_history.setter
    def chat_history(self, value):
        self.chat_sessions[self.active_session_key] = value or []

    def setup_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- 顶部导航栏 (Header) ---
        self.header = QFrame()
        self.header.setObjectName("Header")
        self.header.setFixedHeight(50)
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(15, 0, 15, 0)
        
        # 模型选择器
        self.model_label = QLabel("模型:")
        self.model_label.setObjectName("HeaderLabel")
        self.header_layout.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ModelCombo")
        self.model_combo.addItems([
            "deepseek-chat", 
            "deepseek-reasoner", 
            "gpt-4o", 
            "gpt-4o-mini",
            "gemini-3-pro-preview", 
            "gemini-3-flash-preview",
            "claude-3-5-sonnet",
            "kimi-k2.5"
        ])
        self.model_combo.setFixedWidth(180)
        self.model_combo.currentTextChanged.connect(self.on_model_selection_changed)
        self.header_layout.addWidget(self.model_combo)

        self.header_layout.addStretch()
        
        # 联网搜索开关
        self.web_search_cb = QCheckBox("🌐 联网搜索")
        self.web_search_cb.setObjectName("HeaderCheckbox")
        self.web_search_cb.setToolTip("开启联网搜索增强 (Gemini/Kimi)")
        self.web_search_cb.setVisible(False) # 默认隐藏，仅在选中 gemini 或 kimi 模型时显示
        self.header_layout.addWidget(self.web_search_cb)
        
        # 设置按钮
        self.settings_btn = QPushButton("⚙ 设置")
        self.settings_btn.setObjectName("HeaderBtn")
        self.settings_btn.setFlat(True)
        self.settings_btn.clicked.connect(self.open_settings)
        self.header_layout.addWidget(self.settings_btn)
        
        # 清空按钮
        self.clear_btn = QPushButton("🗑 清空")
        self.clear_btn.setObjectName("HeaderBtn")
        self.clear_btn.setFlat(True)
        self.clear_btn.clicked.connect(self.clear_chat)
        self.header_layout.addWidget(self.clear_btn)
        
        main_layout.addWidget(self.header)

        self.context_frame = QFrame()
        self.context_frame.setObjectName("ContextFrame")
        context_layout = QVBoxLayout(self.context_frame)
        context_layout.setContentsMargins(12, 10, 12, 10)
        context_layout.setSpacing(6)

        top_row = QHBoxLayout()
        self.context_title_label = QLabel("当前上下文")
        self.context_title_label.setObjectName("ContextTitle")
        top_row.addWidget(self.context_title_label)
        self.quick_task_label = QLabel("快捷任务")
        self.quick_task_label.setObjectName("ContextTitle")
        top_row.addWidget(self.quick_task_label)
        top_row.addStretch()
        self.context_refresh_btn = QPushButton("刷新上下文")
        self.context_refresh_btn.setObjectName("ContextBtn")
        self.context_refresh_btn.clicked.connect(self.refresh_context)
        top_row.addWidget(self.context_refresh_btn)
        context_layout.addLayout(top_row)

        self.context_summary_label = QLabel("尚未接入运行上下文")
        self.context_summary_label.setObjectName("ContextSummary")
        self.context_summary_label.setWordWrap(True)
        context_layout.addWidget(self.context_summary_label)

        # --- 券商账户状态栏 ---
        broker_row = QHBoxLayout()
        self.broker_status_icon = QLabel("🔴")
        self.broker_status_icon.setObjectName("BrokerStatusIcon")
        broker_row.addWidget(self.broker_status_icon)
        self.broker_status_label = QLabel("账户: 未连接")
        self.broker_status_label.setObjectName("BrokerStatusLabel")
        broker_row.addWidget(self.broker_status_label)
        broker_row.addStretch()
        self.broker_connect_btn = QPushButton("🔗 连接")
        self.broker_connect_btn.setObjectName("BrokerConnectBtn")
        self.broker_connect_btn.setToolTip("连接券商账户以使用持仓诊断功能（仅查询，不交易）")
        self.broker_connect_btn.clicked.connect(self.on_broker_connect_clicked)
        broker_row.addWidget(self.broker_connect_btn)
        self.broker_disconnect_btn = QPushButton("🔌 断开")
        self.broker_disconnect_btn.setObjectName("BrokerDisconnectBtn")
        self.broker_disconnect_btn.setVisible(False)
        self.broker_disconnect_btn.clicked.connect(self.on_broker_disconnect_clicked)
        broker_row.addWidget(self.broker_disconnect_btn)
        context_layout.addLayout(broker_row)

        quick_row = QHBoxLayout()
        self.quick_symbol_btn = QPushButton("分析当前标的")
        self.quick_symbol_btn.setObjectName("QuickTaskBtn")
        self.quick_symbol_btn.clicked.connect(self.on_quick_symbol_analysis_clicked)
        quick_row.addWidget(self.quick_symbol_btn)
        self.quick_watchlist_btn = QPushButton("巡检当前分组")
        self.quick_watchlist_btn.setObjectName("QuickTaskBtn")
        self.quick_watchlist_btn.clicked.connect(
            lambda: self.run_quick_task(TASK_MODE_WATCHLIST_SCAN)
        )
        quick_row.addWidget(self.quick_watchlist_btn)
        self.quick_position_btn = QPushButton("持仓诊断")
        self.quick_position_btn.setObjectName("QuickTaskBtn")
        self.quick_position_btn.clicked.connect(
            lambda: self.run_quick_task(TASK_MODE_POSITION_DIAGNOSIS)
        )
        quick_row.addWidget(self.quick_position_btn)
        self.quick_trade_decision_btn = QPushButton("📊 交易决策")
        self.quick_trade_decision_btn.setObjectName("QuickTaskBtn")
        self.quick_trade_decision_btn.setToolTip("基于多维分析生成结构化交易决策（含风控审核）")
        self.quick_trade_decision_btn.clicked.connect(
            lambda: self.run_quick_task(TASK_MODE_TRADE_DECISION)
        )
        quick_row.addWidget(self.quick_trade_decision_btn)
        quick_row.addStretch()
        context_layout.addLayout(quick_row)

        main_layout.addWidget(self.context_frame)
        
        # --- 中间对话区域 (统一聊天显示) ---
        self.chat_display = ChatDisplayWidget(theme=self.theme)
        self.chat_display.setObjectName("ChatDisplay")
        main_layout.addWidget(self.chat_display, stretch=10) # 给对话区域最大的权重
        
        # --- 底部输入区域 (Input Area) ---
        self.input_frame = QFrame()
        self.input_frame.setObjectName("InputFrame")
        input_vbox = QVBoxLayout(self.input_frame)
        input_vbox.setContentsMargins(10, 5, 10, 10)
        input_vbox.setSpacing(2) # 极小间距
        
        # 附件和工具栏 (尽可能调小)
        tool_layout = QHBoxLayout()
        tool_layout.setContentsMargins(0, 0, 0, 0)
        self.attach_btn = QToolButton()
        self.attach_btn.setObjectName("AttachBtn")
        self.attach_btn.setText("+ 附件")
        self.attach_btn.setToolTip("上传文件或图片")
        self.attach_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        
        # Create attachment menu
        attach_menu = QMenu(self)
        select_file_action = attach_menu.addAction("📁 选择文件...")
        select_file_action.triggered.connect(self.on_select_file_clicked)
        self.attach_btn.setMenu(attach_menu)
        
        tool_layout.addWidget(self.attach_btn)
        
        tool_layout.addStretch()
        input_vbox.addLayout(tool_layout)

        # 附件预览区域 (缩略图列表)
        self.attachment_scroll = QScrollArea()
        self.attachment_scroll.setFixedHeight(95)
        self.attachment_scroll.setWidgetResizable(True)
        self.attachment_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.attachment_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.attachment_scroll.setStyleSheet("background-color: transparent; border: none;")
        self.attachment_scroll.setVisible(False) # 初始隐藏
        
        self.attachment_container = QWidget()
        self.attachment_layout = QHBoxLayout(self.attachment_container)
        self.attachment_layout.setContentsMargins(0, 5, 0, 5)
        self.attachment_layout.setSpacing(10)
        self.attachment_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.attachment_scroll.setWidget(self.attachment_container)
        input_vbox.addWidget(self.attachment_scroll)
        
        # 文本输入框 (使用自定义 MessageInput)
        input_hbox = QHBoxLayout()
        input_hbox.setContentsMargins(0, 0, 0, 0)
        self.message_input = MessageInput()
        self.message_input.setObjectName("MessageInput")
        self.message_input.files_dropped.connect(self.handle_files_dropped)
        self.message_input.image_pasted.connect(self.handle_image_pasted)
        self.message_input.setPlaceholderText("输入消息，或者直接拖入/粘贴图片或文件... (Enter 发送)")
        self.message_input.setMaximumHeight(120)
        input_hbox.addWidget(self.message_input)
        
        # 发送按钮
        self.send_btn = QPushButton("发送")
        self.send_btn.setObjectName("SendBtn")
        self.send_btn.setFixedWidth(60)
        self.send_btn.setFixedHeight(40)
        self.send_btn.clicked.connect(self.send_message)
        input_hbox.addWidget(self.send_btn, alignment=Qt.AlignmentFlag.AlignBottom)
        
        input_vbox.addLayout(input_hbox)
        main_layout.addWidget(self.input_frame)
        
        # 初始应用主题样式
        self.update_widget_styles()
        
        # 快捷键
        self.message_input.installEventFilter(self)

    def on_attach_clicked(self):
        """点击附件按钮（保留兼容性）"""
        self.on_select_file_clicked()

    def on_select_file_clicked(self):
        """选择文件附件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择附件", "", "图片和文件 (*.png *.jpg *.jpeg *.txt *.csv *.py);;所有文件 (*.*)"
        )
        if file_paths:
            self.add_attachments(file_paths)

    def on_model_selection_changed(self, model_name):
        """当模型选择变化时，显示/隐藏联网搜索开关"""
        model_lower = model_name.lower()
        is_gemini = "gemini" in model_lower
        is_kimi = "kimi" in model_lower
        supports_web_search = is_gemini or is_kimi
        self.web_search_cb.setVisible(supports_web_search)
        if not supports_web_search:
            self.web_search_cb.setChecked(False)

    def set_context_provider(self, provider):
        """注入运行时上下文提供者。provider() -> dict"""
        self.context_provider = provider
        self.refresh_context()

    def refresh_context(self):
        raw_context = {}
        if callable(self.context_provider):
            try:
                raw_context = self.context_provider() or {}
            except Exception as exc:
                logger.error(f"Failed to collect agent context: {exc}")
        self.current_context = AgentContextService.from_raw(raw_context)
        self._refresh_context_panel()
        self._maybe_switch_session()

    def _refresh_context_panel(self):
        lines = self.current_context.to_summary_lines()
        self.context_summary_label.setText("\n".join(lines))
        has_symbol = self.current_context.symbol.is_available
        has_watchlist = self.current_context.watchlist.visible_count > 0
        has_broker = self.current_context.broker.connected
        self.quick_symbol_btn.setEnabled(has_symbol)
        self.quick_watchlist_btn.setEnabled(has_watchlist)
        self.quick_position_btn.setEnabled(has_broker)
        
        # Update broker status UI
        if has_broker:
            self.broker_status_icon.setText("🟢")
            self.broker_status_label.setText("账户: 已连接")
            self.broker_connect_btn.setVisible(False)
            self.broker_disconnect_btn.setVisible(True)
        else:
            self.broker_status_icon.setText("🔴")
            self.broker_status_label.setText("账户: 未连接")
            self.broker_connect_btn.setVisible(True)
            self.broker_disconnect_btn.setVisible(False)

    def _build_session_key(self) -> str:
        mode = self.active_task_mode
        if mode == TASK_MODE_SYMBOL_ANALYSIS:
            return f"{mode}:{self.current_context.symbol.code or 'none'}"
        if mode == TASK_MODE_WATCHLIST_SCAN:
            return f"{mode}:{self.current_context.watchlist.group_name or self.current_context.watchlist.source_tab or 'default'}"
        return mode

    def _maybe_switch_session(self):
        session_key = self._build_session_key()
        if session_key == self.active_session_key:
            return
        self.active_session_key = session_key
        self.chat_sessions.setdefault(session_key, [])
        self._reload_session_messages()

    def _reload_session_messages(self):
        self.chat_display.clear_messages()
        for msg in self.chat_history:
            content = msg.get("content", "")
            if not isinstance(content, (str, dict)):
                content = "[多模态消息]"
            role = msg.get("role", "user")
            if role == "system_display":
                role = "system"
            self.chat_display.add_message(role, content)

    def _switch_task_mode(self, task_mode: str):
        self.active_task_mode = task_mode
        self._maybe_switch_session()

    def on_quick_symbol_analysis_clicked(self):
        self.run_quick_task(TASK_MODE_SYMBOL_ANALYSIS)

    def on_broker_connect_clicked(self):
        """Handle broker connect button click - 直接读取配置文件并连接"""
        try:
            broker_service = get_broker_session_service()
            config = broker_service.get_config()
            path = config.get("qmt_path", "").strip()
            account = config.get("account", "").strip()

            # 验证配置是否完整
            if not path or not account:
                QMessageBox.warning(
                    self,
                    "配置缺失",
                    "未找到券商账户配置信息。\n\n"
                    "请先配置券商账户信息到 trading_app/config/broker_config.json:\n"
                    "{\n"
                    '  "qmt_path": "D:\\\\QMT\\\\userdata_mini",\n'
                    '  "account": "您的资金账号"\n'
                    "}"
                )
                return

            # 验证路径是否存在
            if not os.path.exists(path):
                QMessageBox.warning(
                    self,
                    "路径错误",
                    f"配置的QMT路径不存在：\n{path}\n\n"
                    "请检查 trading_app/config/broker_config.json 中的 qmt_path 配置。"
                )
                return

            # 使用异步连接
            success = broker_service.connect_async(path, account)
            if success:
                # 连接已启动，等待连接结果
                self._broker_connecting = True
                broker_service.connection_changed.connect(self._on_broker_connection_changed)
            else:
                QMessageBox.warning(self, "连接失败", "无法启动券商连接，请检查配置")
        except Exception as e:
            QMessageBox.critical(self, "连接错误", f"连接过程中发生错误：{str(e)}")

    def _on_broker_connection_changed(self, connected: bool, message: str):
        """处理券商连接状态变化"""
        if connected:
            QMessageBox.information(self, "连接成功", message)
            self.refresh_context()  # Refresh to update position diagnosis button
        else:
            # 只在连接失败时显示错误，连接中的状态不显示
            if not getattr(self, '_broker_connecting', False):
                QMessageBox.warning(self, "连接失败", message)
        self._broker_connecting = False

    def on_broker_disconnect_clicked(self):
        """Handle broker disconnect button click"""
        try:
            broker_service = get_broker_session_service()
            broker_service.disconnect()
            QMessageBox.information(self, "断开成功", "券商账户已断开连接")
            self.refresh_context()  # Refresh to update position diagnosis button
        except Exception as e:
            QMessageBox.warning(self, "断开失败", f"断开连接时发生错误：{str(e)}")

    def run_quick_task(self, task_mode: str):
        self._switch_task_mode(task_mode)
        self.refresh_context()
        if task_mode == TASK_MODE_WATCHLIST_SCAN:
            self._run_watchlist_scan_task()
            return
        prompt = AgentPromptBuilder.build_quick_task_prompt(
            task_mode,
            self.current_context,
        )
        if not prompt:
            return
        self._send_task_prompt(prompt, prompt)

    def _run_watchlist_scan_task(self):
        payload = AgentWatchlistScanService.build_scan_prompt(self.current_context.raw)
        if not payload:
            QMessageBox.warning(self, "提示", "当前分组缺少可巡检的标的或历史数据不足")
            return
        self._send_task_prompt(payload["user_display"], payload["prompt"])

    def _send_task_prompt(self, user_display: str, prompt: str):
        if not prompt:
            return
        self.refresh_context()
        model = self.model_combo.currentText()
        config = self.model_configs.get(model, {})
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", "")
        if not api_key:
            QMessageBox.warning(self, "警告", f"请先在设置中配置模型 {model} 的 API Key")
            self.open_settings()
            return

        self.message_input.clear()
        self.attached_files = []
        for i in reversed(range(self.attachment_layout.count())):
            item = self.attachment_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.attachment_scroll.setVisible(False)
        self.message_input.setEnabled(False)
        self.send_btn.setEnabled(False)

        prepared_request = self._prepare_agent_request(
            user_content=prompt,
            task_mode=self.active_task_mode,
        )

        self.append_to_display("user", user_display)
        self.chat_history.append({"role": "user", "content": prepared_request.augmented_user_content})
        self._append_tool_trace(prepared_request)
        self.append_to_display("assistant", "", is_new=True)

        self.chat_thread = ChatThread(
            api_key,
            base_url,
            model,
            prepared_request.system_prompt,
            prepared_request.messages,
            use_web_search=self.web_search_cb.isChecked(),
        )
        self.chat_thread.message_received.connect(self.on_message_received)
        self.chat_thread.finished_signal.connect(self.on_chat_finished)
        self.chat_thread.start()

    def _prepare_agent_request(self, user_content, task_mode: str | None = None, base_system_prompt: str | None = None):
        resolved_task_mode = task_mode or self.active_task_mode
        runtime_system_prompt = AgentPromptBuilder.build_system_prompt(
            base_system_prompt if base_system_prompt is not None else self.system_prompt,
            self.current_context,
            task_mode=resolved_task_mode,
        )
        model_history = [
            message for message in self.chat_history
            if message.get("role") in {"user", "assistant"}
        ]
        prepared_request = self.agent_runtime.prepare_request(
            base_system_prompt=runtime_system_prompt,
            context=self.current_context,
            task_mode=resolved_task_mode,
            chat_history=model_history,
            latest_user_content=user_content,
        )
        if prepared_request.evidence_report_path:
            logger.info(f"Agent evidence saved to: {prepared_request.evidence_report_path}")
        return prepared_request

    def _append_tool_trace(self, prepared_request):
        if not prepared_request.executed_tools:
            return
        trace_payload = {
            "kind": "tool_trace",
            "title": "本轮已自动调用工具并补充证据",
            "items": [
                {
                    "title": item.title,
                    "summary": item.summary,
                }
                for item in prepared_request.evidence_items
            ],
            "report_path": prepared_request.evidence_report_path,
        }
        self.append_to_display("system", trace_payload)
        self.chat_history.append({"role": "system_display", "content": trace_payload})

    def handle_files_dropped(self, file_paths):
        """处理鼠标拖入的文件"""
        if file_paths:
            self.add_attachments(file_paths)
            logger.info(f"Dropped files: {file_paths}")

    def handle_image_pasted(self, pixmap):
        """处理剪切板粘贴的图片"""
        import tempfile
        import time
        
        # 创建临时文件保存图片
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{TEMP_PASTED_PREFIX}{int(time.time())}.png")
        
        if pixmap.save(file_path, "PNG"):
            self.add_attachments([file_path])
            logger.info(f"Pasted image saved to: {file_path}")

    def add_attachments(self, file_paths):
        """统一添加附件并显示缩略图"""
        for path in file_paths:
            if path not in self.attached_files:
                self.attached_files.append(path)
                thumb = AttachmentThumbnail(path, theme=self.theme)
                thumb.deleted.connect(self.remove_attachment)
                self.attachment_layout.addWidget(thumb)
        
        self.update_attachment_visibility()

    def remove_attachment(self, file_path):
        """移除特定附件"""
        if file_path in self.attached_files:
            self.attached_files.remove(file_path)
            
            # 从布局中移除对应的 widget
            for i in range(self.attachment_layout.count()):
                item = self.attachment_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), AttachmentThumbnail):
                    if item.widget().file_path == file_path:
                        item.widget().deleteLater()
                        break
        
        self.update_attachment_visibility()

    def update_attachment_visibility(self):
        """根据是否有附件决定预览区是否显示"""
        has_files = len(self.attached_files) > 0
        self.attachment_scroll.setVisible(has_files)

    def update_theme(self, theme_name):
        """设置主题并更新样式"""
        self.theme = theme_name
        self.update_widget_styles()
        
        # 更新聊天显示区的主题
        self.chat_display.set_theme(theme_name)

    def update_widget_styles(self):
        """更新组件的具体样式表，强制覆盖全局主题"""
        if self.theme == "light":
            bg_main = "#ffffff"
            bg_panel = "#f8f9fa"
            bg_input = "#ffffff"
            text_main = "#333333"
            text_dim = "#666666"
            border_color = "#e0e0e0"
            combo_bg = "#ffffff"
            header_btn_hover = "#e9ecef"
        else:
            bg_main = "#1e1e1e"
            bg_panel = "#252526"
            bg_input = "#3c3c3c"
            text_main = "#ffffff"
            text_dim = "#aaaaaa"
            border_color = "#3c3c3c"
            combo_bg = "#3c3c3c"
            header_btn_hover = "#3c3c3c"

        # 强制设置背景色，并确保不继承父窗口的深色背景
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        style = f"""
            AIAgentWidget {{
                background-color: {bg_main};
                color: {text_main};
            }}
            QFrame#Header {{
                background-color: {bg_panel};
                border-bottom: 1px solid {border_color};
            }}
            QLabel#HeaderLabel {{
                color: {text_main};
                font-weight: bold;
                background: transparent;
            }}
            QComboBox#ModelCombo {{
                background-color: {combo_bg};
                color: {text_main};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 2px 10px;
                min-width: 150px;
            }}
            QComboBox#ModelCombo::drop-down {{
                border: none;
            }}
            QComboBox#ModelCombo QAbstractItemView {{
                background-color: {combo_bg};
                color: {text_main};
                selection-background-color: #0078d4;
            }}
            QComboBox#TaskModeCombo {{
                background-color: {combo_bg};
                color: {text_main};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 2px 10px;
                min-width: 120px;
            }}
            QCheckBox#HeaderCheckbox {{
                color: {text_dim};
                font-size: 12px;
                margin-right: 10px;
                background: transparent;
            }}
            QCheckBox#HeaderCheckbox:hover {{
                color: {text_main};
            }}
            QPushButton#HeaderBtn {{
                background-color: transparent;
                color: {text_dim};
                border: none;
                padding: 5px 10px;
                font-weight: normal;
            }}
            QPushButton#HeaderBtn:hover {{
                background-color: {header_btn_hover};
                color: {text_main};
                border-radius: 4px;
            }}
            ChatDisplayWidget {{
                background-color: {bg_main};
                border: none;
            }}
            QFrame#InputFrame {{
                background-color: {bg_panel};
                border-top: 1px solid {border_color};
            }}
            QFrame#ContextFrame {{
                background-color: {bg_panel};
                border-bottom: 1px solid {border_color};
            }}
            QLabel#ContextTitle {{
                color: {text_main};
                font-weight: bold;
                font-size: 13px;
            }}
            QLabel#ContextSummary {{
                color: {text_dim};
                font-size: 12px;
                line-height: 1.5;
            }}
            QPushButton#ContextBtn, QPushButton#QuickTaskBtn {{
                background-color: {combo_bg};
                color: {text_main};
                border: 1px solid {border_color};
                border-radius: 5px;
                padding: 4px 10px;
            }}
            QPushButton#ContextBtn:hover, QPushButton#QuickTaskBtn:hover {{
                background-color: {header_btn_hover};
            }}
            QPushButton#QuickTaskBtn:disabled {{
                background-color: {bg_panel};
                color: {text_dim};
            }}
            QToolButton#AttachBtn {{
                background-color: transparent;
                color: {text_dim};
                border: 1px solid {border_color};
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 11px;
            }}
            QToolButton#AttachBtn:hover {{
                background-color: {header_btn_hover};
                color: {text_main};
            }}
            QTextEdit#MessageInput {{
                background-color: {bg_input};
                color: {text_main};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 10px;
                font-size: 13px;
                line-height: 1.5;
            }}
            QPushButton#SendBtn {{
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                padding: 8px;
            }}
            QPushButton#SendBtn:hover {{
                background-color: #1a8cdb;
            }}
            QPushButton#SendBtn:disabled {{
                background-color: {border_color};
                color: {text_dim};
            }}
            QLabel {{
                background: transparent;
                color: {text_main};
            }}
        """
        self.setStyleSheet(style)

    def open_settings(self):
        """打开设置对话框"""
        models = [self.model_combo.itemText(i) for i in range(self.model_combo.count())]
        dialog = AISettingsDialog(models, self.model_configs, self.system_prompt, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_config = dialog.get_config()
            self.model_configs = new_config["model_configs"]
            self.system_prompt = new_config["system_prompt"]
            self.save_config()

    def load_config(self):
        """加载 API 配置"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.model_configs = config.get("model_configs", {})
                    self.system_prompt = config.get("system_prompt", "你是一个专业的股票投资顾问。")
                    
                    # 兼容旧版本配置
                    if not self.model_configs and "api_key" in config:
                        model = config.get("model", "deepseek-chat")
                        self.model_configs[model] = {
                            "api_key": config.get("api_key", ""),
                            "base_url": config.get("base_url", "")
                        }

                    # 设置当前模型
                    model = config.get("selected_model", config.get("model", ""))
                    index = self.model_combo.findText(model)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                        self.on_model_selection_changed(model)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        self.refresh_context()

    def save_config(self):
        """保存 API 配置"""
        config = {
            "model_configs": self.model_configs,
            "selected_model": self.model_combo.currentText(),
            "system_prompt": self.system_prompt
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def clear_chat(self):
        """清空对话历史"""
        self.chat_history = []
        # 清空聊天显示区
        self.chat_display.clear_messages()

    def append_to_display(self, role, content, is_new=True):
        """在显示区域添加消息组件"""
        if is_new:
            # 添加新消息到聊天显示区
            self.chat_display.add_message(role, content)
        else:
            # 流式追加更新最后一条消息
            self.chat_display.update_last_message(content)

    def send_message(self):
        """发送消息并获取回复"""
        self._switch_task_mode(TASK_MODE_GENERAL)
        self.refresh_context()
        content = self.message_input.toPlainText().strip()
        if not content and not self.attached_files:
            return
            
        model = self.model_combo.currentText()
        config = self.model_configs.get(model, {})
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", "")
        
        if not api_key:
            QMessageBox.warning(self, "警告", f"请先在设置中配置模型 {model} 的 API Key")
            self.open_settings()
            return
            
        # 准备消息内容 (支持多模态)
        message_content = []
        if content:
            message_content.append({"type": "text", "text": content})
            
        # 处理附件
        for file_path in self.attached_files:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('image/'):
                # 图片附件 - Base64 编码
                try:
                    with open(file_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        message_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_string}"
                            }
                        })
                except Exception as e:
                    logger.error(f"Failed to encode image {file_path}: {e}")
            else:
                # 文本附件 - 读取内容并拼接到文本中
                try:
                    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                        file_text = f.read()
                        file_name = os.path.basename(file_path)
                        text_part = f"\n\n[文件附件: {file_name}]\n{file_text}"
                        # 如果 message_content 里已有文本，追加到第一个文本部分
                        if message_content and message_content[0]["type"] == "text":
                            message_content[0]["text"] += text_part
                        else:
                            message_content.insert(0, {"type": "text", "text": text_part})
                except Exception as e:
                    logger.error(f"Failed to read file {file_path}: {e}")

        # 如果只有文本且没有图片，可以简化为纯字符串结构以兼容某些模型
        final_content = message_content
        has_images = any(part["type"] == "image_url" for part in message_content)
        if not has_images and len(message_content) == 1 and message_content[0]["type"] == "text":
            final_content = message_content[0]["text"]

        # 禁用输入
        self.message_input.clear()
        # 清空已选附件及其 UI
        self.attached_files = []
        for i in reversed(range(self.attachment_layout.count())):
            item = self.attachment_layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()
        self.attachment_scroll.setVisible(False)
        
        self.message_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # 添加到显示和历史
        display_text = content if content else "[发送了附件]"
        self.append_to_display("user", display_text)
        prepared_request = self._prepare_agent_request(user_content=final_content)
        self.chat_history.append({"role": "user", "content": prepared_request.augmented_user_content})
        self._append_tool_trace(prepared_request)

        # 准备 AI 回复组件
        self.append_to_display("assistant", "", is_new=True)

        # 获取联网搜索设置
        use_web_search = self.web_search_cb.isChecked()

        # 启动后台线程
        self.chat_thread = ChatThread(
            api_key, base_url, model, prepared_request.system_prompt, prepared_request.messages,
            use_web_search=use_web_search
        )
        self.chat_thread.message_received.connect(self.on_message_received)
        self.chat_thread.finished_signal.connect(self.on_chat_finished)
        self.chat_thread.start()

    def on_message_received(self, content, is_error):
        """处理流式返回的内容"""
        if not is_error:
            # 先累加到历史记录
            if not self.chat_history or self.chat_history[-1]["role"] != "assistant":
                self.chat_history.append({"role": "assistant", "content": content})
            else:
                self.chat_history[-1]["content"] += content
            
            # 用完整内容更新显示（update_last_message 需要完整内容）
            full_content = self.chat_history[-1]["content"]
            self.append_to_display("assistant", full_content, is_new=False)
        else:
            # 错误消息直接追加显示
            if self.chat_history and self.chat_history[-1]["role"] == "assistant":
                self.chat_history[-1]["content"] += content
                full_content = self.chat_history[-1]["content"]
                self.append_to_display("assistant", full_content, is_new=False)
            else:
                self.append_to_display("assistant", content, is_new=False)

    def on_chat_finished(self):
        """对话结束"""
        self.message_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.message_input.setFocus()
        self.save_config()

        assistant_text = self._get_latest_assistant_text()
        self._maybe_handle_agent_action(assistant_text)

        if self.active_task_mode == TASK_MODE_TRADE_DECISION:
            self._process_trade_decision_response()

    def _get_latest_assistant_text(self) -> str:
        for msg in reversed(self.chat_history):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _maybe_handle_agent_action(self, assistant_text: str):
        intent = AgentActionService.extract_intent(assistant_text)
        if intent is None:
            return

        detail = intent.reason or "AI 建议执行该动作"
        reply = QMessageBox.question(
            self,
            "确认执行智能体动作",
            f"智能体请求执行：{intent.label}\n\n原因：{detail}\n\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self.append_to_display("system", {
                "kind": "trade_decision_status",
                "title": "智能体动作已取消",
                "message": f"已取消执行：{intent.label}",
            })
            return

        self._execute_agent_action(intent.action, detail)

    def _execute_agent_action(self, action: str, reason: str = ""):
        broker_service = get_broker_session_service()
        title = ""
        success = False
        message = ""

        if action == "open_qmt":
            title = "启动 miniQMT"
            success, message, _status = broker_service.launch_client()
        elif action == "login_qmt":
            title = "登录 miniQMT"
            success, message, _status = broker_service.login_client()
        elif action == "close_qmt":
            title = "关闭 miniQMT"
            if broker_service.is_connected:
                broker_service.disconnect()
            success, message, _status = broker_service.close_client()
        elif action == "connect_broker":
            title = "连接券商"
            config = broker_service.get_config()
            qmt_path = config.get("qmt_path", "").strip()
            account = config.get("account", "").strip()
            if not qmt_path or not account:
                success = False
                message = "券商配置缺失，请先完善 broker_config.json 或在券商配置界面中填写。"
            else:
                success = broker_service.connect_async(qmt_path, account)
                message = "已开始建立券商连接，请稍候查看连接状态。" if success else "券商连接未能启动，请检查当前状态。"
        else:
            return

        status_title = title if title else "智能体动作执行"
        self.append_to_display("system", {
            "kind": "trade_decision_status",
            "title": status_title,
            "message": f"{message}" if not reason else f"{message}\n原因: {reason}",
        })
        if not success:
            QMessageBox.warning(self, status_title, message)

    def _process_trade_decision_response(self):
        """Extract structured decision from the LLM response and run risk checks."""
        assistant_text = ""
        for msg in reversed(self.chat_history):
            if msg.get("role") == "assistant":
                assistant_text = msg.get("content", "")
                break

        if not assistant_text:
            return

        decision = TradeDecisionExtractor.extract(assistant_text)
        if decision is None:
            self.append_to_display("system", {
                "kind": "trade_decision_error",
                "title": "决策提取失败",
                "message": "未能从 AI 回复中提取有效的交易决策 JSON，请检查回复内容或重新生成。",
            })
            return

        if not decision.symbol_code and self.current_context.symbol.is_available:
            decision.symbol_code = self.current_context.symbol.code
        if not decision.symbol_name and self.current_context.symbol.name:
            decision.symbol_name = self.current_context.symbol.name
        if decision.current_price <= 0 and self.current_context.symbol.latest_close > 0:
            decision.current_price = self.current_context.symbol.latest_close

        risk_result = self.risk_guard.evaluate(decision, self.current_context.broker)

        self._show_decision_card(decision, risk_result)

    def _show_decision_card(self, decision: TradeDecision, risk_result):
        """Display decision card in chat and optionally show approval dialog."""
        action_label = TRADE_ACTION_LABELS.get(decision.action, decision.action)
        risk_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(
            risk_result.overall_risk_level, "⚪"
        )

        card_payload = {
            "kind": "trade_decision_card",
            "title": f"{action_label} {decision.symbol_name}({decision.symbol_code})",
            "items": [
                {"title": "操作", "summary": action_label},
                {"title": "置信度", "summary": f"{decision.confidence:.0%}"},
                {"title": "目标价", "summary": f"{decision.target_price:.2f}" if decision.target_price > 0 else "-"},
                {"title": "止损价", "summary": f"{decision.stop_loss_price:.2f}" if decision.stop_loss_price > 0 else "-"},
                {"title": "建议仓位", "summary": f"{decision.position_pct:.0%}"},
                {"title": "风险评分", "summary": f"{decision.risk_score:.2f}"},
                {"title": "风控结果", "summary": f"{risk_icon} {risk_result.overall_risk_level.upper()}"},
            ],
        }
        if risk_result.blocked_reasons:
            card_payload["items"].append({
                "title": "⛔ 风控拦截",
                "summary": "; ".join(risk_result.blocked_reasons),
            })
        if risk_result.warnings:
            card_payload["items"].append({
                "title": "⚠ 风控警告",
                "summary": "; ".join(risk_result.warnings),
            })

        self.append_to_display("system", card_payload)
        self.chat_history.append({"role": "system_display", "content": card_payload})

        if decision.is_actionable:
            self._show_approval_dialog(decision, risk_result)

    def _show_approval_dialog(self, decision: TradeDecision, risk_result):
        """Show the trade decision approval dialog."""
        dialog = TradeDecisionApprovalDialog(decision, risk_result, parent=self)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            outcome = DecisionOutcome.APPROVED.value
            record = self.decision_tracker.save_decision(
                decision, risk_result, outcome,
                user_remark=dialog.remark_text,
            )
            self._execute_decision(decision, risk_result, record)
        else:
            outcome = DecisionOutcome.REJECTED_BY_USER.value
            self.decision_tracker.save_decision(
                decision, risk_result, outcome,
                user_remark=dialog.remark_text,
            )
            self.append_to_display("system", {
                "kind": "trade_decision_status",
                "title": "决策已驳回",
                "message": f"用户驳回了 {decision.action_label} {decision.symbol_name} 的交易决策",
            })

    def _execute_decision(self, decision: TradeDecision, risk_result, record):
        """Delegate order execution to TradingBridge via MainWindow."""
        main_window = self._find_main_window()
        if main_window is None or not hasattr(main_window, "trading_bridge"):
            self.append_to_display("system", {
                "kind": "trade_decision_status",
                "title": "决策已记录（未执行）",
                "message": "券商桥接不可用，决策已保存但未自动下单。可在交易窗口中手动执行。",
            })
            return

        bridge = main_window.trading_bridge
        if not hasattr(bridge, "execute_agent_decision"):
            self.append_to_display("system", {
                "kind": "trade_decision_status",
                "title": "决策已记录（未执行）",
                "message": "交易桥接暂不支持智能体下单，决策已保存。",
            })
            return

        result = bridge.execute_agent_decision(
            decision,
            risk_result=risk_result,
            decision_record_id=record.record_id,
        )
        if result.success:
            self.decision_tracker.update_outcome(
                record.record_id,
                outcome=DecisionOutcome.EXECUTED.value,
                broker_order_id=result.broker_order_id,
            )
            if decision.action in ("sell", "reduce"):
                self.decision_tracker.auto_close_by_symbol(
                    decision.symbol_code,
                    decision.current_price,
                    broker_order_id=result.broker_order_id,
                )
            self.append_to_display("system", {
                "kind": "trade_decision_status",
                "title": "下单成功",
                "message": result.message,
            })
        else:
            self.decision_tracker.update_outcome(
                record.record_id,
                outcome=DecisionOutcome.EXECUTION_FAILED.value,
            )
            self.append_to_display("system", {
                "kind": "trade_decision_status",
                "title": "下单失败",
                "message": result.message,
            })

    def _find_main_window(self):
        widget = self.parent()
        while widget is not None:
            if widget.__class__.__name__ == "MainWindow":
                return widget
            widget = widget.parent() if hasattr(widget, "parent") and callable(widget.parent) else None
        return None

    def eventFilter(self, obj, event):
        """处理输入框的 Enter 发送快捷键"""
        if obj is self.message_input and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                self.send_message()
                return True
        return super().eventFilter(obj, event)


class TradeDecisionApprovalDialog(QDialog):
    """Modal dialog for reviewing and approving/rejecting a trade decision."""

    def __init__(self, decision: TradeDecision, risk_result, parent=None):
        super().__init__(parent)
        self.decision = decision
        self.risk_result = risk_result
        self.remark_text = ""
        self.setWindowTitle("交易决策审批")
        self.setMinimumWidth(520)
        self.setMinimumHeight(480)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        action_label = TRADE_ACTION_LABELS.get(self.decision.action, self.decision.action)
        title = QLabel(f"📊 {action_label}  {self.decision.symbol_name}({self.decision.symbol_code})")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 4px 0;")
        layout.addWidget(title)

        info_group = QGroupBox("决策详情")
        info_layout = QFormLayout(info_group)
        info_layout.setSpacing(6)

        info_layout.addRow("操作方向:", QLabel(action_label))
        info_layout.addRow("当前价格:", QLabel(f"{self.decision.current_price:.2f}" if self.decision.current_price > 0 else "-"))
        info_layout.addRow("目标价:", QLabel(f"{self.decision.target_price:.2f}" if self.decision.target_price > 0 else "-"))
        info_layout.addRow("止损价:", QLabel(f"{self.decision.stop_loss_price:.2f}" if self.decision.stop_loss_price > 0 else "-"))

        ret = self.decision.expected_return_pct
        if ret is not None:
            color = "green" if ret > 0 else "red"
            info_layout.addRow("预期收益:", QLabel(f"<span style='color:{color}'>{ret:+.2f}%</span>"))

        loss = self.decision.max_loss_pct
        if loss is not None:
            info_layout.addRow("最大亏损:", QLabel(f"<span style='color:red'>{loss:.2f}%</span>"))

        info_layout.addRow("置信度:", QLabel(f"{self.decision.confidence:.0%}"))
        info_layout.addRow("建议仓位:", QLabel(f"{self.decision.position_pct:.0%}"))
        info_layout.addRow("风险评分:", QLabel(f"{self.decision.risk_score:.2f}"))
        info_layout.addRow("持有周期:", QLabel(self.decision.horizon_label))

        if self.decision.reasoning:
            reason_label = QLabel(self.decision.reasoning)
            reason_label.setWordWrap(True)
            info_layout.addRow("决策理由:", reason_label)

        if self.decision.invalidation:
            inv_label = QLabel(self.decision.invalidation)
            inv_label.setWordWrap(True)
            info_layout.addRow("失效条件:", inv_label)

        layout.addWidget(info_group)

        risk_group = QGroupBox("风控审核结果")
        risk_layout = QVBoxLayout(risk_group)

        risk_icon = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(
            self.risk_result.overall_risk_level, "⚪"
        )
        level_label = QLabel(f"{risk_icon} 综合风险等级: {self.risk_result.overall_risk_level.upper()}")
        level_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        risk_layout.addWidget(level_label)

        for check in self.risk_result.checks:
            icon = "✅" if check.passed else ("⛔" if check.level == "block" else "⚠️")
            check_label = QLabel(f"  {icon} {check.name}: {check.message}")
            check_label.setWordWrap(True)
            risk_layout.addWidget(check_label)

        if self.risk_result.blocked_reasons:
            block_label = QLabel("⛔ 风控拦截: " + "; ".join(self.risk_result.blocked_reasons))
            block_label.setStyleSheet("color: red; font-weight: bold;")
            block_label.setWordWrap(True)
            risk_layout.addWidget(block_label)

        layout.addWidget(risk_group)

        remark_label = QLabel("备注（可选）:")
        layout.addWidget(remark_label)
        self.remark_input = QPlainTextEdit()
        self.remark_input.setMaximumHeight(60)
        self.remark_input.setPlaceholderText("输入审批备注...")
        layout.addWidget(self.remark_input)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        reject_btn = QPushButton("驳回")
        reject_btn.setFixedWidth(100)
        reject_btn.clicked.connect(self._on_reject)
        btn_layout.addWidget(reject_btn)

        approve_btn = QPushButton("确认执行")
        approve_btn.setFixedWidth(120)
        approve_btn.setStyleSheet(
            "QPushButton { background-color: #0078d4; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 12px; }"
            "QPushButton:hover { background-color: #106ebe; }"
        )
        if not self.risk_result.passed:
            approve_btn.setEnabled(False)
            approve_btn.setToolTip("风控未通过，无法执行")
        approve_btn.clicked.connect(self._on_approve)
        btn_layout.addWidget(approve_btn)

        layout.addLayout(btn_layout)

    def _on_approve(self):
        self.remark_text = self.remark_input.toPlainText().strip()
        self.accept()

    def _on_reject(self):
        self.remark_text = self.remark_input.toPlainText().strip()
        self.reject()


class StockAnalysisThread(QThread):
    """Background thread for stock analysis API calls"""
    message_received = pyqtSignal(str, bool)  # content, is_error
    analysis_finished = pyqtSignal(str, str, str, bool)  # result, stock_code, stock_name, success

    def __init__(self, api_key, base_url, model, system_prompt, messages, stock_code, stock_name):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.messages = messages
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.full_result = ""

    def run(self):
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            # Build messages
            full_messages = [{"role": "system", "content": self.system_prompt}]
            full_messages.extend(self.messages)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    self.full_result += content
                    self.message_received.emit(content, False)
            
            logger.info(f"Stock analysis completed for {self.stock_code}. Length: {len(self.full_result)}")
            self.analysis_finished.emit(self.full_result, self.stock_code, self.stock_name, True)
            
        except Exception as e:
            logger.error(f"Stock analysis error: {e}")
            self.message_received.emit(f"\n错误: {str(e)}", True)
            self.analysis_finished.emit("", self.stock_code, self.stock_name, False)


class AnalysisHistoryDialog(QDialog):
    """Dialog to view and manage analysis history"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("历史分析记录")
        self.setMinimumSize(600, 500)
        self.setup_ui()
        self.load_history()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Filter row
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("筛选股票代码:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("输入股票代码筛选（留空显示全部）")
        self.filter_input.textChanged.connect(self.load_history)
        filter_layout.addWidget(self.filter_input)
        layout.addLayout(filter_layout)
        
        # History list
        self.history_list = QListWidget()
        self.history_list.setAlternatingRowColors(True)
        self.history_list.itemDoubleClicked.connect(self.view_detail)
        layout.addWidget(self.history_list)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        view_btn = QPushButton("查看详情")
        view_btn.clicked.connect(self.view_detail)
        btn_layout.addWidget(view_btn)
        
        copy_btn = QPushButton("复制内容")
        copy_btn.clicked.connect(self.copy_content)
        btn_layout.addWidget(copy_btn)
        
        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self.delete_record)
        btn_layout.addWidget(delete_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def load_history(self):
        """Load analysis history from files"""
        self.history_list.clear()
        
        filter_code = self.filter_input.text().strip() if hasattr(self, 'filter_input') else ""
        
        analyzer = get_analyzer()
        records = analyzer.get_analysis_history(stock_code=filter_code if filter_code else None)
        
        for record in records:
            item_text = f"{record['datetime_str']} - {record['stock_name']} ({record['stock_code']})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, record)
            self.history_list.addItem(item)
        
        if not records:
            item = QListWidgetItem("暂无历史记录")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.history_list.addItem(item)
    
    def get_selected_record(self):
        """Get currently selected record"""
        current = self.history_list.currentItem()
        if current:
            return current.data(Qt.ItemDataRole.UserRole)
        return None
    
    def view_detail(self):
        """View selected analysis detail"""
        record = self.get_selected_record()
        if not record:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        
        analyzer = get_analyzer()
        content = analyzer.read_analysis_result(record['filepath'])
        
        if content:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"分析详情 - {record['stock_name']} ({record['stock_code']})")
            dialog.setMinimumSize(700, 600)
            
            layout = QVBoxLayout(dialog)
            
            text_edit = QPlainTextEdit()
            text_edit.setPlainText(content)
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)
            
            close_btn = QPushButton("关闭")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.exec()
        else:
            QMessageBox.warning(self, "错误", "无法读取分析文件")
    
    def copy_content(self):
        """Copy selected analysis content to clipboard"""
        record = self.get_selected_record()
        if not record:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        
        analyzer = get_analyzer()
        content = analyzer.read_analysis_result(record['filepath'])
        
        if content:
            from PyQt6.QtWidgets import QApplication
            QApplication.clipboard().setText(content)
            QMessageBox.information(self, "成功", "分析内容已复制到剪贴板")
        else:
            QMessageBox.warning(self, "错误", "无法读取分析文件")
    
    def delete_record(self):
        """Delete selected analysis record"""
        record = self.get_selected_record()
        if not record:
            QMessageBox.information(self, "提示", "请先选择一条记录")
            return
        
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除 {record['stock_name']} ({record['stock_code']}) 的分析记录吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            analyzer = get_analyzer()
            if analyzer.delete_analysis_result(record['filepath']):
                self.load_history()
                QMessageBox.information(self, "成功", "记录已删除")
            else:
                QMessageBox.warning(self, "错误", "删除失败")


class GuideEditorDialog(QDialog):
    """Dialog to edit the analysis guide file"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑股票分析指导文件")
        self.setMinimumSize(800, 600)
        self.setup_ui()
        self.load_guide()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel("编辑分析指导文件，AI将根据此文件的方法论对股票进行分析：")
        layout.addWidget(info_label)
        
        # Text editor
        self.editor = QPlainTextEdit()
        self.editor.setStyleSheet("""
            QPlainTextEdit {
                font-family: 'Consolas', 'Monaco', 'Source Code Pro', monospace;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        layout.addWidget(self.editor)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_guide)
        btn_layout.addWidget(save_btn)
        
        reload_btn = QPushButton("重新加载")
        reload_btn.clicked.connect(self.load_guide)
        btn_layout.addWidget(reload_btn)
        
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
    
    def load_guide(self):
        """Load guide content from file"""
        analyzer = get_analyzer()
        content = analyzer.reload_guide()
        self.editor.setPlainText(content)
    
    def save_guide(self):
        """Save guide content to file"""
        content = self.editor.toPlainText()
        
        analyzer = get_analyzer()
        guide_path = analyzer.guide_path
        
        try:
            # Ensure directory exists
            guide_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(guide_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Reload the guide in analyzer
            analyzer.reload_guide()
            
            QMessageBox.information(self, "成功", "指导文件已保存")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")

