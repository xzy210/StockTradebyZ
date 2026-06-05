# index_list_widget.py - Index list widget
"""
Index list widget for displaying and selecting indices
"""
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QLineEdit, QPushButton,
    QApplication, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

# Import index service
try:
    from ..services.index_service import fetch_index_data, update_all_indices
except ImportError:
    from trading_app.services.market_data.index_service import fetch_index_data, update_all_indices

from common.data_portal import get_data_portal


class IndexUpdateThread(QThread):
    """Thread for updating index data"""
    progress = pyqtSignal(int, int, str, bool)  # current, total, code, success
    finished_signal = pyqtSignal(dict)  # results
    
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
    
    def run(self):
        results = update_all_indices(
            self.data_dir,
            progress_callback=lambda cur, tot, code, succ: self.progress.emit(cur, tot, code, succ)
        )
        self.finished_signal.emit(results)


class IndexListWidget(QWidget):
    """Index list widget"""
    
    # Signal: Index selected (code, name)
    indexSelected = pyqtSignal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Index data
        self.index_list = get_data_portal().list_assets(asset_type="index", include_status=False)
        self.index_name_map = get_data_portal().get_name_map(asset_type="index")
        self.filtered_list = []
        
        # Update thread
        self.update_thread = None
        
        self.setupUI()
        self.populate_list()
    
    def setupUI(self):
        """Setup UI layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        # Search box
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索指数...")
        self.search_input.textChanged.connect(self.on_search_changed)
        search_layout.addWidget(self.search_input)
        
        # Update button
        self.update_btn = QPushButton("更新")
        self.update_btn.setToolTip("更新所有指数数据")
        self.update_btn.clicked.connect(self.update_index_data)
        self.update_btn.setMaximumWidth(60)
        search_layout.addWidget(self.update_btn)
        
        layout.addLayout(search_layout)
        
        # Index list
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.currentItemChanged.connect(self.on_item_changed)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3c3c3c;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QListWidget::item:hover:!selected {
                background-color: #3c3c3c;
            }
            QListWidget::item:alternate {
                background-color: #252525;
            }
        """)
        layout.addWidget(self.list_widget)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)
    
    def populate_list(self, filter_text: str = ""):
        """Populate the list with indices"""
        self.list_widget.clear()
        self.filtered_list = []
        
        filter_text = filter_text.lower()
        
        for item in self.index_list:
            code = item["code"]
            name = item["name"]
            
            # Apply filter
            if filter_text:
                if filter_text not in code.lower() and filter_text not in name.lower():
                    continue
            
            self.filtered_list.append(code)
            
            # Create list item
            display_text = f"{code}  {name}"
            list_item = QListWidgetItem(display_text)
            list_item.setData(Qt.ItemDataRole.UserRole, code)
            self.list_widget.addItem(list_item)
        
        self.status_label.setText(f"共 {len(self.filtered_list)} 只指数")
    
    def on_search_changed(self, text: str):
        """Handle search text change"""
        self.populate_list(text)
    
    def on_item_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle item selection change"""
        if current is None:
            return
        
        code = current.data(Qt.ItemDataRole.UserRole)
        name = self.index_name_map.get(code, "")
        self.indexSelected.emit(code, name)
    
    def select_index(self, code: str):
        """Select an index by code"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == code:
                self.list_widget.setCurrentItem(item)
                return True
        return False
    
    def get_selected_index(self) -> Optional[str]:
        """Get currently selected index code"""
        item = self.list_widget.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None
    
    def select_previous(self):
        """Select previous item in list"""
        current_row = self.list_widget.currentRow()
        if current_row > 0:
            self.list_widget.setCurrentRow(current_row - 1)
    
    def select_next(self):
        """Select next item in list"""
        current_row = self.list_widget.currentRow()
        if current_row < self.list_widget.count() - 1:
            self.list_widget.setCurrentRow(current_row + 1)
    
    def update_index_data(self):
        """Update all index data from network"""
        reply = QMessageBox.question(
            self, "确认更新",
            "即将从网络更新所有指数数据，这可能需要一些时间。\n是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self.update_btn.setEnabled(False)
        self.status_label.setText("正在更新指数数据...")
        
        # Get data_dir from parent window
        data_dir = "../data"
        try:
            # Try to get data_dir from main window
            main_window = self.window()
            if hasattr(main_window, 'data_dir'):
                data_dir = main_window.data_dir
        except:
            pass
        
        self.update_thread = IndexUpdateThread(data_dir)
        self.update_thread.progress.connect(self.on_update_progress)
        self.update_thread.finished_signal.connect(self.on_update_finished)
        self.update_thread.start()
    
    def on_update_progress(self, current, total, code, success):
        """Handle update progress"""
        name = self.index_name_map.get(code, code)
        status = "✓" if success else "✗"
        self.status_label.setText(f"更新中 ({current}/{total}): {name} {status}")
        QApplication.processEvents()
    
    def on_update_finished(self, results):
        """Handle update completion"""
        self.update_btn.setEnabled(True)
        
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        self.status_label.setText(f"更新完成: {success_count}/{total_count} 成功")
        
        QMessageBox.information(
            self, "更新完成",
            f"指数数据更新完成\n成功: {success_count}/{total_count}"
        )
