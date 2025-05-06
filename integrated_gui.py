import sys
import os
import pandas as pd
import shutil
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                          QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                          QProgressBar, QTextEdit, QFileDialog, QGroupBox, 
                          QFormLayout, QCheckBox, QComboBox, QTableWidget,
                          QTableWidgetItem, QHeaderView, QSplitter, QScrollArea,
                          QSpinBox, QTabWidget, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData, QTimer
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QColor
import logging
import psutil
import traceback

# 从原始的audioedit导入处理线程
from audioedit.gui import ProcessingThread

class FlacRenameThread(QThread):
    """FLAC文件重命名的后台线程"""
    progress_updated = pyqtSignal(float)
    log_updated = pyqtSignal(str)
    processing_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, excel_path, flac_folder, output_folder):
        super().__init__()
        self.excel_path = excel_path
        self.flac_folder = flac_folder
        self.output_folder = output_folder
    
    def run(self):
        try:
            # 确保输出文件夹存在
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                self.log_updated.emit(f"已创建输出文件夹: {self.output_folder}")
            
            # 读取Excel表格
            df = pd.read_excel(self.excel_path)
            
            # 确保表格中有必要的列
            if '伴奏ID' not in df.columns or 'AI干声文件名' not in df.columns:
                self.error_occurred.emit("表格中缺少'伴奏ID'或'AI干声文件名'列")
                return
            
            # 获取所有.flac文件并按创建时间排序
            flac_files = []
            for file in os.listdir(self.flac_folder):
                if file.lower().endswith('.flac'):
                    full_path = os.path.join(self.flac_folder, file)
                    creation_time = os.path.getctime(full_path)
                    flac_files.append((full_path, creation_time))
            
            # 按创建时间从早到晚排序
            flac_files.sort(key=lambda x: x[1])
            
            # 检查文件数量是否匹配
            if len(flac_files) != len(df):
                self.log_updated.emit(f"警告：flac文件数量({len(flac_files)})与表格行数({len(df)})不匹配")
            
            # 创建复制映射
            copy_map = []
            for i, (file_path, _) in enumerate(flac_files):
                if i < len(df):
                    _, ext = os.path.splitext(file_path)
                    new_name = df.iloc[i]['AI干声文件名'] + ext
                    new_path = os.path.join(self.output_folder, new_name)
                    copy_map.append((file_path, new_path))
            
            # 执行复制
            total_files = len(copy_map)
            for index, (old_path, new_path) in enumerate(copy_map):
                if os.path.exists(new_path):
                    self.log_updated.emit(f"警告：文件'{os.path.basename(new_path)}'已存在，跳过复制")
                    continue
                try:
                    shutil.copy2(old_path, new_path)
                    self.log_updated.emit(f"已复制: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
                except Exception as e:
                    self.log_updated.emit(f"复制失败: {os.path.basename(old_path)}, 错误: {str(e)}")
                
                # 更新进度条
                progress = (index + 1) / total_files * 100
                self.progress_updated.emit(progress)
                
            self.log_updated.emit("复制重命名完成！")
            self.processing_complete.emit()
            
        except Exception as e:
            error_message = f"处理过程中出现错误：{str(e)}"
            self.log_updated.emit(error_message)
            self.error_occurred.emit(error_message)

class IntegratedAppGUI(QMainWindow):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.setAcceptDrops(True)
        self.init_ui()
        
        # 添加性能监控定时器
        self.performance_timer = QTimer(self)
        self.performance_timer.timeout.connect(self.update_performance_stats)
        self.performance_timer.start(2000)  # 每2秒更新一次
        
    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle("音频处理工具集")
        self.setMinimumSize(800, 950)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 添加音频对齐裁剪页面
        self.audio_aligner_tab = self.create_audio_aligner_tab()
        self.tab_widget.addTab(self.audio_aligner_tab, "音频对齐裁剪")
        
        # 添加FLAC重命名页面
        self.flac_rename_tab = self.create_flac_rename_tab()
        self.tab_widget.addTab(self.flac_rename_tab, "导唱批量命名")
        
        # 添加状态栏性能信息
        self.statusBar().showMessage("就绪")
        self.performance_label = QLabel()
        self.statusBar().addPermanentWidget(self.performance_label)
        
    def create_audio_aligner_tab(self):
        """创建音频对齐裁剪标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 文件夹选择区域
        folder_group = QGroupBox("文件夹选择")
        folder_layout = QFormLayout()
        
        # 文件夹A选择行
        folder_a_layout = QHBoxLayout()
        self.folder_a_entry = QLineEdit()
        folder_a_button = QPushButton("浏览...")
        folder_a_button.clicked.connect(self.browse_folder_a)
        folder_a_layout.addWidget(self.folder_a_entry)
        folder_a_layout.addWidget(folder_a_button)
        folder_layout.addRow("音频文件夹A (参考长度):", folder_a_layout)
        
        # 文件夹B选择行
        folder_b_layout = QHBoxLayout()
        self.folder_b_entry = QLineEdit()
        folder_b_button = QPushButton("浏览...")
        folder_b_button.clicked.connect(self.browse_folder_b)
        folder_b_layout.addWidget(self.folder_b_entry)
        folder_b_layout.addWidget(folder_b_button)
        folder_layout.addRow("音频文件夹B (需裁剪):", folder_b_layout)
        
        # 输出文件夹选择行
        output_folder_layout = QHBoxLayout()
        self.output_folder_entry = QLineEdit()
        output_folder_button = QPushButton("浏览...")
        output_folder_button.clicked.connect(self.browse_output_folder)
        output_folder_layout.addWidget(self.output_folder_entry)
        output_folder_layout.addWidget(output_folder_button)
        folder_layout.addRow("输出文件夹:", output_folder_layout)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # 设置区域
        settings_group = QGroupBox("设置")
        settings_layout = QFormLayout()
        
        # 命名格式
        self.naming_format = QLineEdit()
        self.naming_format.setText("{file_id}-导唱.{extension}")
        settings_layout.addRow("输出文件命名格式:", self.naming_format)
        
        # 添加命名格式说明标签
        naming_help = QLabel("支持的变量: {original_name}=原文件名, {file_id}=文件ID, {extension}=扩展名, {timestamp}=时间戳")
        naming_help.setWordWrap(True)
        settings_layout.addRow("", naming_help)
        
        # 音频格式转换设置
        self.convert_to_wav = QCheckBox("转换为WAV格式")
        self.convert_to_wav.setChecked(True)
        settings_layout.addRow("格式转换:", self.convert_to_wav)
        
        # 采样率选择
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(["44100Hz", "48000Hz"])
        settings_layout.addRow("采样率:", self.sample_rate)
        
        # 声道数选择
        self.channels = QComboBox()
        self.channels.addItems(["立体声", "单声道"])
        settings_layout.addRow("声道数:", self.channels)
        
        # 位深选择
        self.bit_depth = QComboBox()
        self.bit_depth.addItems(["16bit", "24bit", "32bit"])
        settings_layout.addRow("位深:", self.bit_depth)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)
        
        # 性能状态显示
        self.performance_label = QLabel("系统状态: 准备就绪")
        layout.addWidget(self.performance_label)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        start_button = QPushButton("开始处理")
        start_button.clicked.connect(self.start_audio_processing)
        button_layout.addStretch()
        button_layout.addWidget(start_button)
        layout.addLayout(button_layout)
        
        # 创建结果区域分隔器
        results_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(results_splitter, 1)  # 分配更多空间给结果区域
        
        # 匹配结果框
        match_group = QGroupBox("匹配结果")
        match_layout = QVBoxLayout()
        
        # 匹配统计信息（使用滚动区域使其可以显示长文本）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        self.match_stats_label = QLabel("等待处理...")
        self.match_stats_label.setWordWrap(True)
        self.match_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        scroll_area.setWidget(self.match_stats_label)
        match_layout.addWidget(scroll_area)
        
        match_group.setLayout(match_layout)
        results_splitter.addWidget(match_group)
        
        # 处理结果表格
        results_group = QGroupBox("处理结果")
        results_layout = QVBoxLayout()
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(["参考文件", "处理文件", "时长差值(秒)", "处理结果"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        results_splitter.addWidget(results_group)
        
        # 设置分隔器初始大小比例
        results_splitter.setSizes([100, 300])
        
        # 保留一个隐藏的日志文本框，用于兼容日志处理器
        self.log_text = QTextEdit()
        self.log_text.setVisible(False)
        layout.addWidget(self.log_text)
        
        return tab
    
    def create_flac_rename_tab(self):
        """创建FLAC重命名标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Excel文件选择
        excel_group = QGroupBox("文件选择")
        excel_layout = QFormLayout()
        
        # Excel文件选择行
        excel_path_layout = QHBoxLayout()
        self.excel_path_entry = QLineEdit()
        excel_path_button = QPushButton("浏览...")
        excel_path_button.clicked.connect(self.browse_excel_file)
        excel_path_layout.addWidget(self.excel_path_entry)
        excel_path_layout.addWidget(excel_path_button)
        excel_layout.addRow("Excel表格路径:", excel_path_layout)
        
        # FLAC文件夹选择行
        flac_folder_layout = QHBoxLayout()
        self.flac_folder_entry = QLineEdit()
        flac_folder_button = QPushButton("浏览...")
        flac_folder_button.clicked.connect(self.browse_flac_folder)
        flac_folder_layout.addWidget(self.flac_folder_entry)
        flac_folder_layout.addWidget(flac_folder_button)
        excel_layout.addRow("FLAC文件夹路径:", flac_folder_layout)
        
        # 输出文件夹选择行
        rename_output_layout = QHBoxLayout()
        self.rename_output_entry = QLineEdit()
        rename_output_button = QPushButton("浏览...")
        rename_output_button.clicked.connect(self.browse_rename_output)
        rename_output_layout.addWidget(self.rename_output_entry)
        rename_output_layout.addWidget(rename_output_button)
        excel_layout.addRow("输出文件夹路径:", rename_output_layout)
        
        excel_group.setLayout(excel_layout)
        layout.addWidget(excel_group)
        
        # 进度区域
        rename_progress_group = QGroupBox("处理进度")
        rename_progress_layout = QVBoxLayout()
        
        self.rename_progress_bar = QProgressBar()
        rename_progress_layout.addWidget(self.rename_progress_bar)
        
        # 开始处理按钮
        rename_start_button = QPushButton("开始重命名")
        rename_start_button.clicked.connect(self.start_flac_rename)
        rename_progress_layout.addWidget(rename_start_button)
        
        rename_progress_group.setLayout(rename_progress_layout)
        layout.addWidget(rename_progress_group)
        
        # 日志区域
        rename_log_group = QGroupBox("处理日志")
        rename_log_layout = QVBoxLayout()
        
        self.rename_log_text = QTextEdit()
        self.rename_log_text.setReadOnly(True)
        rename_log_layout.addWidget(self.rename_log_text)
        
        rename_log_group.setLayout(rename_log_layout)
        layout.addWidget(rename_log_group)
        
        return tab
    
    def browse_folder_a(self):
        folder = QFileDialog.getExistingDirectory(self, "选择音频文件夹A")
        if folder:
            self.folder_a_entry.setText(folder)
    
    def browse_folder_b(self):
        folder = QFileDialog.getExistingDirectory(self, "选择音频文件夹B")
        if folder:
            self.folder_b_entry.setText(folder)
    
    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder_entry.setText(folder)
    
    def browse_excel_file(self):
        filename = QFileDialog.getOpenFileName(self, "选择Excel文件", "", "Excel文件 (*.xlsx)")[0]
        if filename:
            self.excel_path_entry.setText(filename)
    
    def browse_flac_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择FLAC文件夹")
        if folder:
            self.flac_folder_entry.setText(folder)
    
    def browse_rename_output(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.rename_output_entry.setText(folder)
    
    def log(self, message):
        """记录日志到当前活动的Tab页"""
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index == 0:  # 音频对齐裁剪Tab
            self.log_text.append(message)
            self.log_text.ensureCursorVisible()
        elif current_tab_index == 1:  # FLAC重命名Tab
            self.rename_log_text.append(message)
            self.rename_log_text.ensureCursorVisible()
    
    def update_performance_stats(self):
        """更新性能统计信息"""
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_percent = process.cpu_percent()
        stats_text = f"内存: {memory_usage:.1f} MB | CPU: {cpu_percent:.1f}%"
        self.performance_label.setText(stats_text)
    
    def start_audio_processing(self):
        # 获取用户输入
        folder_a = self.folder_a_entry.text()
        folder_b = self.folder_b_entry.text()
        output_folder = self.output_folder_entry.text()
        naming_format = self.naming_format.text()
        
        # 验证输入
        if not folder_a or not folder_b or not output_folder:
            QMessageBox.warning(self, "输入错误", "请选择所有必要的文件夹!")
            return
        
        # 禁用处理按钮，防止重复点击
        start_button = self.sender()
        start_button.setEnabled(False)
        start_button.setText("处理中...")
        
        # 重置进度条
        self.progress_bar.setValue(0)
        
        # 清空结果表格和匹配信息
        self.results_table.setRowCount(0)
        self.match_stats_label.setText("正在分析文件...")
        
        # 设置音频处理参数
        self.processor.convert_to_wav = self.convert_to_wav.isChecked()
        self.processor.default_sample_rate = int(self.sample_rate.currentText().replace('Hz', ''))
        self.processor.default_channels = 1 if "单声道" in self.channels.currentText() else 2
        bit_depth_text = self.bit_depth.currentText()
        if "24" in bit_depth_text:
            self.processor.default_bit_depth = 24
        elif "32" in bit_depth_text:
            self.processor.default_bit_depth = 32
        else:
            self.processor.default_bit_depth = 16
        
        # 开始性能更新
        self.performance_timer.start(1000)  # 处理时更频繁地更新
        
        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(
            self.processor, folder_a, folder_b, output_folder, naming_format
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)
        self.processing_thread.matching_info_updated.connect(self.update_matching_info)
        self.processing_thread.file_processed.connect(self.file_processed)
        self.processing_thread.start()
    
    def update_progress(self, value):
        """更新进度条"""
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index == 0:  # 音频对齐裁剪Tab
            self.progress_bar.setValue(int(value * 100))
        elif current_tab_index == 1:  # FLAC重命名Tab
            self.rename_progress_bar.setValue(int(value))
    
    def update_matching_info(self, matched_files, unmatched_files):
        """更新匹配信息区域"""
        # 更新匹配统计信息
        matched_count = len(matched_files)
        unmatched_a_count = len(unmatched_files['unmatched_a'])
        unmatched_b_count = len(unmatched_files['unmatched_b'])
        
        # 设置匹配统计文本
        match_info = f"匹配成功: {matched_count} 个文件\n"
        
        # 文件夹A中未匹配文件
        if unmatched_a_count > 0:
            match_info += f"文件夹A中未匹配: {unmatched_a_count} 个文件: "
            match_info += ", ".join(unmatched_files['unmatched_a'])
            match_info += "\n"
        else:
            match_info += "文件夹A中未匹配: 0 个文件\n"
        
        # 文件夹B中未匹配文件
        if unmatched_b_count > 0:
            match_info += f"文件夹B中未匹配: {unmatched_b_count} 个文件: "
            match_info += ", ".join(unmatched_files['unmatched_b'])
        else:
            match_info += "文件夹B中未匹配: 0 个文件"
        
        self.match_stats_label.setText(match_info)
    
    def file_processed(self, result):
        """处理单个文件结果的显示"""
        # 如果结果包含错误信息，则显示错误
        if 'error' in result:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(result['file']))
            self.results_table.setItem(row, 1, QTableWidgetItem(result.get('matched_with', '未知')))
            self.results_table.setItem(row, 2, QTableWidgetItem("错误"))
            
            error_item = QTableWidgetItem(result['error'])
            error_item.setBackground(QColor(255, 200, 200))  # 淡红色背景
            self.results_table.setItem(row, 3, error_item)
            return
        
        # 添加成功处理的文件结果到表格
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # 参考文件
        self.results_table.setItem(row, 0, QTableWidgetItem(result['file']))
        
        # 处理文件
        self.results_table.setItem(row, 1, QTableWidgetItem(result.get('matched_with', '未知')))
        
        # 时长差值
        diff = result['original_duration'] - result['new_duration']
        diff_text = f"{diff:.2f}"
        diff_item = QTableWidgetItem(diff_text)
        # 根据差值大小设置颜色
        if abs(diff) > 1.0:
            diff_item.setBackground(QColor(255, 235, 156))  # 黄色
        self.results_table.setItem(row, 2, diff_item)
        
        # 处理结果
        if result['operation'] == 'trimmed':
            result_text = f"已裁剪: {result['original_duration']:.2f}秒 -> {result['new_duration']:.2f}秒"
            result_item = QTableWidgetItem(result_text)
            result_item.setBackground(QColor(200, 230, 200))  # 淡绿色
        elif result['operation'] == 'extended':
            result_text = f"已延长: {result['original_duration']:.2f}秒 -> {result['new_duration']:.2f}秒"
            result_item = QTableWidgetItem(result_text)
            result_item.setBackground(QColor(200, 200, 255))  # 淡蓝色
        else:  # unchanged
            result_text = f"保持原样: {result['original_duration']:.2f}秒"
            result_item = QTableWidgetItem(result_text)
        self.results_table.setItem(row, 3, result_item)
        
        # 滚动到最新行，但不要每次都滚动，避免频繁刷新
        if row % 5 == 0:  # 每5行滚动一次
            self.results_table.scrollToBottom()
        
        # 立即处理事件，保持UI响应
        QApplication.processEvents()
    
    def processing_finished(self, results):
        # 处理完成后更新状态
        self.log("\n处理完成!")
        self.log(f"共处理 {len(results)} 个文件")
        
        # 更新匹配信息标签，添加处理完成提示
        current_text = self.match_stats_label.text()
        self.match_stats_label.setText(current_text + "\n\n处理完成!")
        
        # 恢复性能更新频率
        self.performance_timer.start(2000)
        
        # 重新启用处理按钮
        for button in self.findChildren(QPushButton):
            if button.text() == "开始处理":
                button.setEnabled(True)
        
        # 结果排序
        self.results_table.sortItems(0)  # 按第一列排序
        
        # 完成提示
        QMessageBox.information(self, "处理完成", f"成功处理了 {len(results)} 个文件")
    
    def processing_error(self, error_message):
        self.log(f"处理过程中出错: {error_message}")
        
        # 启用开始按钮
        for button in self.findChildren(QPushButton):
            if button.text() == "开始处理":
                button.setEnabled(True)
        
        # 更新匹配信息
        self.match_stats_label.setText(f"处理错误: {error_message}")
        
        # 在表格中显示错误信息
        self.results_table.setRowCount(1)
        self.results_table.setItem(0, 0, QTableWidgetItem("处理错误"))
        self.results_table.setItem(0, 3, QTableWidgetItem(error_message))
    
    def start_flac_rename(self):
        """开始FLAC重命名处理"""
        # 获取用户输入
        excel_path = self.excel_path_entry.text()
        flac_folder = self.flac_folder_entry.text()
        output_folder = self.rename_output_entry.text()
        
        # 输入验证
        if not all([excel_path, flac_folder, output_folder]):
            QMessageBox.warning(self, "输入错误", "请填写所有必要的输入字段")
            return
        
        # 验证文件和文件夹
        if not os.path.exists(excel_path):
            QMessageBox.warning(self, "路径错误", f"Excel文件不存在: {excel_path}")
            return
        if not os.path.exists(flac_folder):
            QMessageBox.warning(self, "路径错误", f"FLAC文件夹不存在: {flac_folder}")
            return
        
        # 禁用开始按钮
        start_button = self.sender()
        start_button.setEnabled(False)
        
        # 清空日志
        self.rename_log_text.clear()
        
        # 创建并启动重命名线程
        self.rename_thread = FlacRenameThread(excel_path, flac_folder, output_folder)
        self.rename_thread.progress_updated.connect(self.update_progress)
        self.rename_thread.log_updated.connect(self.rename_log_text.append)
        self.rename_thread.processing_complete.connect(self.rename_processing_finished)
        self.rename_thread.error_occurred.connect(self.rename_processing_error)
        
        self.rename_log_text.append("开始处理FLAC文件...")
        self.rename_thread.start()
    
    def rename_processing_finished(self):
        """FLAC重命名处理完成后更新状态"""
        QMessageBox.information(self, "完成", "FLAC文件重命名任务已完成！")
        self.rename_progress_bar.setValue(100)
        
        # 启用开始按钮
        for button in self.findChildren(QPushButton):
            if button.text() == "开始重命名":
                button.setEnabled(True)
    
    def rename_processing_error(self, error_message):
        """FLAC重命名处理错误"""
        QMessageBox.critical(self, "处理错误", error_message)
        
        # 启用开始按钮
        for button in self.findChildren(QPushButton):
            if button.text() == "开始重命名":
                button.setEnabled(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """处理拖放事件"""
        urls = event.mimeData().urls()
        if not urls:
            return
            
        # 获取当前活动的标签页
        current_tab_index = self.tab_widget.currentIndex()
        
        # 获取当前鼠标位置下的控件
        pos = event.pos()
        target_widget = self.childAt(pos)
        
        # 根据不同的标签页处理拖放
        if current_tab_index == 0:  # 音频对齐裁剪Tab
            # 检查是否有文件夹被拖放
            for url in urls:
                path = url.toLocalFile()
                if os.path.isdir(path):
                    # 根据拖放位置决定填充哪个输入框，无论是否已有内容
                    if target_widget == self.folder_a_entry or (target_widget and target_widget.parent() == self.folder_a_entry):
                        self.folder_a_entry.setText(path)
                    elif target_widget == self.folder_b_entry or (target_widget and target_widget.parent() == self.folder_b_entry):
                        self.folder_b_entry.setText(path)
                    elif target_widget == self.output_folder_entry or (target_widget and target_widget.parent() == self.output_folder_entry):
                        self.output_folder_entry.setText(path)
                    else:
                        # 如果没有拖放到特定输入框，则按顺序填充空输入框
                        if not self.folder_a_entry.text():
                            self.folder_a_entry.setText(path)
                        elif not self.folder_b_entry.text():
                            self.folder_b_entry.setText(path)
                        elif not self.output_folder_entry.text():
                            self.output_folder_entry.setText(path)
                    break
        
        elif current_tab_index == 1:  # FLAC重命名Tab
            for url in urls:
                path = url.toLocalFile()
                # 检查是否是Excel文件
                if os.path.isfile(path) and path.lower().endswith(('.xlsx', '.xls')):
                    # Excel文件总是更新到Excel路径输入框
                    self.excel_path_entry.setText(path)
                # 检查是否是文件夹
                elif os.path.isdir(path):
                    # 根据拖放位置决定填充哪个输入框，无论是否已有内容
                    if target_widget == self.flac_folder_entry or (target_widget and target_widget.parent() == self.flac_folder_entry):
                        self.flac_folder_entry.setText(path)
                    elif target_widget == self.rename_output_entry or (target_widget and target_widget.parent() == self.rename_output_entry):
                        self.rename_output_entry.setText(path)
                    else:
                        # 如果没有拖放到特定输入框，则按顺序填充空输入框
                        if not self.flac_folder_entry.text():
                            self.flac_folder_entry.setText(path)
                        elif not self.rename_output_entry.text():
                            self.rename_output_entry.setText(path)
                    break 