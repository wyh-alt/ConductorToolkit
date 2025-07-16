import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QProgressBar, QTextEdit, QFileDialog, QGroupBox, 
                             QFormLayout, QCheckBox, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter, QScrollArea,
                             QSpinBox, QTabWidget, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData, QTimer
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import logging
from datetime import datetime
import psutil
import traceback

class ProcessingThread(QThread):
    """处理音频的后台线程"""
    progress_updated = pyqtSignal(float)
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    matching_info_updated = pyqtSignal(dict, dict)
    file_processed = pyqtSignal(dict)  # 添加单个文件处理完成的信号
    
    def __init__(self, processor, folder_a, folder_b, output_folder, naming_format, high_precision=True):
        super().__init__()
        self.processor = processor
        self.folder_a = folder_a
        self.folder_b = folder_b
        self.output_folder = output_folder
        self.naming_format = naming_format
        self.high_precision = high_precision
        self.memory_threshold = 90  # 内存使用阈值，达到此百分比时释放内存
        
    def free_memory(self):
        """尝试释放内存"""
        import gc
        gc.collect()  # 触发垃圾回收
        
    def run(self):
        try:
            # 先释放内存，确保开始处理前有足够内存
            self.free_memory()
            
            # 获取文件夹信息并发送匹配信息
            files_a = self.processor.get_audio_files(self.folder_a)
            files_b = self.processor.get_audio_files(self.folder_b)
            
            # 创建ID到文件路径的映射
            files_a_dict = {}
            files_b_dict = {}
            
            for file_a in files_a:
                file_id = self.processor.extract_file_id(file_a)
                if file_id not in files_a_dict:
                    files_a_dict[file_id] = file_a
            
            for file_b in files_b:
                file_id = self.processor.extract_file_id(file_b)
                if file_id not in files_b_dict:
                    files_b_dict[file_id] = file_b
            
            # 找出匹配和未匹配的文件
            common_ids = set(files_a_dict.keys()) & set(files_b_dict.keys())
            unmatched_a = set(files_a_dict.keys()) - common_ids
            unmatched_b = set(files_b_dict.keys()) - common_ids
            
            # 创建匹配和未匹配的详细信息
            matched_files = {}
            for file_id in common_ids:
                matched_files[file_id] = {
                    'file_a': os.path.basename(files_a_dict[file_id]),
                    'file_b': os.path.basename(files_b_dict[file_id])
                }
            
            unmatched_files = {
                'unmatched_a': [os.path.basename(files_a_dict[file_id]) for file_id in unmatched_a],
                'unmatched_b': [os.path.basename(files_b_dict[file_id]) for file_id in unmatched_b]
            }
            
            # 发送匹配信息更新
            self.matching_info_updated.emit(matched_files, unmatched_files)
            
            # 自定义处理文件的实现，不使用processor.process_folders
            results = []
            total = len(common_ids)
            processed = 0
            
            # 确保输出文件夹存在
            os.makedirs(self.output_folder, exist_ok=True)
            
            # 将ID列表转换为列表以便处理
            id_list = list(common_ids)
            
            # 分批处理，避免内存不足
            batch_size = 3  # 小批量处理，减小内存压力
            for i in range(0, len(id_list), batch_size):
                # 获取当前批次的ID
                batch_ids = id_list[i:i+batch_size]
                
                for file_id in batch_ids:
                    file_a = files_a_dict[file_id]
                    file_b = files_b_dict[file_id]
                    base_name_a = os.path.basename(file_a)
                    base_name_b = os.path.basename(file_b)
                    
                    # 获取A文件的长度
                    try:
                        duration_a = self.processor.get_audio_duration(file_a)
                        
                        # 生成输出文件名
                        original_name = os.path.splitext(base_name_b)[0]
                        extension = os.path.splitext(file_b)[1][1:]
                        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                        
                        # 支持多种命名变量
                        output_filename = self.naming_format.format(
                            original_name=original_name,
                            extension=extension,
                            file_id=file_id,
                            timestamp=timestamp
                        )
                        output_path = os.path.join(self.output_folder, output_filename)
                        
                        # 裁剪B文件
                        try:
                            original_duration = self.processor.get_audio_duration(file_b)
                            operation = self.processor.trim_audio(file_b, duration_a, output_path, high_precision=self.high_precision)
                            
                            result = {
                                'file': base_name_a,
                                'matched_with': base_name_b,
                                'original_duration': original_duration,
                                'new_duration': duration_a,
                                'operation': operation
                            }
                            
                            # 发送单个文件处理结果信号
                            self.file_processed.emit(result)
                            
                            # 添加到结果列表
                            results.append(result)
                        except MemoryError:
                            error_msg = f"处理文件 {base_name_b} 时内存不足"
                            error_result = {
                                'file': base_name_a,
                                'matched_with': base_name_b,
                                'error': error_msg
                            }
                            self.file_processed.emit(error_result)
                            self.free_memory()  # 尝试释放内存
                        except Exception as e:
                            # 处理裁剪错误
                            error_result = {
                                'file': base_name_a,
                                'matched_with': base_name_b,
                                'error': str(e)
                            }
                            self.file_processed.emit(error_result)
                    except Exception as e:
                        # 处理获取长度错误
                        error_result = {
                            'file': base_name_a,
                            'matched_with': base_name_b,
                            'error': str(e)
                        }
                        self.file_processed.emit(error_result)
                    
                    processed += 1
                    self.update_progress(processed / total)
                
                # 每完成一批，释放内存
                self.free_memory()
                
                # 检查内存使用率，如果过高，强制释放
                mem_percent = psutil.virtual_memory().percent
                if mem_percent > self.memory_threshold:
                    self.free_memory()
            
            # 处理完成后发送所有结果
            self.processing_complete.emit(results)
        except MemoryError:
            self.error_occurred.emit("处理过程中内存不足。请关闭其他应用程序或增加系统虚拟内存，然后重试。")
        except Exception as e:
            self.error_occurred.emit(f"处理出错: {str(e)}")
    
    def update_progress(self, value):
        self.progress_updated.emit(value)


class AudioAlignerGUI(QMainWindow):
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
        self.setWindowTitle("音频匹配处理工具")
        self.setMinimumSize(800, 600)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # ===== 主要功能选项卡 =====
        main_tab = QWidget()
        main_tab_layout = QVBoxLayout(main_tab)
        
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
        main_tab_layout.addWidget(folder_group)
        
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
        self.bit_depth.addItems(["16 Bit", "24 Bit", "32 Bit Float"])
        settings_layout.addRow("位深:", self.bit_depth)
        
        # 高精度模式选择
        self.high_precision = QCheckBox("高精度模式（减少误差，处理更慢）")
        self.high_precision.setChecked(True)  # 默认开启高精度模式
        self.high_precision.setToolTip("启用高精度模式可以显著减少音频时长误差，但处理速度会变慢")
        settings_layout.addRow("精度设置:", self.high_precision)
        
        settings_group.setLayout(settings_layout)
        main_tab_layout.addWidget(settings_group)
        
        # 进度条
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignCenter)
        main_tab_layout.addWidget(self.progress)
        
        # 性能状态显示
        self.performance_label = QLabel("系统状态: 准备就绪")
        main_tab_layout.addWidget(self.performance_label)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        self.process_button = QPushButton("开始处理")
        self.process_button.clicked.connect(self.start_processing)
        button_layout.addStretch()
        button_layout.addWidget(self.process_button)
        main_tab_layout.addLayout(button_layout)
        
        # 创建结果区域分隔器
        results_splitter = QSplitter(Qt.Vertical)
        main_tab_layout.addWidget(results_splitter, 1)  # 分配更多空间给结果区域
        
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
        self.results_table.setHorizontalHeaderLabels(["参考文件", "处理文件", "时长差值(秒,微秒级)", "处理结果"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        results_splitter.addWidget(results_group)
        
        # 设置分隔器初始大小比例
        results_splitter.setSizes([100, 300])
        
        # 添加主选项卡
        tab_widget.addTab(main_tab, "主要功能")
        
        # ===== 高级设置选项卡 =====
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        # 性能设置组
        performance_group = QGroupBox("性能设置")
        performance_layout = QFormLayout()
        
        # CPU核心数设置
        self.cpu_cores = QSpinBox()
        self.cpu_cores.setMinimum(1)
        self.cpu_cores.setMaximum(os.cpu_count() or 8)
        self.cpu_cores.setValue(self.processor.max_workers)
        self.cpu_cores.valueChanged.connect(self.update_processor_settings)
        performance_layout.addRow("并行处理线程数:", self.cpu_cores)
        
        # 批处理大小设置
        self.batch_size = QSpinBox()
        self.batch_size.setMinimum(1)
        self.batch_size.setMaximum(100)
        self.batch_size.setValue(self.processor.batch_size)
        self.batch_size.valueChanged.connect(self.update_processor_settings)
        performance_layout.addRow("批处理大小:", self.batch_size)
        
        # 内存优化设置
        self.memory_optimize = QCheckBox("启用内存优化")
        self.memory_optimize.setChecked(self.processor.optimize_memory)
        self.memory_optimize.stateChanged.connect(self.update_processor_settings)
        performance_layout.addRow("内存优化:", self.memory_optimize)
        
        # 缓存设置
        self.enable_cache = QCheckBox("启用音频长度缓存")
        self.enable_cache.setChecked(True)
        self.enable_cache.stateChanged.connect(self.update_processor_settings)
        performance_layout.addRow("音频长度缓存:", self.enable_cache)
        
        performance_group.setLayout(performance_layout)
        advanced_layout.addWidget(performance_group)
        
        # 系统资源监控
        monitor_group = QGroupBox("系统资源监控")
        monitor_layout = QVBoxLayout()
        
        self.resource_label = QLabel("CPU使用率: 0%, 内存使用率: 0%, 可用内存: 0MB")
        monitor_layout.addWidget(self.resource_label)
        
        monitor_group.setLayout(monitor_layout)
        advanced_layout.addWidget(monitor_group)
        
        # 添加空白占位
        advanced_layout.addStretch(1)
        
        # 添加高级设置选项卡
        tab_widget.addTab(advanced_tab, "高级设置")
        
        # 保留一个隐藏的日志文本框，用于兼容日志处理器
        self.log_text = QTextEdit()
        self.log_text.setVisible(False)
        main_layout.addWidget(self.log_text)
    
    def update_processor_settings(self):
        """更新处理器设置"""
        self.processor.max_workers = self.cpu_cores.value()
        self.processor.batch_size = self.batch_size.value()
        self.processor.optimize_memory = self.memory_optimize.isChecked()
        
        # 清除缓存如果禁用了缓存
        if not self.enable_cache.isChecked():
            self.processor.get_audio_duration.cache_clear()
    
    def update_performance_stats(self):
        """更新性能统计信息"""
        try:
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            mem_percent = memory.percent
            available_mb = memory.available / (1024 * 1024)
            
            # 更新资源标签
            self.resource_label.setText(
                f"CPU使用率: {cpu_percent}%, 内存使用率: {mem_percent}%, 可用内存: {available_mb:.0f}MB"
            )
            
            # 如果正在处理，也更新性能标签
            if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
                self.performance_label.setText(
                    f"处理中... CPU: {cpu_percent}%, 内存: {mem_percent}%"
                )
            else:
                self.performance_label.setText("系统状态: 准备就绪")
        except Exception as e:
            logging.error(f"更新性能统计时出错: {str(e)}")
    
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
    
    def log(self, message):
        # 仅在调试时将日志添加到隐藏的文本框
        self.log_text.append(message)
    
    def start_processing(self):
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
        self.process_button.setEnabled(False)
        self.process_button.setText("处理中...")
        
        # 重置进度条
        self.progress.setValue(0)
        
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
        
        # 高精度模式
        self.processor.high_precision = self.high_precision.isChecked()
        
        # 开始性能更新
        self.performance_timer.start(1000)  # 处理时更频繁地更新
        
        # 创建并启动处理线程
        self.processing_thread = ProcessingThread(
            self.processor, folder_a, folder_b, output_folder, naming_format,
            high_precision=self.high_precision.isChecked()
        )
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)
        self.processing_thread.matching_info_updated.connect(self.update_matching_info)
        self.processing_thread.file_processed.connect(self.file_processed)
        self.processing_thread.start()
    
    def update_progress(self, value):
        self.progress.setValue(int(value * 100))
    
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
        diff_text = f"{diff:.5f}"  # 显示到微秒级精度（5位小数，约0.01ms精度）
        diff_item = QTableWidgetItem(diff_text)
        # 根据差值大小设置颜色 - 调整为微秒级阈值
        if abs(diff) > 0.001:  # 1毫秒阈值
            diff_item.setBackground(QColor(255, 235, 156))  # 黄色
        elif abs(diff) > 0.0001:  # 0.1毫秒阈值
            diff_item.setBackground(QColor(255, 245, 200))  # 浅黄色
        self.results_table.setItem(row, 2, diff_item)
        
        # 处理结果
        if result['operation'] == 'trimmed':
            result_text = f"已裁剪: {result['original_duration']:.5f}秒 -> {result['new_duration']:.5f}秒"
            result_item = QTableWidgetItem(result_text)
            result_item.setBackground(QColor(200, 230, 200))  # 淡绿色
        elif result['operation'] == 'extended':
            result_text = f"已延长: {result['original_duration']:.5f}秒 -> {result['new_duration']:.5f}秒"
            result_item = QTableWidgetItem(result_text)
            result_item.setBackground(QColor(200, 200, 255))  # 淡蓝色
        else:  # unchanged
            result_text = f"保持原样: {result['original_duration']:.5f}秒"
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
        self.process_button.setEnabled(True)
        self.process_button.setText("开始处理")
        
        # 结果排序
        self.results_table.sortItems(0)  # 按第一列排序
        
        # 完成提示
        QMessageBox.information(self, "处理完成", f"成功处理了 {len(results)} 个文件")
    
    def processing_error(self, error_message):
        self.log(f"处理过程中出错: {error_message}")
        self.process_button.setEnabled(True)
        
        # 更新匹配信息
        self.match_stats_label.setText(f"处理错误: {error_message}")
        
        # 在表格中显示错误信息
        self.results_table.setRowCount(1)
        self.results_table.setItem(0, 0, QTableWidgetItem("处理错误"))
        self.results_table.setItem(0, 3, QTableWidgetItem(error_message))
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if len(urls) > 0:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                # 根据拖放位置决定填充哪个输入框
                pos = event.pos()
                widget = self.childAt(pos)
                if widget == self.folder_a_entry or widget.parent() == self.folder_a_entry:
                    self.folder_a_entry.setText(path)
                elif widget == self.folder_b_entry or widget.parent() == self.folder_b_entry:
                    self.folder_b_entry.setText(path)
                elif widget == self.output_folder_entry or widget.parent() == self.output_folder_entry:
                    self.output_folder_entry.setText(path)
                else:
                    # 默认填充到文件夹A
                    self.folder_a_entry.setText(path)