import sys
import os
import subprocess
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from audio_processor import AudioProcessor
from gui import AudioAlignerGUI
import logging
import asyncio

if os.name == 'nt':
    _old_popen = subprocess.Popen
    def _new_popen(*args, **kwargs):
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        return _old_popen(*args, **kwargs)
    subprocess.Popen = _new_popen

class LogSignaler(QObject):
    log_signal = pyqtSignal(str)

class GUILogHandler(logging.Handler):
    def __init__(self, gui_window):
        super().__init__()
        self.gui_window = gui_window
        self.signaler = LogSignaler()
        self.signaler.log_signal.connect(self.gui_window.log)

    def emit(self, record):
        msg = self.format(record)
        self.signaler.log_signal.emit(msg)

def main():
    try:
        # 禁用控制台窗口 - 使用更精确的方法
        if hasattr(sys, 'frozen'):
            # 仅在Windows上使用更精确的控制台窗口隐藏方法
            import ctypes
            kernel32 = ctypes.WinDLL('kernel32')
            user32 = ctypes.WinDLL('user32')
            # 获取当前进程的控制台窗口句柄
            hwnd = kernel32.GetConsoleWindow()
            if hwnd:
                # 仅当存在控制台窗口时才隐藏它
                user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
            
        # 创建应用程序
        app = QApplication(sys.argv)
        
        # 创建音频处理器实例
        processor = AudioProcessor()
        
        # 创建GUI窗口
        window = AudioAlignerGUI(processor)
        window.show()
        
        # 配置日志处理器 - 先移除所有现有处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]: 
            root_logger.removeHandler(handler)
        
        # 添加GUI日志处理器
        gui_handler = GUILogHandler(window)
        gui_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root_logger.addHandler(gui_handler)
        root_logger.setLevel(logging.INFO)
        
        # 解决可能的exec函数问题
        try:
            exit_code = app.exec_()
            sys.exit(exit_code)
        except TypeError as e:
            if "argument 'code' must be code, not str" in str(e):
                # 如果遇到特定类型错误，尝试使用数值退出码
                print("捕获到exec错误，尝试使用数值退出码")
                sys.exit(0)
            else:
                raise
    except Exception as e:
        print(f"Application failed to start: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()