import PyInstaller.__main__
import os
import shutil
import sys

def build_executable():
    """构建可执行文件"""
    print("开始构建整合版音频处理工具...")
    
    # 定义应用名称
    app_name = "音频处理工具集"
    
    # 清理旧的构建文件
    for cleanup_dir in ['build', 'dist']:
        if os.path.exists(cleanup_dir):
            print(f"清理目录: {cleanup_dir}")
            shutil.rmtree(cleanup_dir)
    
    # 图标路径
    icon_path = 'audioedit/app_icon.ico'
    if not os.path.exists(icon_path):
        print(f"警告: 未找到图标文件: {icon_path}")
        icon_path = None
    
    # 构建参数
    build_args = [
        'integrated_app.py',  # 主程序文件
        '--noconfirm',        # 不确认覆盖
        '--clean',            # 清理构建文件
        '--name', app_name,   # 应用名称
        '--onedir',           # 单目录模式
        '--noconsole',        # 不显示控制台
    ]
    
    # 添加图标 (如果存在)
    if icon_path:
        build_args.extend(['--icon', icon_path])
    
    # 添加必要的数据文件
    build_args.extend([
        '--add-data', f'audioedit{os.pathsep}audioedit',  # 包含原始音频编辑模块
    ])
    
    # 执行构建
    print("执行PyInstaller命令...")
    PyInstaller.__main__.run(build_args)
    
    print(f"构建完成! 可执行文件位于 dist/{app_name}/ 目录")

if __name__ == "__main__":
    build_executable() 