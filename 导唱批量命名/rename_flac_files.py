import os
import pandas as pd
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class FlacRenameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FLAC文件重命名工具")
        self.root.geometry("600x400")
        
        # 创建主框架
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Excel文件选择
        ttk.Label(main_frame, text="Excel表格路径:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.excel_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.excel_path, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(main_frame, text="浏览", command=self.select_excel).grid(row=0, column=2, padx=5, pady=5)
        
        # FLAC文件夹选择
        ttk.Label(main_frame, text="FLAC文件夹路径:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.flac_folder = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.flac_folder, width=50).grid(row=1, column=1, pady=5)
        ttk.Button(main_frame, text="浏览", command=self.select_flac_folder).grid(row=1, column=2, padx=5, pady=5)
        
        # 输出文件夹选择
        ttk.Label(main_frame, text="输出文件夹路径:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_folder = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_folder, width=50).grid(row=2, column=1, pady=5)
        ttk.Button(main_frame, text="浏览", command=self.select_output_folder).grid(row=2, column=2, padx=5, pady=5)
        
        # 进度显示
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(main_frame, length=400, mode='determinate', variable=self.progress_var)
        self.progress.grid(row=3, column=0, columnspan=3, pady=20)
        
        # 日志显示
        self.log_text = tk.Text(main_frame, height=10, width=60)
        self.log_text.grid(row=4, column=0, columnspan=3, pady=5)
        
        # 开始按钮
        ttk.Button(main_frame, text="开始重命名", command=self.start_rename).grid(row=5, column=1, pady=10)
        
    def select_excel(self):
        filename = filedialog.askopenfilename(filetypes=[("Excel文件", "*.xlsx")])
        if filename:
            self.excel_path.set(filename)
            
    def select_flac_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.flac_folder.set(folder)
            
    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder.set(folder)
            
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def start_rename(self):
        # 检查所有路径是否已选择
        if not all([self.excel_path.get(), self.flac_folder.get(), self.output_folder.get()]):
            messagebox.showerror("错误", "请选择所有必要的路径")
            return
            
        try:
            # 确保输出文件夹存在
            if not os.path.exists(self.output_folder.get()):
                os.makedirs(self.output_folder.get())
                self.log(f"已创建输出文件夹: {self.output_folder.get()}")
            
            # 读取Excel表格
            df = pd.read_excel(self.excel_path.get())
            
            # 确保表格中有必要的列
            if '伴奏ID' not in df.columns or 'AI干声文件名' not in df.columns:
                messagebox.showerror("错误", "表格中缺少'伴奏ID'或'AI干声文件名'列")
                return
            
            # 获取所有.flac文件并按修改时间排序（而不是创建时间）
            flac_files = []
            for file in os.listdir(self.flac_folder.get()):
                if file.lower().endswith('.flac'):
                    full_path = os.path.join(self.flac_folder.get(), file)
                    # 使用修改时间而不是创建时间
                    modified_time = os.path.getmtime(full_path)
                    flac_files.append((full_path, modified_time))
            
            # 按修改时间从早到晚排序
            flac_files.sort(key=lambda x: x[1])
            
            # 检查文件数量是否匹配
            if len(flac_files) != len(df):
                self.log(f"警告：flac文件数量({len(flac_files)})与表格行数({len(df)})不匹配")
            
            # 创建复制映射
            copy_map = []
            for i, (file_path, _) in enumerate(flac_files):
                if i < len(df):
                    _, ext = os.path.splitext(file_path)
                    new_name = df.iloc[i]['AI干声文件名'] + ext
                    new_path = os.path.join(self.output_folder.get(), new_name)
                    copy_map.append((file_path, new_path))
            
            # 执行复制
            total_files = len(copy_map)
            for index, (old_path, new_path) in enumerate(copy_map):
                if os.path.exists(new_path):
                    self.log(f"警告：文件'{new_path}'已存在，跳过复制")
                    continue
                try:
                    shutil.copy2(old_path, new_path)
                    self.log(f"已复制: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
                except Exception as e:
                    self.log(f"复制失败: {os.path.basename(old_path)}, 错误: {str(e)}")
                
                # 更新进度条
                progress = (index + 1) / total_files * 100
                self.progress_var.set(progress)
                
            self.log("复制重命名完成！")
            messagebox.showinfo("完成", "文件重命名完成！")
            
        except Exception as e:
            messagebox.showerror("错误", f"处理过程中出现错误：{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FlacRenameApp(root)
    root.mainloop()