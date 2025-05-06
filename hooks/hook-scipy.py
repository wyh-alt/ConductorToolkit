
from PyInstaller.utils.hooks import collect_all, collect_submodules

# 收集所有scipy子模块
hiddenimports = collect_submodules('scipy')

# 收集数据和二进制文件
datas, binaries, _ = collect_all('scipy')
