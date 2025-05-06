
from PyInstaller.utils.hooks import collect_all, collect_submodules

# 收集所有scipy.signal子模块
hiddenimports = collect_submodules('scipy.signal')

# 显式添加容易被遗漏的模块
hiddenimports.extend([
    'scipy.signal.windows',
    'scipy.signal.windows._windows',
    'scipy.linalg',
    'scipy.linalg._basic',
    'scipy.linalg._decomp',
    'scipy._lib._util',
    'scipy._lib._docscrape',
    'scipy.fft',
    'scipy.special'
])

# 收集数据和二进制文件
datas, binaries, _ = collect_all('scipy.signal')
