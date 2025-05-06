
from PyInstaller.utils.hooks import collect_all, collect_submodules

# �ռ�����scipy.signal��ģ��
hiddenimports = collect_submodules('scipy.signal')

# ��ʽ������ױ���©��ģ��
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

# �ռ����ݺͶ������ļ�
datas, binaries, _ = collect_all('scipy.signal')
