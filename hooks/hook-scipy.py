
from PyInstaller.utils.hooks import collect_all, collect_submodules

# �ռ�����scipy��ģ��
hiddenimports = collect_submodules('scipy')

# �ռ����ݺͶ������ļ�
datas, binaries, _ = collect_all('scipy')
