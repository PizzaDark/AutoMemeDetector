# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# =========================================================
# 资源路径
# =========================================================
datas = [
    ('models', 'models'),
    ('sounds', 'sounds'),
    ('meme_detector.ico', '.')
]

# RapidOCR 可能会下载模型到用户目录，为了离线运行，
# 建议你运行一次脚本，找到它下载的 .onnx 文件 (通常在 site-packages/rapidocr_onnxruntime/models 或用户目录下)
# 然后把这些 .onnx 模型文件也打包进去。
# 这里使用 collect_all 尝试自动收集 rapidocr 库内的资源
tmp_ret = collect_all('rapidocr_onnxruntime')
datas += tmp_ret[0] 

# =========================================================
# 隐式导入
# =========================================================
hiddenimports = [
    'rapidocr_onnxruntime',
    'onnxruntime', 
    'pygame',
    'pygame.mixer',
    'keyboard',
    'pypinyin',
    'mss',
    'vosk',
    'sounddevice',
    'pyaudio',
    'pyaudiowpatch',
    'numpy',
    'cv2',
    'PIL',
    'PIL.Image',
    'cffi',
    'queue',
    'threading',
    'json',
    'winreg',
    'ctypes',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets'
]

# 收集 Vosk
tmp_ret_vosk = collect_all('vosk')
datas += tmp_ret_vosk[0]
hiddenimports += tmp_ret_vosk[2]

# 收集 cv2 (OpenCV)
try:
    tmp_ret_cv2 = collect_all('cv2')
    datas += tmp_ret_cv2[0]
    hiddenimports += tmp_ret_cv2[2]
except:
    pass  # 如果收集失败，继续执行

# 收集 PyAudioWPatch
try:
    tmp_ret_pyaudio = collect_all('pyaudiowpatch')
    datas += tmp_ret_pyaudio[0]
    hiddenimports += tmp_ret_pyaudio[2]
except:
    pass

# 收集 numpy (确保包含所有必要的二进制文件)
try:
    tmp_ret_numpy = collect_all('numpy')
    datas += tmp_ret_numpy[0]
except:
    pass

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'IPython', 'pandas', 'scipy', 'paddle', 'paddleocr', 'pytest', 'setuptools'], # 排除不需要的大型库
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AutoMeme',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False, # UPX压缩
    upx_exclude=['vcruntime140.dll', 'python3.dll', 'msvcp140.dll'],  # 排除系统DLL避免UPX压缩问题
    runtime_tmpdir=None,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='meme_detector.ico'
)