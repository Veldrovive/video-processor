# -*- mode: python ; coding: utf-8 -*-

from os import path

block_cipher = None

site_packages = 'C:\\Users\\Aidan Windows\\anaconda3\\envs\\vidProc\\lib\\site-packages'

a = Analysis(['main.py'],
             pathex=['C:\\Users\\Aidan Windows\\Documents\\video-processor'],
             binaries=[],
             datas=[
                ('icons', 'icons'),
                ('uis\\*.qml', 'uis'),
                ('landmark_detection\models', 'landmark_detection\models'),
                (path.join(site_packages,"torch"), "torch"),
                (path.join(site_packages,"torchvision"), "torchvision")
             ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main')
