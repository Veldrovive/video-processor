# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

site_packages = 'C:\\Users\\Aidan Windows\\anaconda3\\envs\\vidProc\\lib\\site-packages'

a = Analysis(['main.py'],
             pathex=['C:\\Users\\Aidan Windows\\Documents\\video-processor'],
             binaries=[],
             datas=[
                ('icons', 'icons'),
                ('uis\\*.qml', 'uis'),
                ('landmark_detection\models', 'landmark_detection\models')
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
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='main',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
