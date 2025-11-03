# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['ataraxai.hegemonikon_py', 'chromadb.telemetry.product.posthog', 'chromadb.api.rust']
hiddenimports += collect_submodules('fastapi')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('ataraxai')


a = Analysis(
    ['api.py'],
    pathex=[],
    binaries=[('/home/xenon/Atarax-AI/build/ataraxai/hegemonikon/hegemonikon_py.cpython-312-x86_64-linux-gnu.so', 'ataraxai'), ('/usr/lib/x86_64-linux-gnu/libpython3.12.so', '_internal')],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pytest', 'mypy', 'ruff', 'IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='api',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='api',
)
