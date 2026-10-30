import subprocess,sys
from pathlib import Path
if sys.platform != "win32" or sys.maxsize <= 2**32 or sys.version_info[:3] < (3,12,10):
    raise SystemExit("Build using 64-bit Windows Python 3.12.10 or newer.")
root=Path(__file__).resolve().parent
command=[sys.executable,"-m","PyInstaller","--noconfirm","--clean","--onefile","--windowed","--name","YoloDatasetViewer"]
for unused in ("numpy","torch","cv2","IPython","PySide6","PyQt5"):
    command.extend(["--exclude-module",unused])
command.append(str(root/"dataset_viewer.py"))
subprocess.run(command,cwd=root,check=True)
