r"""Lab 5: does Ultralytics YOLO install and run on this laptop? (C9 and Task 4)
Keeps downloads (weights, COCO8 data, results) inside Lab 5\\yolo_check.
Uses the GPU if available, otherwise the CPU.
Run from the Lab 5 folder:  python scripts\\c9_yolo_check.py > evidence\\c9_yolo_check.txt 2>&1
The script lives in scripts\\ because the cloned repo folder Lab 5\\ultralytics would otherwise
shadow the installed package (Python adds the script's own folder to the import path).
"""

import datetime
import os
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "yolo_check"  # Lab 5\yolo_check
HERE.mkdir(exist_ok=True)
os.chdir(HERE)  # model weights download into the current folder

print("date:", datetime.datetime.now().isoformat(timespec="seconds"))
print("os:", platform.platform())
print("python:", sys.version.split()[0], "| executable:", sys.executable)

import torch

import ultralytics
from ultralytics import YOLO, settings
from ultralytics.utils import ASSETS

print("torch:", torch.__version__, "| ultralytics:", ultralytics.__version__)
print(
    "ultralytics imported from:", Path(ultralytics.__file__).parent
)  # shows whether the fork (editable install) is used
print("cuda available:", torch.cuda.is_available())
DEVICE = 0 if torch.cuda.is_available() else "cpu"  # use the GPU when torch can see it
print("device:", DEVICE)

# Note: this updates Ultralytics' own settings file so its folders point into Lab 5\yolo_check
settings.update(
    {"datasets_dir": str(HERE / "datasets"), "weights_dir": str(HERE / "weights"), "runs_dir": str(HERE / "runs")}
)
print("datasets_dir:", settings["datasets_dir"])

model = YOLO("yolo26n.pt")  # smallest model from the README example

# Functionality 1: detect objects in the sample image shipped with the package
res = model(str(ASSETS / "bus.jpg"), device=DEVICE, verbose=False)[0]
names = [res.names[int(c)] for c in res.boxes.cls]
print("predict bus.jpg:", len(names), "boxes:", names)

# Functionality 2: validation on COCO8 (8 labeled images; checks the pipeline, not real accuracy)
m = model.val(data="coco8.yaml", device=DEVICE, imgsz=640, plots=False, verbose=False)
print(f"val coco8: mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
