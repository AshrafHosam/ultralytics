r"""Lab 5, C9 check: does timm install and run on this laptop?
Prints the environment, then runs a small model on CPU and (if available) on the GPU.
Run from the Lab 5 folder:  python scripts\\c9_timm_check.py > evidence\\c9_timm_check.txt 2>&1.
"""

import datetime
import platform
import sys
import time

print("date:", datetime.datetime.now().isoformat(timespec="seconds"))
print("os:", platform.platform())
print("python:", sys.version.split()[0], "| executable:", sys.executable)

import timm
import torch

print("torch:", torch.__version__, "| timm:", timm.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), "| torch cuda build:", torch.version.cuda)

MODEL = "resnet18"  # small model, as the catalog advises for laptop runs
print("models in timm:", len(timm.list_models()))


def run(device, pretrained=False):
    model = timm.create_model(MODEL, pretrained=pretrained).eval().to(device)
    x = torch.randn(2, 3, 224, 224, device=device)  # random input: checks the code runs, not accuracy
    with torch.no_grad():
        model(x)  # warm-up
        t = time.perf_counter()
        out = model(x)
        dt = time.perf_counter() - t
    print(f"[{device}] {MODEL} pretrained={pretrained}: output shape {tuple(out.shape)}, forward {dt * 1000:.1f} ms")


run("cpu")
if torch.cuda.is_available():
    run("cuda")
else:
    print("[cuda] skipped: no CUDA-enabled torch or no GPU visible")

try:
    run("cpu", pretrained=True)  # downloads resnet18 weights from the Hugging Face Hub
except Exception as e:
    print("pretrained download failed:", type(e).__name__, e)
