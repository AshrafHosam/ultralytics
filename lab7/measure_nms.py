# Lab 7: print CPU time and memory of non_max_suppression on a fixed, realistic-sized YOLO output.
# Same input every time (fixed seed): 1 image, 8400 boxes, 80 classes (the size of a 640x640 YOLO output).
import resource
import sys
import time

import torch

from ultralytics.utils.nms import non_max_suppression

torch.manual_seed(0)
xy = torch.rand(1, 2, 8400) * 640  # box centers
wh = torch.rand(1, 2, 8400) * 90 + 10  # box sizes, 10 to 100 pixels
scores = torch.rand(1, 80, 8400) * 0.3  # class scores, up to 0.3
pred = torch.cat([xy, wh, scores], 1)

CALLS = 20
non_max_suppression(pred, conf_thres=0.25)  # warm-up
wall, cpu = time.perf_counter(), time.process_time()
for _ in range(CALLS):
    kept = non_max_suppression(pred, conf_thres=0.25)[0].shape[0]
wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux reports KB
label = sys.argv[1] if len(sys.argv) > 1 else "this version"
print(
    f"{label}: kept {kept} boxes | time per call {wall / CALLS * 1000:.1f} ms | "
    f"CPU time {cpu:.2f} s | CPU {cpu / wall * 100:.0f}% | peak memory {peak_mb:.0f} MB"
)
