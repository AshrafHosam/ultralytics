# Lab 7: check the confidence rule of non_max_suppression on fixed boxes (CPU only, no model needed).
import torch

from ultralytics.utils.nms import non_max_suppression

# Normal NMS input: 4 boxes as (center x, center y, width, height) with one confidence each.
# A = 0.90, B = 0.80 (overlaps A), C = 0.25 (exactly the threshold), D = 0.10
BOXES = torch.tensor([[50, 50, 40, 40], [52, 52, 40, 40], [200, 200, 40, 40], [350, 350, 40, 40]], dtype=torch.float32)
NMS_INPUT = torch.cat([BOXES.T, torch.tensor([[0.90, 0.80, 0.25, 0.10]])])[None]

# End-to-end input (default for YOLO26 models): rows of (x1, y1, x2, y2, confidence, class)
E2E_INPUT = torch.tensor([[[10, 10, 50, 50, 0.90, 0], [100, 100, 140, 140, 0.25, 0], [200, 200, 240, 240, 0.10, 0]]])


def kept(pred):
    """Run NMS with the default threshold 0.25 and return the confidences of the kept boxes."""
    scores = [round(s, 2) for s in non_max_suppression(pred, conf_thres=0.25)[0][:, 4].tolist()]
    print("kept confidences:", scores)
    return scores


def test_box_at_threshold():
    """Expected result for the chosen case: the box exactly at 0.25 is kept (only boxes below are dropped)."""
    for pred in (NMS_INPUT, E2E_INPUT):
        assert kept(pred) == [0.90, 0.25]  # new result
        assert kept(pred) != [0.90]  # the old result no longer holds


def test_unchanged_rules():
    """Must stay true in every version: duplicates removed, low boxes dropped, best box unchanged."""
    out = non_max_suppression(NMS_INPUT, conf_thres=0.25)[0]
    assert 0.80 not in kept(NMS_INPUT)  # B is removed as a duplicate of A
    assert 0.10 not in kept(NMS_INPUT) and 0.10 not in kept(E2E_INPUT)  # D is below the threshold
    assert out[0, :4].tolist() == [30, 30, 70, 70]  # A keeps its box (converted to x1, y1, x2, y2)
