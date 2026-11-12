# Lab #5 - System Selection and Reproducibility Record

> Supporting files are in `lab5/` in the fork branch (`lab5/evidence/`, `lab5/scripts/`). Commands were run in my local `Lab 5` folder, where these folders are `evidence\` and `scripts\`, next to the cloned repo `ultralytics\` and the venv `.venv\` (not committed). Evidence logs in the branch have terminal colour codes removed.

## People and collaboration

- Working mode: [x] Solo  [ ] Pair
- Student A (name / GitHub): Ashraf Hossam / [AshrafHosam](https://github.com/AshrafHosam)
- Student B (name / GitHub; if pair): Not applicable
- Fork owner: AshrafHosam
- Collaborator invitation accepted? Not applicable
- Individual work differentiation: Not applicable (solo)

## First shortlist - catalogue only

| Candidate | Catalogue-based reason | One concern to investigate |
|---|---|---|
| A. timm (PyTorch Image Models): image classification models and training scripts | Core CV; Apache-2.0; training scripts | Accuracy needs labelled images; weights have separate terms |
| B. Segmentation Models PyTorch (SMP): U-Net and other segmentation models | Core CV; mostly MIT; CPU forward pass feasible | Needs real pixel masks; toy masks only test metrics |
| C. Ultralytics YOLO: detection, classification, segmentation, tracking, evaluation | Built-in validation; COCO8 demo with labels | AGPL-3.0; COCO8 only checks the pipeline |
| | | |

### Selection criteria (catalogue stage)

Included:
1. Computer vision only (my choice): catalogue systems #15–21.
2. Laptop-friendly: preferred the catalogue's **Core** label; only timm and SMP have it among CV systems.
3. A way to evaluate quality (Lab 8): added YOLO (Advanced) for its built-in validation and labelled COCO8 demo.
4. Solo shortlist size: 3 (handout asks for 2–3).

Excluded (catalogue risk notes):

| System | Reason |
|---|---|
| Anomalib | Advanced; large downloads; MVTec AD data is non-commercial and must not be uploaded |
| FiftyOne | Advanced; separate app/backend (Node/Yarn); an evaluation platform, not a model to test |
| TorchVision | Advanced; torch/torchvision versions must match; source build can be complex |
| MMDetection | Advanced; MMCV build/compiler constraints; some operators missing in CPU-only builds |
| 14 recommender/retrieval systems | Not computer vision |

## Evidence-based candidate assessment

Use Y = Yes; P = Partial; N = No; U = Unknown. Checked on 12 Nov 2026 (default branch, GitHub API, official docs). Raw numbers: `lab5/evidence/github_repo_stats_2026-11-12.json`.
Repos: [timm](https://github.com/huggingface/pytorch-image-models) · [SMP](https://github.com/qubvel-org/segmentation_models.pytorch) · [YOLO](https://github.com/ultralytics/ultralytics)

| Criterion | Candidate A + evidence (timm) | Candidate B + evidence (SMP) | Candidate C + evidence (YOLO) | Candidate D + evidence (pairs) |
|---|---|---|---|---|
| C1 Public repository / licence / data rights | **Y.** Public; Apache-2.0. ImageNet-trained weights for non-commercial research; some CC-BY-NC 4.0 ([licences](https://github.com/huggingface/pytorch-image-models#licenses)) | **Y.** Public; mostly MIT; some files differ, e.g. SegFormer encoder NVIDIA non-commercial ([LICENSES.md](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/licenses/LICENSES.md)) | **P.** Public; AGPL-3.0; Enterprise licence for commercial use ([LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.4.211/LICENSE)) | |
| C2 Pinned revision / runtime | **Y.** Tag v1.0.31 (3 Nov 2026); Python ≥3.10; venv + `pip install -e .` documented ([install](https://huggingface.co/docs/timm/installation)) | **P.** Tag v0.6.0 (26 Oct 2026); Python ≥3.11; no editable install in docs ([install](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/docs/install.rst)) | **P.** Tag v8.4.211 (11 Nov 2026); 3 releases on 10–11 Nov, so pinning is essential; Python ≥3.8; `pip install -e .` documented ([quickstart](https://docs.ultralytics.com/quickstart/)) | |
| C3 Nontrivial source + refactoring | **Y.** 304 Python files, 5.5 MB | **Y.** 91 Python files, 0.5 MB | **Y.** 232 Python files, 3.7 MB | |
| C4 Tests and expected-output **oracle** | **Y.** 27 Python files in `tests/`; oracle checks output shape | **Y.** 31 Python files in `tests/`; oracle checks output shape | **Y.** 12 Python files in `tests/`; oracle checks output shape | |
| C5 AI/recommender **ground-truth** source and terms | **P.** No labels in repo; built-in CIFAR-10/100 download; pretrained weights use ImageNet's 1,000 classes | **P.** No labels in repo; CamVid in examples; terms not checked | **P.** COCO8 (8 labelled images, demo); pretrained models trained on COCO and evaluated on COCO val2017 ([README](https://github.com/ultralytics/ultralytics#readme)); COCO terms not checked | |
| C6 Evaluation procedure and metrics | **Y.** `validate.py`: top-1 / top-5 accuracy | **Y.** `smp.metrics`: IoU, F1, precision, recall ([metrics](https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/docs/metrics.rst)) | **Y.** Val mode: mAP50, mAP50-95, precision, recall ([val](https://docs.ultralytics.com/modes/val/)) | |
| C7 Upstream GitHub history | **Y.** 3,393 commits, 222 contributors, 1,051 issues, 903 PRs, since 2019 | **Y.** 898 commits, 79 contributors, 725 issues, 661 PRs, since 2019 | **Y.** 5,796 commits, 494 contributors, 13,003 issues, 8,385 PRs, since 2022 | |
| C8 CI/DevOps possibility | **Y.** Tests on push/PR; Ubuntu only, Python 3.10 and 3.14 | **Y.** Tests on push/PR; Ubuntu, macOS, Windows; Python 3.12–3.14; coverage | **Y.** 12 workflows; CI on push/PR and daily; Ubuntu, macOS, Windows ([workflows](https://github.com/ultralytics/ultralytics/tree/main/.github/workflows)) | |
| C9 Laptop setup feasibility | **Y.** Runs on CPU (global Python, torch 2.11.0+cpu, ~40 ms) and GPU (venv, torch 2.15.0+cu132, 4.4 ms) ([CPU](lab5/evidence/c9_timm_check.txt), [GPU](lab5/evidence/c9_timm_check_gpu.txt)) | **U.** Not tried | **Y.** Runs on GPU (ultralytics 8.4.211, torch 2.15.0+cu132) ([evidence](lab5/evidence/c9_yolo_check.txt)) | |
| C10 Continuation to Labs 6-9 / credible question | **P.** Fits Labs 6, 7, 9; Lab 8 needs fine-tuning on a labelled set (e.g. CIFAR-10) | **P.** Lab 8 needs a dataset with checked terms | **Y.** Labs 6–9 all supported (see below); e.g. "How accurate is the pretrained detector on COCO val2017?" | |

### Ground-truth verification notes

- Candidate A (timm): functional test assertion: `assert outputs.shape[0] == batch_size` ([tests/test_models.py](https://github.com/huggingface/pytorch-image-models/blob/v1.0.31/tests/test_models.py)); label source: ImageNet-1k for pretrained weights; CIFAR-10/100 via torchvision in `timm/data/dataset_factory.py`; train/test protocol: `validate.py` on a dataset split (default "validation"); data terms: ImageNet non-commercial research ([licences](https://github.com/huggingface/pytorch-image-models#licenses)); CIFAR terms not checked.
- Candidate B (SMP): functional test assertion: `self.assertEqual(output.shape[1], classes)` ([tests/models/base.py](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.6.0/tests/models/base.py)); label source: CamVid pixel masks, cloned from `alexgkendall/SegNet-Tutorial` in the multiclass notebook ([examples](https://github.com/qubvel-org/segmentation_models.pytorch/tree/main/examples)); train/test protocol: set in the example notebooks (not checked); data terms: not checked.
- Candidate C (YOLO): functional test assertion: `assert backend(torch.zeros(1, 3, 32, 32)).shape == (1, 4, 30, 30)` ([tests/test_python.py](https://github.com/ultralytics/ultralytics/blob/v8.4.211/tests/test_python.py)); label source: COCO 2017 boxes, 80 classes ([coco.yaml](https://github.com/ultralytics/ultralytics/blob/v8.4.211/ultralytics/cfg/datasets/coco.yaml)); train/test protocol: models trained on COCO, evaluated on val2017, 5,000 images ([README](https://github.com/ultralytics/ultralytics#readme)); COCO8 = 4 train + 4 val images, pipeline check only ([coco8.yaml](https://github.com/ultralytics/ultralytics/blob/v8.4.211/ultralytics/cfg/datasets/coco8.yaml)); data terms: COCO terms not checked yet.
- Candidate D: Not applicable (solo).

## Decision

- Final SUT and short description: **Ultralytics YOLO** v8.4.211, a Python library for object detection and other vision tasks, with built-in validation.
- Why it is better suited than rejected alternatives: its pretrained models are trained on COCO and evaluated on COCO val2017, so predictions can be checked against real labels without training first (Lab 8). It also has 12 CI workflows, logging hooks and a large upstream history (Labs 7, 9), and runs on my laptop GPU.
- Rejected alternatives and reasons: timm (kept as backup): Lab 8 needs fine-tuning first, since its weights use ImageNet's 1,000 classes. SMP: no editable install in its docs, CamVid terms unchecked, not tried on my laptop.
- Backup SUT and trigger for switching: **timm** v1.0.31; switch if the YOLO fork cannot be installed or run from source, or if COCO data cannot be used.
- Biggest unresolved risk / mitigation: COCO data terms and AGPL-3.0 obligations; check the COCO terms before Lab 8, keep the fork public, and download only val2017 (about 1 GB) instead of the full 20.3 GB set.

## Repository provenance and pair access

- Original upstream repo URL: https://github.com/ultralytics/ultralytics
- Selected original tag (if any): `v8.4.211`, commit `d2b7876e79ad2e02fd77221622774b511a8f0cb3` (same SHA on the upstream tag and my branch, checked via GitHub API)
- Fork URL: https://github.com/AshrafHosam/ultralytics
- Working branch: [semester-investigation](https://github.com/AshrafHosam/ultralytics/tree/semester-investigation)
- Git and runtime versions: Windows 11 (10.0.26300), Python 3.13.1 (venv), torch 2.15.0+cu132, torchvision 0.29.2+cu132, ultralytics 8.4.211, NVIDIA RTX 3060 Laptop GPU 6 GB (driver 621.14); Git 2.35.1.windows.2
- Applicable code licence (link + note): [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/v8.4.211/LICENSE); the fork stays public; commercial use needs an Enterprise licence
- External dataset licence/limits: COCO8 (Ultralytics, first 8 images of COCO train2017) for pipeline checks only; COCO val2017 for Lab 8, terms to check; datasets and weights are not committed to the fork
- Partner collaborator invitation accepted? (pair only): Not applicable

## Actual execution - not just installation

| Check | Official documentation link | Exact local command / path + input | Observed output / short log + evidence link | Status |
|---|---|---|---|---|
| Install/build source-aware baseline | [Quickstart](https://docs.ultralytics.com/quickstart/) | In `Lab 5\ultralytics` (fork, tag v8.4.211): `..\.venv\Scripts\activate`, `pip uninstall -y ultralytics`, `pip install -e .` (torch 2.15.0+cu132 already in the venv) | `ultralytics imported from: ...\Lab 5\ultralytics\ultralytics`; GPU visible ([evidence](lab5/evidence/task4_yolo_fork.txt)) | Success |
| Functionality 1 (real behaviour) | [Predict mode](https://docs.ultralytics.com/modes/predict/) | `python scripts\c9_yolo_check.py`; model `yolo26n.pt`, input `ultralytics/assets/bus.jpg` | 5 boxes: bus, person ×4 ([evidence](lab5/evidence/task4_yolo_fork.txt)) | Success |
| Functionality 2 (different behaviour) | [Val mode](https://docs.ultralytics.com/modes/val/), [COCO8](https://docs.ultralytics.com/datasets/detect/coco8/) | same script; `model.val(data="coco8.yaml", imgsz=640)` on GPU | 4 images, 17 objects: P 0.552, R 0.961, mAP50 0.943, mAP50-95 0.669; 16.1 ms inference per image ([evidence](lab5/evidence/task4_yolo_fork.txt)) | Success |

- What functionality 1 shows: the pretrained detector finds and labels real objects in a real photo (inference).
- What functionality 2 shows: the evaluation procedure compares predictions with labelled boxes and reports mAP, precision and recall. With only 4 validation images this checks the pipeline, not real accuracy.
- Is the installed code from the local source fork, or only a separately installed wheel/image? From the local fork: editable install, and the script prints the import path inside the fork. An earlier run with the PyPI wheel 8.4.211 gave the same results ([evidence](lab5/evidence/c9_yolo_check.txt)).
- Issue found: running a script from `Lab 5` imported the repo folder `Lab 5\ultralytics` instead of the package (`ImportError ... unknown location`), because Python adds the script's folder to the import path. Fixed by keeping scripts in `Lab 5\scripts`.
- Blockers and fallback date/action: none; if the fork stops working, switch to timm (backup).

## Approval / action

- Instructor status: Pending
- Changes needed before Lab #6: check COCO data terms; push this file to the `semester-investigation` branch.
- Date of record: 2026-11-12

## Future-lab feasibility (LLM-assisted, verified)

Initial assessment by an LLM (Claude), checked against the repository and docs. ✅ = verified, ⚠️ = assumption still to test.

| Future lab | Suitable? | Possible investigation / tools | Main concern |
|---|---|---|---|
| #6 Static analysis / refactoring | Yes | ✅ 232 Python files, 12 test files to protect a refactoring. ⚠️ Lizard, Radon, Ruff (not run yet) | Large codebase: pick one module |
| #7 DevOps / CI/CD | Yes | ✅ 12 workflows ([workflows](https://github.com/ultralytics/ultralytics/tree/main/.github/workflows)); ✅ logging hooks for TensorBoard, MLflow, W&B, ClearML, Comet, DVC ([callbacks](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/utils/callbacks)). ⚠️ Workflows must be reviewed and enabled in the fork | Large CI; use a small part |
| #8 AI quality | Yes | ✅ Val mode metrics ([val](https://docs.ultralytics.com/modes/val/)); ✅ COCO val2017, 5,000 images (~1 GB per `coco.yaml`); ✅ ran on my GPU with COCO8 | COCO terms; download only val2017 |
| #9 Empirical study | Yes | ✅ 5,796 commits, 8,385 PRs, 13,003 issues (upstream, via GitHub API) | Very large: sample; PRs and issues are not in the fork |

Corrected or unsupported LLM suggestions: `coco.yaml` also downloads train2017 (19 GB), so it is not laptop-friendly as is; COCO128 comes from COCO train2017, so it is not a fair test set; COCO8 only checks the pipeline.

Decision: confident that YOLO can support Labs 6–9. Remaining risks: COCO terms, AGPL obligations, keeping the pinned version.
