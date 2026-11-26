# Lab #7 — From Code to Operation: Investigation Record

**Team:** individual  
**Student(s) / GitHub handle(s):** Ashraf Hossam — [@AshrafHosam](https://github.com/AshrafHosam)  
**Fork / working branch:** https://github.com/AshrafHosam/ultralytics/tree/semester-investigation (work done on `lab7-ashraf`)  
**Upstream project and pinned version:** https://github.com/ultralytics/ultralytics, tag `v8.4.211`, commit [`d2b7876`](https://github.com/AshrafHosam/ultralytics/commit/d2b7876e79ad2e02fd77221622774b511a8f0cb3)

**In short:** I installed my fork on a clean GitHub Ubuntu machine and ran YOLO's box filter on a few fixed boxes. Then I changed one rule: a box exactly at the confidence threshold is now kept. The pipeline showed the old and the new result, and the checks for things that must not change passed both times.

## 1. Stakeholder, goal and selected functionality

**Stakeholder concern:** Sam (release engineer) wants to know that my fork's code really installs and runs on a clean machine, and that a change in detection results shows up in the checks.

**GQM goal:** Analyze **YOLO's box filter (`non_max_suppression`) in my fork** for the purpose of **evaluation**, with respect to **running it on a clean machine and catching a behavior change**, from the viewpoint of **Sam, the release engineer**, in the context of **my fork of Ultralytics v8.4.211 on a GitHub Ubuntu 24.04 runner**.

| Question | Question in your own words                                                              | Evidence / measurement and why it helps                                                              |
| -------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Q1       | Does my fork's code install and run on a clean machine, and can I prove which code ran? | Green run, same commit SHA pushed and checked out, and the place `ultralytics` was loaded from.      |
| Q2       | When I change the rule, do the checks show it, and do the other rules still hold?       | Kept boxes before and after for the same input, and the result of the unchanged checks in both runs. |

**Selected function/component:** [`non_max_suppression` in `ultralytics/utils/nms.py`](https://github.com/AshrafHosam/ultralytics/blob/b64bae8182a903fee4fa67e44a0d0d4765453a7a/ultralytics/utils/nms.py#L13-L157)  
**Why this represents the software's purpose:** YOLO finds objects in images. This function turns the model's raw boxes into the final detections: it drops boxes with low confidence and removes duplicate boxes on the same object. YOLO's predictor calls it on every detection result.  
**Representative input and baseline expectation:** A few fixed boxes, with the default threshold 0.25:

- normal input: A 0.90, B 0.80 (on top of A), C 0.25 (exactly the threshold), D 0.10;
- "end-to-end" input (the format used by the default YOLO26 models): 0.90, 0.25, 0.10.

Today only the 0.90 box is kept in both, because the code keeps only boxes **above** 0.25.

**Planned small behavior change:** Also keep boxes **at** 0.25 (`>` becomes `>=`). The project's own docs say boxes "with confidence **below** this threshold will be disregarded" ([predict-args.md](https://github.com/AshrafHosam/ultralytics/blob/b64bae8182a903fee4fa67e44a0d0d4765453a7a/docs/macros/predict-args.md)), so a box at 0.25 should stay. The change makes the code match the docs.  
**What must still work unchanged?** B is still removed as a duplicate of A, D is still dropped, A keeps its box, and the project's own NMS test still passes.

## 2. Baseline: deploy and execute on GitHub

| Evidence                                                                 | Baseline                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline commit SHA and link                                             | [`b64bae8182a903fee4fa67e44a0d0d4765453a7a`](https://github.com/AshrafHosam/ultralytics/commit/b64bae8182a903fee4fa67e44a0d0d4765453a7a)                                                                                                                                      |
| Actions workflow file link                                               | [`lab7-deploy.yml`](https://github.com/AshrafHosam/ultralytics/blob/b64bae8182a903fee4fa67e44a0d0d4765453a7a/.github/workflows/lab7-deploy.yml)                                                                                                                               |
| Ubuntu/runtime and relevant dependency versions                          | Ubuntu 24.04.5, Python 3.12.16, torch 2.15.0 (CPU), pytest 9.2.0                                                                                                                                                                                                              |
| How the fork's code is installed/built and verified as the executed code | `pip install -e .` from the checked-out commit. The log shows the same SHA for "pushed" and "checked out", and `ultralytics 8.4.211 from /home/runner/work/ultralytics/...`, which is the checkout, not a downloaded package. It also prints the rule lines (`> conf_thres`). |
| Functional case executed (command/script/test or local endpoint)         | `pytest tests/test_lab7_nms.py` ([test](https://github.com/AshrafHosam/ultralytics/blob/b64bae8182a903fee4fa67e44a0d0d4765453a7a/tests/test_lab7_nms.py)) plus the project's test `test_nms_end2end_classes_before_max_det`                                                   |
| Expected vs observed behavior                                            | expected `[0.9]`, got `kept confidences: [0.9]` for both inputs                                                                                                                                                                                                               |
| Check result and meaningful measurement(s), if any                       | 3 passed in 0.57 s                                                                                                                                                                                                                                                            |
| Exact GitHub Actions run URL                                             | https://github.com/AshrafHosam/ultralytics/actions/runs/45826190609                                                                                                                                                                                                           |

**What does this establish, and what does it not establish?** My fork's exact commit installs and runs on a clean Ubuntu machine and gives the expected result for my boxes. It says nothing about a real trained model on real images, because I used neither.

## 3. Controlled change and verification

**Source/function changed:** [`ultralytics/utils/nms.py`](https://github.com/AshrafHosam/ultralytics/blob/319a5dbaf72a5f8844e8710b3d4e7b4047f60881/ultralytics/utils/nms.py#L74-L123): the 4 confidence checks (lines 74, 86, 117, 123) went from `>` to `>=`, and the description of `conf_thres` now says boxes below it are dropped.  
**Reason for change:** A box exactly at the threshold was dropped, although the docs say only boxes below it are.  
**Code-change commit(s):** https://github.com/AshrafHosam/ultralytics/commit/319a5dbaf72a5f8844e8710b3d4e7b4047f60881

| Item                                                   | Before change                                                             | After change                                                        |
| ------------------------------------------------------ | ------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Same input/case                                        | A 0.90, B 0.80, C 0.25, D 0.10; end-to-end 0.90, 0.25, 0.10               | same                                                                |
| Expected behavior under each version's stated contract | keep boxes above 0.25: `[0.9]`                                            | keep boxes at or above 0.25: `[0.9, 0.25]`                          |
| Observed behavior on GitHub runner                     | `kept confidences: [0.9]`                                                 | `kept confidences: [0.9, 0.25]`                                     |
| Automated case check/result                            | expects `[0.9]`: PASSED                                                   | expects `[0.9, 0.25]` and "no longer `[0.9]`": PASSED               |
| Unchanged regression/invariant check                   | duplicate removed, low box dropped, A unchanged, project NMS test: PASSED | same tests, not edited: PASSED                                      |
| Exact Actions run URL                                  | https://github.com/AshrafHosam/ultralytics/actions/runs/45826190609       | https://github.com/AshrafHosam/ultralytics/actions/runs/45827681676 |
| Commit SHA                                             | `b64bae8182a903fee4fa67e44a0d0d4765453a7a`                                | `319a5dbaf72a5f8844e8710b3d4e7b4047f60881`                          |

**CPU and memory (asked by the instructor):** Every run gets a fresh machine, but not always the same hardware, so comparing numbers from two different runs is not fair. Instead, one run ([run](https://github.com/AshrafHosam/ultralytics/actions/runs/45831330008), commit [`8ae37cb`](https://github.com/AshrafHosam/ultralytics/commit/8ae37cb97cf76f703ce93025584917cbfa79a60e)) measures **both versions on the same machine**: it swaps in the old `nms.py` from `b64bae8`, measures, puts the new one back, measures, 3 rounds. The [script](https://github.com/AshrafHosam/ultralytics/blob/8ae37cb97cf76f703ce93025584917cbfa79a60e/lab7/measure_nms.py) runs the filter 20 times on a fixed, realistic-sized input (8,400 boxes, 80 classes, the size of a 640×640 YOLO output).

| Same machine: 4 CPU cores, 15.7 GB RAM | Before (`>`, `b64bae8`)  | After (`>=`, `8ae37cb`)  |
| -------------------------------------- | ------------------------ | ------------------------ |
| Time per call, rounds 1 / 2 / 3        | 928.3 / 914.5 / 918.9 ms | 929.4 / 910.3 / 918.5 ms |
| Average time per call                  | 920.6 ms                 | 919.4 ms                 |
| Average CPU time (20 calls)            | 33.42 s                  | 33.37 s                  |
| CPU use                                | 181–182%                 | 181–182%                 |
| Peak memory                            | 375 MB                   | 375 MB                   |
| Boxes kept                             | 300 (the default limit)  | 300 (the default limit)  |

**Result:** no measurable effect. The difference between the versions (about 1 ms) is smaller than the difference between rounds of the same version (up to 19 ms), and memory is the same. This makes sense: `>=` costs the same as `>`, and with realistic scores almost no box is exactly at 0.25.

Consumption Measurement Run: https://github.com/AshrafHosam/ultralytics/actions/runs/45831330008/job/139948563860

**How do your checks distinguish the two versions?** The case check expects a different result in each version, and the new version also checks that the old result is gone. The unchanged checks were not edited and pass in both runs, so I did not just rewrite every expectation. The log also prints the rule line (`>` or `>=`), so you can see which code ran.

**Did anything fail unexpectedly? What did you learn or fix?** Nothing failed. While planning I found that the default YOLO26 models use a different, shorter part of the function, so I changed the rule in all 4 places and tested both input formats.

## 4. Engineering judgment and submission

**Answer Q1:** Yes. Both runs were green on a clean Ubuntu machine, the SHAs match, and `ultralytics` was loaded from my fork's checkout. The printed rule (`>` before, `>=` after) matches each commit.  
**Answer Q2:** Yes. For the same boxes, the result changed from `[0.9]` to `[0.9, 0.25]`, the case check caught it, and the unchanged checks passed in both runs. Measured on the same machine, the change did not change CPU time or memory.

**Release decision:** Accept the new version. It ran from the right commit, it now matches the docs, nothing that should stay the same changed in my checks, and it uses the same CPU and memory as before.  
**One important limitation or missing check:** I only used a few hand-made boxes, not a real model on real images. Also, the multi-label part (line 117) changed too, but normal prediction doesn't use it and I didn't test it.

| GitHub evidence                               | Link                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Lab #7 pull request (within your fork)        | [TO FILL after opening the PR]                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Merged commit** on `semester-investigation` | [TO FILL after merging]                                                                                                                                                                                                                                                                                                                                                                                                             |
| Final `LAB7_ANALYSIS.md` on merged branch     | https://github.com/AshrafHosam/ultralytics/blob/semester-investigation/LAB7_ANALYSIS.md                                                                                                                                                                                                                                                                                                                                             |
| Student 1 meaningful commit(s)                | Ashraf Hossam: [`b64bae8`](https://github.com/AshrafHosam/ultralytics/commit/b64bae8182a903fee4fa67e44a0d0d4765453a7a) (pipeline and baseline check), [`319a5db`](https://github.com/AshrafHosam/ultralytics/commit/319a5dbaf72a5f8844e8710b3d4e7b4047f60881) (rule change and updated check), [`8ae37cb`](https://github.com/AshrafHosam/ultralytics/commit/8ae37cb97cf76f703ce93025584917cbfa79a60e) (CPU and memory measurement) |
| Student 2 meaningful commit(s), if a pair     | individual                                                                                                                                                                                                                                                                                                                                                                                                                          |

**LLM assistance (if used):** Claude helped me write the workflow, and the test. I checked the docs text and the rule lines myself, read both GitHub logs, and reviewed every diff before committing.

**Submission reminder:** Each student submits `Lab7_YOUR_GITHUB_HANDLE.txt` to the Lab #7 MS Teams submit folder, containing their name, fork URL, merged-commit URL and final analysis URL.
