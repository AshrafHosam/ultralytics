# Lab #6 - Investigation Record

**Team / student(s):** Ashraf Hossam ([@AshrafHosam](https://github.com/AshrafHosam))  
**Mode:** individual  
**Approved upstream project:** https://github.com/ultralytics/ultralytics  
**Fork / working branch:** [`semester-investigation`](https://github.com/AshrafHosam/ultralytics/tree/semester-investigation)  
**Original Lab #5 version:** tag `v8.4.211`, commit [`d2b7876e79ad2e02fd77221622774b511a8f0cb3`](https://github.com/AshrafHosam/ultralytics/commit/d2b7876e79ad2e02fd77221622774b511a8f0cb3)  
**Lab #6 pre-refactoring baseline:** [`8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6`](https://github.com/AshrafHosam/ultralytics/commit/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6) (only adds the scan workflows; the code is the same as Lab 5)

**What I did, in short:** I scanned the `ultralytics/` code on GitHub, found that `check_cfg` (the function that checks every setting you give YOLO) was long and had a check copied 3 times, split it into smaller functions, and showed with the same tests before and after that it still works the same.

## Task 1 - Stakeholder, GQM, and Code Smell Selection

**Selected stakeholder and exact fictional concern from handout:** Scenario A - Alex, Software Maintainer: "Our software keeps evolving. Future contributors need to understand and modify its components, but certain design choices might make that work unnecessarily difficult. Please investigate possible code smells, identify maintainability concerns worth addressing, and demonstrate a focused improvement without changing the software's intended behavior."

**GQM goal:** Analyze **the Python code of the Ultralytics package (`ultralytics/`)** for the purpose of **finding and fixing code that is hard to maintain**, with respect to **code smells that make code hard to understand and change**, from the viewpoint of **Alex, the software maintainer**, in the context of **an open-source computer-vision library that keeps changing (my fork of v8.4.211), scanned with GitHub Actions**.

| Initial question | Question wording | Relevant metric(s) / observation(s) | Why useful? |
|---|---|---|---|
| **Q1** | Which functions are the hardest to understand and change? | Ruff warnings: complexity (limit 10), branches (limit 12), statements (limit 50), arguments (limit 5); jscpd copied code. Then I read the code. | Tells me **where** to look. |
| **Q2** | Which of those are changed often, so they slow people down the most? | Number of commits to the file in the last 12 months (19 Nov 2025 to 12 Nov 2026), from GitHub history. | Tells me **which one** is worth fixing first. |

**Code-smell types chosen from the lecture:** 3 (individual), from slide 32.

| Smell type | Lecture slide(s) | Why relevant to our GQM? | How we plan to investigate it (tool/rule/metric/inspection) |
|---|---|---|---|
| 1. Long Method (Bloater) | 32, 34-37 | A method that does too much is hard to read and change. | Ruff complexity, branches and statements warnings, then reading the code. |
| 2. Long Parameter List (Bloater) | 32, 43-44 | Many parameters make every call and every change harder. | Ruff "too many arguments" warning, then checking if the arguments are really needed. |
| 3. Duplicated Code (Dispensable) | 32, 72-77 | Copies must all be changed together, or they drift apart. | jscpd, then checking if the copy is real code or just similar text. |

**Primary GitHub-integrated tool:** Ruff on GitHub Actions ([docs](https://docs.astral.sh/ruff/)), the course example workflow set to `ultralytics/`  
**Complementary tool (only if needed):** jscpd on GitHub Actions ([repo](https://github.com/kucherenko/jscpd)), the course example workflow limited to Python files in `ultralytics/`  
**Why these tools match our questions and selected smells:** Ruff covers smells 1 and 2 but cannot find copied code, so I added jscpd for smell 3.

## Task 2 - Detect and Investigate Code Smells

**Analysis configuration:**

- Fork branch and **baseline SHA**: `semester-investigation`, `8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6`
- Source directory/directories analyzed: `ultralytics/` (232 Python files)
- Exclusions (generated/test/vendor/weight data, etc.): tests, docs and examples are not scanned. I did not pick anything in `ultralytics/trackers/`, because a classmate on the same project works on tracking.
- Tool(s), versions and relevant enabled rules/settings: Ruff 0.14.4 (rules C901, PLR0912, PLR0913, PLR0915, default limits); jscpd 5.3.2 (Python only, default settings). The same settings before and after.
- **Baseline GitHub scan / Actions run link(s):** Ruff #1 and Duplication #1, on commit `8c983bb` (see the run lists: [Ruff](https://github.com/AshrafHosam/ultralytics/actions/workflows/lab6-python-ruff.yml), [Duplication](https://github.com/AshrafHosam/ultralytics/actions/workflows/lab6-duplication.yml))

| Selected smell type | Found / not found / not assessable | Tool result or source link | Interpretation / limitations |
|---|---|---|---|
| Long Method | Found | 102 functions flagged. The worst: `Exporter.__call__` (complexity 88) and `check_cfg` (complexity 39). | A warning only means "look here"; I had to read the code to judge it. |
| Long Parameter List | Found, mostly harmless | 230 functions with more than 5 arguments. | A warning is not always a real problem; see `plot_images` below. |
| Duplicated Code | Found | 216 copied blocks (2.44% of the code). | jscpd did not report the copies inside `check_cfg`, because each copy is smaller than its minimum size (50 tokens). |

**Candidate issues inspected in source (at least two):**

| Candidate ID | File/entity permalink | What was actually observed? | Why it might matter / why uncertain |
|---|---|---|---|
| C1 | [`check_cfg` in `ultralytics/cfg/__init__.py`](https://github.com/AshrafHosam/ultralytics/blob/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6/ultralytics/cfg/__init__.py#L398-L537) | One long `if/elif` chain that checks every setting, with the same "invalid float" check copied 3 times ([1](https://github.com/AshrafHosam/ultralytics/blob/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6/ultralytics/cfg/__init__.py#L442-L446), [2](https://github.com/AshrafHosam/ultralytics/blob/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6/ultralytics/cfg/__init__.py#L459-L463), [3](https://github.com/AshrafHosam/ultralytics/blob/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6/ultralytics/cfg/__init__.py#L480-L484)). | Every training, validation, prediction and export goes through it, and the file had 70 commits in a year. But some of its length is normal, because there are many settings. |
| C2 | [`Exporter.__call__` in `ultralytics/engine/exporter.py`](https://github.com/AshrafHosam/ultralytics/blob/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6/ultralytics/engine/exporter.py#L649-L1108) | About 460 lines of checks and special cases for many export formats (`if fmt == "engine"`, `"coreml"`, `"rknn"`, ...) in one method. | The most changed file (227 commits in a year). But keeping the export steps in one place may be on purpose. |
| C3 | [`Segment.forward`](https://github.com/AshrafHosam/ultralytics/blob/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6/ultralytics/nn/modules/head.py#L358-L380) and [`YOLOESegment.forward`](https://github.com/AshrafHosam/ultralytics/blob/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6/ultralytics/nn/modules/head.py#L1417-L1439) | The same code in two model classes (28 copied lines). | A fix in one copy can be forgotten in the other. But the two classes may be meant to change separately. |

**One warning we challenged, or one potentially missed issue:** Ruff says [`plot_images`](https://github.com/AshrafHosam/ultralytics/blob/8c983bbdfea87ba72f290a0ae3ceb2d2892eb4d6/ultralytics/utils/plotting.py#L881-L895) has too many arguments (13). I don't think that is a real problem: 12 of them are optional, and each of its 10 callers only sets the few it needs. Changing it would mean changing all those calls for very little gain. Also, jscpd **missed** the 3 copies of the float check in `check_cfg` (they are too small for it); I only found them by reading the code.

## Task 3 - Prioritize Technical Debt and Plan Refactoring

| Candidate | Expected future impact / relevance | Estimated effort and behavior risk | Prioritize now or defer? Why? |
|---|---|---|---|
| C1 | High: it grows with every new setting, and the file changes often. | Low: a small change in one file, easy to test without models or a GPU. | **Fix now.** Useful and safe. |
| C2 | High: most changed and most complex. | High: testing needs exports to many formats. | **Later.** Too big and risky for now. |
| C3 | Medium: two copies can drift apart. | Medium: needs changes in two model classes. | **Later.** A good next step. |

**Selected documented issue(s) and individual ownership:** C1 (`check_cfg`), done by me, Ashraf Hossam (@AshrafHosam).

| Contributor / GitHub handle | Candidate ID | Planned refactoring | Expected benefit | GitHub regression check planned |
|---|---|---|---|---|
| Ashraf Hossam / @AshrafHosam | C1 | **Extract Method** (slide 36): move the `scale` and fraction checks into their own small functions. **Remove the copied code** (slides 73, 75): put the float check in one shared function. | `check_cfg` becomes shorter and simpler, and the float check lives in one place. | A `Lab6 - Tests` workflow on GitHub, run before and after: 2 existing project tests plus a new test (`tests/test_lab6_check_cfg.py`) that I added first, to record how the parts I change behave today. |

## Task 4 - Refactor, Verify, and Commit

**Baseline test/check run(s) on GitHub:** Tests #1 (see the [Tests run list](https://github.com/AshrafHosam/ultralytics/actions/workflows/lab6-tests.yml)) on commit [`629f1ad`](https://github.com/AshrafHosam/ultralytics/commit/629f1add57696628089e04ccf5b51fb345d2da93): green, 6 tests passed (old code)  
**After-refactoring SHA(s):** [`21b3bfdf442092019d022f9d353a8734521638ed`](https://github.com/AshrafHosam/ultralytics/commit/21b3bfdf442092019d022f9d353a8734521638ed)  
**After-refactoring scan / workflow run(s):** [Ruff #3](https://github.com/AshrafHosam/ultralytics/actions/runs/44537860014); Duplication #3 (see the [Duplication run list](https://github.com/AshrafHosam/ultralytics/actions/workflows/lab6-duplication.yml))  
**After-refactoring test/check run(s):** [Tests #2](https://github.com/AshrafHosam/ultralytics/actions/runs/44537859924) on commit `21b3bfd`: green, the same 6 tests passed (new code)

**What I changed** (only in [`ultralytics/cfg/__init__.py`](https://github.com/AshrafHosam/ultralytics/blob/21b3bfdf442092019d022f9d353a8734521638ed/ultralytics/cfg/__init__.py#L398-L578)):

1. **Extract Method:** the `scale` check is now its own function, `_check_scale()`, and the fraction check is now `_check_fraction()`. `check_cfg` just calls them.
2. **Removed the copied code:** the float check that was written 3 times is now one function, `_to_float()`, used in all 3 places.

Everything else stays the same: the order of the checks, the errors and the error messages.

| Issue / contributor | Before (same scope and measurement) | After | Test / verification result | What does this actually establish? |
|---|---|---|---|---|
| C1 Long Method / Ashraf | complexity **39**, branches **39**, statements **73** | complexity **23**, branches **23**, statements under 50 (that warning is gone) | 6 tests pass before and after | `check_cfg` is about 40% less complex, and it still behaves the same in the tests. It is still above Ruff's limits for complexity and branches. |
| C1 copied code / Ashraf | the float check written **3** times | written **once** | 6 tests pass before and after | A future change to this check only has to be made in one place. |
| Whole package | Ruff 443 warnings, jscpd 216 copies | Ruff 442 warnings, jscpd 216 copies | - | Nothing else changed. jscpd shows no change because it never found this copy. |

**Where to find the proof:**

- **Ruff numbers:** download `lab6-ruff-report` from Ruff #1 (before, in the [Ruff run list](https://github.com/AshrafHosam/ultralytics/actions/workflows/lab6-python-ruff.yml)) and [Ruff #3](https://github.com/AshrafHosam/ultralytics/actions/runs/44537860014) (after) and open `ruff.json`. `check_cfg` is the first entry in both:
  - before: `` `check_cfg` is too complex (39 > 10) ``, `Too many branches (39 > 12)`, `Too many statements (73 > 50)`, all at row 398;
  - after: `` `check_cfg` is too complex (23 > 10) ``, `Too many branches (23 > 12)`, both at row 485. There is no statements warning at row 485 any more.
  - Row 398 (old) and row 485 (new) are both the line where `check_cfg` starts. The other "Too many statements" warnings in the after file belong to other functions, which you can tell by their different row numbers.
- **The new functions:** `_to_float`, `_check_scale` and `_check_fraction` do not appear in the after file at all, so each one is under all of Ruff's limits.
- **Tests:** in Tests #1 and [Tests #2](https://github.com/AshrafHosam/ultralytics/actions/runs/44537859924), open the job `cfg-tests`, step "Run the check_cfg tests". Both list the same 6 tests as PASSED.
- **The code change:** the [refactoring commit](https://github.com/AshrafHosam/ultralytics/commit/21b3bfdf442092019d022f9d353a8734521638ed) shows the old and new code side by side.

**Answers to the original GQM questions:**

- **Q1:** The tools flag a lot of functions. When I read the code, the clearest problems were in long central functions like `Exporter.__call__` and `check_cfg`. Some warnings are not real problems (like `plot_images`), so the raw counts alone are not enough.
- **Q2:** Of the files I looked at, `engine/exporter.py` (227 commits) and `cfg/__init__.py` (70 commits) change the most. I fixed `check_cfg` first because it changes often, everything uses it, and the fix is small and safe. `Exporter.__call__` changes even more, but fixing it safely needs much bigger testing, so I left it for later.

**Conclusion and limits:**

- **Better:** `check_cfg` is simpler (complexity 39 to 23, statements under the limit), and the float check exists once instead of 3 times.
- **Still not perfect:** `check_cfg` is still above Ruff's complexity and branch limits. Fixing that fully would mean redesigning the whole function, which is a bigger job.
- **Worse:** the file is 41 lines longer (90 added, 49 removed), mostly the new functions' descriptions.
- **What I can't claim:** I can't prove the code is now easier to maintain, because I didn't measure how long changes take. The tests only check the cases written in them.

### Individual implementation commits - permanent GitHub links

| Contributor / GitHub account | Documented issue (C1, C2, ...) | What was refactored? | **Permanent commit link** | PR or GitHub Actions push-run link |
|---|---|---|---|---|
| Ashraf Hossam / @AshrafHosam | C1 | `check_cfg` split into `_check_scale`, `_check_fraction` and one shared `_to_float` | https://github.com/AshrafHosam/ultralytics/commit/21b3bfdf442092019d022f9d353a8734521638ed | [Ruff #3](https://github.com/AshrafHosam/ultralytics/actions/runs/44537860014), [Tests #2](https://github.com/AshrafHosam/ultralytics/actions/runs/44537859924) |
| Ashraf Hossam / @AshrafHosam (test, not a refactoring) | C1 | Added the test `tests/test_lab6_check_cfg.py` and the `Lab6 - Tests` workflow | https://github.com/AshrafHosam/ultralytics/commit/629f1add57696628089e04ccf5b51fb345d2da93 | Tests #1 (see the [Tests run list](https://github.com/AshrafHosam/ultralytics/actions/workflows/lab6-tests.yml)) |

**Completed template location on `semester-investigation`:** https://github.com/AshrafHosam/ultralytics/blob/semester-investigation/LAB6_ANALYSIS.md  
**Template/report commit link:** this file is added in the last commit of the branch, see its [history](https://github.com/AshrafHosam/ultralytics/commits/semester-investigation/LAB6_ANALYSIS.md).

**Evidence preservation note:** GitHub deletes the scan files (artifacts) after 30 days, so I copied all the important numbers into this file.
