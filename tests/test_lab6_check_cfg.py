# Lab 6 characterization test for ultralytics.cfg.check_cfg.
# It records what check_cfg does at the Lab 6 baseline (commit 8c983bbd), so the refactoring of
# check_cfg can be checked for unchanged behavior. It covers the branches the existing tests do not run.

import pytest

from ultralytics.cfg import check_cfg


def checked(key, value, hard=True):
    """Run check_cfg on a one-key config and return the (possibly converted) value."""
    cfg = {key: value}
    check_cfg(cfg, hard=hard)
    return cfg[key]


def test_float_keys():
    """Float keys accept int or float; other types raise when hard, and are converted when not hard."""
    assert checked("box", 7) == 7
    with pytest.raises(TypeError, match="degrees"):
        checked("degrees", "10")
    value = checked("degrees", "10", hard=False)
    assert value == 10.0 and type(value) is float


def test_scale():
    """Scale accepts a number in [0, 1] or a pair of numbers."""
    assert checked("scale", (0.5, 2.0)) == (0.5, 2.0)
    with pytest.raises(TypeError, match="scale"):
        checked("scale", [1, 2, 3])
    assert checked("scale", [1, 2, 3], hard=False) == [1, 2, 3]  # left unchanged when not hard
    with pytest.raises(TypeError, match="scale"):
        checked("scale", "0.5")
    assert checked("scale", "0.5", hard=False) == 0.5
    with pytest.raises(ValueError, match="scale"):
        checked("scale", 2.0)


def test_fraction_keys():
    """Fraction keys use [0, 1]; the dataset 'fraction' also accepts counts and split lists."""
    with pytest.raises(TypeError, match="conf"):
        checked("conf", "0.5")
    assert checked("conf", "0.5", hard=False) == 0.5
    with pytest.raises(ValueError, match="conf"):
        checked("conf", 1.5)
    value = checked("fraction", 1)
    assert value == 1.0 and type(value) is float
    assert checked("fraction", 500) == 500
    assert checked("fraction", "0.5", hard=False) == 0.5
    with pytest.raises(ValueError, match="fraction"):
        checked("fraction", 0)
    with pytest.raises(ValueError, match="fraction"):
        checked("fraction", [1])


def test_other_keys():
    """The remaining branches of the loop keep their behavior."""
    value = checked("epochs", 1e2)
    assert value == 100 and type(value) is int
    with pytest.raises(TypeError, match="epochs"):
        checked("epochs", 1.5)
    with pytest.raises(ValueError, match="epochs"):
        checked("epochs", 0)
    assert checked("save", "true", hard=False) is True
    with pytest.raises(TypeError, match="save"):
        checked("save", "true")
    assert checked("optimizer", 5, hard=False) == "5"
    with pytest.raises(ValueError, match="tile"):
        checked("tile", 0)
    assert checked("compile", 1, hard=False) is True
    with pytest.raises(TypeError, match="compile"):
        checked("compile", 1)
    assert checked("amp", "FP16") == "fp16"
    with pytest.raises(ValueError, match="amp"):
        checked("amp", "int8")
    with pytest.raises(TypeError, match="degrees"):
        checked("degrees", None)
