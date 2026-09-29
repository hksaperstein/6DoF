from pathlib import Path

import pytest

from config import load_config

ARM_YAML = Path(__file__).parent / "arm.yaml"


def test_loads_six_rows_in_joint_order():
    cfg = load_config(ARM_YAML)
    assert [r.joint for r in cfg.rows] == [0, 1, 2, 3, 4, 5]


def test_resolves_parameter_references_to_values():
    cfg = load_config(ARM_YAML)
    assert cfg.rows[1].a == pytest.approx(200.0)      # a1_link_upper
    assert cfg.rows[5].d == pytest.approx(112.0)      # d5_tool_length
    assert cfg.rows[0].a == pytest.approx(0.0)        # literal 0, not a name


def test_alpha_converted_to_radians():
    import math
    cfg = load_config(ARM_YAML)
    assert cfg.rows[0].alpha == pytest.approx(math.pi / 2)
    assert cfg.rows[4].alpha == pytest.approx(-math.pi / 2)
    assert cfg.rows[2].alpha == pytest.approx(0.0)


def test_value_looks_up_by_name():
    cfg = load_config(ARM_YAML)
    assert cfg.value("a4_yaw_to_roll") == pytest.approx(20.0)


def test_with_override_rebuilds_rows_and_leaves_original_untouched():
    cfg = load_config(ARM_YAML)
    bumped = cfg.with_override(a4_yaw_to_roll=250.0)
    assert bumped.rows[4].a == pytest.approx(250.0)
    assert cfg.rows[4].a == pytest.approx(20.0)


def test_unknown_parameter_reference_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "units: {length: mm, angle: deg}\n"
        "parameters:\n"
        "  a1: {value: 1.0, status: settled, rationale: x}\n"
        "dh:\n"
        "  convention: classical\n"
        "  rows:\n"
        "    - {joint: 0, d: 0, a: nonexistent_param, alpha: 0}\n"
    )
    with pytest.raises(KeyError, match="nonexistent_param"):
        load_config(bad)


def test_override_of_unknown_parameter_is_an_error():
    cfg = load_config(ARM_YAML)
    with pytest.raises(KeyError, match="not_a_param"):
        cfg.with_override(not_a_param=1.0)


def test_null_value_warns_and_loads_as_zero(tmp_path):
    bad = tmp_path / "null.yaml"
    bad.write_text(
        "units: {length: mm, angle: deg}\n"
        "parameters:\n"
        "  d_open: {value: null, status: open, rationale: unresolved}\n"
        "dh:\n"
        "  convention: classical\n"
        "  rows:\n"
        "    - {joint: 0, d: d_open, a: 0, alpha: 0}\n"
    )
    with pytest.warns(UserWarning, match="d_open"):
        cfg = load_config(bad)
    assert cfg.rows[0].d == pytest.approx(0.0)


def test_suspect_status_warns_even_with_a_value(tmp_path):
    bad = tmp_path / "suspect.yaml"
    bad.write_text(
        "units: {length: mm, angle: deg}\n"
        "parameters:\n"
        "  d_sus: {value: 5.0, status: suspect, rationale: doubtful}\n"
        "dh:\n"
        "  convention: classical\n"
        "  rows:\n"
        "    - {joint: 0, d: d_sus, a: 0, alpha: 0}\n"
    )
    with pytest.warns(UserWarning, match="suspect"):
        load_config(bad)


def test_current_config_loads_without_warnings(recwarn):
    load_config(ARM_YAML)
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []
