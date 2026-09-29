# Kinematic Model and Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a verified forward-kinematics model and interactive 3D visualization of the custom 6-DoF arm, so its kinematic configuration can be validated numerically and `d0_base_height` settled before geometry goes to CAD.

**Architecture:** The DH table lives in `arm.yaml` as data, loaded into a frozen `ArmConfig`. `kinematics.py` is pure numpy with no I/O and computes every frame in the chain. `test_kinematics.py` verifies the table through pose-independent invariants and parameter-sensitivity checks rather than trusting it. `plot_arm.py` renders the chain with sliders.

**Tech Stack:** Python 3, numpy, matplotlib, PyYAML. `roboticstoolbox-python` as a dev-only cross-check.

**Spec:** `docs/superpowers/specs/2026-09-28-kinematics-model-design.md`

## Global Constraints

- Branch: `kinematics-model`, already created off `develop`. Do not touch `src/robot/`, `docker/`, or `.github/workflows/` — legacy AR4 code, deliberately left alone.
- Joint indices are **0-based** everywhere: document, config, code, future URDF. Never renumber to 1-based.
- Lengths are **millimetres** throughout, config and runtime alike.
- Angles are **degrees in `arm.yaml` only**; every angle crossing a Python interface is **radians**. Conversion happens once, in `config.py`.
- Classical DH convention: `Rot(z, theta) · Trans(z, d) · Trans(x, a) · Rot(x, alpha)`.
- Joint `i`'s axis is the z axis of the frame **preceding** it: `frames[i]`, where `frames[0]` is the base. This off-by-one is the most likely source of a silent error in the whole codebase.
- **Never invent a mechanical value.** If a number is not in `arm.yaml` or the spec, it is open — fail or warn, never pick a plausible default.
- `kinematics.py` imports no I/O, no yaml, no matplotlib.

---

### Task 1: Configuration loader

**Files:**
- Create: `utilities/kinematics/arm.yaml`
- Create: `utilities/kinematics/config.py`
- Create: `utilities/kinematics/requirements.txt`
- Create: `utilities/kinematics/requirements-dev.txt`
- Create: `utilities/kinematics/.gitignore`
- Test: `utilities/kinematics/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `load_config(path: Path) -> ArmConfig`; `ArmConfig` with `.parameters: dict[str, Parameter]`, `.rows: tuple[DHRow, ...]`, `.value(name: str) -> float`, `.with_override(**kwargs) -> ArmConfig`. `DHRow` has `.joint: int`, `.d: float`, `.a: float`, `.alpha: float` (radians). `Parameter` has `.name`, `.value`, `.status`, `.rationale`.

- [ ] **Step 1: Create the dependency files**

`utilities/kinematics/requirements.txt`:
```
numpy>=1.24
matplotlib>=3.7
PyYAML>=6.0
```

`utilities/kinematics/requirements-dev.txt`:
```
-r requirements.txt
pytest>=7.4
roboticstoolbox-python>=1.1
```

`utilities/kinematics/.gitignore`:
```
.venv/
__pycache__/
```

- [ ] **Step 2: Create `arm.yaml`**

Copy verbatim — the values and rationales are the spec's, and `d3_roll_offset` is intentionally absent (removed; see spec finding F6).

```yaml
# Kinematic configuration for the custom 6-DoF arm.
# Lengths in mm. Angles in DEGREES here only; converted to radians on load.
# Joint indices are 0-based and never renumbered.
units: {length: mm, angle: deg}

parameters:
  d0_base_height:        {value: 200.0, status: placeholder, rationale: "To be explored; sweep it"}
  a1_link_upper:         {value: 200.0, status: settled,     rationale: "Equal-length links as starting point"}
  a2_link_forearm:       {value: 200.0, status: settled,     rationale: "Equal to upper arm; maximizes dexterous workspace"}
  a3_wrist_pitch_to_yaw: {value: 0.0,   status: assumed,     rationale: "J3/J4 coincident; revisit when wrist is packaged"}
  a4_yaw_to_roll:        {value: 20.0,  status: estimate,    rationale: "THE non-spherical offset (F3); ratified at 20 mm 2026-09-28, ~10% of a link; stays an estimate until the wrist is packaged"}
  d5_tool_length:        {value: 112.0, status: settled,     rationale: "Hiwonder gripper, flange to grasp point"}

dh:
  convention: classical   # Rot(z,theta) Trans(z,d) Trans(x,a) Rot(x,alpha)
  # theta is omitted: row i always takes joint variable q[i]. A constant
  # theta_offset per joint is added here once a home pose is chosen.
  rows:
    - {joint: 0, d: d0_base_height, a: 0,                     alpha:  90}
    - {joint: 1, d: 0,              a: a1_link_upper,         alpha:   0}
    - {joint: 2, d: 0,              a: a2_link_forearm,       alpha:   0}
    - {joint: 3, d: 0,              a: a3_wrist_pitch_to_yaw, alpha:  90}
    - {joint: 4, d: 0,              a: a4_yaw_to_roll,        alpha: -90}
    - {joint: 5, d: d5_tool_length, a: 0,                     alpha:   0}
```

- [ ] **Step 3: Write the failing tests**

`utilities/kinematics/test_config.py`:
```python
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
    assert cfg.rows[4].a == pytest.approx(100.0)


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
```

- [ ] **Step 4: Run the tests to verify they fail**

```bash
cd utilities/kinematics
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 5: Write `config.py`**

```python
"""Load the arm's kinematic configuration from arm.yaml.

The DH table is data, not code: rows reference parameters by name so a
parameter can change without touching kinematics.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

NOISY_STATUSES = {"open", "suspect"}


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    status: str
    rationale: str


@dataclass(frozen=True)
class DHRow:
    joint: int
    d: float
    a: float
    alpha: float  # radians


@dataclass(frozen=True)
class ArmConfig:
    parameters: dict[str, Parameter]
    rows: tuple[DHRow, ...]
    _raw_rows: tuple[dict, ...]

    def value(self, name: str) -> float:
        if name not in self.parameters:
            raise KeyError(f"unknown parameter {name!r}")
        return self.parameters[name].value

    def with_override(self, **overrides: float) -> "ArmConfig":
        params = dict(self.parameters)
        for name, value in overrides.items():
            if name not in params:
                raise KeyError(f"cannot override unknown parameter {name!r}")
            params[name] = replace(params[name], value=float(value))
        return ArmConfig(
            parameters=params,
            rows=_build_rows(self._raw_rows, params),
            _raw_rows=self._raw_rows,
        )


def _resolve(entry, params: dict[str, Parameter]) -> float:
    """A row field is either a literal number or a parameter name."""
    if isinstance(entry, (int, float)):
        return float(entry)
    if entry not in params:
        raise KeyError(f"row references unknown parameter {entry!r}")
    return params[entry].value


def _build_rows(raw_rows, params: dict[str, Parameter]) -> tuple[DHRow, ...]:
    return tuple(
        DHRow(
            joint=int(r["joint"]),
            d=_resolve(r["d"], params),
            a=_resolve(r["a"], params),
            alpha=math.radians(float(r["alpha"])),
        )
        for r in raw_rows
    )


def load_config(path: Path) -> ArmConfig:
    data = yaml.safe_load(Path(path).read_text())

    params: dict[str, Parameter] = {}
    for name, spec in data["parameters"].items():
        value = spec.get("value")
        status = spec.get("status", "unknown")
        if value is None:
            warnings.warn(
                f"parameter {name!r} has no value (status {status!r}); "
                f"loading as 0.0",
                UserWarning,
                stacklevel=2,
            )
            value = 0.0
        elif status in NOISY_STATUSES:
            warnings.warn(
                f"parameter {name!r} has status {status!r}; "
                f"its value {value} is not trustworthy",
                UserWarning,
                stacklevel=2,
            )
        params[name] = Parameter(
            name=name,
            value=float(value),
            status=status,
            rationale=spec.get("rationale", ""),
        )

    raw_rows = tuple(data["dh"]["rows"])
    rows = _build_rows(raw_rows, params)
    if [r.joint for r in rows] != list(range(len(rows))):
        raise ValueError("dh.rows must be in ascending joint order starting at 0")
    return ArmConfig(parameters=params, rows=rows, _raw_rows=raw_rows)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest test_config.py -v`
Expected: 10 passed.

- [ ] **Step 7: Commit**

```bash
git add utilities/kinematics/
git commit -m "Add kinematic configuration loader and arm.yaml"
```

---

### Task 2: Forward kinematics core

**Files:**
- Create: `utilities/kinematics/kinematics.py`
- Test: `utilities/kinematics/test_kinematics.py`

**Interfaces:**
- Consumes: `load_config`, `ArmConfig` from Task 1.
- Produces: `dh_transform(theta, d, a, alpha) -> ndarray(4,4)`; `forward_kinematics(cfg: ArmConfig, q: ndarray[6]) -> Pose`; `axis_distance(p1, d1, p2, d2) -> float`. `Pose` exposes `.frames: ndarray(7,4,4)`, `.joint_origins: ndarray(7,3)`, `.joint_axes: ndarray(6,3)`, `.tcp: ndarray(4,4)`, `.tcp_position: ndarray(3,)`, and `.joint_axis(i) -> (point, direction)`.

- [ ] **Step 1: Write the failing tests**

`utilities/kinematics/test_kinematics.py`:
```python
import math
from pathlib import Path

import numpy as np
import pytest

from config import load_config
from kinematics import axis_distance, dh_transform, forward_kinematics

ARM_YAML = Path(__file__).parent / "arm.yaml"


@pytest.fixture
def cfg():
    return load_config(ARM_YAML)


def test_dh_transform_identity_for_all_zero_args():
    assert np.allclose(dh_transform(0.0, 0.0, 0.0, 0.0), np.eye(4))


def test_dh_transform_pure_translation_along_x():
    T = dh_transform(0.0, 0.0, 7.0, 0.0)
    assert np.allclose(T[:3, 3], [7.0, 0.0, 0.0])


def test_dh_transform_pure_translation_along_z():
    T = dh_transform(0.0, 5.0, 0.0, 0.0)
    assert np.allclose(T[:3, 3], [0.0, 0.0, 5.0])


def test_frames_shape_and_base_is_identity(cfg):
    pose = forward_kinematics(cfg, np.zeros(6))
    assert pose.frames.shape == (7, 4, 4)
    assert np.allclose(pose.frames[0], np.eye(4))


def test_frames_are_valid_rigid_transforms(cfg):
    pose = forward_kinematics(cfg, np.array([0.3, 0.4, -0.5, 0.6, 0.7, 0.2]))
    for T in pose.frames:
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])


def test_joint_origins_and_axes_shapes(cfg):
    pose = forward_kinematics(cfg, np.zeros(6))
    assert pose.joint_origins.shape == (7, 3)
    assert pose.joint_axes.shape == (6, 3)
    assert np.allclose(np.linalg.norm(pose.joint_axes, axis=1), 1.0)


def test_joint_zero_axis_is_world_z_in_every_pose(cfg):
    rng = np.random.default_rng(0)
    for _ in range(20):
        pose = forward_kinematics(cfg, rng.uniform(-math.pi, math.pi, 6))
        assert np.allclose(pose.joint_axes[0], [0.0, 0.0, 1.0])


def test_base_height_places_joint_one_origin(cfg):
    pose = forward_kinematics(cfg, np.zeros(6))
    assert np.allclose(pose.joint_origins[1], [0.0, 0.0, cfg.value("d0_base_height")])


def test_tcp_matches_last_frame(cfg):
    pose = forward_kinematics(cfg, np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert np.allclose(pose.tcp, pose.frames[6])
    assert np.allclose(pose.tcp_position, pose.frames[6][:3, 3])


def test_q_length_must_be_six(cfg):
    with pytest.raises(ValueError, match="6"):
        forward_kinematics(cfg, np.zeros(5))


def test_axis_distance_of_parallel_offset_lines():
    d = np.array([1.0, 0.0, 0.0])
    assert axis_distance(np.zeros(3), d, np.array([0.0, 3.0, 4.0]), d) == pytest.approx(5.0)


def test_axis_distance_of_intersecting_lines_is_zero():
    assert axis_distance(
        np.zeros(3), np.array([1.0, 0.0, 0.0]),
        np.zeros(3), np.array([0.0, 1.0, 0.0]),
    ) == pytest.approx(0.0)


def test_axis_distance_of_skew_lines():
    # x-axis through origin, and the y-axis line lifted to z=2
    assert axis_distance(
        np.zeros(3), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 2.0]), np.array([0.0, 1.0, 0.0]),
    ) == pytest.approx(2.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest test_kinematics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kinematics'`.

- [ ] **Step 3: Write `kinematics.py`**

```python
"""Forward kinematics for the custom 6-DoF arm.

Pure numpy. No file I/O, no plotting, no globals.

Frame indexing, which is the single most likely source of a silent error:
`frames[0]` is the base and `frames[i+1]` is the frame after applying DH row
`i`. Joint `i`'s axis is the z axis of the frame PRECEDING it, `frames[i]`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import ArmConfig


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Classical DH: Rot(z,theta) Trans(z,d) Trans(x,a) Rot(x,alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


@dataclass(frozen=True)
class Pose:
    frames: np.ndarray  # (7, 4, 4): base, then one per DH row

    @property
    def joint_origins(self) -> np.ndarray:
        """(7, 3) — base origin, each joint origin, then the TCP."""
        return self.frames[:, :3, 3]

    @property
    def joint_axes(self) -> np.ndarray:
        """(6, 3) unit vectors. Joint i's axis is z of the frame before it."""
        return self.frames[:6, :3, 2]

    @property
    def tcp(self) -> np.ndarray:
        return self.frames[6]

    @property
    def tcp_position(self) -> np.ndarray:
        return self.frames[6][:3, 3]

    def joint_axis(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(point on the axis, unit direction) for joint i."""
        return self.frames[i][:3, 3], self.frames[i][:3, 2]


def forward_kinematics(cfg: ArmConfig, q) -> Pose:
    q = np.asarray(q, dtype=float)
    if q.shape != (len(cfg.rows),):
        raise ValueError(f"q must have {len(cfg.rows)} elements, got {q.shape}")

    frames = np.empty((len(cfg.rows) + 1, 4, 4))
    frames[0] = np.eye(4)
    T = np.eye(4)
    for i, row in enumerate(cfg.rows):
        T = T @ dh_transform(q[i], row.d, row.a, row.alpha)
        frames[i + 1] = T
    return Pose(frames=frames)


def axis_distance(p1, d1, p2, d2) -> float:
    """Minimum distance between two infinite lines, parallel case handled."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d1 = np.asarray(d1, float) / np.linalg.norm(d1)
    d2 = np.asarray(d2, float) / np.linalg.norm(d2)
    n = np.cross(d1, d2)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-9:  # parallel
        w = p2 - p1
        return float(np.linalg.norm(w - np.dot(w, d1) * d1))
    return float(abs(np.dot(p2 - p1, n)) / norm_n)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest test_kinematics.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add utilities/kinematics/kinematics.py utilities/kinematics/test_kinematics.py
git commit -m "Add forward kinematics core for the 6-DoF arm"
```

---

### Task 3: Verification suite

This task is the reason the work exists: the DH table was derived by hand and has already been wrong once. These tests are the permanent guard.

**Files:**
- Create: `utilities/kinematics/test_verification.py`

**Interfaces:**
- Consumes: `load_config`, `ArmConfig.with_override`, `forward_kinematics`, `axis_distance`, `Pose.joint_axis` from Tasks 1–2.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the pose-independent invariant tests**

`utilities/kinematics/test_verification.py`:
```python
"""Verification of the DH table itself, not of the code that evaluates it.

The table was derived by hand and is unverified. Every test here is
pose-independent, because the joint zero offsets are still unassigned: we
cannot yet say which q means "arm straight up".
"""
import math
from pathlib import Path

import numpy as np
import pytest

from config import load_config
from kinematics import axis_distance, forward_kinematics

ARM_YAML = Path(__file__).parent / "arm.yaml"
POSE_COUNT = 40


@pytest.fixture
def cfg():
    return load_config(ARM_YAML)


@pytest.fixture
def poses():
    rng = np.random.default_rng(20260928)
    return rng.uniform(-math.pi, math.pi, size=(POSE_COUNT, 6))


def test_upper_arm_length_is_a1_in_every_pose(cfg, poses):
    expected = cfg.value("a1_link_upper")
    for q in poses:
        p = forward_kinematics(cfg, q).joint_origins
        assert np.linalg.norm(p[2] - p[1]) == pytest.approx(expected)


def test_forearm_length_is_a2_in_every_pose(cfg, poses):
    expected = cfg.value("a2_link_forearm")
    for q in poses:
        p = forward_kinematics(cfg, q).joint_origins
        assert np.linalg.norm(p[3] - p[2]) == pytest.approx(expected)


def test_pitch_axes_j1_j2_j3_are_mutually_parallel(cfg, poses):
    for q in poses:
        axes = forward_kinematics(cfg, q).joint_axes
        for a, b in ((1, 2), (2, 3), (1, 3)):
            assert np.linalg.norm(np.cross(axes[a], axes[b])) == pytest.approx(0.0, abs=1e-9)


def test_tcp_lies_on_the_j5_roll_axis(cfg, poses):
    """Required by the handoff: an off-axis tool swings through a circle
    when J5 rolls instead of spinning in place."""
    for q in poses:
        pose = forward_kinematics(cfg, q)
        point, direction = pose.joint_axis(5)
        offset = pose.tcp_position - point
        perpendicular = offset - np.dot(offset, direction) * direction
        assert np.linalg.norm(perpendicular) == pytest.approx(0.0, abs=1e-9)


def test_tool_length_separates_j5_origin_from_tcp(cfg, poses):
    expected = cfg.value("d5_tool_length")
    for q in poses:
        pose = forward_kinematics(cfg, q)
        assert np.linalg.norm(pose.tcp_position - pose.joint_origins[5]) == pytest.approx(expected)
```

- [ ] **Step 2: Run them and confirm they pass**

Run: `python -m pytest test_verification.py -v`
Expected: 5 passed. **If any fail, stop and report** — a failure here means the DH table is wrong, which is information, not a bug to code around.

- [ ] **Step 3: Add the sphericity tests**

Append to `test_verification.py`:
```python
def _wrist_axis_separations(cfg, q):
    """Perpendicular distances between consecutive wrist axes J3, J4, J5."""
    pose = forward_kinematics(cfg, q)
    p3, d3 = pose.joint_axis(3)
    p4, d4 = pose.joint_axis(4)
    p5, d5 = pose.joint_axis(5)
    return (
        axis_distance(p3, d3, p4, d4),
        axis_distance(p4, d4, p5, d5),
    )


def test_wrist_is_non_spherical_with_the_current_config(cfg, poses):
    """a4_yaw_to_roll is non-zero, so J4 and J5 never meet. This is what
    forces numerical IK."""
    expected = cfg.value("a4_yaw_to_roll")
    assert expected != 0.0
    for q in poses:
        _, d45 = _wrist_axis_separations(cfg, q)
        assert d45 == pytest.approx(expected)


def test_j3_and_j4_are_coincident_while_a3_is_zero(cfg, poses):
    assert cfg.value("a3_wrist_pitch_to_yaw") == 0.0
    for q in poses:
        d34, _ = _wrist_axis_separations(cfg, q)
        assert d34 == pytest.approx(0.0, abs=1e-9)


def test_zeroing_a4_makes_the_wrist_spherical(cfg, poses):
    """The pose-invariant definition: all three wrist axes meet at a point."""
    spherical = cfg.with_override(a4_yaw_to_roll=0.0)
    for q in poses:
        d34, d45 = _wrist_axis_separations(spherical, q)
        assert d34 == pytest.approx(0.0, abs=1e-9)
        assert d45 == pytest.approx(0.0, abs=1e-9)


def test_a4_is_exactly_the_j4_to_j5_perpendicular_distance(cfg, poses):
    for a4 in (10.0, 45.0, 100.0, 250.0):
        variant = cfg.with_override(a4_yaw_to_roll=a4)
        for q in poses[:5]:
            _, d45 = _wrist_axis_separations(variant, q)
            assert d45 == pytest.approx(a4)
```

- [ ] **Step 4: Run and confirm**

Run: `python -m pytest test_verification.py -v`
Expected: 9 passed.

- [ ] **Step 5: Add the parameter-sensitivity tests**

This is the generalized form of the check that caught the original error: a parameter that provably cannot influence the quantity it is named for is wrong regardless of its value.

Append to `test_verification.py`:
```python
DELTA = 37.0  # arbitrary, non-round, so a coincidental match is unlikely


def _measure(cfg, name, q):
    """The geometric quantity each parameter claims to control."""
    pose = forward_kinematics(cfg, q)
    p = pose.joint_origins
    if name == "d0_base_height":
        return float(np.linalg.norm(p[1] - p[0]))
    if name == "a1_link_upper":
        return float(np.linalg.norm(p[2] - p[1]))
    if name == "a2_link_forearm":
        return float(np.linalg.norm(p[3] - p[2]))
    if name == "a3_wrist_pitch_to_yaw":
        return float(np.linalg.norm(p[4] - p[3]))
    if name == "a4_yaw_to_roll":
        return axis_distance(*pose.joint_axis(4), *pose.joint_axis(5))
    if name == "d5_tool_length":
        return float(np.linalg.norm(pose.tcp_position - p[5]))
    raise AssertionError(f"no measurement defined for {name!r}")


@pytest.mark.parametrize("name", [
    "d0_base_height",
    "a1_link_upper",
    "a2_link_forearm",
    "a3_wrist_pitch_to_yaw",
    "a4_yaw_to_roll",
    "d5_tool_length",
])
def test_every_parameter_moves_the_quantity_it_is_named_for(cfg, poses, name):
    q = poses[0]
    before = _measure(cfg, name, q)
    after = _measure(cfg.with_override(**{name: cfg.value(name) + DELTA}), name, q)
    assert after - before == pytest.approx(DELTA), (
        f"{name} changed by {DELTA} but its quantity moved by {after - before}. "
        f"A parameter that cannot influence what it is named for is misplaced "
        f"in the DH table."
    )


def test_every_parameter_is_actually_referenced_by_the_table(cfg):
    """Guards against a parameter that exists in arm.yaml but is wired to
    nothing — the state d3_roll_offset was in before it was removed."""
    for name in cfg.parameters:
        bumped = cfg.with_override(**{name: cfg.value(name) + DELTA})
        moved = any(
            not np.allclose(
                forward_kinematics(cfg, q).frames,
                forward_kinematics(bumped, q).frames,
            )
            for q in (np.zeros(6), np.full(6, 0.3))
        )
        assert moved, f"{name} is declared but changes nothing in the chain"
```

- [ ] **Step 6: Run and confirm**

Run: `python -m pytest test_verification.py -v`
Expected: 16 passed.

- [ ] **Step 7: Add the independent cross-check**

Append to `test_verification.py`:
```python
def test_hand_written_fk_matches_roboticstoolbox(cfg, poses):
    """An independent implementation built from the same table. Two
    derivations that agree are evidence; one is a guess."""
    # importorskip MUST be inside the test. At module level it would skip
    # this whole file, silently taking the invariant and sensitivity tests
    # with it whenever the dev dependency is absent.
    roboticstoolbox = pytest.importorskip(
        "roboticstoolbox", reason="dev-only cross-check dependency"
    )
    robot = roboticstoolbox.DHRobot([
        roboticstoolbox.RevoluteDH(d=r.d, a=r.a, alpha=r.alpha) for r in cfg.rows
    ])
    for q in poses:
        assert np.allclose(forward_kinematics(cfg, q).tcp, robot.fkine(q).A, atol=1e-7)
```

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest -v`
Expected: all pass (the cross-check skips if `roboticstoolbox` is unavailable; that is acceptable, a skip is not a failure).

- [ ] **Step 9: Commit**

```bash
git add utilities/kinematics/test_verification.py
git commit -m "Add DH table verification suite with parameter sensitivity guards"
```

---

### Task 4: Interactive 3D visualization

**Files:**
- Create: `utilities/kinematics/plot_arm.py`
- Modify: `README.md` (add a pointer to `utilities/kinematics/`)

**Interfaces:**
- Consumes: `load_config`, `ArmConfig.with_override`, `forward_kinematics`, `axis_distance` from Tasks 1–2.
- Produces: a `python plot_arm.py` entry point. No importable API other tasks depend on.

- [ ] **Step 1: Write `plot_arm.py`**

```python
"""Interactive 3D stick-figure view of the arm.

Run: python plot_arm.py

Sliders drive the six joints plus the two parameters this tool exists to
inform: d0_base_height (the value to settle) and a4_yaw_to_roll (the
non-spherical offset). Watch the wrist separation readout fall to zero as
a4 approaches zero -- that is the wrist becoming spherical.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from config import load_config
from kinematics import axis_distance, forward_kinematics

ARM_YAML = Path(__file__).parent / "arm.yaml"
AXIS_QUIVER_LENGTH = 60.0
WORK_SURFACE_HALF_WIDTH = 500.0
JOINT_LABELS = ["J0", "J1", "J2", "J3", "J4", "J5"]


def _set_equal_aspect(ax, reach):
    """A 200 mm arm drawn in a stretched box misleads about reach."""
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(0.0, 2.0 * reach)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def main():
    base_cfg = load_config(ARM_YAML)
    reach = sum(base_cfg.value(n) for n in
                ("a1_link_upper", "a2_link_forearm", "a4_yaw_to_roll", "d5_tool_length"))

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0.05, right=0.72, bottom=0.05, top=0.95)

    sliders = {}
    for i, label in enumerate(JOINT_LABELS):
        sliders[label] = Slider(
            fig.add_axes([0.78, 0.88 - 0.05 * i, 0.18, 0.03]),
            label, -180.0, 180.0, valinit=0.0,
        )
    sliders["d0_base_height"] = Slider(
        fig.add_axes([0.78, 0.50, 0.18, 0.03]),
        "d0 base", 0.0, 600.0, valinit=base_cfg.value("d0_base_height"),
    )
    sliders["a4_yaw_to_roll"] = Slider(
        fig.add_axes([0.78, 0.44, 0.18, 0.03]),
        "a4 offset", 0.0, 100.0, valinit=base_cfg.value("a4_yaw_to_roll"),
    )

    def redraw(_=None):
        cfg = base_cfg.with_override(
            d0_base_height=sliders["d0_base_height"].val,
            a4_yaw_to_roll=sliders["a4_yaw_to_roll"].val,
        )
        q = np.radians([sliders[label].val for label in JOINT_LABELS])
        pose = forward_kinematics(cfg, q)
        p = pose.joint_origins

        ax.clear()

        # Work surface, at the base of the arm.
        w = WORK_SURFACE_HALF_WIDTH
        gx, gy = np.meshgrid([-w, w], [-w, w])
        ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.15, color="tab:green")

        # The arm itself: base -> each joint -> TCP.
        ax.plot(p[:, 0], p[:, 1], p[:, 2], "-o", color="tab:blue",
                linewidth=3, markersize=6, label="links")
        ax.scatter(*pose.tcp_position, color="tab:red", s=90, label="TCP")

        # Joint axes, drawn because the wrist offset is invisible in a top
        # view and cannot be confirmed by looking at a model.
        for i in range(6):
            point, direction = pose.joint_axis(i)
            v = direction * AXIS_QUIVER_LENGTH
            ax.quiver(*(point - v / 2), *v, color="tab:orange", linewidth=1.2)
            ax.text(*point, f" {JOINT_LABELS[i]}", fontsize=8)

        wrist_gap = axis_distance(*pose.joint_axis(4), *pose.joint_axis(5))
        x, y, z = pose.tcp_position
        ax.set_title(
            f"TCP  x={x:7.1f}  y={y:7.1f}  z={z:7.1f} mm\n"
            f"J4-J5 axis separation = {wrist_gap:6.1f} mm "
            f"({'spherical wrist' if wrist_gap < 1e-6 else 'non-spherical'})"
        )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
        ax.legend(loc="upper left")
        _set_equal_aspect(ax, reach)
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(redraw)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Launch it and confirm it renders**

```bash
cd utilities/kinematics && . .venv/bin/activate && python plot_arm.py
```
Check, and report what you observe:
1. The arm renders as a connected blue polyline from the origin to a red TCP marker.
2. Moving the `J0` slider rotates the whole arm about the vertical axis.
3. Moving `d0 base` raises and lowers the whole arm.
4. Dragging `a4 offset` to 0 makes the title read `spherical wrist` and the separation read `0.0`; restoring it to 100 reads `non-spherical`.
5. Orange quivers appear at each joint, and J1/J2/J3 quivers stay mutually parallel as the arm moves.

If the environment is headless and no window can open, say so explicitly rather than reporting success — set `MPLBACKEND=Agg`, call `redraw()` and `fig.savefig("/tmp/arm.png")` in a scratch script, and confirm from the image instead.

- [ ] **Step 3: Add a pointer in the repo README**

Append to `README.md` (outer fence is tildes only so the nested bash block survives — do not copy the tilde lines):

~~~~markdown

## Kinematics

`utilities/kinematics/` holds the arm's kinematic configuration (`arm.yaml`),
a hand-written forward-kinematics model, and an interactive 3D view:

```bash
cd utilities/kinematics
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest               # verify the DH table
python plot_arm.py             # interactive view
```

See `docs/superpowers/specs/2026-09-28-kinematics-model-design.md` for the
design and for known-open parameters.
~~~~

- [ ] **Step 4: Run the full test suite one more time**

Run: `cd utilities/kinematics && python -m pytest -v`
Expected: all pass or skip; no failures.

- [ ] **Step 5: Commit**

```bash
git add utilities/kinematics/plot_arm.py README.md
git commit -m "Add interactive 3D visualization of the arm"
```

---

## Out of scope

Do not build these. They are deferred in the spec for stated reasons:

- **Workspace sweep / reachability scatter.** Blocked on joint limits, which are undefined for all six joints. A sweep without them produces a plot that looks authoritative and means nothing.
- **Inverse kinematics.** The wrist is non-spherical, so IK must be numerical with a convergence-failure path — its own piece of work.
- **URDF export**, and the printed XYZ axis indicator as a second tool transform.
- **Removing the legacy AR4 code.** A separate change that has not been scheduled.
