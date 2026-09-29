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


def test_zero_q_pose_pins_every_parameter_to_its_dh_slot(cfg):
    """The suite's primary guard on DH slot assignment and alpha signs.

    At q=0 the arm lies extended along +x, so each expected coordinate is
    checkable by hand. Moving any length between an `a` and a `d` slot, or
    flipping the sign of any alpha, changes these numbers.

    This is the zero-q pose, which coincides with the home pose only while every
    theta_offset is zero. When a home pose is chosen and offsets are introduced
    (see the spec's open items), these expectations must be RE-DERIVED for the
    new zero — not weakened or deleted. Nothing else in the suite covers the
    sign of an alpha.
    """
    d0 = cfg.value("d0_base_height")
    a1 = cfg.value("a1_link_upper")
    a2 = cfg.value("a2_link_forearm")
    a4 = cfg.value("a4_yaw_to_roll")
    d5 = cfg.value("d5_tool_length")
    pose = forward_kinematics(cfg, np.zeros(6))
    assert np.allclose(pose.joint_origins[2], [a1, 0.0, d0])
    assert np.allclose(pose.joint_origins[3], [a1 + a2, 0.0, d0])
    assert np.allclose(pose.tcp_position, [a1 + a2 + a4, -d5, d0])


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


_PARAM_NAMES = sorted(load_config(ARM_YAML).parameters)


@pytest.mark.parametrize("name", _PARAM_NAMES)
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
    """Guards a parameter declared in arm.yaml but wired into no DH row, so that
    changing it moves nothing at all.

    Note this is NOT the d3_roll_offset failure: that parameter WAS wired (into
    row 3's d slot) and did move geometry — just not the geometry it was named
    for. That class is caught by
    test_every_parameter_moves_the_quantity_it_is_named_for.
    """
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


def test_consecutive_joint_axes_are_separated_by_exactly_alpha(cfg, poses):
    """alpha_i is the twist between joint axis i and axis i+1, and is
    pose-independent. This pins every alpha magnitude in the table without
    reference to any home pose. Signs are pinned by the zero-q test."""
    for q in poses:
        axes = forward_kinematics(cfg, q).joint_axes
        for i, row in enumerate(cfg.rows[:-1]):
            cos = float(np.dot(axes[i], axes[i + 1]))
            assert cos == pytest.approx(math.cos(row.alpha), abs=1e-9)


def test_closed_form_dh_matches_elementary_factorization(cfg, poses):
    """Independent derivation of the same transform. The classical DH row is
    DEFINED as Rot(z,theta) Trans(z,d) Trans(x,a) Rot(x,alpha); dh_transform
    ships the multiplied-out closed form. Building it from the four elementary
    matrices instead exercises a different derivation path, so a transcription
    error in any entry of the closed-form 4x4 cannot survive both."""
    def rot_z(t):
        c, s = np.cos(t), np.sin(t)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        return T

    def rot_x(t):
        c, s = np.cos(t), np.sin(t)
        T = np.eye(4)
        T[1:3, 1:3] = [[c, -s], [s, c]]
        return T

    def trans_z(d):
        T = np.eye(4)
        T[2, 3] = d
        return T

    def trans_x(a):
        T = np.eye(4)
        T[0, 3] = a
        return T

    for q in poses:
        expected = np.eye(4)
        for i, row in enumerate(cfg.rows):
            expected = expected @ rot_z(q[i]) @ trans_z(row.d) @ trans_x(row.a) @ rot_x(row.alpha)
        assert np.allclose(forward_kinematics(cfg, q).tcp, expected, atol=1e-9)
