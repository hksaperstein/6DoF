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


def test_home_pose_frames_match_hand_computed_geometry(cfg):
    """At q=0 the arm lies extended along +x, so these are checkable by
    inspection: shoulder at base height, then two 200 mm links out along x,
    the 20 mm wrist offset adding to x, and the tool hanging along -y."""
    d0 = cfg.value("d0_base_height")
    a1 = cfg.value("a1_link_upper")
    a2 = cfg.value("a2_link_forearm")
    a4 = cfg.value("a4_yaw_to_roll")
    d5 = cfg.value("d5_tool_length")
    pose = forward_kinematics(cfg, np.zeros(6))
    assert np.allclose(pose.joint_origins[2], [a1, 0.0, d0])
    assert np.allclose(pose.joint_origins[3], [a1 + a2, 0.0, d0])
    assert np.allclose(pose.tcp_position, [a1 + a2 + a4, -d5, d0])


def test_frame_accumulation_is_left_to_right(cfg):
    """Guards the exact regression the composition order invites: building
    the chain as dh @ T instead of T @ dh."""
    q = np.array([0.3, -0.4, 0.5, -0.6, 0.7, -0.2])
    pose = forward_kinematics(cfg, q)
    expected = np.eye(4)
    for i, row in enumerate(cfg.rows):
        expected = expected @ dh_transform(q[i], row.d, row.a, row.alpha)
        assert np.allclose(pose.frames[i + 1], expected)
