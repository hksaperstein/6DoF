#!/usr/bin/env python3
"""
AR4 MK5 Inverse Kinematics Node

Subscribes to a target end-effector pose and publishes the IK solution
as joint angles using a damped-least-squares Jacobian solver with
multi-start restarts to avoid local minima.

Topics
------
  Subscribed:  ~/target_pose  (geometry_msgs/PoseStamped)
               ~/joint_states (sensor_msgs/JointState)   -- used as warm start
  Published:   ~/ik_joint_states (sensor_msgs/JointState)
               ~/ik_success      (std_msgs/Bool)
"""

import math
import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


# ---------------------------------------------------------------------------
#  DH-based kinematic solver
# ---------------------------------------------------------------------------

class AR4Kinematics:
    """
    Forward/inverse kinematics for the Annin AR4 MK5 using Standard DH.

    DH table row: [a (m), d (m), alpha (rad), theta_offset (rad)]

    Default values match the physical AR4 MK5 design (all lengths in metres).
    Every value is overridable via the dict passed to __init__.
    """

    # [a,       d,        alpha,        theta_offset]
    _DEFAULT_DH = [
        [0.0,     0.16977,  math.pi / 2,  0.0],          # J1  base rotation
        [0.17278, 0.0,      0.0,           0.0],          # J2  shoulder
        [0.0,     0.0,      math.pi / 2,  -math.pi / 2], # J3  elbow (−90° offset)
        [0.0,     0.22263, -math.pi / 2,   0.0],          # J4  forearm roll
        [0.0,     0.0,      math.pi / 2,   0.0],          # J5  wrist bend
        [0.0,     0.03625,  0.0,            0.0],          # J6  wrist roll
    ]

    # [lower_limit (rad), upper_limit (rad)]
    _DEFAULT_LIMITS = [
        (-math.pi,       math.pi      ),  # J1  ±180°
        (-2.35619,       2.35619      ),  # J2  ±135°
        (-2.35619,       2.35619      ),  # J3  ±135°
        (-math.pi,       math.pi      ),  # J4  ±180°
        (-2.35619,       2.35619      ),  # J5  ±135°
        (-math.pi,       math.pi      ),  # J6  ±180°
    ]

    JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3',
                   'joint_4', 'joint_5', 'joint_6']

    def __init__(self, dh=None, limits=None):
        self.dh = dh if dh is not None else list(self._DEFAULT_DH)
        self.limits = limits if limits is not None else list(self._DEFAULT_LIMITS)

    # ------------------------------------------------------------------
    #  Forward kinematics
    # ------------------------------------------------------------------

    @staticmethod
    def _dh_mat(a, d, alpha, theta):
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([
            [ct,  -st * ca,  st * sa,  a * ct],
            [st,   ct * ca, -ct * sa,  a * st],
            [0.0,       sa,       ca,       d],
            [0.0,      0.0,      0.0,     1.0],
        ])

    def fk(self, q, n=6):
        """Return 4x4 homogeneous transform for the first *n* joints."""
        T = np.eye(4)
        for i in range(n):
            a, d, alpha, off = self.dh[i]
            T = T @ self._dh_mat(a, d, alpha, float(q[i]) + off)
        return T

    # ------------------------------------------------------------------
    #  Jacobian (numerical finite-difference)
    # ------------------------------------------------------------------

    def _jacobian(self, q, eps=1e-6):
        T0 = self.fk(q)
        p0 = T0[:3, 3]
        R0 = T0[:3, :3]
        J = np.zeros((6, 6))
        for i in range(6):
            qp = q.copy()
            qp[i] += eps
            T1 = self.fk(qp)
            # Linear part
            J[:3, i] = (T1[:3, 3] - p0) / eps
            # Angular part via skew-symmetric dR/dq * R^T
            dR = (T1[:3, :3] - R0) / eps
            skew = dR @ R0.T
            J[3, i] = skew[2, 1]
            J[4, i] = skew[0, 2]
            J[5, i] = skew[1, 0]
        return J

    # ------------------------------------------------------------------
    #  Pose error (position + axis-angle orientation)
    # ------------------------------------------------------------------

    @staticmethod
    def _pose_error(T_cur, T_tgt):
        dp = T_tgt[:3, 3] - T_cur[:3, 3]
        R_err = T_tgt[:3, :3] @ T_cur[:3, :3].T
        cos_a = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        angle = math.acos(cos_a)
        if angle < 1e-9:
            dr = np.zeros(3)
        else:
            s = math.sin(angle)
            dr = (angle / (2.0 * s)) * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])
        return np.concatenate([dp, dr])

    # ------------------------------------------------------------------
    #  Joint limit helpers
    # ------------------------------------------------------------------

    def _clamp(self, q):
        return np.array([
            np.clip(qi, lo, hi)
            for qi, (lo, hi) in zip(q, self.limits)
        ])

    def within_limits(self, q):
        return all(lo <= qi <= hi for qi, (lo, hi) in zip(q, self.limits))

    # ------------------------------------------------------------------
    #  Numerical IK  (damped least squares)
    # ------------------------------------------------------------------

    def _solve_from_seed(self, T_tgt, q0,
                         max_iter=300, tol=5e-5,
                         damping=0.05, step=0.5):
        q = self._clamp(np.array(q0, dtype=float))
        for _ in range(max_iter):
            T_cur = self.fk(q)
            err = self._pose_error(T_cur, T_tgt)
            if np.linalg.norm(err) < tol:
                return q if self.within_limits(q) else None
            J = self._jacobian(q)
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + damping ** 2 * np.eye(6), err)
            q = self._clamp(q + step * dq)
        return None

    def solve(self, T_tgt, q_warm=None, n_restarts=10, rng_seed=0):
        """
        Try multiple starting configurations.
        Returns the valid solution closest to *q_warm* (or zero), or None.
        """
        seeds = []
        if q_warm is not None:
            seeds.append(np.array(q_warm, dtype=float))
        seeds.append(np.zeros(6))

        rng = np.random.default_rng(rng_seed)
        for _ in range(n_restarts):
            seeds.append(np.array([
                rng.uniform(lo, hi) for lo, hi in self.limits
            ]))

        ref = np.array(q_warm, dtype=float) if q_warm is not None else np.zeros(6)
        best, best_dist = None, float('inf')

        for seed in seeds:
            sol = self._solve_from_seed(T_tgt, seed)
            if sol is not None:
                dist = float(np.sum((sol - ref) ** 2))
                if dist < best_dist:
                    best, best_dist = sol, dist

        return best


# ---------------------------------------------------------------------------
#  ROS 2 node
# ---------------------------------------------------------------------------

def _quaternion_to_matrix(qx, qy, qz, qw):
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw),  1 - 2*(qx**2+qz**2), 2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2+qy**2)],
    ])


class AR4IKNode(Node):

    def __init__(self):
        super().__init__('ar4_mk5_ik_node')

        dh = self._declare_dh_params()
        self._solver = AR4Kinematics(dh=dh)
        self._q_current = None  # warm-start from joint_states

        self._sub_pose = self.create_subscription(
            PoseStamped, 'target_pose', self._on_target_pose, 10)
        self._sub_js = self.create_subscription(
            JointState, 'joint_states', self._on_joint_states, 10)

        self._pub_js = self.create_publisher(JointState, 'ik_joint_states', 10)
        self._pub_ok = self.create_publisher(Bool, 'ik_success', 10)

        self.get_logger().info('AR4 MK5 IK node ready')
        self.get_logger().info(
            f'DH params (a, d, alpha, offset): {self._solver.dh}')

    # ------------------------------------------------------------------

    def _declare_dh_params(self):
        """Declare and retrieve per-joint DH parameters from ROS params."""
        defaults = AR4Kinematics._DEFAULT_DH
        dh = []
        for i, row in enumerate(defaults):
            j = i + 1
            a     = self.declare_parameter(f'dh.j{j}.a',            row[0]).value
            d     = self.declare_parameter(f'dh.j{j}.d',            row[1]).value
            alpha = self.declare_parameter(f'dh.j{j}.alpha',        row[2]).value
            off   = self.declare_parameter(f'dh.j{j}.theta_offset', row[3]).value
            dh.append([a, d, alpha, off])
        return dh

    # ------------------------------------------------------------------

    def _on_joint_states(self, msg: JointState):
        """Cache the current joint positions as a warm start for IK."""
        names = AR4Kinematics.JOINT_NAMES
        if len(msg.position) < 6:
            return
        try:
            q = [msg.position[msg.name.index(n)] for n in names]
            self._q_current = q
        except ValueError:
            pass

    def _on_target_pose(self, msg: PoseStamped):
        o = msg.pose.orientation
        p = msg.pose.position
        R = _quaternion_to_matrix(o.x, o.y, o.z, o.w)
        T_tgt = np.eye(4)
        T_tgt[:3, :3] = R
        T_tgt[0, 3] = p.x
        T_tgt[1, 3] = p.y
        T_tgt[2, 3] = p.z

        solution = self._solver.solve(T_tgt, q_warm=self._q_current)

        ok_msg = Bool(data=solution is not None)
        self._pub_ok.publish(ok_msg)

        if solution is None:
            self.get_logger().warn(
                f'IK: no solution for pose '
                f'({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
            return

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.header.frame_id = 'base_link'
        js.name = list(AR4Kinematics.JOINT_NAMES)
        js.position = solution.tolist()
        self._pub_js.publish(js)

        deg = [math.degrees(q) for q in solution]
        self.get_logger().info(
            f'IK solved [deg]: '
            f'J1={deg[0]:+.1f}  J2={deg[1]:+.1f}  J3={deg[2]:+.1f}  '
            f'J4={deg[3]:+.1f}  J5={deg[4]:+.1f}  J6={deg[5]:+.1f}')


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = AR4IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
