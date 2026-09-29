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


D0_SLIDER_MAX = 600.0


def main():
    base_cfg = load_config(ARM_YAML)
    reach = D0_SLIDER_MAX + sum(base_cfg.value(n) for n in
                ("a1_link_upper", "a2_link_forearm", "a4_yaw_to_roll", "d5_tool_length"))

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=0.05, right=0.72, bottom=0.05, top=0.95)

    sliders = {}
    for i, label in enumerate(JOINT_LABELS):
        # +/-180 deg is a UI range, not a joint limit; limits are undefined
        # (see spec Open items).
        sliders[label] = Slider(
            fig.add_axes([0.78, 0.88 - 0.05 * i, 0.18, 0.03]),
            label, -180.0, 180.0, valinit=0.0,
        )
    sliders["d0_base_height"] = Slider(
        fig.add_axes([0.78, 0.50, 0.18, 0.03]),
        "d0 base", 0.0, D0_SLIDER_MAX, valinit=base_cfg.value("d0_base_height"),
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
