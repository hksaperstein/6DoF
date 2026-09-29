# 6DoF
Playground for a 6DoF robotic manipulator with ROS2

## Kinematics

`utilities/kinematics/` holds the arm's kinematic configuration (`arm.yaml`),
a hand-written forward-kinematics model, and an interactive 3D view:

```bash
cd utilities/kinematics
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
PYTHONPATH= python -m pytest   # verify the DH table
python plot_arm.py             # interactive view
```

The empty `PYTHONPATH=` matters in a ROS-sourced shell: ROS leaks a broken
`launch_testing` pytest plugin onto the path, which crashes collection before
any test runs. Clearing it for that one command avoids the problem without
disturbing the ROS environment.

See `docs/superpowers/specs/2026-09-28-kinematics-model-design.md` for the
design and for known-open parameters.
