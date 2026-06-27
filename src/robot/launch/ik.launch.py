from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ik_node = Node(
        package='robot',
        executable='ar4_mk5_ik_node.py',
        name='ar4_mk5_ik_node',
        output='screen',
        parameters=[{
            # AR4 MK5 Standard DH parameters (metres / radians).
            # Override here or via a YAML param file if your build differs.
            'dh.j1.a':            0.0,
            'dh.j1.d':            0.16977,
            'dh.j1.alpha':        1.5707963,   # pi/2
            'dh.j1.theta_offset': 0.0,

            'dh.j2.a':            0.17278,
            'dh.j2.d':            0.0,
            'dh.j2.alpha':        0.0,
            'dh.j2.theta_offset': 0.0,

            'dh.j3.a':            0.0,
            'dh.j3.d':            0.0,
            'dh.j3.alpha':        1.5707963,   # pi/2
            'dh.j3.theta_offset': -1.5707963,  # -pi/2

            'dh.j4.a':            0.0,
            'dh.j4.d':            0.22263,
            'dh.j4.alpha':        -1.5707963,  # -pi/2
            'dh.j4.theta_offset': 0.0,

            'dh.j5.a':            0.0,
            'dh.j5.d':            0.0,
            'dh.j5.alpha':        1.5707963,   # pi/2
            'dh.j5.theta_offset': 0.0,

            'dh.j6.a':            0.0,
            'dh.j6.d':            0.03625,
            'dh.j6.alpha':        0.0,
            'dh.j6.theta_offset': 0.0,
        }],
    )

    return LaunchDescription([ik_node])
