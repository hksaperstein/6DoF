

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution
)
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
def generate_launch_description():

    ar_model_arg = DeclareLaunchArgument(
        name="ar_model",
        default_value="mk5",
        choices=["mk1", "mk2", "mk3", "mk4", "mk5"],
        description="Model of AR4"
    )

    tf_prefix_arg = DeclareLaunchArgument(
        name="tf_prefix",
        default_value="",
        description="tf_tree prefix"
    )

    initial_joint_controllers = PathJoinSubstitution([
        FindPackageShare("annin_ar4_driver"), "config", "controllers.yaml"
    ])
    
    ar_model_config = LaunchConfiguration("ar_model")
    tf_prefix_config = LaunchConfiguration("tf_prefix")
    
    robot_description_content = Command([
        FindExecutable(name="xacro"),
        " ",
        PathJoinSubstitution([
            FindPackageShare("annin_ar4_description"),
            "urdf",
            "ar_gazebo.urdf.xacro",
        ]),
        " ",
        "ar_model:=",
        ar_model_config,
        " ",
        "tf_prefix:=",
        tf_prefix_config,
        " ",
        "simulation_controllers:=",
        initial_joint_controllers,
    ])
    robot_state_publish_parameters = {
        "robot_description": robot_description_content,
        # "use_sim_time": "True"
    }
    
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_state_publish_parameters],
        
    )
    
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2"
    )

    # Gazebo setup
    gz_empty_world = PathJoinSubstitution([
        FindPackageShare("annin_ar4_gazebo"), "worlds", "empty.world"
    ])
    gazebo_node = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare("ros_gz_sim"),
            "launch",
            "gz_sim.launch.py"
        ]),
        launch_arguments = {
            'gz_args': ["-r -v 4 --physics-engine gz-physics-bullet-featherstone-plugin ", gz_empty_world],
            'on_exit_shutdown': "True"
        }.items()
    )

    ros_gz_sim_node = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-name",
            "ar_model_config",
            "-topic",
            "robot_description"
        ]
    )

    # Ros<->GZ Bridge
    ros_gz_bridge_config = PathJoinSubstitution([
        FindPackageShare("robot"),
        "config",
        "gz_bridge.yaml"
    ])
    ros_gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            "config_file": ros_gz_bridge_config
        }]
    )

    # Control Nodes
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster", "-c", "/controller_manager",
            "--controller-manager-timeout", "60"
        ],
    )



    return LaunchDescription([
        ar_model_arg,
        tf_prefix_arg,
        robot_state_publisher_node,
        gazebo_node,
        ros_gz_sim_node,
        ros_gz_bridge_node,
        joint_state_broadcaster_spawner,
        rviz2_node

    ])