#  Copyright (c) 2021 Franka Emika GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# This file is an adapted version of
# https://github.com/ros-planning/moveit_resources/blob/ca3f7930c630581b5504f3b22c40b4f82ee6369d/panda_moveit_config/launch/demo.launch.py

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, Shutdown)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, FindExecutable, LaunchConfiguration,
                                   PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def validate_launch_args(context):
    use_isaac_sim = LaunchConfiguration('use_isaac_sim').perform(context).lower() == 'true'
    use_fake_hardware = LaunchConfiguration('use_fake_hardware').perform(context).lower() == 'true'
    robot_ip = LaunchConfiguration('robot_ip').perform(context).strip()

    if not use_isaac_sim and not use_fake_hardware and not robot_ip:
        raise RuntimeError(
            "robot_ip must be set when use_isaac_sim:=false and use_fake_hardware:=false"
        )

    return []


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    load_gripper_parameter_name = 'load_gripper'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_isaac_sim = LaunchConfiguration('use_isaac_sim')


    # Command-line arguments

    db_arg = DeclareLaunchArgument(
        'db', default_value='False', description='Database flag'
    )

    # planning_context — uses hand_without_gripper_collision.xacro instead of hand.xacro
    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots',
                                     'panda_arm_without_gripper_collision.urdf.xacro')
    robot_description_config = Command(
        [FindExecutable(name='xacro'), ' ', franka_xacro_file, ' hand:=', load_gripper,
         ' robot_ip:=', robot_ip, ' use_fake_hardware:=', use_fake_hardware,
         ' fake_sensor_commands:=', fake_sensor_commands,
         ' use_isaac_sim:=', use_isaac_sim])

    robot_description = {'robot_description': robot_description_config}

    franka_semantic_xacro_file = os.path.join(get_package_share_directory('franka_moveit_config'),
                                              'srdf',
                                              'panda_arm.srdf.xacro')
    robot_description_semantic_config = Command(
        [FindExecutable(name='xacro'), ' ', franka_semantic_xacro_file, ' hand:=', load_gripper]
    )
    robot_description_semantic = {
        'robot_description_semantic': robot_description_semantic_config
    }

    kinematics_yaml = load_yaml(
        'franka_moveit_config', 'config/kinematics.yaml'
    )

    joint_limits_yaml = {
        'robot_description_planning': load_yaml(
            'franka_moveit_config', 'config/joint_limits.yaml'
        )
    }

    # Planning Functionality
    ompl_planning_pipeline_config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl_planning_yaml = load_yaml(
        'franka_moveit_config', 'config/ompl_planning.yaml'
    )
    ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)

    # Trajectory Execution Functionality
    # Two controller configs: with gripper (load_gripper:=true) and without (load_gripper:=false).
    # Using the wrong one crashes move_group when joints referenced in the config don't exist
    # in the robot model.
    moveit_controllers_with_gripper = {
        'moveit_simple_controller_manager': load_yaml(
            'franka_moveit_config', 'config/panda_controllers.yaml'),
        'moveit_controller_manager': 'moveit_simple_controller_manager'
                                     '/MoveItSimpleControllerManager',
    }
    moveit_controllers_with_gripper_isaac = {
        'moveit_simple_controller_manager': load_yaml(
            'franka_moveit_config', 'config/panda_controllers_isaac.yaml'),
        'moveit_controller_manager': 'moveit_simple_controller_manager'
                                     '/MoveItSimpleControllerManager',
    }
    moveit_controllers_no_gripper = {
        'moveit_simple_controller_manager': load_yaml(
            'franka_moveit_config', 'config/panda_controllers_no_gripper.yaml'),
        'moveit_controller_manager': 'moveit_simple_controller_manager'
                                     '/MoveItSimpleControllerManager',
    }

    trajectory_execution_default = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }
    trajectory_execution_isaac = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.05,
    }

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    common_move_group_params = [
        robot_description,
        robot_description_semantic,
        kinematics_yaml,
        ompl_planning_pipeline_config,
        trajectory_execution_default,
        planning_scene_monitor_parameters,
    ]
    common_move_group_params_isaac = common_move_group_params.copy()
    common_move_group_params_isaac.insert(3, joint_limits_yaml)
    common_move_group_params_isaac[5] = trajectory_execution_isaac

    use_non_isaac_with_gripper = PythonExpression(
        ["'true' if '", load_gripper, "' == 'true' and '", use_isaac_sim, "' == 'false' else 'false'"])
    use_non_isaac_no_gripper = PythonExpression(
        ["'true' if '", load_gripper, "' == 'false' and '", use_isaac_sim, "' == 'false' else 'false'"])
    use_isaac_with_gripper = PythonExpression(
        ["'true' if '", load_gripper, "' == 'true' and '", use_isaac_sim, "' == 'true' else 'false'"])
    use_isaac_no_gripper = PythonExpression(
        ["'true' if '", load_gripper, "' == 'false' and '", use_isaac_sim, "' == 'true' else 'false'"])

    # Start the actual move_group node/action server
    use_isaac_with_gripper = PythonExpression(
        ["'true' if '", use_isaac_sim, "'.lower() == 'true' and '",
         load_gripper, "'.lower() == 'true' else 'false'"])
    use_non_isaac_with_gripper = PythonExpression(
        ["'true' if '", use_isaac_sim, "'.lower() == 'false' and '",
         load_gripper, "'.lower() == 'true' else 'false'"])

    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=common_move_group_params + [moveit_controllers_with_gripper],
        condition=IfCondition(use_non_isaac_with_gripper),
    )
    run_move_group_node_no_gripper = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=common_move_group_params + [moveit_controllers_no_gripper],
        condition=IfCondition(use_non_isaac_no_gripper),
    )
    run_move_group_node_isaac = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=common_move_group_params_isaac + [moveit_controllers_with_gripper_isaac],
        condition=IfCondition(use_isaac_with_gripper),
    )
    run_move_group_node_isaac_no_gripper = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=common_move_group_params_isaac + [moveit_controllers_no_gripper],
        condition=IfCondition(use_isaac_no_gripper),
    )

    # RViz
    rviz_base = os.path.join(get_package_share_directory('franka_moveit_config'), 'rviz')
    rviz_full_config = os.path.join(rviz_base, 'moveit.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_full_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            kinematics_yaml,
        ],
        condition=UnlessCondition(use_isaac_sim),
    )
    rviz_node_isaac = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_full_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            kinematics_yaml,
            joint_limits_yaml,
        ],
        condition=IfCondition(PythonExpression(
            ["'true' if '", use_isaac_sim, "'.lower() == 'true' and '",
             LaunchConfiguration('rviz'), "'.lower() == 'true' else 'false'"])),
    )

    # Publish TF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    ros2_controllers_path = os.path.join(
        get_package_share_directory('franka_moveit_config'),
        'config',
        'panda_ros_controllers.yaml',
    )
    ros2_controllers_path_fake = os.path.join(
        get_package_share_directory('franka_moveit_config'),
        'config',
        'panda_ros_controllers_fake.yaml',
    )
    ros2_controllers_path_isaac = os.path.join(
        get_package_share_directory('franka_moveit_config'),
        'config',
        'panda_ros_controllers_isaac.yaml',
    )

    # Three-way conditions (PythonExpression is safe in Humble;
    # AndSubstitution/NotSubstitution only exist in Iron+)
    use_real_hardware = PythonExpression(
        ["'true' if '", use_fake_hardware, "' == 'false' and '",
         use_isaac_sim, "' == 'false' else 'false'"])
    use_fake_only = PythonExpression(
        ["'true' if '", use_fake_hardware, "' == 'true' and '",
         use_isaac_sim, "' == 'false' else 'false'"])

    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, ros2_controllers_path],
        remappings=[('joint_states', 'franka/joint_states')],
        output={
            'stdout': 'screen',
            'stderr': 'screen',
        },
        on_exit=Shutdown(),
        condition=IfCondition(use_real_hardware),
    )
    ros2_control_node_fake = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, ros2_controllers_path_fake],
        remappings=[('joint_states', 'franka/joint_states')],
        output={
            'stdout': 'screen',
            'stderr': 'screen',
        },
        on_exit=Shutdown(),
        condition=IfCondition(use_fake_only),
    )
    ros2_control_node_isaac = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[ros2_controllers_path_isaac],
        remappings=[('/controller_manager/robot_description', '/robot_description')],
        output={
            'stdout': 'screen',
            'stderr': 'screen',
        },
        on_exit=Shutdown(),
        condition=IfCondition(use_isaac_sim),
    )

    # Load controllers
    load_controllers = [
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['panda_arm_controller', '-c', '/controller_manager'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '-c', '/controller_manager'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['panda_gripper', '-c', '/controller_manager'],
            output='screen',
            condition=IfCondition(use_isaac_with_gripper),
        ),
    ]

    # Warehouse mongodb server
    db_config = LaunchConfiguration('db')
    # mongodb_server_node = Node(
    #     package='warehouse_ros_mongo',
    #     executable='mongo_wrapper_ros.py',
    #     parameters=[
    #         {'warehouse_port': 33829},
    #         {'warehouse_host': 'localhost'},
    #         {'warehouse_plugin': 'warehouse_ros_mongo::MongoDatabaseConnection'},
    #     ],
    #     output='screen',
    #     condition=IfCondition(db_config)
    # )

    # In Isaac mode the joint_state_broadcaster publishes /joint_states directly;
    # this node is only needed for real/fake hardware where sources are remapped.
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'source_list': ['franka/joint_states', 'panda_gripper/joint_states'], 'rate': 30}],
        condition=UnlessCondition(use_isaac_sim),
    )
    use_isaac_sim_arg = DeclareLaunchArgument(
        'use_isaac_sim',
        default_value='false',
        description='Use Isaac Sim as physics backend via topic_based_ros2_control')

    robot_arg = DeclareLaunchArgument(
        robot_ip_parameter_name,
        default_value='',
        description='Hostname or IP address of the robot (not required for fake/Isaac Sim modes).')

    use_fake_hardware_arg = DeclareLaunchArgument(
        use_fake_hardware_parameter_name,
        default_value='false',
        description='Use fake hardware')
    load_gripper_arg = DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='true',
            description='Use Franka Gripper as an end-effector, otherwise, the robot is loaded '
                        'without an end-effector.')

    fake_sensor_commands_arg = DeclareLaunchArgument(
        fake_sensor_commands_parameter_name,
        default_value='false',
        description="Fake sensor commands. Only valid when '{}' is true".format(
            use_fake_hardware_parameter_name))
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Launch RViz2'
    )
    gripper_launch_file = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([PathJoinSubstitution(
            [FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])]),
        launch_arguments={'robot_ip': robot_ip,
                          use_fake_hardware_parameter_name: use_fake_hardware}.items(),
        condition=IfCondition(use_non_isaac_with_gripper)
    )
    return LaunchDescription(
        [robot_arg,
         use_fake_hardware_arg,
         use_isaac_sim_arg,
         fake_sensor_commands_arg,
         load_gripper_arg,
         db_arg,
         rviz_arg,
         rviz_node,
         rviz_node_isaac,
         robot_state_publisher,
         run_move_group_node,
         run_move_group_node_no_gripper,
         run_move_group_node_isaac,
         run_move_group_node_isaac_no_gripper,
         ros2_control_node,
         ros2_control_node_fake,
         ros2_control_node_isaac,
        #  mongodb_server_node,
         joint_state_publisher,
         gripper_launch_file,
         ]
        + load_controllers
    )
