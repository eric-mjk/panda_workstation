# Isaac Sim Config Locations

This note points to the files that control the `use_isaac_sim` behavior.

## Main toggle

The launch argument is declared and routed here:

- `ros2_ws/src/panda_ros2/franka_moveit_config/launch/moveit.launch.py`

What to look for:

- `use_isaac_sim` launch argument
- non-Isaac vs Isaac `move_group` branches
- non-Isaac vs Isaac RViz branches
- real/fake/Isaac `ros2_control_node` selection

## Hardware plugin selection

The hardware backend chosen by `use_isaac_sim` is here:

- `ros2_ws/src/panda_ros2/franka_description/robots/panda_arm.ros2_control.xacro`

Behavior:

- `use_isaac_sim=true`:
  - uses `topic_based_ros2_control/TopicBasedSystem`
  - uses `/isaac_joint_commands` and `/isaac_joint_states`
  - uses `position` command interface
- `use_isaac_sim=false`:
  - falls back to the original path
  - `use_fake_hardware=true` uses `fake_components/GenericSystem`
  - `use_fake_hardware=false` uses `franka_hardware/FrankaHardwareInterface`

The launch file passes the flag into the robot description here:

- `ros2_ws/src/panda_ros2/franka_description/robots/panda_arm.urdf.xacro`

## Controller configs

### When `use_isaac_sim=true`

- `ros2_ws/src/panda_ros2/franka_moveit_config/config/panda_ros_controllers_isaac.yaml`

This is the Isaac-specific `ros2_control` controller config.

### When `use_isaac_sim=false`

- `ros2_ws/src/panda_ros2/franka_moveit_config/config/panda_ros_controllers.yaml`
- `ros2_ws/src/panda_ros2/franka_moveit_config/config/panda_ros_controllers_fake.yaml`

These are the original real-hardware and fake-hardware controller configs.

## MoveIt controller configs

These are used regardless of Isaac, depending on whether the gripper is loaded:

- `ros2_ws/src/panda_ros2/franka_moveit_config/config/panda_controllers.yaml`
- `ros2_ws/src/panda_ros2/franka_moveit_config/config/panda_controllers_no_gripper.yaml`

## Isaac-only planning tweaks

These are only applied on the Isaac path from `moveit.launch.py`:

- `ros2_ws/src/panda_ros2/franka_moveit_config/config/joint_limits.yaml`

Non-Isaac mode now uses the original workflow again:

- no Isaac joint-limits injection
- original trajectory start tolerance

## Related topic-flow note

If you want the ROS topic bridge overview for Isaac Sim, see:

- `IssacSettings.md`
