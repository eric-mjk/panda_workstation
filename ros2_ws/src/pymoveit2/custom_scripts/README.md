# Custom Panda Scripts

This folder is for custom control scripts developed on top of `pymoveit2`.

Build from the ROS workspace:

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select pymoveit2
source install/setup.bash
```

Start the Panda MoveIt launch first:

```bash
ros2 launch franka_moveit_config moveit.launch.py
```

These scripts default to the group/link names used by that launch:

```text
group_name: panda_arm
base_link_name: panda_link0
end_effector_name: panda_link8
```

Then run a custom script:

```bash
ros2 run pymoveit2 panda_joint_goal.py
```

Move to the Panda SRDF `ready` named state:

```bash
ros2 run pymoveit2 panda_ready.py
```

The script also supports the SRDF `extended` state:

```bash
ros2 run pymoveit2 panda_ready.py --ros-args -p named_state:=extended
```

```bash
ros2 run pymoveit2 panda_pose_goal.py --ros-args \
  -p position:="[0.45, 0.0, 0.35]" \
  -p quat_xyzw:="[1.0, 0.0, 0.0, 0.0]"
```

Gripper open/close:

```bash
ros2 run pymoveit2 panda_gripper_control.py --ros-args -p action:=open
ros2 run pymoveit2 panda_gripper_control.py --ros-args -p action:=close
ros2 run pymoveit2 panda_gripper_control.py --ros-args -p action:=toggle
```

The default gripper action server is `panda_gripper/gripper_action`, which is
the name used by `franka_gripper` in `franka_moveit_config/moveit.launch.py`.
For Isaac Sim, use:

```bash
ros2 run pymoveit2 panda_gripper_control.py --ros-args \
  -p action:=open \
  -p action_name:=panda_gripper/gripper_cmd
```

Keyboard Cartesian jogging:

```bash
ros2 run pymoveit2 panda_keyboard_cartesian_jog.py
```

WASD sends small planned end-effector moves in the `panda_link0` frame.
Press `q` to quit.

By default:

```text
w/s: +/-X
a/d: +/-Y
r/f: +/-Z
step_meters: 0.03
cartesian: false
```

With `cartesian:=false`, the script still commands Cartesian EE pose targets,
but uses the regular MoveIt pose planner. Set `cartesian:=true` only if your
`compute_cartesian_path` service is responding reliably.

You can change the Cartesian step size:

```bash
ros2 run pymoveit2 panda_keyboard_cartesian_jog.py --ros-args \
  -p step_meters:=0.01 \
  -p max_velocity:=0.1 \
  -p max_acceleration:=0.1
```

If you later want to command the hand TCP instead of the flange link, pass matching
MoveIt group/link names with ROS parameters.
