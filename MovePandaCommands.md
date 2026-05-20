## Get Joint
python3 /workspace/scriptsget_joints.py

==============


## Panda Ready
ros2 run pymoveit2 panda_ready.py --ros-args -p named_state:=ready


## Panda Gripper
ros2 run pymoveit2 panda_gripper_control.py --ros-args -p action:=open
ros2 run pymoveit2 panda_gripper_control.py --ros-args -p action:=close
ros2 run pymoveit2 panda_gripper_control.py --ros-args -p action:=toggle

==============


## Panda Joint Goal
ros2 run pymoveit2 panda_joint_goal.py --ros-args \
  -p joint_positions:="[0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]"


## Panda Pose Goal
ros2 run pymoveit2 panda_pose_goal.py --ros-args \
  -p position:="[0.45, 0.0, 0.35]" \
  -p quat_xyzw:="[1.0, 0.0, 0.0, 0.0]"



## Panda Planner
ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args -p preset:=ready

ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args \
  -p joint_positions:="[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]"