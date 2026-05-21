# Setup Robot
- Franka FCI
Activate FCI / But press the stop trigger

- Move to target pose manually

- Unpress the trigger

# Setup Moveit
- Camera Launch
ros2 launch realsese_bringup realrobot_camera.launch.py

- Moveit Rviz Launch
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true

# Record the gripper pose
- copy joint states
ros2 topic echo /joint_states --once | python3 -c "
import sys, yaml
msg = yaml.safe_load(sys.stdin.read().replace('---',''))
print(msg['position'][:7])
"

[0.010833520069456938, 0.2770359245405734, 0.03475776986997473, -2.4844073507571984, 0.011520724211080594, 2.7799502530157776, 2.0290077956688726]