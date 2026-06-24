========================
# REAL ROBOT
## Start Camera
ros2 launch realsese_bringup realrobot_camera.launch.py

## Start MoveIt
### Standard
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true
### Without Gripper Collision
ros2 launch franka_moveit_config moveit_without_gripper_collision.launch.py robot_ip:=172.16.0.2 load_gripper:=true


========================
# SIM / FAKE ROBOT
## Start Camera
ros2 launch realsese_bringup sim_camera.launch.py


## Start MoveIt
### For Fake hardware
ros2 launch franka_moveit_config moveit.launch.py use_fake_hardware:=true load_gripper:=true
## For Isaac Sim
ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true