# ros2 launch — fake hardware
ros2 launch franka_moveit_config moveit.launch.py use_fake_hardware:=true load_gripper:=true

# ros2 launch — Isaac Sim (start Isaac Sim first, press Play, then run this)
ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=false

ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true

ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true rviz:=false

# ros2 launch — real robot
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true
