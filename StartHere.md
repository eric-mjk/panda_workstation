# Start Camera

## Real robot camera
ros2 launch realsese_bringup realrobot_camera.launch.py

## Isaac Sim camera
ros2 launch realsese_bringup sim_camera.launch.py


# Start MoveIt

## Fake hardware
ros2 launch franka_moveit_config moveit.launch.py use_fake_hardware:=true load_gripper:=true

## Isaac Sim
Start Isaac Sim first, press Play, then run one of these:

ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=false

ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true

ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true rviz:=false

## Real robot
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true

ros2 launch franka_moveit_config moveit_without_gripper_collision.launch.py robot_ip:=172.16.0.2 load_gripper:=true

# Start ThinkGrasp

## Local computer
ros2 run pymoveit2 grasp_pipeline_realrobot.py --ros-args -p instruction:="pick up the mustard bottle"

### For sim
ros2 run pymoveit2 sim_grasp_pipeline_realrobot.py --ros-args -p instruction:="pick up the red bowl" -p grasp_depth_offset_m:=0.0


## Server computer
export OPENAI_API_KEY="sk-xxxxx"
cd /workspace/thinkgrasp/ThinkGrasp

export THINKGRASP_SHOW_MATPLOTLIB=0
export THINKGRASP_SHOW_OPEN3D=0
python realarm_upload_server.py
