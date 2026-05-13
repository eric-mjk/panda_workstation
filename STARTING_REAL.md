# Start Camera
ros2 launch pymoveit2 thinkgrasp_camera.launch.py



# Start Moveit
## fake hardware
ros2 launch franka_moveit_config moveit.launch.py use_fake_hardware:=true load_gripper:=true

## real robot
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true



# Start Thinkgrasp
## local computer
ros2 run pymoveit2 grasp_pipeline_realrobot.py --ros-args -p instruction:="pick up the mustard bottle"

## server computer
export OPENAI_API_KEY="sk-xxxxx"
cd /workspace/thinkgrasp/ThinkGrasp

export THINKGRASP_SHOW_MATPLOTLIB=0
export THINKGRASP_SHOW_OPEN3D=0
python realarm_upload_server.py