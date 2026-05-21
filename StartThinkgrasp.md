# Start ThinkGrasp
===========================
## Local computer
### Real ROBOT
ros2 run pymoveit2 grasp_pipeline_realrobot.py --ros-args -p instruction:="pick up the mustard bottle"
### Sim ROBOT
ros2 run pymoveit2 sim_grasp_pipeline_realrobot.py --ros-args -p instruction:="pick up the red bowl" -p grasp_depth_offset_m:=0.0

===========================
## Server computer
export OPENAI_API_KEY="sk-xxxxx"
cd /workspace/thinkgrasp/ThinkGrasp

export THINKGRASP_SHOW_MATPLOTLIB=0
export THINKGRASP_SHOW_OPEN3D=0
python realarm_upload_server.py