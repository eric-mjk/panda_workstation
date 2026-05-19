# panda_workstation

ROS 2 Humble workspace for the Franka Emika Panda arm. Supports fake hardware, Isaac Sim, real robot, and a ThinkGrasp vision-language grasping pipeline.

---

## Setup

### 1. Clone

```bash
git clone https://github.com/eric-mjk/panda_workstation.git ~/Eric/panda_workstation
cd ~/Eric/panda_workstation
git submodule update --init --recursive
```

### 2. Start the container

```bash
docker run -it -d \
  --gpus all --ipc host --net host --privileged \
  -v /dev:/dev -v /dev/bus/usb:/dev/bus/usb \
  --name panda \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Eric/panda_workstation:/workspace \
  ericmjk/panda_ws:latest
```

### 3. First-time build (inside container)

```bash
sudo apt install ros-humble-ros-testing
cd /workspace/ros2_ws
colcon build
source install/setup.bash
```

---

## Running

Source the workspace before any `ros2` command:
```bash
source /workspace/ros2_ws/install/setup.bash
```

### Fake hardware
No robot or simulator needed.
```bash
ros2 launch franka_moveit_config moveit.launch.py use_fake_hardware:=true load_gripper:=true
```

### Isaac Sim
Open [sim.usda](https://drive.google.com/file/d/1yR3XmFyKMNpFvo22lVXZy3M6fGLrfBA_/view?usp=sharing) in Isaac Sim and press **Play**, then:
```bash
ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true
```

### Real robot
```bash
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true
```

### ThinkGrasp pipeline
Requires `ericmjk/panda_thinkgrasp:sim` on the GPU server. Run each step in a separate terminal.

```bash
# 1. Camera (local PC)
ros2 launch realsese_bringup realrobot_camera.launch.py

# 2. MoveIt (local PC)
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true

# 3. Grasp pipeline (local PC)
ros2 run pymoveit2 grasp_pipeline_realrobot.py --ros-args -p instruction:="pick up the mustard bottle"

# 4. ThinkGrasp server (GPU server PC)
export OPENAI_API_KEY="sk-..."
cd /workspace/thinkgrasp/ThinkGrasp
python realarm_upload_server.py
```

---

## Docker Images

| Image | Use |
|---|---|
| [`ericmjk/panda_ws:latest`](https://hub.docker.com/r/ericmjk/panda_ws) | ROS 2 + MoveIt + libfranka |
| `ericmjk/panda_ws:vanilla` | Minimal base |
| `ericmjk/panda_thinkgrasp:sim` | ThinkGrasp + SAM3 (GPU server) |
