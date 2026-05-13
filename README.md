# Panda_ws

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/eric-mjk/panda_ws.git ~/Eric/panda_ws
```

### 2. Start the container
```
docker run -it -d --net=host --ipc=host \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --gpus all\
  --privileged\
  --name eric_panda\
  -v ~/Eric/panda_ws:/workspace \
  ericmjk/panda_thinkgrasp:sim
```

### 3. First-time setup (inside container)
```bash
cd /workspace
git submodule update --init --recursive

sudo apt update
sudo apt install ros-humble-ros-testing
```

### 4. Build (inside container, from `/workspace/ros2_ws`)
```bash
colcon build
source /workspace/ros2_ws/install/setup.bash
```

---

## Running

### Fake hardware (no robot or simulator needed)
```bash
ros2 launch franka_moveit_config moveit.launch.py use_fake_hardware:=true load_gripper:=true
```

### Isaac Sim
Download the scene file: [sim.usda](https://drive.google.com/file/d/1yR3XmFyKMNpFvo22lVXZy3M6fGLrfBA_/view?usp=sharing)

Open the `.usda` file in Isaac Sim and press **Play**, then run:
```bash
ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true
```

---

## Docker Images

- [`ericmjk/panda_ws`](https://hub.docker.com/repository/docker/ericmjk/panda_ws/general)
- [`ericmjk/panda_thinkgrasp`](https://hub.docker.com/repository/docker/ericmjk/panda_thinkgrasp/general)
