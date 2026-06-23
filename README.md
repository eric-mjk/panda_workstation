# panda_workstation

ROS 2 Humble workspace for the Franka Emika Panda arm. Supports fake hardware, Isaac Sim, real robot, and a ThinkGrasp vision-language grasping pipeline.

---

## Quick Start

```bash
git clone https://github.com/eric-mjk/panda_workstation.git ~/Eric/panda_workstation
cd ~/Eric/panda_workstation
git submodule update --init --recursive
```

Start the container (mounts the repo as `/workspace`):

```bash
docker run -it -d \
  --ipc host --net host --privileged \
  -v /dev:/dev -v /dev/bus/usb:/dev/bus/usb \
  --name eric_panda_workstation \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Eric/panda_workstation:/workspace \
  ericmjk/panda_ws:mod_vanilla
```
for systems with nvidia gpus
```

```bash
docker run -it -d \
  --ipc host --net host --privileged \
  --gpus all \
  -v /dev:/dev -v /dev/bus/usb:/dev/bus/usb \
  --name eric_panda_workstation \
  -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/.Xauthority:/root/.Xauthority:rw \
  -e XAUTHORITY=/root/.Xauthority \
  -v ~/Eric/panda_workstation:/workspace \
  ericmjk/panda_ws:mod_vanilla
```


First-time build (inside container):

```bash
sudo apt install ros-humble-ros-testing
cd /workspace/ros2_ws
colcon build
source install/setup.bash
```

---

## File Structure

```
panda_workstation/
├── ros2_ws/                         # ROS 2 colcon workspace
│   └── src/
│       ├── panda_ros2/              # [submodule] Franka hardware driver
│       ├── pymoveit2/               # [submodule] Python MoveIt 2 client
│       ├── topic_based_ros2_control/ # [submodule] Isaac Sim ↔ ros2_control bridge
│       ├── realsese_bringup/        # RealSense camera driver and TF publisher
│       └── tsdf_voxel/              # TSDF 3D reconstruction from RGBD streams
│
├── thinkgrasp/
│   └── ThinkGrasp/                  # Vision-language grasp detection (CoRL 2024)
│
├── scripts/                         # Standalone utilities (no ROS runtime needed)
│   ├── project_to_image.py          # Project a 3D world-frame point onto a camera image
│   └── save_camera_image.py         # Save a RealSense snapshot
│
└── ClientServerTests/               # TCP client/server skeletons for local↔GPU comms
    ├── ClientServerTutorial.md      # Protocol docs + working examples
    ├── test_client_local.py
    └── test_server_local.py
```

### ros2_ws/src/panda_ros2

Franka Emika driver packages for `ros2_control`. Bridges `libfranka` with the ROS 2 control ecosystem.

| Package | What it does |
|---|---|
| `franka_hardware` | `ros2_control` hardware plugin; wraps libfranka; exposes joint state/command interfaces |
| `franka_control2` | Main control node; real-time `SCHED_FIFO` executor; runs the read/update/write loop |
| `franka_msgs` | Franka-specific messages, services (`SetJointStiffness`, `ErrorRecovery`), and actions (`Grasp`, `Homing`) |
| `franka_example_controllers` | 8+ example controllers: joint velocity/position/impedance, Cartesian velocity, gravity compensation |
| `franka_robot_state_broadcaster` | Publishes `FrankaState` (joints, forces, torques, Cartesian pose, collision data) |
| `franka_gripper` | Action server for the Franka Hand (Grasp, Move, Homing) |
| `franka_description` | URDF/Xacro robot descriptions; single-arm and dual-arm |
| `franka_bringup` | Launch files and controller YAML configs |
| `franka_moveit_config` | MoveIt 2 config; supports `use_fake_hardware`, `use_isaac_sim`, and real-robot modes |

### ros2_ws/src/pymoveit2

Forked Python MoveIt 2 client. Key classes: `MoveIt2`, `MoveIt2Gripper`, `MoveIt2Servo`.

Additional scripts added in this fork:

| Path | What it does |
|---|---|
| `custom_scripts/panda_ready.py` | Move arm to home position |
| `custom_scripts/panda_joint_goal.py` | Send a joint-space goal |
| `custom_scripts/panda_pose_goal.py` | Send a Cartesian pose goal |
| `custom_scripts/panda_gripper_control.py` | Open/close gripper |
| `custom_scripts/sim_panda_gripper_control.py` | Gripper control for Isaac Sim |
| `thinkgrasp/grasp_pipeline_realrobot.py` | Full grasp pipeline for real robot |
| `thinkgrasp/sim_grasp_pipeline_realrobot.py` | Full grasp pipeline for Isaac Sim |
| `thinkgrasp/send_grasp_request_realrobot.py` | Send a grasp request to the ThinkGrasp server |

Run any script with `ros2 run pymoveit2 <script_name>`.

### ros2_ws/src/realsese_bringup

RealSense camera integration for ROS 2.

| File | What it does |
|---|---|
| `launch/realrobot_camera.launch.py` | Bringup for real RealSense camera |
| `launch/sim_camera.launch.py` | Bringup for Isaac Sim virtual camera |
| `realsese_bringup/realsense_publisher.py` | Publishes color, depth, and aligned-depth topics from hardware |
| `realsese_bringup/camera_tf_publisher.py` | Publishes camera TF relative to `panda_link0` using forward kinematics |
| `realsese_bringup/rgbd_pointcloud.py` | Converts real RGBD frames to a `sensor_msgs/PointCloud2` |
| `realsese_bringup/sim_rgbd_pointcloud.py` | Same, but for Isaac Sim camera topics |
| `realsese_bringup/transforms.py` | Shared TF / geometry utilities |

### ros2_ws/src/tsdf_voxel

TSDF-based 3D reconstruction using Open3D. Subscribes to RGB + aligned depth topics, integrates frames into a TSDF volume, and exports meshes as `.ply` files.

| File | What it does |
|---|---|
| `tsdf_voxel/tsdf_integrator.py` | Continuous integration node — integrates every incoming frame |
| `tsdf_voxel/tsdf_integrator_once.py` | Keyboard-triggered node — press a key to integrate, `s` to save mesh |
| `launch/tsdf_integrator.launch.py` | Launch file for the continuous integrator |
| `config/real_robot_camera.yaml` | Camera intrinsics + extrinsics for the real RealSense |
| `config/sim_robot_camera.yaml` | Camera intrinsics + extrinsics for Isaac Sim |
| `config/tsdf_debug.yaml` | TSDF volume params (voxel size, truncation, world bounds) |

Meshes are saved to `/workspace/ros2_ws/tsdf_meshes/`. Requires `numpy<1.25`.

### thinkgrasp/ThinkGrasp

Vision-language grasp detection system (CoRL 2024). Uses LangSAM for segmentation and FGC-GraspNet for 6-DOF grasp pose estimation. Runs on the GPU server PC, communicating with the local PC over TCP port 5050.

Key entry point: `realarm_upload_server.py` — starts the Flask/socket server that receives RGBD frames from the local PC, runs segmentation and grasp planning, and returns grasp poses.

### scripts/

Standalone utilities that run on the host without a full ROS session.

| File | What it does |
|---|---|
| `project_to_image.py` | Projects a 3D world-frame point onto a camera image given a manually-supplied camera pose (from `tf2_echo`) |
| `save_camera_image.py` | Saves a color + depth snapshot from a RealSense camera |

### ClientServerTests/

TCP client/server skeletons for the local ↔ GPU server communication pattern. `ClientServerTutorial.md` documents the wire protocol and has working examples of streaming RGBD + joint state from the local PC to the GPU server for SAM3 inference.

---

## Docker Images

| Image | Use |
|---|---|
| `ericmjk/panda_ws:latest` | ROS 2 + MoveIt + libfranka (development) |
| `ericmjk/panda_ws:vanilla` | Minimal ROS 2 base |
| `ericmjk/panda_thinkgrasp:sim` | ThinkGrasp + SAM3 (GPU server) |
