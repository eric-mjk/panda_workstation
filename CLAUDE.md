# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Docker Environment

This workspace is designed to run inside a Docker container. The host directory `~/Eric/panda_workstation` is mounted as `/workspace` inside the container.

**Start the container:**
```bash
docker run -it -d \
  --gpus all \
  --ipc host \
  --net host \
  --privileged \
  -v /dev:/dev \
  -v /dev/bus/usb:/dev/bus/usb \
  --name panda \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
  -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Eric/panda_workstation:/workspace \
  ericmjk/panda_ws:vanilla
```

Available images:
- `ericmjk/panda_ws:vanilla` / `ericmjk/panda_ws:latest` — ROS 2 + MoveIt + libfranka
- `ericmjk/panda_thinkgrasp:sim` — includes ThinkGrasp + SAM3 (use for grasp pipeline)

## Workspace Layout

```
/workspace/   (= ~/Eric/panda_workstation on host)
  ros2_ws/          ← ROS 2 colcon workspace
    src/
      panda_ros2/   ← Franka hardware driver packages (forked submodule)
      pymoveit2/    ← Python MoveIt 2 client library (submodule)
      topic_based_ros2_control/  ← Isaac Sim ↔ ros2_control bridge via topics (submodule)
      realsese_bringup/          ← RealSense camera ROS 2 bringup and camera helpers
  thinkgrasp/ThinkGrasp/        ← vision-language grasp detection system
  ClientServerTests/            ← socket client/server skeletons and tests
```

All `src/` directories are git submodules. Initialize with:
```bash
git submodule update --init --recursive
```

Submodule URLs:
- `ros2_ws/src/panda_ros2` → `https://github.com/eric-mjk/_forked_panda_ros2.git` (branch: humble)
- `ros2_ws/src/pymoveit2` → `https://github.com/eric-mjk/_forked_pymoveit2.git` (branch: main)
- `ros2_ws/src/topic_based_ros2_control` → `https://github.com/PickNikRobotics/topic_based_ros2_control.git`

## Build & Test Commands

All commands run **inside the container** from `/workspace/ros2_ws`.

```bash
# First-time only: install test dependency
sudo apt install ros-humble-ros-testing

# Build
colcon build --cmake-args -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCHECK_TIDY=ON

# Build a single package
colcon build --packages-select <package_name>

# Run tests
colcon test
colcon test-result

# Run tests for a single package
colcon test --packages-select <package_name>
```

Before running anything, source the workspace: `source /workspace/ros2_ws/install/setup.bash`

> All paths below use `/workspace` (the in-container path).

## Launch Commands

```bash
# Fake hardware (no robot needed — for development/simulation)
ros2 launch franka_moveit_config moveit.launch.py use_fake_hardware:=true load_gripper:=true

# Isaac Sim (start Isaac Sim first and press Play, then run this)
# Scene file: https://drive.google.com/file/d/1yR3XmFyKMNpFvo22lVXZy3M6fGLrfBA_/view
ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true
ros2 launch franka_moveit_config moveit.launch.py use_isaac_sim:=true load_gripper:=true rviz:=false

# Real robot (IP: 172.16.0.2)
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true

# Hardware-only bringup (without MoveIt)
ros2 launch franka_bringup franka.launch.py robot_ip:=172.16.0.2 use_fake_hardware:=false
```

Controller management (after bringup):
```bash
# List available/active controllers
ros2 control list_controllers

# Spawn and activate a controller
ros2 control load_controller --set-state active joint_velocity_example_controller

# Switch active controller
ros2 control switch_controllers --activate joint_impedance_example_controller \
  --deactivate joint_velocity_example_controller
```

## ThinkGrasp Pipeline

Full pipeline for real-robot grasping. Run each step in a separate terminal.

**1. Start camera** (local/robot PC, inside container):
```bash
ros2 launch realsese_bringup realrobot_camera.launch.py
```

**2. Start MoveIt** (local/robot PC):
```bash
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=172.16.0.2 load_gripper:=true
```

**3. Run grasp pipeline** (local/robot PC):
```bash
ros2 run pymoveit2 grasp_pipeline_realrobot.py --ros-args -p instruction:="pick up the mustard bottle"
```

**4. Start ThinkGrasp server** (GPU server PC, inside container):
```bash
export OPENAI_API_KEY="sk-..."
cd /workspace/thinkgrasp/ThinkGrasp
export THINKGRASP_SHOW_MATPLOTLIB=0
export THINKGRASP_SHOW_OPEN3D=0
python realarm_upload_server.py
```

## Linting & Formatting

- **C++ formatting**: Chromium style, C++14, column limit 100 (`src/panda_ros2/.clang-format`)
- **C++ static analysis**: Comprehensive rules in `src/panda_ros2/.clang-tidy` (lower_case variables, CamelCase classes/structs)
- Linting is enforced via `ament_lint_auto` which runs: `ament_clang_format`, `ament_clang_tidy`, `ament_cppcheck`, `ament_copyright`, `ament_flake8`, `ament_pep257`, `ament_xmllint`

## Architecture Overview

### ROS 2 Control Stack (panda_ros2)

This is a **ROS 2 Humble hardware driver and control framework** for Franka Emika Panda robot arms. It bridges `libfranka` (Franka's C++ SDK) with the `ros2_control` ecosystem.

**Control flow:**
```
libfranka (robot hardware)
    ↓
FrankaHardwareInterface (franka_hardware) — plugin loaded by ros2_control
    ↓ read()
Controller Manager — runs the real-time control loop
    ↓ update()
Example Controllers (franka_example_controllers)
    ↓ write()
FrankaHardwareInterface → libfranka → robot
```

The main control node (`franka_control2`) creates a multi-threaded executor with `SCHED_FIFO` real-time scheduling (priority 50). The hardware interface is loaded as a `pluginlib` plugin.

**Package responsibilities** (all under `ros2_ws/src/panda_ros2/`):

| Package | Role |
|---|---|
| `franka_hardware` | `ros2_control` hardware interface plugin; wraps `libfranka`; exposes state/command interfaces; hosts parameter services (stiffness, load, frames, collision behavior) and error recovery |
| `franka_control2` | Main control node binary; sets up `ControllerManager`, real-time executor, period-based `read/update/write` loop |
| `franka_msgs` | All Franka-specific ROS 2 message, service (`SetJointStiffness`, `ErrorRecovery`, etc.), and action (`Grasp`, `Homing`, `Move`) definitions |
| `franka_example_controllers` | 8+ example `ros2_control` controllers covering joint velocity/position/impedance, Cartesian velocity, gravity compensation; dual-arm variants included |
| `franka_robot_state_broadcaster` | Publishes `FrankaState` topic with full robot state (joints, forces, torques, Cartesian info, collision data) |
| `franka_semantic_components` | Adapter layer translating `ros2_control` hardware interfaces into Franka-specific semantic types |
| `franka_gripper` | Action server for Franka Hand gripper (Grasp, Move, Homing actions) |
| `franka_description` | URDF/Xacro robot descriptions; single-arm (`panda_arm.urdf.xacro`) and dual-arm (`dual_panda_arm.urdf.xacro`); `ros2_control` xacro configs |
| `franka_bringup` | Launch files and controller YAML configs for single/dual-arm bringup; MoveIt2 integration |
| `franka_moveit_config` | MoveIt 2 motion planning configuration; supports `use_fake_hardware`, `use_isaac_sim`, and real-robot modes |

**Additional workspace packages** (under `ros2_ws/src/`):

| Package | Role |
|---|---|
| `topic_based_ros2_control` | `ros2_control` hardware interface that bridges to/from ROS topics; used with Isaac Sim (`use_isaac_sim:=true`) so MoveIt communicates with the simulator over joint state/command topics |

**Key design patterns:**
- **Hardware parameters at runtime**: `franka_hardware` hosts ROS 2 parameter services so controllers and users can change robot behavior (stiffness, collision thresholds, TCP frame) without restart.
- **Error recovery**: Service servers in `franka_hardware` expose error recovery without restarting the control node.
- **Dual-arm support**: `FrankaMultiHardwareInterface` and dual-arm example controllers handle synchronized multi-robot configurations.
- **Real-time constraints**: The control loop uses `SCHED_FIFO` scheduling. Avoid allocations or blocking calls in controller `update()` methods.

### pymoveit2

Python client library (`ros2_ws/src/pymoveit2/`) providing async MoveIt 2 interfaces. Key classes: `MoveIt2` (arm planning/execution), `MoveIt2Gripper`, `MoveIt2Servo`. Used to drive the Panda from Python nodes without writing C++ controllers.

### Client-Server Architecture

The system splits compute across two machines connected over LAN (port 5050):

- **Local PC** — ROS 2 + camera + robot; runs `pymoveit2`, captures RealSense frames, acts as the TCP client
- **GPU Server** — runs heavy vision/ML models (ThinkGrasp, SAM3, etc.); acts as the TCP server

Wire protocol: each message is `[4-byte big-endian length][payload bytes]`. For image streams the header is `[4-byte meta length][meta JSON][8-byte: rgb_len + depth_len][rgb JPEG bytes][depth uint16 bytes]`. Skeletons in `ClientServerTests/`.

### ThinkGrasp

Vision-language grasp detection system (`thinkgrasp/ThinkGrasp/`) — CoRL 2024. Uses LangSAM for segmentation and FGC-GraspNet for 6-DOF grasp pose estimation. Runs in PyBullet simulation or real-world via Flask API.

When populated:
- **CUDA environment** required — set `CUDA_HOME=/usr/local/cuda-11.8`, add `$CUDA_HOME/bin` to `PATH` and `$CUDA_HOME/lib64` to `LD_LIBRARY_PATH` before running.
- **Asset issue**: Many `unseen_objects_40` URDFs were patched to replace missing `textured.obj` with available collision meshes. See `thinkgrasp_edits.txt` for the full patch log. To properly fix, re-download assets from the ThinkGrasp HuggingFace dataset.

## Critical Dependencies — Do Not Corrupt

### libfranka 0.8.0 (source build)

- **Version**: 0.8.0 — **incompatible with the newer apt package** `ros-humble-libfranka` (0.20.4) which is also present on the system
- **Source**: `/opt/libfranka/` (do not delete)
- **Installed to**: `/usr/local/lib/libfranka.so.0.8`, headers at `/usr/local/include/franka`, cmake at `/usr/local/lib/cmake/Franka/`
- **How it wins over the apt version**: `CMAKE_PREFIX_PATH=/usr/local:...` and `LD_LIBRARY_PATH=/usr/local/lib:...` are set so CMake and the linker find the source-built 0.8.0 **before** the apt-installed 0.20.4 at `/opt/ros/humble`

**What would break it**:
- `sudo apt upgrade` or `apt install ros-humble-*` that modifies `ros-humble-libfranka`
- Reordering `CMAKE_PREFIX_PATH` so `/opt/ros/humble` comes before `/usr/local`
- Running `sudo make install` in any other libfranka build directory (overwrites `/usr/local`)
- Deleting `/opt/libfranka/`

**Verify**:
```bash
ldconfig -p | grep franka          # should show /usr/local/lib/libfranka.so.0.8
find /usr/local -name "FrankaConfig.cmake"
```

### librealsense 2.53.1 (source build)

- **Installed to**: `/usr/local/lib`, headers at `/usr/local/include/librealsense2`, cmake at `/usr/local/lib/cmake/realsense2`

**Verify**:
```bash
ldconfig -p | grep realsense
find /usr/local -name "*realsense*Config.cmake"
python3 -c "import pyrealsense2 as rs; print('OK')"
```

### MoveIt 2.5.9 (apt)

- **Install method**: `apt` — `ros-humble-moveit 2.5.9` at `/opt/ros/humble/`
- Do not run `sudo apt remove ros-humble-moveit*` or manually downgrade

**Verify**:
```bash
dpkg -l | grep ros-humble-moveit   # should show 2.5.9
```

### Required environment variables

Must be set before building or running (check `~/.bashrc`):
```bash
export CMAKE_PREFIX_PATH=/usr/local:/opt/openrobots/lib/cmake:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

For ThinkGrasp:
```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Pinocchio (robotpkg)

- **Install location**: `/opt/openrobots`
- Verify: `ls /opt/openrobots/lib/cmake`
