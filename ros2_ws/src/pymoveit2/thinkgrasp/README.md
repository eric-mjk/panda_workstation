# ThinkGrasp Flask Client

ros2 run pymoveit2 grasp_pipeline_realrobot.py --ros-args -p instruction:="pick up the mustard bottle"



This folder contains ROS 2 snapshot clients for `thinkgrasp/ThinkGrasp/realarm.py`.
Start the Flask server first:

```bash
cd /workspace/thinkgrasp/ThinkGrasp
python3 realarm.py
```

There are two variants depending on your hardware setup:

| File | Setup | Transport |
|---|---|---|
| `send_grasp_request.py` | Isaac Sim | Sends file paths as JSON (server must share filesystem) |
| `send_grasp_request_realrobot.py` | Real robot (RealSense) | Uploads files as multipart form-data |

---

## Isaac Sim

```bash
ros2 run pymoveit2 send_grasp_request.py --ros-args -p instruction:="pick up the banana"
```

Default topics: `/isaac_rgb`, `/isaac_depth`

The node sends file paths to the server as JSON:

```json
{
  "image_path": "/tmp/thinkgrasp/rgb_YYYYMMDD_HHMMSS.png",
  "depth_path": "/tmp/thinkgrasp/depth_YYYYMMDD_HHMMSS.png",
  "text_path":  "/tmp/thinkgrasp/instruction_YYYYMMDD_HHMMSS.txt"
}
```

---

## Real Robot (RealSense)

```bash
ros2 run pymoveit2 send_grasp_request_realrobot.py --ros-args -p instruction:="pick up the banana"
```

Default topics: `/camera/color/image_raw`, `/camera/depth/image_raw`

Files are uploaded directly as `multipart/form-data` — the server does not need filesystem access.
RGB and depth may have different resolutions; both are independently resized to `target_width × target_height` before upload.

---

## Build

```bash
cd /workspace/ros2_ws
colcon build --symlink-install --packages-select pymoveit2
source install/setup.bash
```

## Controls

- **`g`** — snapshot latest RGB/depth frames, write instruction, send to ThinkGrasp
- **`q`** — quit

## Parameters (both nodes)

```bash
-p instruction:="pick up the red cup"   # required
-p rgb_topic:=<topic>
-p depth_topic:=<topic>
-p server_url:=http://127.0.0.1:5000
-p timeout_sec:=300.0
-p output_dir:=/tmp/thinkgrasp
-p target_width:=640
-p target_height:=480
-p trigger_key:=g
```

## Response

On success, the server response is printed as JSON:

```json
{
  "xyz": [0.0, 0.0, 0.0],
  "rot": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "dep": 0.0
}
```
