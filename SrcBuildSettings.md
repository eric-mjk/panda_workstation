# 🐼 Panda ROS2 Setup (Docker + Source Build)

## 🧩 Environment

- **OS (container)**: Ubuntu 22.04
- **ROS2**: Humble
- **Python**: 3.10
- **Docker**:
  - `--net host`
  - `--ipc host`
  - `--privileged`

---

## 🔧 Installed Components

### 📷 librealsense

- **Version**: `2.53.1`
- **Install Method**: Source build
- **Install Location**:
  - Libraries: `/usr/local/lib`
  - Headers: `/usr/local/include/librealsense2`
  - CMake: `/usr/local/lib/cmake/realsense2`

#### ✔ Verify

```bash
# C++ libraries
ldconfig -p | grep realsense

# headers
ls /usr/local/include | grep librealsense

# cmake config
find /usr/local -name "*realsense*Config.cmake"

# python binding
python3 - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
print("devices:", len(ctx.devices))
PY

# pip package
python3 -m pip show pyrealsense2
```

---

### 🤖 libfranka

- **Version**: `0.8.0`
- **Install Method**: Source build
- **Install Location**:
  - Libraries: `/usr/local/lib`
  - Headers: `/usr/local/include/franka`
  - CMake: `/usr/local/lib/cmake/Franka`

#### ✔ Verify

```bash
# runtime library
ldconfig -p | grep franka

# cmake config
find /usr/local -name "FrankaConfig.cmake"

# headers
ls /usr/local/include | grep franka
```

---

### 🧠 Pinocchio

- **Install Method**: robotpkg
- **Install Location**: `/opt/openrobots`

#### ✔ Verify

```bash
ls /opt/openrobots/lib/cmake
```

---

## ⚙️ Environment Variables

```bash
export CMAKE_PREFIX_PATH=/usr/local:/opt/openrobots/lib/cmake:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

#### ✔ Check

```bash
echo $CMAKE_PREFIX_PATH
echo $LD_LIBRARY_PATH
```

---

## 📁 Directory Structure

```text
/opt/
  ├── libfranka
  └── librealsense

/workspace/   (host volume mount)
  └── panda_ros2_ws
```

---

## 🐳 Docker Runtime

```bash
docker run -it \
  --ipc host \
  --net host \
  -d \
  --privileged \
  -v /dev:/dev \
  -v /dev/bus/usb:/dev/bus/usb \
  --name panda \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/Eric/panda_ws:/workspace \
  ericmjk/panda_ws:latest
```

---

## 🧪 Quick Sanity Check

```bash
# librealsense
ldconfig -p | grep realsense

# libfranka
ldconfig -p | grep franka

# python binding
python3 - <<'PY'
import pyrealsense2 as rs
print("pyrealsense OK")
PY
```

---

## 🧠 Notes

- 모든 핵심 라이브러리는 **source build → `/usr/local` 설치**
- ROS workspace는 `/workspace` (host와 공유)
- 외부 라이브러리는 `/opt`에 분리
- `libfranka`는 apt 설치 금지 (충돌 방지)
- RealSense는 `RSUSB backend` 사용

---
