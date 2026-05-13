# After Cloning panda_ws
git submodule update --init --recursive

sudo apt update
sudo apt install ros-humble-ros-testing


# Franka Panda - ROS2

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

  
# Images
checkpoint:latest
ericmjk/panda_ws:setup_end
ericmjk/panda_ws:latest
ericmjk/panda_ws:thinkgrasp
osrf/ros:humble-desktop
ericmjk/panda_thinkgrasp:sim