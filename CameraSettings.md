The main launch file is rs_launch.py. For the L515 publishing color + depth:


source /workspace/ros2_ws/install/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  device_type:=l515 \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true


What you'll get:

Topic	Resolution	Notes
/camera/color/image_raw	1920×1080	RGB color
/camera/depth/image_raw	1024×768	Raw depth in depth frame
/camera/aligned_depth_to_color/image_raw	1920×1080	Depth warped into color frame
/camera/color/camera_info	—	Intrinsics (valid for color + aligned depth)


align_depth.enable is false by default — add it if you need depth and color pixels to correspond 1:1 (e.g. for grasp detection with ThinkGrasp). If you only need raw depth without reprojection, you can drop that argument