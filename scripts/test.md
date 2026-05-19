Done. Here's how to use it:

1. Get the camera pose while the robot is running:


ros2 run tf2_ros tf2_echo panda_link0 camera_color_optical_frame
Copy the translation [tx, ty, tz] and quaternion [qx, qy, qz, qw] from the output.

2. Run the script (no ROS needed):


python scripts/project_to_image.py \
    --point 0.45 0.02 0.30 \
    --image /tmp/snapshot.png \
    --cam_pos 0.12 -0.34 0.56 \
    --cam_quat 0.01 0.70 0.02 0.71 \
    --fx 909.7 --fy 909.4 --cx 641.4 --cy 361.6 \
    --output /tmp/result.png
A few things to note:

--fx/fy/cx/cy default to typical D435 values at 1280×720. Get exact values from ros2 topic echo /camera/color/camera_info once and reuse them.
The script prints the pixel (u, v) and depth, and warns if the point projects outside the image.
If you omit --output it opens an OpenCV window instead.