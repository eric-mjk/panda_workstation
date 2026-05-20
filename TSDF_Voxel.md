# View the saved voxel
python3 -c "import open3d as o3d; mesh=o3d.io.read_triangle_mesh('tsdf_meshes/debug_tsdf_mesh_1779201245800577484.ply'); mesh.compute_vertex_normals(); o3d.visualization.draw_geometries([mesh])"

================================================================
Then use the keyboard:

p — move to next candidate, settle 1s, integrate TSDF frame
m — extract and publish mesh
r — remove collision object
q — quit


# Publish voxel as ROS2 topic
##  Sim
ros2 run tsdf_voxel tsdf_integrator_once --ros-args \
  -p camera_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/sim_robot_camera.yaml \
  -p tsdf_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/tsdf_debug.yaml

##  Real
ros2 run tsdf_voxel tsdf_integrator_once --ros-args \
  -p camera_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/real_robot_camera.yaml \
  -p tsdf_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/tsdf_debug.yaml

===============================================================

# Using view candidate
## Sim
ros2 run tsdf_voxel view_scan_executor --ros-args \
  -p camera_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/sim_robot_camera.yaml \
  -p tsdf_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/tsdf_debug.yaml \
  -p view_scan_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/view_scan_config.yaml


## Real
ros2 run tsdf_voxel view_scan_executor --ros-args \
  -p camera_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/real_robot_camera.yaml \
  -p tsdf_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/tsdf_debug.yaml \
  -p view_scan_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/view_scan_config.yaml


===============================================================

# Start panda planner (moveit planning sceene)
ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args \
  -p preset:=ready

ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args \
  -p planner_id:=PRMkConfigDefault \
  -p joint_positions:="[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]"


===============================================================
