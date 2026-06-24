0. Setup 5 terminals (moveit, camera, tsdf, view candidate, commands)
1. StartHere (setup like normal)
2. TSDF & View Candidate -> TSDF terminal should be open!!! Wait for 'm' button

3. Shutdown robot and move to grasp pose
4. Activate robot & StartHere

5. Grasp object
6. Press 'm' button
7. Run the RRT planner
================================================================
# TSDF & Publish to ROS2 Planning Scene
##  Sim
ros2 run tsdf_voxel tsdf_integrator_once --ros-args \
  -p camera_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/sim_robot_camera.yaml \
  -p tsdf_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/tsdf_debug.yaml

##  Real
ros2 run tsdf_voxel tsdf_integrator_once --ros-args \
  -p camera_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/real_robot_camera.yaml \
  -p tsdf_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/tsdf_debug.yaml


Then use the keyboard:
p — move to next candidate, settle for 1 second, integrate TSDF frame
m — extract and publish mesh
r — remove collision object
q — quit
===============================================================
# Move robot to desired view candidates
python3 /workspace/scripts/run_view_candidates.py
============================================================
# Start panda planner (moveit planning sceene)
ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args \
  -p preset:=ready

ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args \
  -p planner_id:=PRMkConfigDefault \
  -p joint_positions:="
===============================================================
# View the saved voxel
python3 -c "import open3d as o3d; mesh=o3d.io.read_triangle_mesh('tsdf_meshes/debug_tsdf_mesh_1779201245800577484.ply'); mesh.compute_vertex_normals(); o3d.visualization.draw_geometries([mesh])"
