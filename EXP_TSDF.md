0. Setup 5 terminals

1. Follow StartHere.md (this takes up 2 terminals)

2. Run View Candiate Mover terminal (1 terminal)
=============================================================
# Move robot to desired view candidates
python3 /workspace/scripts/run_view_candidates.py
===========================================================

3. Run TSDF terminal (1 terminal)
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


keyboard keys:
p — move to next candidate, settle for 1 second, integrate TSDF frame
m — extract and publish mesh
r — remove collision object
q — quit
===============================================================

4. Shutdown panda & close the first two terminals setup by StartHere.md (DONT TURN OFF TSDF terminal)

5. Move panda to grasp pose and activate panda

6. Follow StartHere.md again

7. Grasp object (in 5th terminal)
ros2 run pymoveit2 panda_gripper_control

8. Press 'm' button in TSDF terminal (to add voxel to planning scene)

7. Run the RRT planner
============================================================
# Start panda planner (moveit planning sceene)
ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args \
  -p preset:=ready

ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args \
  -p planner_id:=PRMkConfigDefault \
  -p joint_positions:="
===============================================================

8. If needed
# View the saved voxel
python3 -c "import open3d as o3d; mesh=o3d.io.read_triangle_mesh('tsdf_meshes/debug_tsdf_mesh_1779201245800577484.ply'); mesh.compute_vertex_normals(); o3d.visualization.draw_geometries([mesh])"
