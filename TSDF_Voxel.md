python3 -c "import open3d as o3d; mesh=o3d.io.read_triangle_mesh('tsdf_meshes/debug_tsdf_mesh_1779201245800577484.ply'); mesh.compute_vertex_normals(); o3d.visualization.draw_geometries([mesh])"



ros2 run tsdf_voxel tsdf_integrator_once --ros-args \
  -p camera_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/sim_robot_camera.yaml \
  -p tsdf_config_file:=/workspace/ros2_ws/src/tsdf_voxel/config/tsdf_debug.yaml