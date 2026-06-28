## 0. Setup
export FETCHBENCH_EXP=no_georun


## 1. Open3D visualization
python3 /workspace/ros2_ws/visualize_occupancy.py \
  /workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/occupancy_grid.ply \
  --states occupied

--states all       # show everything
--states occupied  # show only occupied voxels
--states unknown   # show only unknown voxels
--states free      # show only free voxels
--states known     # show free + occupied, hide unknown

python3 /workspace/ros2_ws/visualize_direction.py \
  /workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/aggregate_direction.ply


## 2. Rviz visualization
ros2 run fetchbench_real fetchbench_publish_ply_visualization --ros-args \
  -p occupancy_ply:=/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/occupancy_grid.ply \
  -p publish_occupancy:=true \
  -p occupancy_states:=occupied \
  -p publish_direction_set:=false \
  -p frame_id:=panda_link0 \
  -p voxel_size_m:=0.02

then add /fetchbench_execute/occupancy_ply in rviz2


## 3. Direction visualization
ros2 run fetchbench_real fetchbench_publish_ply_visualization --ros-args \
  -p occupancy_ply:=/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/occupancy_grid.ply \
  -p direction_ply:=/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/aggregate_direction.ply \
  -p result_json:=/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/final_3d_direction.json \
  -p publish_occupancy:=true \
  -p occupancy_states:=occupied \
  -p publish_direction_set:=true \
  -p frame_id:=panda_link0 \
  -p voxel_size_m:=0.02

In RViz add:

/fetchbench_execute/occupancy_ply
/fetchbench_execute/best_direction_set


## 4. TSDF + Sweeping Object visualization

ros2 run fetchbench_real fetchbench_publish_debug_geometry --ros-args \
  -p experiment_name:=${FETCHBENCH_EXP} \
  -p frame_id:=panda_link0 \
  -p occupancy_states:=all \
  -p voxel_size_m:=0.02

In RViz add:

/fetchbench_debug/original_occupancy
/fetchbench_debug/tsdf_without_target
/fetchbench_debug/target_surface_points
/fetchbench_debug/scoring_points_100
/fetchbench_debug/sweep_points
/fetchbench_execute/best_direction_set

### other sweep plys
-p sweep_ply:=/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/debug_geometry/sweep_geometry_only_direction.ply
-p sweep_ply:=/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/debug_geometry/sweep_vlm_only_direction.ply
-p sweep_ply:=/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/debug_geometry/sweep_aggregate_direction.ply
