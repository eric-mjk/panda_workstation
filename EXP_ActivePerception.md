==============================================
# Enable Robot Motion

## Real robot
ros2 run fetchbench_real fetchbench_active_perception --ros-args \
  --params-file /workspace/ros2_ws/src/fetchbench_real/config/active_perception_real.yaml \
  -p dry_run_motion:=false \
  -p experiment_name:=ex1 \
  -p max_steps:=4
## Isaac Sim
ros2 run fetchbench_real fetchbench_active_perception --ros-args \
  --params-file /workspace/ros2_ws/src/fetchbench_real/config/active_perception_sim.yaml \
  -p dry_run_motion:=false \
  -p experiment_name:=ex3 \
  -p max_steps:=4
=============================================

=============================================
# Extra: Keyboard Controls

p - process latest RGB-D frame, update AP occupancy grid, publish voxel markers
n - select the next active-perception view
m - move to the selected view and settle
v - republish voxel markers
w - write and save
q - quit

With dry_run_motion:=true, m will not move the robot.

Normal manual loop:

p -> n -> m -> p -> n -> m -> ...

=============================================
# Extra: RViz Occupancy Grid Topics

Add Marker displays for:

/fetchbench_active_perception/occupied_voxels
/fetchbench_active_perception/unknown_voxels
/fetchbench_active_perception/free_voxels
/fetchbench_active_perception/next_best_view

occupied_voxels, unknown_voxels, and free_voxels are enabled by default.
next_best_view is a red marker published when n selects a candidate.
