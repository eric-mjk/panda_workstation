python3 /workspace/ros2_ws/visualize_occupancy.py \
  /workspace/ros2_ws/ours_experiment/ex3/occupancy_final.ply \
  --states occupied

--states all       # show everything
--states occupied  # show only occupied voxels
--states unknown   # show only unknown voxels
--states free      # show only free voxels
--states known     # show free + occupied, hide unknown



python3 /workspace/ros2_ws/visualize_direction.py \
  /workspace/ros2_ws/ours_experiment/ex3/offline/best_direction.ply