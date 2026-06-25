ros2 run fetchbench_real fetchbench_pull_best_direction --ros-args \
    -p experiment_name:=ex3



ros2 run fetchbench_real fetchbench_publish_ply_visualization --ros-args \
    -p experiment_name:=ex3 \
    -p publish_occupancy:=true \
    -p publish_direction_set:=true \
    -p occupancy_states:=occupied
