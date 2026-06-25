==============================================
# Offline VLM + Geometry

Use this after collecting AP views with:

p -> n -> m -> p -> n -> m -> ...

Expected input folder:

/ours_experiment/ex1/
  summary.json
  occupancy_final.ply
  views/
    rgb/
    depth/
    depth_preview/
    intrinsics.json
    pose.json

==============================================
# 1. Prepare VLM Inputs From A Clicked Pixel

ros2 run fetchbench_real fetchbench_offline_pipeline \
  --experiment-name ex1 \
  --target-view-index 0 \
  --target-pixel 640 360 \
  --prepare-only

This creates yellow-circle images in:

/ours_experiment/ex1/offline/vlm_inputs/

==============================================
# 2. Prepare VLM Inputs From A Known 3D Grasp Point

ros2 run fetchbench_real fetchbench_offline_pipeline \
  --experiment-name ex1 \
  --grasp-world 0.45 0.02 0.10 \
  --prepare-only

==============================================
# 3. Run VLM API + Geometry Fusion

Make sure GEMINI_API_KEY is set.

ros2 run fetchbench_real fetchbench_offline_pipeline \
  --experiment-name ex1 \
  --target-view-index 0 \
  --target-pixel 640 360 \
  --call-api

Or with a known 3D grasp point:

ros2 run fetchbench_real fetchbench_offline_pipeline \
  --experiment-name ex1 \
  --grasp-world 0.45 0.02 0.10 \
  --call-api

==============================================
# 4. Use Cached VLM Results

After the first API run, the per-view responses are cached.

ros2 run fetchbench_real fetchbench_offline_pipeline \
  --experiment-name ex1 \
  --target-view-index 0 \
  --target-pixel 640 360

==============================================
# 5. Outputs

/ours_experiment/ex1/offline/
  target_point.json
  prepared_views.json
  vlm_inputs/
  vlm_results/
  final_3d_direction.json
  best_direction.ply

Important file:

/ours_experiment/ex1/offline/final_3d_direction.json

Check:

best_direction
best_score
vlm_best_direction
geometry_best_direction

==============================================
# 6. Change Experiment Name

ros2 run fetchbench_real fetchbench_offline_pipeline \
  --experiment-name ex2 \
  --target-view-index 0 \
  --target-pixel 640 360 \
  --prepare-only

==============================================
