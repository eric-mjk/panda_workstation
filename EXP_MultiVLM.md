After running EXP_ActivePerception.md
==============================================
# Run VLM API + Geometry Fusion

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

Or to just see the prepared input images

ros2 run fetchbench_real fetchbench_offline_pipeline \
  --experiment-name ex1 \
  --grasp-world 0.45 0.02 0.10 \
  --call-api \
  --prepare-only

==============================================
# Use Cached VLM Results

After the first API run, the per-view responses are cached.

Run again without --call-api:

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
