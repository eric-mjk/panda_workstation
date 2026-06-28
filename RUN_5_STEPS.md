# Running The ROS Pipeline

Pipeline:

```text
ap -> prep -> mask -> vlm -> direction -> execute
```

Build first:

```bash
cd /workspace/ros2_ws
colcon build --packages-select fetchbench_real
source install/setup.bash
```

Set the experiment name once:

```bash
export FETCHBENCH_EXP=ex4
```

To rerun prep/mask/VLM/direction without recollecting AP data, clean the experiment back to AP outputs only.

Preview first:

```bash
ros2 run fetchbench_real fetchbench_clean \
  --experiment-name ${FETCHBENCH_EXP}
```

Actually delete non-AP outputs:

```bash
ros2 run fetchbench_real fetchbench_clean \
  --experiment-name ${FETCHBENCH_EXP} \
  --yes
```

This preserves:

```text
rgb/
depth/
depth_preview/
intrinsics.json
pose.json
occupancy_grid.ply
```

## Config Files

Use the real config when running against the physical robot/camera:

```text
/workspace/ros2_ws/src/fetchbench_real/config/active_perception_real.yaml
```

This points to:

```text
/workspace/ros2_ws/src/fetchbench_real/config/real_robot_camera.yaml
```

Check the real camera topics and frame:

```yaml
rgb_topic: /camera/color/image_raw
depth_topic: /camera/aligned_depth_to_color/image_raw
frame_id: camera_link
depth_scale: 4000.0
```

Use the sim config when running against Isaac Sim:

```text
/workspace/ros2_ws/src/fetchbench_real/config/active_perception_sim.yaml
```

This points to:

```text
/workspace/ros2_ws/src/fetchbench_real/config/sim_robot_camera.yaml
```

Check the sim topics and frame:

```yaml
rgb_topic: /isaac_rgb
depth_topic: /isaac_depth
frame_id: l515_camera
depth_scale: 1.0
sync_slop: 0.5
```

For a no-mask dry run, the robot/camera is still expected to move in real or sim during AP. The dry-run part is only that we skip mask/TSDF geometry later with `--skip-geometry`.

## 1. AP

Real robot AP, moving the robot to collect initial + 15 AP views:

```bash
ros2 run fetchbench_real fetchbench_ap --ros-args \
  --params-file /workspace/ros2_ws/src/fetchbench_real/config/active_perception_real.yaml \
  -p keyboard_control:=false \
  -p dry_run_motion:=false \
  -p experiment_name:=${FETCHBENCH_EXP} \
  -p max_steps:=15
```

Isaac Sim AP, moving the simulated robot to collect initial + 15 AP views:

```bash
ros2 run fetchbench_real fetchbench_ap --ros-args \
  --params-file /workspace/ros2_ws/src/fetchbench_real/config/active_perception_sim.yaml \
  -p keyboard_control:=false \
  -p dry_run_motion:=false \
  -p experiment_name:=${FETCHBENCH_EXP} \
  -p max_steps:=15
```

Selection-only AP debug, without robot motion:

```bash
ros2 run fetchbench_real fetchbench_ap --ros-args \
  --params-file /workspace/ros2_ws/src/fetchbench_real/config/active_perception_real.yaml \
  -p keyboard_control:=false \
  -p dry_run_motion:=true \
  -p experiment_name:=${FETCHBENCH_EXP}_debug \
  -p max_steps:=15
```

The selection-only command is only for checking initial capture and NBV selection. It will not collect the full 15 moved AP views.

Expected outputs:

```text
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/rgb/
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/depth/
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/depth_preview/
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/intrinsics.json
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/pose.json
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/occupancy_grid.ply
```

## 2. Prep

Record the AP view list, choose 4 VLM views, create marked VLM inputs, and create the mask placeholder.

```bash
ros2 run fetchbench_real fetchbench_prep \
  --experiment-name ${FETCHBENCH_EXP} \
  --grasp-world 0.45 0.02 0.10
```

Expected outputs:

```text
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/vlm_subset.json
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/rgb_vlm_in/
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/masks/masks_manifest.json
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/target_point.json
```

At this point, `masks/` only contains a placeholder manifest. Run the mask stage to fill it.

## 3. Mask

Generate class-mask PNGs for every AP view listed in `vlm_subset.json`.

Default ROS pipeline mode uses the text prompt plus an automatically projected target point. The point is computed by projecting `grasp_position_world` from `vlm_subset.json` into each original RGB image, so each AP view gets its own image-frame point:

```bash
ros2 run fetchbench_real fetchbench_mask \
  --experiment-name ${FETCHBENCH_EXP} \
  --server-ip 192.168.0.71 \
  --prompt "the mustard bottle" \
  --overwrite
```

Use pure SAM3 text grounding, with no point cue:

```bash
ros2 run fetchbench_real fetchbench_mask \
  --experiment-name ${FETCHBENCH_EXP} \
  --server-ip 192.168.0.71 \
  --prompt "the mustard bottle" \
  --point-mode none \
  --overwrite
```

Use point-only segmentation from the projected target point:

```bash
ros2 run fetchbench_real fetchbench_mask \
  --experiment-name ${FETCHBENCH_EXP} \
  --server-ip 192.168.0.71 \
  --prompt "" \
  --point-mode auto \
  --overwrite
```

Use a fixed image-frame point for every view. `X` is pixel column from the left edge and `Y` is pixel row from the top edge of the RGB image:

```bash
ros2 run fetchbench_real fetchbench_mask \
  --experiment-name ${FETCHBENCH_EXP} \
  --server-ip 192.168.0.71 \
  --prompt "the mustard bottle" \
  --point 520 375 \
  --overwrite
```

Expected outputs:

```text
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/masks/0000.png
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/masks/0001.png
...
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/masks/masks_manifest.json
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/masks/raw_sam3/
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/masks/sam3_results/
```

The mask PNGs are class masks:

```text
0 = background
1 = target object
```

`masks/raw_sam3/` stores the raw SAM3 union masks in RGB image resolution. `masks/0000.png`, `masks/0001.png`, etc. are resized to the corresponding depth image resolution, because the geometry stage indexes these masks against depth pixels.

The prompt should name the real target object, for example:

```bash
--prompt "the mustard bottle"
--prompt "the red cup"
--prompt "the cracker box"
```

For no-mask dry runs, skip this stage and use `--skip-geometry` in the direction stage.

## 4. VLM

Call the VLM API:

```bash
ros2 run fetchbench_real fetchbench_vlm \
  --experiment-name ${FETCHBENCH_EXP} \
  --call-api
```

Use cached VLM responses:

```bash
ros2 run fetchbench_real fetchbench_vlm \
  --experiment-name ${FETCHBENCH_EXP}
```

Expected outputs:

```text
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/rgb_vlm_out/
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/rgb_vlm_out/vlm_results.json
```

Each VLM image should have:

```text
idx_XXXX_response.json
idx_XXXX_scores.png
```

## 5. Direction

Current no-mask dry run. This assumes AP already moved the robot/camera and collected views. Only geometry is skipped here:

```bash
ros2 run fetchbench_real fetchbench_direction \
  --experiment-name ${FETCHBENCH_EXP} \
  --skip-geometry
```

Full geometry run, after masks are available:

```bash
ros2 run fetchbench_real fetchbench_direction \
  --experiment-name ${FETCHBENCH_EXP}
```

Expected outputs:

```text
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/final_3d_direction.json
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/geometry_only_direction.json
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/vlm_only_direction.json
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/aggregate_direction.json
```

In `--skip-geometry` mode, `aggregate_direction` is effectively VLM-only.

## 6. Execute

Execute the aggregate direction:

```bash
ros2 run fetchbench_real fetchbench_execute --ros-args \
  -p experiment_name:=${FETCHBENCH_EXP}
```

Execute an ablation direction:

```bash
ros2 run fetchbench_real fetchbench_execute --ros-args \
  -p experiment_name:=${FETCHBENCH_EXP} \
  -p direction_key:=vlm_only_direction
```

```bash
ros2 run fetchbench_real fetchbench_execute --ros-args \
  -p experiment_name:=${FETCHBENCH_EXP} \
  -p direction_key:=geometry_only_direction
```

The execute stage reads:

```text
/workspace/ros2_ws/ours_experiment/${FETCHBENCH_EXP}/directions/final_3d_direction.json
```
