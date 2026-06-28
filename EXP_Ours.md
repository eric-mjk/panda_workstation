# The 5-Step ROS Pipeline

1. ap : active perception
2. prep : preprocess images (generate mask with SAM3 & select 4 input images for vlm)
3. mask : send 15 (all) images to SAM3 and get the masked image
4. vlm : send 4 images to vlm and get the vlm direction scores
5. direction : calculate the geometry direction scores and aggregate scores for final post-grasp direction (geo, vlm, geo+vlm)
6. execute : execute the motion

# STARTHERE
## 0. Setup
### set the experiment name
export FETCHBENCH_EXP=no_georun
### set the gemini api key
export GEMINI_API_KEY=" "
##### TO DELETE EXCEPT AP
ros2 run fetchbench_real fetchbench_clean \
  --experiment-name ${FETCHBENCH_EXP} \
  --yes
============================================
## 1. AP
### Real robot
```bash
ros2 run fetchbench_real fetchbench_ap --ros-args \
  --params-file /workspace/ros2_ws/src/fetchbench_real/config/active_perception_real.yaml \
  -p keyboard_control:=false \
  -p dry_run_motion:=false \
  -p experiment_name:=${FETCHBENCH_EXP} \
  -p max_steps:=8
```
### IsaacSim
```bash
ros2 run fetchbench_real fetchbench_ap --ros-args \
  --params-file /workspace/ros2_ws/src/fetchbench_real/config/active_perception_sim.yaml \
  -p keyboard_control:=false \
  -p dry_run_motion:=false \
  -p experiment_name:=${FETCHBENCH_EXP} \
  -p max_steps:=15
```
Press p to process latest RGB-D frame and publish voxel markers
Press n to select the next active-perception view
Press m to move to the selected view and settle
Press v to republish voxel markers
Press w to write summary and PLY outputs
Press q to quit
============================================
## 2. Prep
ros2 run fetchbench_real fetchbench_prep \
  --experiment-name ${FETCHBENCH_EXP} \
  --grasp-world 0.5 0.0 0.12
============================================
## 3. Mask
### FIRST RUN THE SERVER... Then
####  Prompt + point is the default
```bash
ros2 run fetchbench_real fetchbench_mask \
  --experiment-name ${FETCHBENCH_EXP} \
  --server-ip 192.168.0.71 \
  --prompt "mustard bottle" \
  --overwrite
```
####  Prompt only:
```bash
ros2 run fetchbench_real fetchbench_mask \
  --experiment-name ${FETCHBENCH_EXP} \
  --server-ip 192.168.0.71 \
  --prompt "the mustard bottle" \
  --point-mode none \
  --overwrite
```
####  Point only:
```bash
ros2 run fetchbench_real fetchbench_mask \
  --experiment-name ${FETCHBENCH_EXP} \
  --server-ip 192.168.0.71 \
  --prompt "" \
  --point-mode auto \
  --overwrite
```
============================================
## 3. VLM
```bash
ros2 run fetchbench_real fetchbench_vlm \
  --experiment-name ${FETCHBENCH_EXP} \
  --call-api
```
### Using the cached VLM response without calling api
```bash
ros2 run fetchbench_real fetchbench_vlm \
  --experiment-name ${FETCHBENCH_EXP}
```
===========================================
## 4. Direction
```bash
ros2 run fetchbench_real fetchbench_direction \
  --experiment-name ${FETCHBENCH_EXP} \
  --skip-geometry
```
### Full geometry run
```bash
ros2 run fetchbench_real fetchbench_direction \
  --experiment-name ${FETCHBENCH_EXP}
```
============================================
## 5. Execute
```bash
ros2 run fetchbench_real fetchbench_execute --ros-args \
  -p experiment_name:=${FETCHBENCH_EXP}
```
### Ablation direction:
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
