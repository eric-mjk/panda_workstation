# fetchbench_real

ROS 2 package for adapting FetchBench methods to the Panda workstation.

## Layout

```text
fetchbench_real/
  real_active_perception/
    core.py
    coordinator.py
    view_candidates/

  org_sim_src/
    src/
      vlm_aggregate.py
      active_perception/
      mv_vlm/
    baselines/
```

`real_active_perception/` is the runtime implementation. It is self-contained
and does not import from `org_sim_src/`.

`org_sim_src/` is a read-only reference copy of the original simulation source
files. It is kept only for comparison while implementing the real robot version.

## Real Active Perception

The ROS coordinator keeps active-perception state in process:

```text
ROS RGB-D + TF
  -> occupancy voxel update
  -> unknown voxel scoring
  -> candidate next-best-view selection
  -> optional Panda joint goal execution
  -> summary.json
```

Dry-run motion is enabled by default:

```bash
ros2 run fetchbench_real fetchbench_active_perception --ros-args \
  --params-file /home/mechanical/Eric/panda_workstation/ros2_ws/src/fetchbench_real/config/active_perception_real.yaml
```

For Isaac Sim camera topics, use:

```bash
ros2 run fetchbench_real fetchbench_active_perception --ros-args \
  --params-file /home/mechanical/Eric/panda_workstation/ros2_ws/src/fetchbench_real/config/active_perception_sim.yaml
```

Set `dry_run_motion:=false` only when the camera, TF, and MoveIt2 stack are
already running and the workspace is clear.
