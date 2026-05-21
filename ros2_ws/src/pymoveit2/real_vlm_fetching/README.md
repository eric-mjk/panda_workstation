# real_vlm_fetching

Isolated scaffold for real-world VLM-guided fetching baseline experiments.

**This folder does not modify any existing robot code.**
No files outside `real_vlm_fetching/` are touched.

---

## Intended future pipeline

```
capture image
    |
    v
make VLM overlay  (draw clock overlay on image)
    |
    v
run VLM           (query model, write result JSON with "best_clock" field)
    |
    v
inspect JSON      (verify clock and confidence before touching the robot)
    |
    v
execute fetch trial
    python scripts/run_fetch_trial_from_json.py --vlm-result <path>
    (currently prints displacement only — robot commands TBD)
```

---

## Structure

```
real_vlm_fetching/
  configs/          -- experiment configuration files (YAML / JSON)
  poses/            -- saved robot poses for experiments
  prompts/          -- VLM prompt templates
  scripts/          -- runnable entry points
  real_vlm_fetching/-- importable Python package
    pose_io.py      -- JSON save/load helpers
    frame_conventions.py -- clock-to-world coordinate mapping
    trial_logger.py -- per-trial directory and log management
  runs/             -- trial output directories (git-ignored except .gitkeep)
```

---

## Frame convention

Clock positions map to world XY directions:

| Clock | World direction |
|-------|----------------|
| 12    | +Y             |
| 3     | +X             |
| 6     | -Y             |
| 9     | -X             |

Increments are 30 degrees clockwise.

---

## Quick start

```bash
# Create a minimal VLM result JSON
echo '{"best_clock": 12}' > /tmp/vlm_result.json

# Dry-run: print displacement without commanding the robot
python real_vlm_fetching/scripts/run_fetch_trial_from_json.py \
    --vlm-result /tmp/vlm_result.json \
    --horizontal-distance-m 0.10 \
    --vertical-clearance-m 0.05
```
