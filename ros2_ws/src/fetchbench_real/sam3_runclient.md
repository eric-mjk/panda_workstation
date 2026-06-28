export SAM3_CLIENT=/workspace/ros2_ws/src/fetchbench_real/sam3_client.py
export SAM3_IMAGE=/workspace/ros2_ws/ours_experiment/no_georun/rgb_vlm_in/idx_0000_input_with_query.png

## Local PC / Client
```bash
python3 ${SAM3_CLIENT} \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image ${SAM3_IMAGE} \
  --prompt "the mustard bottle" \
  --out-mask test.png
```

To request per-instance masks in the JSON response too:

```bash
python3 ${SAM3_CLIENT} \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image ${SAM3_IMAGE} \
  --prompt "the mustard bottle" \
  --out-mask test.png
  --return-instances
```

To also save per-instance mask PNG files:

```bash
python3 ${SAM3_CLIENT} \
  --server-ip 192.168.0.71 \
  --port 5050 \
  --image ${SAM3_IMAGE} \
  --prompt "the mustard bottle" \
  --out-mask test.png
  --return-instances \
  --out-instances-dir microwave_instances
```