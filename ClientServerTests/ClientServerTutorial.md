
YOU ARE CURRNENTLY A LOCAL COMPUTER

## 현재 세팅 예시

```text
내 컴퓨터 / Local PC
- 카메라, 로봇, ROS2가 붙어있는 쪽
- 필요하면 데이터를 수집해서 host/server로 보냄
- Docker 사용
- 보통 client 역할

상대 컴퓨터 / Host(Server) PC
- GPU 모델, 무거운 연산을 돌리는 쪽
- local에서 보낸 데이터를 받아 처리
- Docker 사용
- 보통 server 역할
```

Docker 실행 예시:

```bash
# Local PC
docker run -it --net host --ipc host --privileged \
  -v /dev:/dev \
  -v /dev/bus/usb:/dev/bus/usb \
  -v ~/my_ws:/workspace \
  my_local_image bash
```

```bash
# Host(Server) PC
docker run -it --net host --gpus all \
  -v ~/server_ws:/workspace \
  my_server_image bash
```

---

## Host(Server) 쪽 skeleton

`server.py`

```python
import socket
import struct
import json

HOST = "0.0.0.0"
PORT = 5050


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


def handle_request(metadata, payload):
    """
    여기에 서버에서 돌릴 연산/모델을 넣으면 됨.
    예: SAM, VLM, trajectory generation, grasp planning 등
    """
    print("Received metadata:", metadata)
    print("Received payload size:", len(payload))

    result = {
        "status": "ok",
        "message": "processed on server",
        "received_index": metadata.get("data_index"),
    }
    return result


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"Server listening on {HOST}:{PORT}")

    conn, addr = server.accept()
    print("Connected from:", addr)

    with conn:
        while True:
            try:
                # 1. metadata 길이 받기
                meta_len_bytes = recv_exact(conn, 4)
                (meta_len,) = struct.unpack(">I", meta_len_bytes)

                # 2. metadata 받기
                meta_bytes = recv_exact(conn, meta_len)
                metadata = json.loads(meta_bytes.decode("utf-8"))

                # 3. payload 길이 받기
                payload_len_bytes = recv_exact(conn, 4)
                (payload_len,) = struct.unpack(">I", payload_len_bytes)

                # 4. payload 받기
                payload = recv_exact(conn, payload_len)

                # 5. 서버 연산
                result = handle_request(metadata, payload)

                # 6. 결과 다시 보내기
                result_bytes = json.dumps(result).encode("utf-8")
                conn.sendall(struct.pack(">I", len(result_bytes)))
                conn.sendall(result_bytes)

            except ConnectionError:
                print("Client disconnected")
                break
```

---

## Local(Client) 쪽 skeleton

`client.py`

```python
import socket
import struct
import json
import time

SERVER_IP = "147.46.118.233"  # host/server 컴퓨터 IP로 변경
PORT = 5050


def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


def make_payload():
    """
    여기에 local에서 보낼 데이터를 만들면 됨.
    예: RGB image bytes, depth bytes, robot state, command 등
    """
    payload = b"example binary data"
    return payload


with socket.create_connection((SERVER_IP, PORT), timeout=5.0) as sock:
    print(f"Connected to server {SERVER_IP}:{PORT}")

    data_index = 0

    while True:
        # 1. metadata 만들기
        metadata = {
            "data_index": data_index,
            "timestamp": time.time(),
            "type": "example_request",
        }

        # 2. payload 만들기
        payload = make_payload()

        # 3. metadata 전송
        meta_bytes = json.dumps(metadata).encode("utf-8")
        sock.sendall(struct.pack(">I", len(meta_bytes)))
        sock.sendall(meta_bytes)

        # 4. payload 전송
        sock.sendall(struct.pack(">I", len(payload)))
        sock.sendall(payload)

        # 5. 서버 결과 받기
        result_len_bytes = recv_exact(sock, 4)
        (result_len,) = struct.unpack(">I", result_len_bytes)

        result_bytes = recv_exact(sock, result_len)
        result = json.loads(result_bytes.decode("utf-8"))

        print("Server result:", result)

        data_index += 1
        time.sleep(1.0)
```

---

## 실행 순서

```bash
# 1. Host(Server) PC에서 먼저 실행
python3 server.py
```

```bash
# 2. Local PC에서 실행
python3 client.py
```

요약하면:

```text
Local client:
  데이터 생성 → bytes로 변환 → server로 전송 → 결과 수신

Host server:
  대기 → 데이터 수신 → 모델/연산 실행 → 결과 전송
```



# WORKING SETUP EXAMPLE
- Local
```
import select
import sys
import termios
import tty
import time
import socket
import struct
import json
from threading import Thread

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node


from pymoveit2 import MoveIt2
from pymoveit2.robots import panda as robot

CAMERA_SERIAL = "f0221251"
SERVER_IP = "147.46.118.233"
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
PORT = 5050

EE_TO_CAMERA = np.array(
	[
		[-0.00655526, -0.999961, 0.00593217, 0.092329],
		[0.999813, -0.006662, -0.0181578, 0.00308126],
		[0.0181966, 0.00581203, 0.999818, 0.0622155],
		[0.0, 0.0, 0.0, 1.0],
	],
	dtype=np.float64,
)

def set_camera(camera_serial: str):
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_device(camera_serial)
	config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, 30)
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	profile = pipeline.start(config)
	align = rs.align(rs.stream.color)

	device = profile.get_device()
	sensors = device.query_sensors()

	color_sensor = None
	depth_sensor = None
	for sensor in sensors:
		name = sensor.get_info(rs.camera_info.name)
		if "RGB" in name:
			color_sensor = sensor
		if "Depth" in name:
			depth_sensor = sensor

	if color_sensor is not None:
		color_sensor.set_option(rs.option.enable_auto_exposure, 0)
		color_sensor.set_option(rs.option.exposure, 1000.0)
		color_sensor.set_option(rs.option.gain, 32.0)

	if depth_sensor is not None:
		depth_sensor.set_option(rs.option.min_distance, 0.0)

	return pipeline, align


def get_frames(camera_pipeline, align):
	frames = camera_pipeline.wait_for_frames()
	return align.process(frames)


def _send_all(sock: socket.socket, data: bytes):
	view = memoryview(data)
	while view:
		sent = sock.send(view)
		if sent == 0:
			raise RuntimeError("Socket connection broken")
		view = view[sent:]


def main():
	rclpy.init()
	node = Node("robot_arm_viewer")
	callback_group = ReentrantCallbackGroup()

	moveit2 = MoveIt2(
		node=node,
		joint_names=robot.joint_names(),
		base_link_name=robot.base_link_name(),
		end_effector_name=robot.end_effector_name(),
		group_name=robot.MOVE_GROUP_ARM,
		callback_group=callback_group,
	)

	executor = rclpy.executors.MultiThreadedExecutor(2)
	executor.add_node(node)
	executor_thread = Thread(target=executor.spin, daemon=True)
	executor_thread.start()

	pipeline, align = set_camera(CAMERA_SERIAL)

	old_tty = termios.tcgetattr(sys.stdin)
	tty.setcbreak(sys.stdin.fileno())
	last_send_time = 0.0
	send_requested = True
	data_index = 0

	try:
		with socket.create_connection((SERVER_IP, PORT), timeout=5.0) as sock:
			print(f"Connected to server {SERVER_IP}:{PORT}")

			while rclpy.ok():
				frames = get_frames(pipeline, align)
				color_frame = frames.get_color_frame()
				depth_frame = frames.get_depth_frame()

				# keyboard-driven send trigger
				ready, _, _ = select.select([sys.stdin], [], [], 0.0)
				if ready:
					key = sys.stdin.read(1)
					if key.lower() == "q":
						send_requested = True
					elif key.lower() == "x":
						break

				if not color_frame or not depth_frame:
					print("No frames received")
					continue

				color = np.asanyarray(color_frame.get_data())
				depth = np.asanyarray(depth_frame.get_data()).astype(np.uint16)

				if time.time() - last_send_time < 3.0 and not send_requested:
					continue

				# Encode RGB as JPEG for bandwidth efficiency
				ok, rgb_encoded = cv2.imencode(".jpg", color)
				if not ok:
					print("Failed to encode RGB frame")
					continue
				rgb_bytes = rgb_encoded.tobytes()
				depth_bytes = depth.tobytes()

				# Joint angles from MoveIt2
				joint_state = moveit2.joint_state
				joint_angles = list(joint_state.position) if joint_state is not None else []

				# Cartesian pose of end-effector through FK
				cartesian_pose = None
				pose_stamped = moveit2.compute_fk(fk_link_names=["panda_hand"])
				if pose_stamped is not None:
					if isinstance(pose_stamped, list):
						pose_stamped = pose_stamped[0]
					p = pose_stamped.pose
					cartesian_pose = {
						"position": [p.position.x, p.position.y, p.position.z],
						"orientation": [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
					}

				metadata = {
					"data_index": data_index,
					"timestamp": time.time(),
					"joint_angles": joint_angles,
					"cartesian_pose": cartesian_pose,
					"rgb_bytes": len(rgb_bytes),
					"depth_bytes": len(depth_bytes),
				}

				meta_json = json.dumps(metadata).encode("utf-8")
				# Send metadata length + metadata, then rgb + depth
				_send_all(sock, struct.pack(">I", len(meta_json)))
				_send_all(sock, meta_json)
				_send_all(sock, struct.pack(">II", len(rgb_bytes), len(depth_bytes)))
				_send_all(sock, rgb_bytes)
				_send_all(sock, depth_bytes)

				last_send_time = time.time()
				send_requested = False
				data_index += 1
				# small sleep to avoid saturating link; tuning optional
				time.sleep(0.05)
	except Exception as e:
		print(f"Error in streaming loop: {e}")
		import traceback
		traceback.print_exc()
	finally:
		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
		pipeline.stop()
		executor.shutdown()
		node.destroy_node()
		rclpy.shutdown()

if __name__ == "__main__":
	main()
```

- Host (server)
```
import os
import sys
import socket
import struct
import json
import time

import torch
import numpy as np
from PIL import Image
import cv2

from colors import colors
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("remote closed")
        buf += chunk
    return buf

HOST = "0.0.0.0"
PORT = 5050
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720

input_dir = "dataset"
output_dir = "output"
text_prompt = "object"

# Load the model
start_time = time.time()
model = build_sam3_image_model(checkpoint_path="/workspace/sam3_panda/facebook/sam3/sam3.pt")
processor = Sam3Processor(model)
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print("listening", HOST, PORT)
    conn, addr = s.accept()
    with conn:
        print("connected", addr)
        while True:
            hdr = recv_exact(conn, 4)
            if not hdr:
                break
            (meta_len,) = struct.unpack(">I", hdr)
            meta_json = recv_exact(conn, meta_len)
            metadata = json.loads(meta_json.decode("utf-8"))

            hdr2 = recv_exact(conn, 8)
            rgb_len, depth_len = struct.unpack(">II", hdr2)

            rgb_bytes = recv_exact(conn, rgb_len)
            depth_bytes = recv_exact(conn, depth_len)

            # decode rgb
            rgb_np = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
            # convert depth (height and width must match client depth stream description)
            depth_arr = np.frombuffer(depth_bytes, dtype=np.uint16).reshape((COLOR_HEIGHT, COLOR_WIDTH))

            joint_angles_str = ", ".join(f"{angle:.3f}" for angle in metadata["joint_angles"])
            print("index:", metadata["data_index"], "meta:", metadata["timestamp"], "joints:", joint_angles_str)

            if not os.path.exists(os.path.join(input_dir, "rgb")):
                os.makedirs(os.path.join(input_dir, "rgb"))
            if not os.path.exists(os.path.join(input_dir, "depth")):
                os.makedirs(os.path.join(input_dir, "depth"))
            if not os.path.exists(os.path.join(input_dir, "depth_colormap")):
                os.makedirs(os.path.join(input_dir, "depth_colormap"))
            if not os.path.exists(os.path.join(output_dir, "mask")):
                os.makedirs(os.path.join(output_dir, "mask"))
            if not os.path.exists(os.path.join(output_dir, "class_ids")):
                os.makedirs(os.path.join(output_dir, "class_ids"))

            # save rgb and depth
            data_index = metadata["data_index"]
            cv2.imwrite(os.path.join(input_dir, "rgb", f"{data_index}.png"), rgb_np)
            np.save(os.path.join(input_dir, "depth", f"{data_index}.npy"), depth_arr)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_arr, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(input_dir, "depth_colormap", f"{data_index}.png"), depth_colormap)

            print("Load image and run SAM3 inference...")
            start_time = time.time()
            image = Image.fromarray(cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB))
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            print(f"Number of masks ({text_prompt}): {masks.shape[0]}, Inference time: {time.time() - start_time:.2f} seconds")

            # masks: (N, 1, H, W), boxes: (N, 4), scores: (N)
            mask_path = os.path.join(output_dir, f"mask_{data_index}.png")
            masks_np = masks.detach().cpu().numpy()
            H, W = masks_np.shape[2], masks_np.shape[3]
            colored_mask = np.zeros((H, W, 3), dtype=np.uint8)
            class_ids = np.zeros((H, W), dtype=np.int32)  # to store class IDs for each pixel

            for i in range(masks_np.shape[0]):
                color = colors[i] * 255
                colored_mask[masks_np[i, 0]] = color.astype(np.uint8)
                class_ids[masks_np[i, 0]] = i + 1  # assign class ID (starting from 1)

            colored_mask_image = Image.fromarray(colored_mask)
            colored_mask_image.save(os.path.join(output_dir, "mask", f"{data_index}.png"))
            np.save(os.path.join(output_dir, "class_ids", f"{data_index}.npy"), class_ids)
```