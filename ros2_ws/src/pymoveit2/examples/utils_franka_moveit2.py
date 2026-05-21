import os
import json

import numpy as np

from utils_math import (
    rotation_matrix_from_quaternion
)
from robot_arm_config import (
    EE_TO_GRIPPER,
    EE_TO_CAMERA,
)

def delete_all_collision_boxes(moveit2) -> None:
    """Delete all collision boxes in the planning scene."""
    try:
        has_scene = moveit2.update_planning_scene()
        if has_scene and moveit2.planning_scene is not None:
            existing_ids = {
                obj.id for obj in moveit2.planning_scene.world.collision_objects
            }
            for obj_id in existing_ids:
                moveit2.remove_collision_object(obj_id)
                print(f"[scene] Removed existing collision box: {obj_id}")
    except Exception as err:
        print(f"[scene] Failed to delete collision boxes: {err}")

def ensure_table_collision_box(moveit2) -> None:
    """Ensure a table collision box exists in the planning scene."""
    try:
        has_scene = moveit2.update_planning_scene()
        if has_scene and moveit2.planning_scene is not None:
            existing_ids = {
                obj.id for obj in moveit2.planning_scene.world.collision_objects
            }
            if "table" in existing_ids:
                print(f"[scene] Table collision box already exists")
                return

        moveit2.add_collision_box(
            id="table",
            position=[0.5, 0.0, 0.0],
            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            size=[0.8, 2.0, 0.04],
        )
        print("[scene] Added table collision box ")
    except Exception as err:
        print(f"[scene] Failed to ensure table collision box: {err}")
		
def ensure_wall_collision_box(moveit2) -> None:
    """Ensure a wall collision box exists in the planning scene."""
    try:
        has_scene = moveit2.update_planning_scene()
        if has_scene and moveit2.planning_scene is not None:
            existing_ids = {
                obj.id for obj in moveit2.planning_scene.world.collision_objects
            }
            if "wall" in existing_ids:
                print(f"[scene] Wall collision box already exists")
                # delete and re-add to ensure correct position/size
                # moveit2.remove_collision_object("wall")
                # print(f"[scene] Removed existing wall collision box for refresh")
                return

        moveit2.add_collision_box(
            id="wall",
            position=[-0.45, 0.0, 0.6],
            quat_xyzw=[0.0, 0.0, 0.0, 0.1],
            size=[0.1, 2.0, 1.2],
        )
        print("[scene] Added wall collision box ")
    except Exception as err:
        print(f"[scene] Failed to ensure wall collision box: {err}")

def pose_to_transform(pose_stamped) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    p = pose_stamped.pose.position
    q = pose_stamped.pose.orientation
    transform[:3, :3] = rotation_matrix_from_quaternion((q.x, q.y, q.z, q.w))
    transform[:3, 3] = np.array([p.x, p.y, p.z], dtype=np.float64)
    return transform

def capture_pose(moveit2, output_dir, index):
	joint_state = moveit2.joint_state
	if joint_state is None:
		print("No joint state received (pose capture skip)")
		return
	else:
		os.makedirs(os.path.join(output_dir, "pose"), exist_ok=True)
		pose_info = {}
		joint_angles = list(joint_state.position)
		pose_info["joint_angles"] = joint_angles

		pose_stamped = moveit2.compute_fk(fk_link_names=["panda_hand"])
		if pose_stamped is not None:
			if isinstance(pose_stamped, list):
				pose_stamped = pose_stamped[0]

			base_to_ee = pose_to_transform(pose_stamped)
			base_to_gripper = base_to_ee @ EE_TO_GRIPPER
			base_to_cam = base_to_ee @ EE_TO_CAMERA

			pose_info["base_to_ee"] = base_to_ee.tolist()
			pose_info["base_to_gripper"] = base_to_gripper.tolist()
			pose_info["base_to_cam"] = base_to_cam.tolist()
			
			p = pose_stamped.pose
			pose_info["ee_position"] = [p.position.x, p.position.y, p.position.z]
			pose_info["ee_orientation"] = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
		else:
			print("Failed to compute FK (pose capture skip)")
		json_path = os.path.join(output_dir, "pose", f"{index:05d}.json")
		with open(json_path, "w") as f:
			json.dump(pose_info, f, indent=4)
		print(f"Saved pose {index:05d} to {json_path}")