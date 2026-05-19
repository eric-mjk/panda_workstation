#!/usr/bin/env python3
"""Project a 3D world-frame point onto a camera image and draw a circle.

You supply the camera pose directly (e.g. from `ros2 run tf2_ros tf2_echo`),
so no ROS session or forward-kinematics are needed at runtime.

Get the camera pose while MoveIt / the camera are running:
    ros2 run tf2_ros tf2_echo panda_link0 camera_color_optical_frame

That command prints the translation + quaternion you pass to this script.

Usage
-----
python project_to_image.py \
    --point   0.45 0.02 0.30 \
    --image   /tmp/snapshot.png \
    --cam_pos 0.12 -0.34 0.56 \
    --cam_quat 0.01 0.70 0.02 0.71 \
    --fx 909.7 --fy 909.4 --cx 641.4 --cy 361.6

Optionally save the result:
    ... --output /tmp/result.png

Dependencies: numpy, opencv-python (cv2)
"""

import argparse
import sys

import cv2
import numpy as np


def quat_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion (x, y, z, w) → 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def world_to_pixel(
    point_world: np.ndarray,
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> tuple[float, float, float]:
    """Return (u, v, depth_m) for a world-frame 3-D point.

    cam_pos  — translation part of the TF (camera origin in world frame)
    cam_rot  — 3x3 rotation matrix (camera axes expressed in world frame)

    The TF convention from `tf2_echo parent child` is:
        p_world = cam_rot @ p_cam + cam_pos
    So the inverse is:
        p_cam = cam_rot.T @ (p_world - cam_pos)
    """
    p_cam = cam_rot.T @ (point_world - cam_pos)

    depth = p_cam[2]
    if depth <= 0:
        raise ValueError(f"Point is behind the camera (depth={depth:.4f} m).")

    u = fx * p_cam[0] / depth + cx
    v = fy * p_cam[1] / depth + cy
    return u, v, depth


def draw_circle(
    image: np.ndarray,
    u: float,
    v: float,
    radius: int = 15,
    color: tuple = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    out = image.copy()
    cv2.circle(out, (int(round(u)), int(round(v))), radius, color, thickness)
    cv2.circle(out, (int(round(u)), int(round(v))), 3, color, -1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--point", nargs=3, type=float, required=True,
                        metavar=("X", "Y", "Z"),
                        help="3-D point in world frame (panda_link0), metres.")
    parser.add_argument("--image", required=True,
                        help="Input image file (PNG, JPG, …).")
    parser.add_argument("--cam_pos", nargs=3, type=float, required=True,
                        metavar=("TX", "TY", "TZ"),
                        help="Camera position in world frame (translation from tf2_echo).")
    parser.add_argument("--cam_quat", nargs=4, type=float, required=True,
                        metavar=("QX", "QY", "QZ", "QW"),
                        help="Camera orientation in world frame (quaternion from tf2_echo).")

    parser.add_argument("--fx", type=float, default=909.7,
                        help="Focal length x (pixels).  Default: D435 @ 1280x720.")
    parser.add_argument("--fy", type=float, default=909.4,
                        help="Focal length y (pixels).  Default: D435 @ 1280x720.")
    parser.add_argument("--cx", type=float, default=641.4,
                        help="Principal point x (pixels).  Default: D435 @ 1280x720.")
    parser.add_argument("--cy", type=float, default=361.6,
                        help="Principal point y (pixels).  Default: D435 @ 1280x720.")

    parser.add_argument("--radius", type=int, default=15,
                        help="Circle radius in pixels (default 15).")
    parser.add_argument("--output", default="",
                        help="Save result to this path instead of displaying it.")

    args = parser.parse_args()

    # --- load image ---
    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: cannot read image '{args.image}'", file=sys.stderr)
        return 1

    # --- build transforms ---
    point_world = np.array(args.point, dtype=float)
    cam_pos     = np.array(args.cam_pos, dtype=float)
    cam_rot     = quat_to_rotation_matrix(*args.cam_quat)

    # --- project ---
    try:
        u, v, depth = world_to_pixel(point_world, cam_pos, cam_rot,
                                     args.fx, args.fy, args.cx, args.cy)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    h, w = img.shape[:2]
    in_frame = (0 <= u < w) and (0 <= v < h)
    print(f"Pixel: u={u:.1f}, v={v:.1f}  depth={depth:.3f} m  "
          f"{'(in frame)' if in_frame else '(OUT OF FRAME)'}")

    # --- draw ---
    result = draw_circle(img, u, v, radius=args.radius)

    if args.output:
        cv2.imwrite(args.output, result)
        print(f"Saved to {args.output}")
    else:
        cv2.imshow("projection", result)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
