"""
Clock-to-base-frame direction mapping using camera extrinsics.

Camera convention: OpenCV optical frame.
  +x = image right
  +y = image down
  +z = optical axis (into scene)

Clock-to-camera formula:
  theta = radians((clock - 3) * 30)
  d_cam = [cos(theta), sin(theta), 0]

The caller supplies R_base_cam (3x3 rotation that maps camera-frame vectors
into the robot base frame), i.e.:
  p_base = R_base_cam @ p_cam + t_base_cam

No external dependencies beyond the Python standard library.
"""
import math


def quaternion_xyzw_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert unit quaternion (x, y, z, w) to a 3x3 rotation matrix.
    Returns a row-major nested list [[r00, r01, r02], ...].
    """
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-10:
        raise ValueError("Quaternion has near-zero norm.")
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return [
        [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ]


def _mat3_vec3(R, v):
    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    ]


def clock_to_cam_direction(clock: int) -> list:
    """Return unit direction in camera frame for a clock position (1-12)."""
    if clock not in range(1, 13):
        raise ValueError(f"clock must be 1-12, got {clock}")
    theta = math.radians((clock - 3) * 30)
    dx = round(math.cos(theta), 10)
    dy = round(math.sin(theta), 10)
    dx = 0.0 if abs(dx) < 1e-9 else dx
    dy = 0.0 if abs(dy) < 1e-9 else dy
    return [dx, dy, 0.0]


def clock_to_base_direction(clock: int, R_base_cam: list) -> dict:
    """
    Map clock direction to robot base frame.

    Parameters
    ----------
    clock : int
        Clock position 1-12.
    R_base_cam : list[list[float]]
        3x3 rotation matrix: maps camera-frame unit vectors into base frame.

    Returns
    -------
    dict with keys:
        d_cam                  – unit vec in camera frame
        d_base_raw             – R_base_cam @ d_cam (may have z component)
        horizontal_direction_base – d_base_raw projected onto XY and normalized
    """
    d_cam = clock_to_cam_direction(clock)
    d_base_raw = _mat3_vec3(R_base_cam, d_cam)

    bx, by = d_base_raw[0], d_base_raw[1]
    norm_xy = math.sqrt(bx*bx + by*by)
    if norm_xy < 1e-9:
        raise ValueError(
            f"Clock {clock} maps to a near-zero XY projection in the base frame "
            f"(d_base_raw={d_base_raw}). Check camera extrinsics."
        )
    return {
        "d_cam": d_cam,
        "d_base_raw": d_base_raw,
        "horizontal_direction_base": [bx / norm_xy, by / norm_xy, 0.0],
    }
