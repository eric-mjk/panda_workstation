import numpy as np

def skew(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )

def quaternion_from_rotation_matrix(rot):
    m00, m01, m02 = rot[0]
    m10, m11, m12 = rot[1]
    m20, m21, m22 = rot[2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = (tr + 1.0) ** 0.5 * 2
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = (1.0 + m00 - m11 - m22) ** 0.5 * 2
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = (1.0 + m11 - m00 - m22) ** 0.5 * 2
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = (1.0 + m22 - m00 - m11) ** 0.5 * 2
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat

def rotation_matrix_from_quaternion(quat_xyzw):
	x, y, z, w = quat_xyzw
	xx = x * x
	yy = y * y
	zz = z * z
	xy = x * y
	xz = x * z
	yz = y * z
	wx = w * x
	wy = w * y
	wz = w * z

	return np.array(
		[
			[1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
			[2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
			[2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
		],
		dtype=np.float64,
	)

def rpy_from_rotation(rotation: np.ndarray):
	# ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
	sy = -rotation[2, 0]
	pitch = float(np.arcsin(np.clip(sy, -1.0, 1.0)))

	if abs(np.cos(pitch)) > 1e-6:
		roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
		yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
	else:
		# Gimbal-lock fallback
		roll = 0.0
		yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))

	return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
