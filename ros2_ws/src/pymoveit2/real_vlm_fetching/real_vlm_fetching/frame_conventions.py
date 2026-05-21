import math


def clock_to_world_direction(clock: int) -> list:
    """
    Map clock position (1-12) to a unit vector in world XY plane.

    Convention:
      12 -> +Y, 3 -> +X, 6 -> -Y, 9 -> -X
    Clock increments are 30 degrees clockwise.
    """
    if clock not in range(1, 13):
        raise ValueError(f"clock must be 1-12, got {clock}")
    # 12 o'clock = 90 degrees from +X axis in standard math frame
    # Each hour clockwise = -30 degrees in standard frame
    angle_deg = 90.0 - (clock - 12) * 30.0  # equivalent: 90 - 30*(clock-12)
    # Normalize to avoid floating-point drift at cardinal directions
    angle_deg = angle_deg % 360.0
    angle_rad = math.radians(angle_deg)
    x = round(math.cos(angle_rad), 10)
    y = round(math.sin(angle_rad), 10)
    # Snap near-zero values to exactly 0.0
    x = 0.0 if abs(x) < 1e-9 else x
    y = 0.0 if abs(y) < 1e-9 else y
    return [x, y, 0.0]


def compute_fetch_displacement(
    clock: int,
    horizontal_distance_m: float,
    vertical_clearance_m: float,
) -> list:
    """
    Compute 3D displacement vector for a fetch motion.

    displacement = horizontal_distance_m * horizontal_dir
                 + vertical_clearance_m * [0, 0, 1]
    """
    h_dir = clock_to_world_direction(clock)
    return [
        horizontal_distance_m * h_dir[0],
        horizontal_distance_m * h_dir[1],
        vertical_clearance_m,
    ]
