#!/usr/bin/env python3
"""Move the Panda arm to a joint goal with a selectable OMPL planner.

Plans first, previews the EE path as a MarkerArray on /panda_planned_path_markers,
waits for Enter, then executes.

Usage
-----
ros2 run pymoveit2 panda_planner_joint_goal.py --ros-args \
  -p planner_id:=RRTConnectkConfigDefault \
  -p joint_positions:=[0.0,-0.785,0.0,-2.356,0.0,1.571,0.785]

Named presets (pass via -p preset:=<name>):
  ready     [0.0, -pi/4, 0.0, -3pi/4, 0.0, pi/2, pi/4]  (default)
  extended  [0.0, 0.0, 0.0, -0.1, 0.0, pi/2, pi/4]
  home      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

Available planners (ompl_planning.yaml):
  RRTConnectkConfigDefault  (default) — bidirectional RRT, fast, general purpose
  RRTkConfigDefault         — single-tree RRT
  RRTstarkConfigDefault     — asymptotically optimal RRT*
  PRMkConfigDefault         — probabilistic roadmap
  PRMstarkConfigDefault     — optimal PRM*
  ESTkConfigDefault         — expansive space tree
  SBLkConfigDefault         — single-query bidirectional lazy
  KPIECEkConfigDefault      — kinodynamic KPIECE
  BKPIECEkConfigDefault     — bidirectional KPIECE
  LBKPIECEkConfigDefault    — lazy bidirectional KPIECE
  TRRTkConfigDefault        — transition RRT (cost-aware)
  BiTRRTkConfigDefault      — bidirectional TRRT
  LBTRRTkConfigDefault      — lower bound TRRT
  FMTkConfigDefault         — fast marching tree
  BFMTkConfigDefault        — bidirectional FMT
  PDSTkConfigDefault        — planning with dynamic shortcuts
  STRIDEkConfigDefault      — STRIDE
  BiESTkConfigDefault       — bidirectional EST
  ProjESTkConfigDefault     — projected EST
  LazyPRMkConfigDefault     — lazy PRM
  LazyPRMstarkConfigDefault — lazy PRM*
  SPARSkConfigDefault       — SPARS roadmap
  SPARStwokConfigDefault    — SPARS2 roadmap
  TrajOptDefault            — trajectory optimisation

RViz setup
----------
Add a MarkerArray display and set the topic to /panda_planned_path_markers.
"""

import time
from math import pi
from threading import Thread

import rclpy
from geometry_msgs.msg import Point
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from pymoveit2 import MoveIt2
from pymoveit2.robots import panda as robot

MARKER_TOPIC = "/panda_planned_path_markers"

DEFAULT_BASE_LINK = "panda_link0"
DEFAULT_END_EFFECTOR_LINK = "panda_link8"
DEFAULT_MOVE_GROUP = "panda_arm"

PRESETS = {
    "ready":    [0.0, -pi / 4.0, 0.0, -3.0 * pi / 4.0, 0.0, pi / 2.0, pi / 4.0],
    "extended": [0.0,  0.0,      0.0, -0.1,              0.0, pi / 2.0, pi / 4.0],
    "home":     [0.0,  0.0,      0.0,  0.0,              0.0, 0.0,      0.0],
}

KNOWN_PLANNERS = {
    "RRTConnectkConfigDefault",
    "RRTkConfigDefault",
    "RRTstarkConfigDefault",
    "PRMkConfigDefault",
    "PRMstarkConfigDefault",
    "ESTkConfigDefault",
    "SBLkConfigDefault",
    "KPIECEkConfigDefault",
    "BKPIECEkConfigDefault",
    "LBKPIECEkConfigDefault",
    "TRRTkConfigDefault",
    "BiTRRTkConfigDefault",
    "LBTRRTkConfigDefault",
    "FMTkConfigDefault",
    "BFMTkConfigDefault",
    "PDSTkConfigDefault",
    "STRIDEkConfigDefault",
    "BiESTkConfigDefault",
    "ProjESTkConfigDefault",
    "LazyPRMkConfigDefault",
    "LazyPRMstarkConfigDefault",
    "SPARSkConfigDefault",
    "SPARStwokConfigDefault",
    "TrajOptDefault",
}

# How long to wait for planning / FK futures (seconds)
_FUTURE_TIMEOUT = 30.0


def _wait_future(future, timeout=_FUTURE_TIMEOUT):
    """Poll a rclpy Future using time.sleep — safe with a background executor."""
    deadline = time.time() + timeout
    while not future.done():
        if time.time() > deadline:
            return False
        time.sleep(0.05)
    return True


def _wait_execution(moveit2, timeout=120.0):
    """Poll MoveIt2 execution state without calling spin_once."""
    deadline = time.time() + timeout
    time.sleep(0.3)  # give the action server time to accept the goal
    while time.time() < deadline:
        # Access name-mangled private flags
        requested = moveit2._MoveIt2__is_motion_requested
        executing = moveit2._MoveIt2__is_executing
        if not requested and not executing:
            return
        time.sleep(0.1)


def _build_markers(ee_points, frame_id):
    """Return a MarkerArray with a line strip path and spheres at each waypoint."""
    now = rclpy.clock.Clock().now().to_msg()
    cyan = ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0)
    yellow = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)

    line = Marker()
    line.header.frame_id = frame_id
    line.header.stamp = now
    line.ns = "planned_path"
    line.id = 0
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = 0.005  # line width in metres
    line.color = cyan
    line.points = ee_points

    spheres = Marker()
    spheres.header.frame_id = frame_id
    spheres.header.stamp = now
    spheres.ns = "planned_path"
    spheres.id = 1
    spheres.type = Marker.SPHERE_LIST
    spheres.action = Marker.ADD
    spheres.scale.x = 0.015
    spheres.scale.y = 0.015
    spheres.scale.z = 0.015
    spheres.color = yellow
    spheres.points = ee_points

    array = MarkerArray()
    array.markers = [line, spheres]
    return array


def main():
    rclpy.init()
    node = Node("panda_planner_joint_goal")

    node.declare_parameter("planner_id", "RRTConnectkConfigDefault")
    node.declare_parameter("preset", "")
    node.declare_parameter(
        "joint_positions",
        [0.0, -pi / 4.0, 0.0, -3.0 * pi / 4.0, 0.0, pi / 2.0, pi / 4.0],
    )
    node.declare_parameter("base_link_name", DEFAULT_BASE_LINK)
    node.declare_parameter("end_effector_name", DEFAULT_END_EFFECTOR_LINK)
    node.declare_parameter("group_name", DEFAULT_MOVE_GROUP)
    node.declare_parameter("max_velocity", 0.5)
    node.declare_parameter("max_acceleration", 0.5)

    planner_id = node.get_parameter("planner_id").value
    preset = node.get_parameter("preset").value
    group_name = node.get_parameter("group_name").value
    ee_link = node.get_parameter("end_effector_name").value

    if planner_id not in KNOWN_PLANNERS:
        node.get_logger().error(
            f"Unknown planner '{planner_id}'. Available: {sorted(KNOWN_PLANNERS)}"
        )
        rclpy.shutdown()
        return

    if preset:
        if preset not in PRESETS:
            node.get_logger().error(
                f"Unknown preset '{preset}'. Available: {sorted(PRESETS)}"
            )
            rclpy.shutdown()
            return
        joint_positions = PRESETS[preset]
        node.get_logger().info(f"Using preset '{preset}': {joint_positions}")
    else:
        joint_positions = list(node.get_parameter("joint_positions").value)

    callback_group = ReentrantCallbackGroup()
    moveit2 = MoveIt2(
        node=node,
        joint_names=robot.joint_names(),
        base_link_name=node.get_parameter("base_link_name").value,
        end_effector_name=ee_link,
        group_name=group_name,
        callback_group=callback_group,
    )
    moveit2.planner_id = planner_id
    moveit2.max_velocity = node.get_parameter("max_velocity").value
    moveit2.max_acceleration = node.get_parameter("max_acceleration").value

    marker_pub = node.create_publisher(MarkerArray, MARKER_TOPIC, 1)

    # Start background executor — all ROS callbacks run here.
    # We never call rclpy.spin_once() after this point to avoid stealing
    # the node away from this executor.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor_thread = Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    time.sleep(1.0)  # wait for action servers to be discovered

    node.get_logger().info(
        f"Planner: {planner_id} | "
        f"Group: {group_name} | "
        f"Target joints: {[round(v, 4) for v in joint_positions]}"
    )

    # --- Plan (async, polled with sleep so the background executor keeps running) ---
    node.get_logger().info("Planning...")
    plan_future = moveit2.plan_async(joint_positions=joint_positions)
    if plan_future is None or not _wait_future(plan_future):
        node.get_logger().error("Planning timed out or failed to start.")
        rclpy.shutdown()
        executor_thread.join()
        return

    trajectory = moveit2.get_trajectory(plan_future)
    if trajectory is None:
        node.get_logger().error("Planning failed — no trajectory returned.")
        rclpy.shutdown()
        executor_thread.join()
        return

    node.get_logger().info(
        f"Planned {len(trajectory.points)} waypoints. Computing FK for preview..."
    )

    # --- FK each waypoint to get EE positions for the MarkerArray ---
    ee_points = []
    joint_names = robot.joint_names()
    for wp in trajectory.points:
        js = JointState()
        js.name = joint_names
        js.position = list(wp.positions)
        fk_future = moveit2.compute_fk_async(joint_state=js, fk_link_names=[ee_link])
        if fk_future is None or not _wait_future(fk_future):
            node.get_logger().warn("FK timed out for a waypoint — skipping.")
            continue
        pose = moveit2.get_compute_fk_result(fk_future, fk_link_names=[ee_link])
        if pose is not None:
            p = pose[0].pose.position if isinstance(pose, list) else pose.pose.position
            ee_points.append(Point(x=p.x, y=p.y, z=p.z))

    # Publish markers several times so RViz catches them
    if ee_points:
        markers = _build_markers(ee_points, DEFAULT_BASE_LINK)
        node.get_logger().info(
            f"Publishing {len(ee_points)} EE waypoints to {MARKER_TOPIC}"
        )
        for _ in range(8):
            marker_pub.publish(markers)
            time.sleep(0.25)
    else:
        node.get_logger().warn("No FK results — MarkerArray will be empty.")

    # --- Confirm before executing ---
    input("Press Enter to execute, or Ctrl-C to abort...")

    node.get_logger().info("Executing...")
    moveit2.execute(trajectory)
    _wait_execution(moveit2)

    node.get_logger().info("Done.")
    rclpy.shutdown()
    executor_thread.join()


if __name__ == "__main__":
    main()
