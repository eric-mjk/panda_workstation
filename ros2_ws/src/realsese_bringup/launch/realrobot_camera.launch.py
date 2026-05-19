#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    start_realsense = LaunchConfiguration("start_realsense")
    start_camera_tf = LaunchConfiguration("start_camera_tf")
    start_pointcloud = LaunchConfiguration("start_pointcloud")

    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    points_topic = LaunchConfiguration("points_topic")
    pointcloud_frame_id = LaunchConfiguration("pointcloud_frame_id")
    depth_scale = LaunchConfiguration("depth_scale")
    max_depth = LaunchConfiguration("max_depth")
    stride = LaunchConfiguration("stride")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_realsense",
                default_value="true",
                description="Start the RealSense camera topic publisher.",
            ),
            DeclareLaunchArgument(
                "start_camera_tf",
                default_value="true",
                description="Start the camera_link TF publisher.",
            ),
            DeclareLaunchArgument(
                "start_pointcloud",
                default_value="true",
                description="Start the RGBD pointcloud publisher.",
            ),
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/camera/color/image_raw",
                description="RGB image topic.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/aligned_depth_to_color/image_raw",
                description="Aligned depth image topic.",
            ),
            DeclareLaunchArgument(
                "points_topic",
                default_value="/camera/rgbd/points",
                description="Output PointCloud2 topic.",
            ),
            DeclareLaunchArgument(
                "pointcloud_frame_id",
                default_value="/camera_link",
                description=(
                    "PointCloud2 frame_id. Use an empty value to use the RGB image frame."
                ),
            ),
            DeclareLaunchArgument(
                "depth_scale",
                default_value="4000.0",
                description="Depth divisor for uint16 depth images.",
            ),
            DeclareLaunchArgument(
                "max_depth",
                default_value="3.0",
                description="Maximum depth in meters to include in the cloud.",
            ),
            DeclareLaunchArgument(
                "stride",
                default_value="4",
                description="Pixel stride for downsampling the cloud.",
            ),
            Node(
                package="realsese_bringup",
                executable="realsense_publisher",
                name="realsense_publisher",
                output="screen",
                condition=IfCondition(start_realsense),
            ),
            Node(
                package="realsese_bringup",
                executable="camera_tf_publisher",
                name="camera_tf_publisher",
                output="screen",
                condition=IfCondition(start_camera_tf),
            ),
            Node(
                package="realsese_bringup",
                executable="rgbd_pointcloud",
                name="rgbd_to_pointcloud",
                output="screen",
                parameters=[
                    {
                        "rgb_topic": rgb_topic,
                        "depth_topic": depth_topic,
                        "points_topic": points_topic,
                        "frame_id": pointcloud_frame_id,
                        "depth_scale": depth_scale,
                        "max_depth": max_depth,
                        "stride": stride,
                    }
                ],
                condition=IfCondition(start_pointcloud),
            ),
        ]
    )
