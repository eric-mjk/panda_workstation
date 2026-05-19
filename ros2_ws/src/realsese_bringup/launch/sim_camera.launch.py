#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    start_camera_tf = LaunchConfiguration("start_camera_tf")
    start_pointcloud = LaunchConfiguration("start_pointcloud")

    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    points_topic = LaunchConfiguration("points_topic")
    pointcloud_frame_id = LaunchConfiguration("pointcloud_frame_id")
    use_message_timestamp = LaunchConfiguration("use_message_timestamp")
    depth_scale = LaunchConfiguration("depth_scale")
    max_depth = LaunchConfiguration("max_depth")
    sync_slop = LaunchConfiguration("sync_slop")
    stride = LaunchConfiguration("stride")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "start_camera_tf",
                default_value="true",
                description="Start the sim l515_camera TF publisher.",
            ),
            DeclareLaunchArgument(
                "start_pointcloud",
                default_value="true",
                description="Start the sim RGBD pointcloud publisher.",
            ),
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/isaac_rgb",
                description="RGB image topic.",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/isaac_depth",
                description="Aligned depth image topic.",
            ),
            DeclareLaunchArgument(
                "points_topic",
                default_value="/camera/rgbd/points",
                description="Output PointCloud2 topic.",
            ),
            DeclareLaunchArgument(
                "pointcloud_frame_id",
                default_value="l515_camera",
                description=(
                    "PointCloud2 frame_id. Use an empty value to use the RGB image frame."
                ),
            ),
            DeclareLaunchArgument(
                "use_message_timestamp",
                default_value="false",
                description="Use the RGB image timestamp for PointCloud2 instead of now.",
            ),
            DeclareLaunchArgument(
                "depth_scale",
                default_value="4000.0",
                description="Depth divisor for uint16 depth images.",
            ),
            DeclareLaunchArgument(
                "max_depth",
                default_value="10.0",
                description="Maximum depth in meters to include in the cloud.",
            ),
            DeclareLaunchArgument(
                "sync_slop",
                default_value="0.5",
                description="Maximum RGB/depth timestamp difference in seconds.",
            ),
            DeclareLaunchArgument(
                "stride",
                default_value="4",
                description="Pixel stride for downsampling the cloud.",
            ),
            Node(
                package="realsese_bringup",
                executable="sim_camera_tf_publisher",
                name="sim_camera_tf_publisher",
                output="screen",
                condition=IfCondition(start_camera_tf),
            ),
            Node(
                package="realsese_bringup",
                executable="sim_rgbd_pointcloud",
                name="sim_rgbd_to_pointcloud",
                output="screen",
                parameters=[
                    {
                        "rgb_topic": rgb_topic,
                        "depth_topic": depth_topic,
                        "points_topic": points_topic,
                        "frame_id": pointcloud_frame_id,
                        "use_message_timestamp": use_message_timestamp,
                        "depth_scale": depth_scale,
                        "max_depth": max_depth,
                        "sync_slop": sync_slop,
                        "stride": stride,
                    }
                ],
                condition=IfCondition(start_pointcloud),
            ),
        ]
    )
