#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    rgb_topic = LaunchConfiguration('rgb_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_config_file = LaunchConfiguration('camera_config_file')
    depth_scale = LaunchConfiguration('depth_scale')
    max_depth = LaunchConfiguration('max_depth')
    voxel_length = LaunchConfiguration('voxel_length')
    sdf_trunc = LaunchConfiguration('sdf_trunc')
    sync_slop = LaunchConfiguration('sync_slop')
    output_dir = LaunchConfiguration('output_dir')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'rgb_topic',
                default_value='/camera/color/image_raw',
                description='RGB image topic.',
            ),
            DeclareLaunchArgument(
                'depth_topic',
                default_value='/camera/aligned_depth_to_color/image_raw',
                description='Aligned depth image topic.',
            ),
            DeclareLaunchArgument(
                'camera_config_file',
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare('tsdf_voxel'),
                        'config',
                        'real_robot_camera.yaml',
                    ]
                ),
                description='YAML file containing hardcoded camera intrinsics and extrinsics.',
            ),
            DeclareLaunchArgument(
                'depth_scale',
                default_value='1000.0',
                description='Depth units per meter (1000.0 for D435, 4000.0 for L515).',
            ),
            DeclareLaunchArgument(
                'max_depth',
                default_value='3.0',
                description='Maximum depth in meters to integrate.',
            ),
            DeclareLaunchArgument(
                'voxel_length',
                default_value='0.005',
                description='TSDF voxel size in meters.',
            ),
            DeclareLaunchArgument(
                'sdf_trunc',
                default_value='0.04',
                description='TSDF truncation distance in meters.',
            ),
            DeclareLaunchArgument(
                'sync_slop',
                default_value='0.1',
                description='Maximum RGB/depth timestamp difference in seconds.',
            ),
            DeclareLaunchArgument(
                'output_dir',
                default_value='/tmp',
                description='Directory to save extracted mesh PLY files.',
            ),
            Node(
                package='tsdf_voxel',
                executable='tsdf_integrator',
                name='tsdf_integrator',
                output='screen',
                parameters=[
                    {
                        'rgb_topic': rgb_topic,
                        'depth_topic': depth_topic,
                        'camera_config_file': camera_config_file,
                        'depth_scale': depth_scale,
                        'max_depth': max_depth,
                        'voxel_length': voxel_length,
                        'sdf_trunc': sdf_trunc,
                        'sync_slop': sync_slop,
                        'output_dir': output_dir,
                    }
                ],
            ),
        ]
    )
