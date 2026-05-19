import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'realsese_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='eric.mjkim35@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'realsense_publisher = realsese_bringup.realsense_publisher:main',
            'camera_tf_publisher = realsese_bringup.camera_tf_publisher:main',
            'rgbd_pointcloud = realsese_bringup.rgbd_pointcloud:main',
            'sim_camera_tf_publisher = realsese_bringup.sim_camera_tf_publisher:main',
            'sim_rgbd_pointcloud = realsese_bringup.sim_rgbd_pointcloud:main',
        ],
    },
)
