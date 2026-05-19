import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'tsdf_voxel'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy<1.25',
        'open3d',
        'PyYAML',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='eric.mjkim35@gmail.com',
    description='Open3D TSDF integration and mesh extraction from RGBD camera streams.',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'tsdf_integrator = tsdf_voxel.tsdf_integrator:main',
            'tsdf_integrator_once = tsdf_voxel.tsdf_integrator_once:main',
        ],
    },
)
