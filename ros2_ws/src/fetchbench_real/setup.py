from glob import glob
from setuptools import find_packages, setup


package_name = "fetchbench_real"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[
        "real_active_perception",
        "real_active_perception.*",
        "real_offline",
        "real_offline.*",
        "real_execute",
        "real_execute.*",
    ]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    package_data={
        "real_active_perception": [
            "view_candidates/*.json",
            "view_candidates/*.png",
        ],
    },
    install_requires=["setuptools", "PyYAML", "Pillow", "google-genai", "open3d"],
    zip_safe=False,
    maintainer="Eric",
    maintainer_email="eric@example.com",
    description="FetchBench method code staged for Panda real-robot adaptation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # Canonical ROS pipeline:
            # ap -> prep -> mask -> vlm -> direction -> execute
            "fetchbench_ap = real_active_perception.coordinator:main",
            "fetchbench_prep = real_offline.prep:main",
            "fetchbench_mask = real_offline.mask:main",
            "fetchbench_vlm = real_offline.vlm:main",
            "fetchbench_direction = real_offline.direction:main",
            "fetchbench_execute = real_execute.pull_best_direction:main",
            "fetchbench_clean = real_offline.clean:main",
            # Backward-compatible aliases.
            "fetchbench_active_perception = real_active_perception.coordinator:main",
            "fetchbench_offline_pipeline = real_offline.pipeline:main",
            "fetchbench_select_subset = real_offline.select_subset:main",
            "fetchbench_pull_best_direction = real_execute.pull_best_direction:main",
            "fetchbench_publish_ply_visualization = real_execute.publish_ply_visualization:main",
            "fetchbench_publish_debug_geometry = real_execute.publish_debug_geometry:main",
        ],
    },
)
