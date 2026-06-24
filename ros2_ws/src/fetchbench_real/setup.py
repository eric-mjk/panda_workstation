from glob import glob
from setuptools import find_packages, setup


package_name = "fetchbench_real"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["real_active_perception", "real_active_perception.*", "real_offline", "real_offline.*"]),
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
    install_requires=["setuptools", "PyYAML", "Pillow", "google-genai"],
    zip_safe=False,
    maintainer="Eric",
    maintainer_email="eric@example.com",
    description="FetchBench method code staged for Panda real-robot adaptation.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fetchbench_active_perception = real_active_perception.coordinator:main",
            "fetchbench_offline_pipeline = real_offline.pipeline:main",
        ],
    },
)
