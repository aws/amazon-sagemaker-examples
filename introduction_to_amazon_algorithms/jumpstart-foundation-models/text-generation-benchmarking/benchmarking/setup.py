import os
from setuptools import find_namespace_packages
from setuptools import setup


def read_content(file_name: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), file_name)) as f:
        return f.read()


package_requirements = read_content("requirements.txt")

setup(
    author="Amazon Web Services",
    name="jumpstart_bench",
    version="0.0.1",
    description="SageMaker JumpStart benchmarking library",
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=package_requirements,
    entry_points={
        "console_scripts": [
            "jumpstart-bench=jumpstart_bench.executable:main",
        ],
    },
)