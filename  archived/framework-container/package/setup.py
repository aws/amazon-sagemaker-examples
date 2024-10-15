from __future__ import absolute_import

import os
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

setup(
    name="custom_framework_training",
    version="1.0.0",
    description="Custom framework container training package.",
    keywords="custom framework contaier training package SageMaker",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    author="Giuseppe A. Porcelli",
    author_email="giu.porcelli@gmail.com",
    license="Apache License 2.0",
    install_requires=["sagemaker-training==3.4.1"],
)
