from setuptools import find_packages, setup

VERSION = "0.1.0"
DESCRIPTION = "This package is built for the ERA assignments"

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="lightning module",
    version=VERSION,
    python_requires=">=3.8",
    install_requires=requirements,
)