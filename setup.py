"""
Setup script for the project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# read the README for the long description
HERE = Path(__file__).parent
LONG_DESC = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="gp-quadratures",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        # core dependencies
        "torch>=1.10.0",
        "numpy>=1.21.0",
        "finufft>=1.2.0",
        "pytorch-finufft",
        # utility
        "pydantic>=1.8",
        # testing
        "pytest>=6.0",
    ],
    python_requires=">=3.10",
    description="A package for Gaussian Process quadratures",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    author="Dan Biderman",
    author_email="danbider@gmail.com",
    url="https://github.com/danbider/gp-quadratures",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=["gaussian-process", "quadrature", "nufft", "torch"],
)
