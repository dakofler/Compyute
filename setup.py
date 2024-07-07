"""Setup file for compyute."""

# template by https://github.com/rochacbruno/python-project-template/blob/main/setup.py

import pathlib

from setuptools import find_packages, setup

setup(
    name="compyute",
    version=pathlib.Path("compyute/VERSION").read_text(encoding="utf-8"),
    description="Deep learning toolbox developed in pure NumPy/CuPy.",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/dakofler/Compyute/",
    author="Daniel Kofler",
    author_email="dkofler@outlook.com",
    license="MIT",
    project_urls={
        "Source Code": "https://github.com/dakofler/Compyute",
        "Issues": "https://github.com/dakofler/Compyute/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10,<3.12",
    install_requires=[
        "cupy_cuda12x>=13.0.0",
        "ipywidgets>=8.1.2",
        "numpy>=1.26.4,<2.0.0",
        "regex>=2023.12.25",
        "tqdm>=4.66.2",
    ],
    packages=find_packages(exclude=["tests", ".github", ".venv", "docs"]),
    include_package_data=True,
)
