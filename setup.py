"""Setup file for compyute."""

import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs) -> str:
    """Read the contents of a text file safely."""
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements() -> list[str]:
    """Reads requirements from requirements.txt."""
    return [
        line.strip()
        for line in read("requirements.txt").split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


# get version from __version__ variable in compyute/__init__.py
source_root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(source_root, "compyute", "_version.py")) as f:
    exec(f.read())


CLASSIFIERS = """\
"Programming Language :: Python :: 3"
"License :: OSI Approved :: MIT License"
"Operating System :: OS Independent"
"""


setup(
    name="compyute",
    version=__version__,
    description="Deep learning toolbox developed in pure NumPy/CuPy.",
    url="https://github.com/dakofler/Compyute/",
    project_urls={
        "Source Code": "https://github.com/cupy/cupy",
        "Issues": "https://github.com/dakofler/Compyute/issues",
    },
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="Daniel Kofler",
    author_email="dkofler@outlook.com",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements(),
    entry_points={"console_scripts": ["project_name = project_name.__main__:main"]},
    python_requires=">=3.11",
)
