# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import pathlib
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

project = "Compyute"
copyright = "2024, Daniel Kofler"
author = "Daniel Kofler"
release = pathlib.Path("../compyute/VERSION").read_text(encoding="utf-8")
version = pathlib.Path("../compyute/VERSION").read_text(encoding="utf-8")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_typehints = "none"

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
