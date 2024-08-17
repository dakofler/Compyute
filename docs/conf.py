# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

project = "Compyute"
author = "Daniel Kofler"
year = datetime.now().year
copyright = f"2022-{year}, Daniel Kofler"

import compyute

release = compyute.__version__
version = compyute.__version__

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    # "logo": {
    #     "image_light": "_static/<add_logo>.svg",
    #     "image_dark": "_static/<add_logo>.svg",
    # },
    "github_url": "https://github.com/dakofler/Compyute",
    "collapse_navigation": True,
    "header_links_before_dropdown": 6,
    "navbar_end": [
        "search-button",
        "theme-switcher",
        "navbar-icon-links",
    ],
    "navbar_persistent": [],
    "show_version_warning_banner": True,
}
html_title = "%s v%s Manual" % (project, version)
html_static_path = ["_static"]
html_context = {"default_mode": "light"}

autodoc_typehints = "none"

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
autodoc_member_order = "groupwise"
