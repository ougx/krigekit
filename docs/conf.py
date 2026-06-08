"""Sphinx configuration for pyKriging documentation."""

import os
import sys

# Make the package importable for type resolution (autoapi doesn't need it,
# but sphinx.ext.intersphinx and napoleon benefit from it).
sys.path.insert(0, os.path.abspath("../src"))

# ---------------------------------------------------------------------------
# Project metadata
# ---------------------------------------------------------------------------
project   = "pyKriging"
copyright = "2026, Michael Ou, mou@sspa.com"
author    = "Michael Ou"
release   = "0.1.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "myst_parser",              # Markdown source files
    "sphinx.ext.mathjax",       # MathJax rendering for $ and $$ math
    "autoapi.extension",        # API reference (no import needed)
    "sphinx.ext.napoleon",      # NumPy/Google docstring styles
    "sphinx.ext.intersphinx",   # Cross-links to numpy, python docs
    "sphinx_copybutton",        # Copy button on code blocks
    "sphinx_design",            # Grid cards, tabs, badges on index page
    "sphinx_gallery.gen_gallery",
]

# ---------------------------------------------------------------------------
# autoapi — generate API pages from Python source without importing
# ---------------------------------------------------------------------------
autoapi_dirs    = ["../src"]
autoapi_type    = "python"
autoapi_root    = "autoapi"   # default; keeps generated pages separate from docs/api/
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files       = True
autoapi_add_toctree_entry = False   # we add it manually in index.md

# ---------------------------------------------------------------------------
# MyST — enable useful Markdown extensions
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # ::: directive syntax
    "deflist",       # definition lists
    "fieldlist",     # field lists
    "dollarmath",    # $...$ inline and $$...$$ display math
    "amsmath",       # \begin{align} ... \end{align} environments
]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
}

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# RST substitutions available in all pages, including autoapi-generated ones.
rst_prolog = """
.. |dt| replace:: :math:`\\Delta t`
"""

# ---------------------------------------------------------------------------
# HTML output — pydata-sphinx-theme
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url":         "https://github.com/ougx/pykriging",
    "navbar_end":         ["navbar-icon-links"],
    "secondary_sidebar_items": ["page-toc"],
    "show_nav_level":     2,
}

html_static_path = ["_static"]
html_css_files   = []   # add custom CSS here if needed

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_numpy_docstring   = True
napoleon_google_docstring  = False
napoleon_use_rtype         = False
napoleon_use_param         = True

_lib_so = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "../src/pykriging/libkriging.so")
_lib_dll = _lib_so.replace("libkriging.so", "kriging.dll")
_lib_available = os.path.exists(_lib_so) or os.path.exists(_lib_dll)

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r".*\.py",
    # Only execute gallery scripts when the compiled library is present.
    "plot_gallery": _lib_available,
}

plot_pre_code = """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pykriging import *
"""