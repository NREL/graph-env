# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from importlib.metadata import version

sys.path.insert(0, os.path.abspath("../.."))

import graphenv
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = "graphenv"
copyright = "2022, Alliance for Sustainable Energy, LLC"
author = "David Biagioni, Charles Edison Tripp, Jeffrey Law, Struan Clark, and Peter St. John"

# The full version, including alpha/beta/rc tags

release = version("graphenv")
version = ".".join(release.split(".")[:3])
# version = graphenv.__version__
# release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    # 'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    # 'nbsphinx',  # Integrate Jupyter Notebooks and Sphinx
    # 'IPython.sphinxext.ipython_console_highlighting',
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "nbsphinx",
]

napoleon_google_docstring = True

# # Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3/", None),
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # ['_static']

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autoclass_content = "both"  # Add __init__ doc (ie. params) to class summaries
html_show_sourcelink = (
    False  # Remove 'view source code' from top of page (for html, not python)
)
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class
set_type_checking_flag = True  # Enable 'expensive' imports for sphinx_autodoc_typehints
nbsphinx_allow_errors = True  # Continue through Jupyter errors
autodoc_typehints = "description"
add_module_names = False  # Remove namespaces from class/method signatures
