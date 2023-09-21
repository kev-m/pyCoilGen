# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from pyCoilGen import __version__
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyCoilGen User Guide'
copyright = '2023, Kevin Meyer, Philipp Amrein'
author = 'Kevin Meyer, Philipp Amrein'
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
source_dir = os.path.abspath(os.path.dirname(__file__))

extensions = [
    # Using 'myst_parser' for MD parsing, as per https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html
    'myst_parser',
]
# Specify the source suffix for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# Myst Extensions
myst_enable_extensions = [
    'deflist',
    ]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'requirements.txt',
    ]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'latest'
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
