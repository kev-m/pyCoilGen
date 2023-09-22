"""Magnetic Field Coil Generator for Python."""

import logging

from pyCoilGen.pyCoilGen_release import pyCoilGen

if __name__ == "__main__":
    # Set up logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Fetch parameters from the command-line
    pyCoilGen(log)