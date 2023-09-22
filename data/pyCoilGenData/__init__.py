"""Extra data for pyCoilGen, the Open source Magnetic Resonance Coil Generator."""
__version__ = "0.0.2"

from os import path

__data_directory = __file__[:-(len('__init.py__'))]

def data_directory():
    """Get the installation directory of pyCoilGenData"""
    return __data_directory