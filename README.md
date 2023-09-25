# pyCoilGen
[![GitHub license](https://img.shields.io/github/license/kev-m/pyCoilGen)](https://github.com/kev-m/pyCoilGen/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pycoilgen?logo=pypi)](https://pypi.org/project/pycoilgen/)
[![semver](https://img.shields.io/badge/semver-2.0.0-blue)](https://semver.org/)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kev-m/pyCoilGen?sort=semver)](https://github.com/kev-m/pyCoilGen/releases)
[![Code style: autopep8](https://img.shields.io/badge/code%20style-autopep8-000000.svg)](https://pypi.org/project/autopep8/)

The **pyCoilGen** project is a community-based tool for the generation of [gradient field coil](https://mriquestions.com/gradient-coils.html) layouts within the
[MRI](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging) and [NMR](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance) environments. **pyCoilGen** is based on a boundary element method and generates interconnected non-overlapping wire-tracks on 3D support structures. 

This Python project is a port of the MATLAB [CoilGen code](https://github.com/Philipp-MR/CoilGen) developed by Philipp Amrein. 

For detailed documentation, refer to the [pyCoilGen Documentation](https://pycoilgen.readthedocs.io/).

## Installation

Refer to the [Installation Guide](https://pycoilgen.readthedocs.io/en/latest/installation.html) for detailed instructions on how to install and set up **pyCoilGen**.

## Examples

The [`examples`](https://github.com/kev-m/pyCoilGen/blob/master/examples) directory contains several examples for how to use **pyCoilGen**. These examples demonstrate different scenarios and configurations for generating coil layouts.

## Acknowledgements

The porting of the code from MATLAB to Python was facilitated by [ChatGPT, May 24 through August 3 Version](https://chat.openai.com) with manual corrections. 

Additional cross-checking was done using [MATLAB Online](https://www.mathworks.com/products/matlab-online.html) provided by MathWorks.

## Contributing

If you'd like to contribute to **pyCoilGen**, follow the guidelines outlined in the [Contributing Guide](https://github.com/kev-m/pyCoilGen/blob/master/CONTRIBUTING.md).

## License

See [`LICENSE.txt`](https://github.com/kev-m/pyCoilGen/blob/master/LICENSE.txt) for more information.

## Contact

For inquiries and discussion, use [pyCoilGen Discussions](https://github.com/kev-m/pyCoilGen/discussions).

## Issues

For issues related to this Python implementation, visit the [Issues](https://github.com/kev-m/pyCoilGen/issues) page.

## Citation

Use the following publication, if you need to cite this work:

- [Amrein, P., Jia, F., Zaitsev, M., & Littin, S. (2022). CoilGen: Open-source MR coil layout generator. Magnetic Resonance in Medicine, 88(3), 1465-1479.](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294)
