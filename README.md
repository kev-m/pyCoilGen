# pyCoilGen
[![GitHub license](https://img.shields.io/github/license/kev-m/pyCoilGen)](https://github.com/kev-m/pyCoilGen/blob/main/LICENSE)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pycoilgen?logo=pypi)](https://pypi.org/project/pycoilgen/)
[![semver](https://img.shields.io/badge/semver-2.0.0-blue)](https://semver.org/)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/kev-m/pyCoilGen?sort=semver)](https://github.com/kev-m/pyCoilGen/releases)
[![Code style: autopep8](https://img.shields.io/badge/code%20style-autopep8-000000.svg)](https://pypi.org/project/autopep8/)

![pyCoilGen logo](https://github.com/kev-m/pyCoilGen/blob/master/docs/source/figures/Logo_small.png)

The **pyCoilGen** project is an open source tool for generating coil winding layouts, such as
[gradient field coils](https://mriquestions.com/gradient-coils.html), within the
[MRI](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging) and
[NMR](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance) environments. **pyCoilGen** is based on a boundary element method and generates interconnected non-overlapping wire-tracks on 3D support structures.

This Python project is a port of the MATLAB [CoilGen code](https://github.com/Philipp-MR/CoilGen) developed by Philipp Amrein. 

For detailed documentation, refer to the [pyCoilGen Documentation](https://pycoilgen.readthedocs.io/).

## Installation

Refer to the [Installation Guide](https://pycoilgen.readthedocs.io/en/latest/installation.html) for detailed instructions on how to install and set up **pyCoilGen**.

## Compatibility

Release 0.3.0 and later require Python >= 3.11. If you are bound to an older Python, pip installs the newest release that supports it, or name a release, for example `pip install pycoilgen==0.2.4`.

| Release | Python | numpy | scipy |
|---------|--------|-------|-------|
| 0.0.8 to 0.1.1 | 3.7 to 3.10 (declared >= 3.6) | ==1.21.6 | ==1.7.3 |
| 0.1.2 to 0.2.4 | 3.6 to 3.12 (declared >= 3.6) | 1.* | 1.* |
| 0.3.0 to 0.6.0 | >= 3.11, tested on 3.11 and 3.13 | >= 1.26, < 3 | >= 1.11, < 2 |

See [Compatibility](https://pycoilgen.readthedocs.io/en/latest/compatibility.html) for details, including the tested environments.

From release 0.6.0, the optional `fast` extra (`pip install pycoilgen[fast]`) compiles the largest calculations with Numba and uses all CPU cores, which makes designs up to 90 times faster than release 0.4.0 (6 to 28 times with numpy alone, from the vectorised steps), with identical results. See [Install Fast Extras](https://pycoilgen.readthedocs.io/en/latest/installation.html#install-fast-extras).

## Known Issues and Bug Fixes

Release 0.4.0 corrects the units of `ohmian_resistance`: it was computed as `L / (A * value)` with the copper *resistivity* (0.018e-6 Ohm·m) as `specific_conductivity_conductor`, giving about 1e12 Ohm. It is now `R = rho * L / A`. The parameter is also renamed `specific_resistivity_conductor`; the old name is deprecated. Results of 0.3.0 and earlier should be recomputed. See [Bug fixes](https://pycoilgen.readthedocs.io/en/latest/bug_fixes.html).

## Examples

The [`examples`](https://github.com/kev-m/pyCoilGen/blob/master/examples) directory contains several examples for how to use **pyCoilGen**. These examples demonstrate different scenarios and configurations for generating coil layouts.

## Acknowledgements

The porting of the code from MATLAB to Python was facilitated by [ChatGPT, May 24 through August 3 Version](https://chat.openai.com) with manual corrections. 

Additional cross-checking was done using [MATLAB Online](https://www.mathworks.com/products/matlab-online.html) provided by MathWorks.

The performance improvements of release 0.6.0 (the vectorised pipeline steps and the optional Numba kernels) were developed with [Claude Sonnet](https://www.anthropic.com/claude) from Anthropic, using Claude Code. The results were checked against the previous implementations and the regression harness.

The circular bi-planar mesh builder, the Gerber and groove plate exporters and the independent field verification (release 0.3.0) are based on ideas from pull request [#82](https://github.com/kev-m/pyCoilGen/pull/82) by [Sairam Geethanath](https://github.com/sairamgeethanath), re-implemented to fit the existing pipeline.

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
