# pyCoilGen
This Python project is a conversion of the [CoilGen MatLab project](https://github.com/Philipp-MR/CoilGen) developed by Philipp Amrein, a community-based tool for the generation of gradient field coil layouts within the MRI/NMR environment. It is based on a boundary element method and generates an interconnected non-overlapping wire-tracks on 3D support structures.

The user specifies a target field and a surface mesh geometry. The code then generates a coil layout in the form of a non-overlapping, interconnected wire trace to achieve the desired field.

It is a community-based tool designed for generating [gradient field coil](https://mriquestions.com/gradient-coils.html) layouts within the
[MRI](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging)/NMR environment.

The tool is based on a boundary element method and creates interconnected, non-overlapping wire-tracks on 3D support structures.

## Features

- Specify a target field (e.g., `bz(x,y,z)=y`) and a surface mesh geometry.
- Supports built-in surface mesh geometries or importing meshes from .stl files.
- Generates a coil layout in the form of a non-overlapping, interconnected wire trace to achieve the desired field, exported as an .stl file.

For a detailed description of the algorithm, refer to the [research paper](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294).

## Examples

The [`examples`](./examples) directory contains several usage examples for pyCoilGen. 

These examples demonstrate different scenarios and configurations for generating coil layouts.

## Acknowledgements

This conversion from MatLab to Python was facilitated using [ChatGPT, May 24 through August 3 Version](https://chat.openai.com) with manual corrections. 

Additional cross-checking was done using the [online MATLAB](https://matlab.mathworks.com/) provided by MathWorks.

## Installation

Please refer to the [Installation Guide](installation.md) for detailed instructions on how to install and set up pyCoilGen.

## Getting Started

To quickly get started with pyCoilGen, refer to the [Quick Start Guide](quick_start.md) for a basic example and an overview of main components.

## Configuration Parameters

Refer to the [Configuration Parameters Guide](configuration.md) for information on how to configure pyCoilGen to suit specific needs. 

This includes details on available settings, options, and their default values.

## Contributing

If you'd like to contribute to pyCoilGen, please follow the guidelines outlined in the [Contributing Guide](CONTRIBUTING.md).

## License

See [`LICENSE.txt`](./LICENSE.txt) for more information.

## Contact

For inquiries and discussion, please use [pyCoilGen Discussions](https://github.com/kev-m/pyCoilGen/discussions).

## Issues

For issues related to this Python implementation, please visit the [Issues](https://github.com/kev-m/pyCoilGen/issues) page.

## Citation

For citation of this work, please refer to the following publication:
- [Amrein, P., Jia, F., Zaitsev, M., & Littin, S. (2022). CoilGen: Open-source MR coil layout generator. Magnetic Resonance in Medicine, 88(3), 1465-1479.](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294)