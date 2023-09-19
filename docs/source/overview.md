# Overview

**pyCoilGen** is a community-based tool for the generation of [gradient field coil](https://mriquestions.com/gradient-coils.html) layouts within the
[MRI](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging) and [NMR](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance) environments. . **pyCoilGen** is based on a boundary element method and generates an interconnected non-overlapping wire-tracks on 3D support structures. 

## Features

- Specify a target field (e.g., `bz(x,y,z)=y`) and a surface mesh geometry.
- Supports built-in surface mesh geometries or 3D meshes defined in `.stl` files.
- Generates a coil layout in the form of a non-overlapping, interconnected wire trace to achieve the desired field, exported as an `.stl` file.

For a detailed description of the algorithm, refer to the research paper [CoilGen: Open-source MR coil layout generator](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294).

## Examples

The [`examples`](https://github.com/kev-m/pyCoilGen/examples) directory contains several usage examples for pyCoilGen. 

These examples demonstrate different scenarios and configurations for generating coil layouts.

## Citation

Use the following publication if you need to cite this work:

- [Amrein, P., Jia, F., Zaitsev, M., & Littin, S. (2022). CoilGen: Open-source MR coil layout generator. Magnetic Resonance in Medicine, 88(3), 1465-1479.](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294)
