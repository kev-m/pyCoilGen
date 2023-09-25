# Overview

**pyCoilGen** is a community-based tool for the generation of [gradient field coil](https://mriquestions.com/gradient-coils.html) layouts within the
[MRI](https://en.wikipedia.org/wiki/Magnetic_resonance_imaging) and [NMR](https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance) environments. 

**pyCoilGen** is based on a boundary element method and generates interconnected non-overlapping wire-tracks on 3D support structures.

The source code for **pyCoilGen** is available on [GitHub](https://github.com/kev-m/pyCoilGen).

```{figure} figures/mesh_shielded_ygradient_swept_3D_copper.png
:scale: 50 %
:align: center
:alt: A 3D rendered view of the `.stl` swept output.

A 3D rendering of the `.stl` output for the `shielded_ygradient_coil.py` example.
```
```{figure} figures/plot_shielded_ygradient_coil_2D.png
:scale: 50 %
:align: center
:alt: A colour plot showing the stream function and the corresponding contour groups.

A colour plot showing the 2D stream function and the corresponding contour groups for the `shielded_ygradient_coil.py` example. 
```

## Features

With **pyCoilGen**, you can:

- Specify a target field (e.g., `bz(x,y,z)=y`) and a surface mesh geometry.
- Use built-in surface mesh geometries or 3D meshes defined in `.stl` files.
- Generate a coil layout in the form of a non-overlapping, interconnected wire track to achieve the desired field, exported as an `.stl` file.

For a detailed description of the algorithm, refer to the research paper [CoilGen: Open-source MR coil layout generator](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294).

## Examples

The [`examples`](https://github.com/kev-m/pyCoilGen/blob/master/examples) directory in the GitHub repository contains several usage examples for **pyCoilGen**. These examples demonstrate different scenarios and configurations for generating coil layouts.

## Citation

Use the following publication if you need to cite this work:

- [Amrein, P., Jia, F., Zaitsev, M., & Littin, S. (2022). CoilGen: Open-source MR coil layout generator. Magnetic Resonance in Medicine, 88(3), 1465-1479.](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294)
