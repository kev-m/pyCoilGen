# Future ideas

Ideas that were investigated or considered, and deliberately **not** done yet. Each entry says what it would give, what
it would cost, and when it would be worth doing. Nothing here is planned. Add to this list rather than losing the
findings.

## GPU acceleration of the Biot-Savart sums (OpenCL)

**Status:** prototyped and measured, not implemented. Worth doing only if someone runs a project that needs it (see
"When it would be worth it").

### What it would speed up

The largest calculations of a design are sums of the Biot-Savart kernel over many sources and many target points:

* the field and the gradient of the final wire path (`wire_kernels.py`, used by `evaluate_field_errors`,
  `calculate_gradient` and `process_raw_loops`),
* the sensitivity matrix and the gradient sensitivity matrix (`sensitivity_kernels.py`).

These are dense, independent, arithmetic-heavy and map well onto a GPU. The rest of the pipeline (contours, loop
grouping, interconnection, the sweep) is branchy geometry and graph code, which a GPU does not help. The dense
Tikhonov solve is limited by double precision matrix products, where consumer GPUs are no faster than a many-core CPU.

### What was measured

A prototype of the wire field and gradient kernels was written in OpenCL (double precision, one work-item per target
point) and compared with the numpy kernels and with the compiled Numba kernels (`pip install pycoilgen[fast]`, which
runs on the CPU and is already in the code). Machine: NVIDIA GeForce RTX 5070 (12 GB, OpenCL 3.0, `cl_khr_fp64`),
16 thread CPU. Problem: 13,202 wire segments and 41,569 target points, the largest call of the
`preoptimzed_svd_coil` example. Results agree to about 1e-14 relative error.

| Kernel | numpy | Numba (CPU) | OpenCL (RTX 5070) |
|--------|-------|-------------|-------------------|
| field | 5.31 s | 0.27 s | 0.20 s |
| gradient | 10.33 s | 0.79 s | 0.33 s |

So on the largest problem the GPU is about 1.4 times (field) to 2.4 times (gradient) faster than the compiled CPU
code, and 25 to 30 times faster than numpy. On smaller problems the two are close, and the CPU code wins on the
smallest. The sensitivity-matrix kernel was not ported to OpenCL, so its gain is not measured; it is expected to be
similar.

### Why it was not done

With the Numba kernels, these sums are already a small part of a run. In the shipped examples they take about 2 s of
a run of 4 to 6 s, so a GPU would save perhaps 20% of the total, not more. The cost is a second optional dependency and
a kernel written in OpenCL C, which is a new language to maintain in the repository.

### When it would be worth it

When a project is large enough that these sums dominate for minutes, not seconds: for example a very fine mesh (many
tens of thousands of nodes), a very dense target region, or a wire path with very many segments, evaluated many times
(optimisation loops, parameter sweeps). Measure first with `PYCOILGEN_NUMBA=auto` on the real project; if the Biot-Savart
stages are still the largest part of the run, this is the next step. A parameter sweep (see the graphical application
below) is a likely case, although reusing the earlier stages of a sweep saves more.

### How to do it

* Optional extra, for example `pip install pycoilgen[gpu]` with `pyopencl`. The wheel is a few MB and needs no vendor SDK:
  the OpenCL driver ships with the GPU driver. Do **not** use CuPy or PyTorch for this: they are NVIDIA-specific or
  hundreds of MB to several GB.
* Add it as a third backend behind `accel.py`, selected by the environment variable `PYCOILGEN_GPU` (off by default).
  The numpy kernels remain the reference and the fallback, and the Numba kernels the CPU fast path.
* Require `cl_khr_fp64` and fall back to the CPU if the device does not have it. Many integrated and mobile GPUs do not.
  Single precision is not enough: the tests compare with the numpy kernels at about 1e-11, and the Tikhonov systems are
  ill-conditioned.
* Consumer GPUs have weak double precision (about 1/64 of single precision). The kernels are cheap enough for this not to
  matter at the sizes above, but do not expect more on very large dense matrix products.
* Upload the target points once and reuse the buffer: the field kernel is called hundreds of times per design with the
  same targets.
* Test with the same comparisons as `tests/test_numba_kernels.py`, and skip when `pyopencl` or a suitable device is missing.

## A graphical application with parameter optimisation

**Status:** an idea only. Nothing has been started.

A separate, top-level application on top of the library, with:

* graphical views of the input (mesh, target region and target field), the intermediate data (stream function, contours,
  loop groups) and the final wire path and its field errors;
* its own project files, which persist whatever is meaningful to the application, so that the library does not need to
  save its internal state;
* parameter studies: sweeps and an optimiser that searches for the best coil, for example the lowest current for a given
  gradient error, or the lowest gradient error for a given current, shown as a trade-off curve (a Pareto front) rather than
  one best design.

### What the library would need first

* **Resuming from a stage.** `pyCoilGen()` runs the whole pipeline in one call. The parameters fall into groups by the
  stage they affect: the mesh and the target region determine the sensitivity matrices (the most expensive stage),
  `tikhonov_reg_factor` affects only the solution of the normal equations, and `levels`, `pot_offset_factor` and the cut
  and interconnection parameters affect only the contours and later stages. A sweep over one of the later parameters
  should reuse the earlier results, which could make it an order of magnitude faster than repeating every run. This needs
  an entry point that resumes from a given stage with cached state.
* **Objectives that do not depend on the pipeline.** The regression harness (`tests/regression/`) already measures the
  gradient deviation and the efficiency (the field per ampere, which determines the current) with an independent
  Biot-Savart calculation of the wire path. Those functions could become the objective functions of the optimiser.
* **Intermediate data without files.** The library returns only the final `CoilSolution`. A per-stage callback or hook
  would give the application the intermediate data without saving it to disk.
* **Plot functions that return figures or data.** The current plotting code opens windows when `debug` is above 0.
* **No shared mutable state between runs.** The debug level (`constants.py`) is a module-level global, and the number of
  Numba threads is a setting of the whole process. Run the points of a sweep in separate processes, each with
  `PYCOILGEN_THREADS=1`, instead of in threads.

### Relation to the other ideas

A sweep multiplies the run time of a design by the number of points: a hundred points of a 10 s design take about 17
minutes. This is the project that would justify the GPU acceleration above, and resuming from a stage should be done
first, since it saves more than a faster kernel.

## Smaller ideas

* **Conductor sweep point location.** After the vectorisation of the sweep, the remaining time is the point location in
  the 2D mesh (`Mesh.get_face_indices`). A spatial index would help for very long wire paths.
