# Known Issues and Bug Fixes

This page records corrections that change the *results* of earlier releases, so that you can judge whether they
affect your work. Fixes that do not change results are only listed in the [CHANGELOG](https://github.com/kev-m/pyCoilGen/blob/master/CHANGELOG.md).

## Contour loop closing fix (0.4.1)

**Affected:** releases 0.4.0 and earlier (and the MATLAB CoilGen original). **Affected results:** the contour loops,
and everything derived from them: the wire path, the field errors of the unconnected contours and of the layout, the
resistance and the wire length. The stream function and the sensitivity matrices are not affected.

**Problem.** The contour lines of the stream function never arrive closed, so `process_raw_loops` closes them by
appending the first point. The MATLAB original only did this if *both* coordinates of the last point differed from those
of the first point, compared for exact equality. A contour that ends on a mesh grid line shares one coordinate with its
start, so it was left open, with a gap that was typically 5-15% of the loop. Whether that happened depended on the last
bit of the contour, i.e. on rounding: a change of the order of 1e-15 in the sensitivity matrix, for example from another
version of NumPy or SciPy, could close one loop and leave another one open.

**Correction.** A loop is now closed unless its end point coincides with its start point, within a tolerance relative to
the size of the loop (`is_loop_closed`). The result no longer depends on the last bit of the contour.

**What you need to do.** Designs without such contours are unchanged (the examples `preoptimzed_breast_coil`,
`preoptimzed_svd_coil`, `ygradient_coil` and others give the same results). For designs with them, expect small changes
of the wire path (in the regression case `biplanar_x_tikhonov` the wire path moved by about 7e-5 m, and the
gradient deviation changed from 15.039% to 15.040%), and larger changes of the field error of the unconnected contours,
because loops that were open are now closed. Rerun designs with 0.4.1 if you compare such values with earlier results.

## Resistance units fix (0.4.0)

**Affected:** releases 0.3.0 and earlier (and the MATLAB CoilGen original). **Affected result:** `ohmian_resistance`
of each coil part, created by `create_sweep_along_surface`.

**Problem.** The parameter `specific_conductivity_conductor` has the default `0.018e-6`. That is the *resistivity* of
copper in Ohm·m, not a conductivity (copper is about 5.8e7 S/m). The resistance of the swept wire was
calculated as `R = L / (A · value)`, i.e. dividing by a resistivity. With the default this gave values of about 1e12 Ohm.

**Correction.** The wire resistance is now `R = ρ · L / A`, with `ρ = specific_resistivity_conductor`. For a typical
design this is now a few milli-Ohm, as in the regression cases.

**What is not affected.** The field, the stream function, the wire path and the Tikhonov resistance matrix
(`calculate_resistance_matrix` already used the value as a resistivity, `ρ / thickness`). Only `ohmian_resistance`
changed.

**What you need to do.**
- If you read `ohmian_resistance` from results of 0.3.0 or earlier, recompute it, or rerun with 0.4.0.
- If you passed a *conductivity* (for example `5.8e7`) to `specific_conductivity_conductor` to work around this,
  pass the resistivity (`1.0 / 5.8e7`) instead.
- The parameter was renamed to `specific_resistivity_conductor` in the same release, because it always was a resistivity. The old name
  `specific_conductivity_conductor` still works, as a command line option and as a dictionary key, but logs a deprecation
  warning. Update your scripts, as the old name will be removed in a later release. See [Configuration](configuration.md).
