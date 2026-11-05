# Known Issues and Bug Fixes

This page records corrections that change the *results* of earlier releases, so that you can judge whether they
affect your work. Fixes that do not change results are only listed in the [CHANGELOG](https://github.com/kev-m/pyCoilGen/blob/master/CHANGELOG.md).

## Resistance units fix (0.3.1)

**Affected:** releases 0.3.0 and earlier (and the MATLAB CoilGen original). **Affected result:** `ohmian_resistance`
of each coil part, created by `create_sweep_along_surface`.

**Problem.** The parameter `specific_conductivity_conductor` has the default `0.018e-6`. That is the *resistivity* of
copper in Ohm·m, not a conductivity (copper is about 5.8e7 S/m). The resistance of the swept wire was
calculated as `R = L / (A · value)`, i.e. dividing by a resistivity. With the default this gave values of about 1e12 Ohm.

**Correction.** The wire resistance is now `R = ρ · L / A`, with `ρ = specific_conductivity_conductor`. For a typical
design this is now a few milli-Ohm, as in the regression cases.

**What is not affected.** The field, the stream function, the wire path and the Tikhonov resistance matrix
(`calculate_resistance_matrix` already used the value as a resistivity, `ρ / thickness`). Only `ohmian_resistance`
changed.

**What you need to do.**
- If you read `ohmian_resistance` from results of 0.3.0 or earlier, recompute it, or rerun with 0.3.1.
- If you passed a *conductivity* (for example `5.8e7`) to `specific_conductivity_conductor` to work around this,
  pass the resistivity (`1.0 / 5.8e7`) instead.
- The parameter keeps its name for compatibility. It is documented as a resistivity in [Configuration](configuration.md).
