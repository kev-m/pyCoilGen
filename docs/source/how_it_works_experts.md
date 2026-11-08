# How It Works: The Algorithm in Detail

This guide describes what **pyCoilGen** computes at each step, the equations behind it, and how the
[configuration parameters](./configuration.md) act on it. It is for readers who know the field (boundary element methods,
gradient coil design) and want to choose parameters with understanding. If you are new to the subject, read
[How It Works: An Introduction](./how_it_works_beginners.md) first.

The method is the CoilGen algorithm of [Amrein et al. (2022)](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294). This
guide describes the **pyCoilGen** implementation, which is what the code does today. Where a statement depends on a
particular design, it says so. SI units are used throughout: metres, tesla, amperes, ohms.

Each section lists the same four things: what the step computes, the key equations, the parameters that control it, and
what goes wrong.

```{contents}
:local:
:depth: 2
```

## 1. Surface mesh and parameterisation

**What it does.** The support surface is a triangle mesh (a built-in builder or an `.stl` file). Each coil part is a
separate mesh. The mesh is *parameterised*: mapped to a plane with coordinates $(u,v)$, so that the contours of the
stream function can be found in two dimensions and mapped back onto the surface.

**Parameters.** The mesh geometry and builders: [Mesh creation](./configuration.md#mesh-creation).
`iteration_num_mesh_refinement` (default 0) subdivides the triangles, which multiplies the number of nodes.

**What goes wrong.**
- The mesh sets the **ceiling on accuracy**. The stream function has one value per node, so the achievable current
  patterns, and hence the field, are limited by the node count and by the node density where the current must vary.
  Section 11 gives measurements.
- The stream function is held constant along each boundary of the mesh, so no current crosses it.

## 2. Target field

**What it does.** The target is a set of points $\mathbf r_k$ in a region, with the wanted field $\mathbf b_k$ at each.
The region is a sphere (`target_region_radius`, default 0.15 m, sampled at the resolution `target_region_resolution`, default 10) or a mesh. The field comes from `field_shape_function`, for example `x`, `y` or `z` for a gradient, or from a
file.

**What is fitted.** Only the $z$ component of the field (the $B_0$ direction) is used in the stream function solve. The
other components do not enter the cost function, so a design whose other components matter is not controlled by the solver.

**What goes wrong.**
- A target region that is large compared with the surface is hard to meet, as the field error rises towards the coil.
  Keep the target region well inside the surface.
- Too few target points (`target_region_resolution`) leaves the field between points unconstrained. Too many costs
  time (the sensitivity matrix grows with the number of points) and, as the problem is usually under-determined
  by the number of nodes, adds little.

## 3. Basis functions and the sensitivity matrix

**What it does.** The stream function $\psi$ is represented with piecewise-linear "hat" functions, one per node:

$$\psi(\mathbf r) = \sum_i \psi_i\,\phi_i(\mathbf r).$$

The surface current density is $\mathbf J = \nabla\psi\times\mathbf n$ (up to a sign convention), so it is constant in
each triangle and its divergence is zero by construction: no charge accumulates. $\psi$ has the unit of current (A),
since the difference of $\psi$ between two contours is the current between them.

The field at target point $k$ follows from the Biot–Savart law, integrated over the triangles of the ring of every node:

$$\mathbf B(\mathbf r_k) = \frac{\mu_0}{4\pi}\int_S \frac{\mathbf J(\mathbf r')\times(\mathbf r_k-\mathbf r')}{|\mathbf r_k-\mathbf r'|^{3}}\,dA'
\quad\Rightarrow\quad b_k = \sum_i S_{ki}\,\psi_i .$$

$S$ is the **sensitivity matrix**, of size (target points) × (nodes). It is built once. A second matrix gives the gradient
of the field in the same way.

**Parameters.** `gauss_order` (default 2) is the number of quadrature points per triangle.

**Cost.** This is typically the largest calculation: it scales with
(target points) × (nodes) × (triangles per node) × (quadrature points). It is the step to watch when you refine the mesh.
Because the field is linear in $\psi$, nothing is iterated here.

**What goes wrong.** Target points close to the surface (a few triangle edge lengths) make the integrand nearly
singular, so the quadrature is poor, and the error rises. Keep the target region away from the surface, or raise `gauss_order`.

## 4. The resistance matrix

**What it does.** The power dissipated in the conducting sheet is a quadratic form in $\psi$:

$$P = \psi^{T} R\,\psi,\qquad R_{ij}\ \propto\ \frac{\rho}{t}\int_S \mathbf J_i\cdot\mathbf J_j\,dA ,$$

where $\rho$ is `specific_resistivity_conductor` (Ohm·m) and $t$ is `conductor_thickness` (m). $R$ has non-zero entries only for
neighbouring nodes, which share triangles.

**What goes wrong.** `conductor_thickness` here is the thickness of the *sheet* used in the optimisation. It scales $R$,
and therefore acts together with the regularisation factor. It is not the cross-section of the final wire (see section 9).

## 5. The stream function solve

**What it does.** The solver finds $\psi$ so that the field $S\psi$ matches the target $\mathbf b$, while keeping the
dissipation small. Each boundary of the mesh carries a single free value, shared by all of its nodes.

**Tikhonov method (`sf_opt_method = 'tikhonov'`, the default).** As implemented, the solution is

$$\psi = \left(S^{T}S + \tilde\lambda^{2}\,R^{T}R\right)^{+} S^{T}\,b,\qquad
\tilde\lambda = \lambda\,\frac{n_\text{targets}}{n_\text{nodes}},$$

with $\lambda$ = `tikhonov_reg_factor` and $n_\text{nodes}$ the number of free values after the boundaries are merged. The
scaling by $n_\text{targets}/n_\text{nodes}$ means that the same factor acts differently on a mesh with more nodes, so a
value that worked for one design is a starting point, not a recipe, for another.

Note two details of the implementation:
- The penalty is $\|R\psi\|^{2}$, not the power $\psi^{T}R\psi$. It is a regulariser of the same kind, but the physical meaning
  of $\lambda$ is not "watts per tesla".
- The pseudo-inverse (`pinv`) returns the minimum-norm solution. The overall constant of $\psi$ does not change the current,
  so it is arbitrary, which is harmless.

**Iterative method (any other `sf_opt_method`).** The Tikhonov solution is used as the starting point, and then
`scipy.optimize.minimize` (`minimize_method`, default `SLSQP`) minimises

$$\|S\psi-b\|^{2} + \tilde\lambda\,\psi^{T}R\,\psi,\qquad |\psi_i|\le 2\max\psi^{\text{Tikhonov}} .$$

Here $\lambda$ enters linearly, and the penalty is the power. **The same `tikhonov_reg_factor` therefore does not give
equivalent designs with the two methods.** The iterative method stops at `maxiter`
(default 100 in the options), whether or not it has converged, and the result then depends on the numerical libraries
as well as on the code. Use it only if you need its bounds.

**Choosing $\lambda$.** Sweep it in factors of 10 and plot the field error against the resistance. A small factor favours
the field, and gives a stream function with large values that vary quickly, so it needs many levels and gives tight wire
spacing. A large factor gives a smooth, low-power coil with low efficiency. A good value is usually near the knee of that curve.

## 6. Contour levels

**What it does.** The stream function is sliced into equal steps. With $L$ = `levels` and $f$ = `pot_offset_factor`
(default 0.5), the step is

$$\Delta\psi = \frac{\psi_\max-\psi_\min}{L-1+2f},\qquad \psi_k = \psi_\min + (f+k)\,\Delta\psi,\quad k=0\dots L-1 .$$

**$\Delta\psi$ is the current in each wire**, in amperes, for the field as scaled by the solve. Doubling the number of
levels halves the current per wire. The field errors are calculated per unit current, so they do not depend on this scaling.

**`level_set_method`.**
- `primary`: $\Delta\psi$ comes from the part with the largest $\psi$ range, and the other parts use the same step, so every
  wire carries the same current, as it must if the parts are driven in series.
- `combined`: one step from the combined range of all parts.
- `independent`: each part has its own step, and therefore its own current per wire.

**`pot_offset_factor`.** It positions the first and last level inside the range. With $f=\tfrac12$ the levels sit at the
centres of the bins. A small $f$ places a level close to an extremum of $\psi$, which makes a tiny loop around the peak.
That loop adds a wire turn that contributes almost nothing and is hard to make.

**What goes wrong.** Too few levels is the largest source of error that you can fix by yourself: the discretisation
replaces the smooth current with a staircase. As a guide, see section 11. Many levels give wires that are closer together
than you can manufacture, and a longer, more resistive wire.

## 7. Contour lines

**What it does.** The contour of each level is found on the flattened mesh. Because $\psi$ is linear in each triangle, the
contour is a **straight segment inside every triangle**. Where it crosses a triangle edge, the crossing point is interpolated
linearly, and the segments are chained into loops. Each loop is then mapped back onto the 3D surface by locating its points
in the triangles and interpolating.

**Closing.** A traced contour does not return to its start point exactly. A loop is closed by appending its first point, unless
its end already coincides with its start, within $10^{-9}$ of the size of the loop. The closing segment lies inside a single
triangle, and is therefore exact. (Earlier versions made this decision by exact floating-point comparison, which left some loops
open. See [Known Issues and Bug Fixes](./bug_fixes.md).)

**Loop significance.** The field of each loop is calculated by the Biot–Savart law, with the wire divided into segments carrying the
current $\Delta\psi$. The *significance* of a loop is

$$\text{significance}_n = 100\ \frac{\max_k |B_{z,n}(\mathbf r_k)|}{\overline{|B_{z}^{\text{all loops}}|}}\quad[\%] .$$

Loops below `min_loop_significance` (default 1) are removed. A loop smaller than `interconnection_cut_width` is also removed,
because the cut that joins it to its neighbour would remove the whole loop.

**Parameters.** `smooth_factor` (default 1, no smoothing) replaces each point with a moving average of this many points.
Smoothing makes the wire easier to make, but it moves the contour off the exact level, and costs field accuracy.

**What goes wrong.** The decisions in this step and in the following ones (which loops exist, which are closed or nested)
are *discrete*. A change of the stream function that is far below the accuracy of the solve (a rounding difference of $10^{-15}$ in the matrix grew to about $5	imes10^{-11}$ in
one Tikhonov solve) can change which triangle a contour crosses, and with it a loop. If a design appears to change between two
machines or library versions, compare the loop count first.

## 8. Topology, cuts and interconnection

This part turns separate loops into a single wire per coil part.

**Grouping.** For every pair of loops, a point-in-polygon test (in $(u,v)$) decides whether one encloses the other. Loops that
enclose each other form a *topological group*, a stack of nested rings. Loops that sit side by side at the same depth (they have
the same enclosing loops but do not enclose each other) belong to different groups. The group centre is
derived from the centres of its loops.

The cost of the inclusion test grows with (number of loops)² × (points per loop), which is noticeable in designs with hundreds of loops.

**Cuts within a group.** To join the rings of a group into a spiral, each ring is cut, and its ends are joined to the
adjacent ring. The cut position is where a *cut plane* through the group centre meets the ring. This gives two points per ring, a "high" and a
"low" one. A *sphere* of diameter `interconnection_cut_width` (default 0.01 m) around the cut point is removed from the loop, and the
two ends are connected to the neighbour.

- `cut_plane_definition`: `nearest` (default) orients the plane towards the closest contour point to the centre, and `B0` aligns it
  with `b_0_direction`. Cutting along the $B_0$ direction is intended to reduce the field distortion that the cuts cause.
- `force_cut_selection`: `high` or `low` for all groups, or one entry per group.

**Joining the groups.** The spirals of different groups are joined by further cuts, using the nesting depth, until
there is one continuous path per coil part with a start and an end. Return paths bring the wire back from the
inside of a spiral, and they cross other loops.

**Shifting the return paths.** Wherever a return path crosses another turn, the crossing points are lifted along the surface normal
by `normal_shift_length` (default 0.001 m), and the shape of the lift is smoothed by `normal_shift_smooth_factors`. This
avoids intersections in 3D. It needs room to work: choose `normal_shift_length` of the order of the wire thickness or more.

**What goes wrong.**
- If the cut width is large compared with the loop spacing, the cut sphere can reach the neighbouring loop, and the connections can be
  badly placed. Keep the cut width *less than the spacing between wires*, and far less than the size of the smallest loop.
- The field of the final wire is slightly different from the field of the loops. The cut and the return paths add current paths that were not in the
  stream function. This is the `layout` error in section 10.

## 9. Sweep and outputs

The wire path is swept into a volumetric conductor, along the surface, with the cross-section given by
`conductor_cross_section_width` and `conductor_cross_section_height`, or by `cross_sectional_points`. The resistance of the wire is

$$R = \rho\,\frac{L}{A},$$

with $L$ the length of the path and $A$ the cross-sectional area. The result can be exported as an `.stl` file, a Gerber file
(planar parts) or a plate with a groove. See [Data exporters](./configuration.md#data-exporters).

## 10. Evaluating the result

pyCoilGen reports the field in three stages, each against the target and against the field of the stream function:

| Field | Meaning |
|---|---|
| stream function | The continuous current, before any discretisation. The best the solve can do. |
| unconnected contours | The separate loops, each carrying $\Delta\psi$. The error here is due to the discretisation (levels, mesh). |
| layout | The final connected wire path. The error added here is due to the cuts and return paths. |

Each is given as a mean and maximum relative error ([SolutionErrors](./results.md#solutionerrors)). Read them in order. If
the *unconnected contours* are far from the *stream function*, the problem is the number of levels or the mesh. If the *layout* is
far from the *unconnected contours*, the problem is the interconnection (cut width, cut plane, return paths).

A weakness of the "unconnected vs stream function" error is that it is dominated by the discretisation of coarse designs, and
it can change by a large factor between neighbouring level counts without any real change in quality. For a gradient coil, prefer the
[independent verification](./results.md#independent-field-verification) of the final wire path: the gradient deviation, in
percent, with a typical tolerance of 5%.

## 11. Choosing values: evidence from one design

The following was measured on one design: two square bi-planar plates, 0.25 m wide, an $x$-gradient, a target sphere of radius 5 cm,
Tikhonov factor 10, and a cut width of 0.01 m. It shows trends, not guarantees. The time is for a single process
on a desktop computer.

| Nodes per side | Levels | Time | Layout error vs target | Gradient max. deviation |
|---|---|---|---|---|
| 12 | 16 | 5 s | 44% | 15.3% |
| 12 | 32 | 8 s | 40% | 15.4% |
| 16 | 32 | 13 s | 25% | 11.2% |
| 20 | 24 | 16 s | 38% | 7.3% |
| 20 | 32 | 20 s | 28% | 6.2% |
| 24 | 16 | 15–19 s | 48% | 8.6% |
| 32 | 16 | 31–39 s | 45% | 8.1% |

What the data suggest:

1. **The mesh sets the gradient accuracy, and levels do not make up for a coarse mesh.** At 12 nodes per side the deviation stays near 15% from 16 to 32
   levels. It only falls below 10% at 20 nodes or more.
2. **There is a minimum number of levels.** At 8 levels, every mesh gave a layout error of 130–180% of the target, so the discretisation is
   too coarse to represent the field at all. Above about 16–24 levels the gain was small compared with the effect of the mesh.
3. **More levels are not always better.** The *unconnected contours vs stream function* error was not monotonic in the number of levels
   (it varied by about a factor of 10 between 56 and 112 levels on one coarse design), so use it only to compare designs of the same resolution.
4. **Time grows quickly with nodes.** The step that grows fastest is the sensitivity matrix (section 3), so refine only as far as the
   accuracy needs.

A practical order of work:

1. Start with a coarse mesh and the default parameters, and check that the result has the right *shape*.
2. Refine the mesh until the gradient deviation stops improving.
3. Choose the number of levels from the wire spacing you can manufacture (section 6), and no higher.
4. Sweep the Tikhonov factor to set the trade-off between field accuracy and resistance (section 5).
5. Set the cut width and normal shift last, to suit the wire (section 8).

## 12. Symptoms and likely causes

| Symptom | Likely cause |
|---|---|
| Gradient deviation above 5% with a smooth stream function | The mesh is too coarse (section 1), or the target region is too large for the surface. |
| Large *unconnected contours* error, small *stream function* error | Too few levels, or too coarse a mesh. |
| Large *layout* error, small *unconnected contours* error | The cuts or return paths. Reduce the cut width, or use `cut_plane_definition = 'B0'`. |
| A stream function with huge values, and very many loops | The Tikhonov factor is too small. |
| Low efficiency (T/m/A) and little current | The Tikhonov factor is too large. |
| Tiny loops around the extrema | `pot_offset_factor` is small, or the levels are too fine near the peaks. |
| Results differ between machines or library versions | A discrete decision (section 7) flipped. Compare the loop counts and the closing of loops. |
| The design stops with an error at the cut | The cut width is large relative to the loop size or spacing (section 8). |
