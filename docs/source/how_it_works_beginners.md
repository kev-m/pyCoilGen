# How It Works: An Introduction

This page explains, in plain terms, what **pyCoilGen** does and why it works. It needs no knowledge of the
software and only a little physics. If you want the details of each step, and advice on choosing the
[configuration](./configuration.md) values, you will find them in [How It Works: The Algorithm in Detail](./how_it_works_experts.md).

## The problem

Many instruments need a magnetic field that is shaped on purpose. An MRI scanner, for example, needs *gradient
coils*: coils that produce a field that gets steadily stronger as you move along one direction, such as
"the field along *z* grows in proportion to *y*". Other coils cancel small unwanted variations in a field ("shim" coils).

A coil is just a wire carrying a current, and a current makes a magnetic field. The designer's question is the
reverse of the usual one: **I know the field I want. Where should the wire go?**

That is hard, because a wire can take almost any path. A few turns in the wrong places make a poor field, and a
long tangle of wire is expensive, heats up, and is hard to build. We need a way to search for good layouts in a
systematic fashion. This is what **pyCoilGen** does.

## The key idea: current as a landscape

Imagine the surface that will carry the wire (a cylinder, or two flat plates) and, instead of thinking about wires,
imagine a continuous layer of current flowing over it, like water over a sheet.

There is a neat way to describe such a flow: draw a *landscape* over the surface, with hills and valleys, and let
the current flow **along the contour lines** of that landscape, as on a walker's map. The height of the
landscape is called the **stream function**.

```{figure} figures/plot_example_ygradient_2D.png
:scale: 60 %
:align: center
:alt: A colour map of a stream function on an unrolled cylinder, with coloured contour lines around two yellow hills and two dark valleys.

A stream function, drawn flat (the cylinder has been unrolled). Yellow is a hill, dark blue is a valley. The
coloured lines are the contours, and the current flows along them.
```

This landscape has three useful properties:

- **Contours never cross**, so the current never has to jump from one path to another.
- **Current flows round the hills in one direction and round the valleys in the other.** That is how a
  gradient coil gets a field that is positive on one side and negative on the other.
- **The wire layout comes for free.** Cut the landscape into equal height steps and draw a contour line at each
  step: each line is one wire, and each wire carries the same current (the height of the step).

So the design problem becomes: *find the right landscape.* Once we have it, the wires follow.

## Step 1: Describe the surface

The computer cannot work with a smooth surface, so the surface is covered with a *mesh* of small triangles. You
can describe the support surface yourself, with an `.stl` file, or use one of the built-in shapes
(a cylinder, a flat plate, or two plates facing each other). The stream function is then described by one number per
corner of every triangle: its height at that point.

The mesh is also flattened out into a plane (the "parameterisation") so that the contour lines can be found in a
simple flat picture and then wrapped back onto the real surface.

You also tell the program the **target**: the points in space where the field should be right (a small sphere in
the middle of the coil, for example, 5 cm in radius), and the field value wanted there.

All lengths in **pyCoilGen** are in metres, fields are in tesla (T), and currents are in amperes (A). A gradient
is therefore measured in tesla per metre (T/m), and the efficiency of a gradient coil in tesla per metre per ampere (T/m/A).

## Step 2: Find the landscape

Magnetic fields add up. If you double the current everywhere, you double the field everywhere. This makes the
calculation tractable. The program works out, once, how much field each corner of the mesh contributes at each
target point *per unit of height*. This table is called the **sensitivity matrix** and is the largest
calculation in the whole process.

With the table in hand, finding the best landscape is a standard mathematical problem: choose the heights so that
the total field at the target points is as close as possible to the field you asked for.

There is a catch. Asking for a perfect field gives landscapes that are impossible to build: they are
extremely wiggly, with huge currents. So the program also **penalises** the power the coil would waste as heat.
The balance between a perfect field and a practical coil is set by a *regularisation* factor
(the Tikhonov factor in the configuration). A small factor favours accuracy, and a large factor favours a coil that
is cheaper to run, at the cost of a less perfect field.

## Step 3: Slice the landscape into wires

Now the program draws the contour lines. You choose how many **levels** to use. More levels mean more wires, each
carrying a smaller current, and a smoother approximation of the ideal continuous flow. Fewer levels mean fewer
wires, each carrying more current.

The result is a set of closed loops, which you can see in the picture above.

## Step 4: Join the loops into one wire

A set of separate loops is not a coil yet. A real coil is usually a single wire with two ends, which you connect
to the power supply, so the program has to join the loops together.

First it sorts the loops into **groups**: loops that sit inside one another, like the rings of a tree. Each group
is turned into a spiral by making a small *cut* in every ring and reconnecting the ends to the next ring. Then
the program joins the groups to each other, and adds **return paths** to bring the wire back from the middle of a
spiral to its edge. The final result is a single, continuous wire path for each part of the coil, with no
crossings.

The program also checks that wires from neighbouring groups do not come too close to each other, and moves them
apart where they do.

## Step 5: Check, and make

Before building anything, you want to know whether the design is any good. **pyCoilGen** recalculates the field of
the final wire path, in an independent calculation, and reports how closely it matches the target. This matters,
because turning a smooth landscape into a finite number of wires is an approximation. It also reports the
*efficiency* (the field gained per ampere) and how resistive the coil is.

If you are happy, the wire path can be exported for manufacture. This can be a 3D model of the wire swept along
the surface (an `.stl` file), a printed circuit board layout (a *Gerber* file), or a plate with a groove to hold the wire.

```{figure} figures/mesh_example_gradient_3D.png
:scale: 60 %
:align: center
:alt: A green cylinder with a copper-coloured wire wound over it in looping spirals.

The finished wire path, swept into a 3D conductor on the surface of a cylinder.
```

## The whole process

```{figure} figures/flow_chart_algorithm_revised.png
:scale: 50 %
:align: center
:alt: A flow chart with the steps: specify the mesh and find the stream function, parameterise the mesh, discretise and build contour lines, find the contour topology, interconnect the contours, manage overlaps, evaluate the result, and manufacture.

The steps of **pyCoilGen**, from the mesh to a coil that can be manufactured. The figure is from the
[CoilGen paper](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294).
```

## What you can control

You do not need to set most settings, as the defaults give a reasonable design. The choices that matter most are:

| You choose | It affects | In the configuration |
|---|---|---|
| The surface and its size | How much room the wires have, and how good the field can be | [Mesh creation](./configuration.md#mesh-creation) |
| The mesh detail | How accurate the calculation is, and how long it takes | [Subdividing the mesh](./configuration.md#coil-meshes) |
| The target field and region | What the coil is designed to do | [Target field](./configuration.md#target-field) |
| The regularisation factor | Field accuracy against power use | [Stream function](./configuration.md#stream-function) |
| The number of levels | The number of wires, and the smoothness of the field | [Contour parameters](./configuration.md#contour-parameters) |

## Things to keep in mind

- **A coarse mesh limits the accuracy.** The mesh is the foundation of everything else. If it is too coarse,
  no amount of tuning will make the field good.
- **A design is a compromise.** Accuracy, efficiency, resistance and the number of wires all pull in different
  directions. There is no single best answer, only the right balance for your use.
- **Always check the result.** The field error of the final wire path, not of the smooth landscape, is the figure
  that counts.

## Where next

- Follow the [Quick Start](./quick_start.md) to design your first coil.
- Look up unfamiliar words in the [Glossary](./glossary.md).
- Read the original paper: [CoilGen: Open-source MR coil layout generator](https://onlinelibrary.wiley.com/doi/10.1002/mrm.29294).
