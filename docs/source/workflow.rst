Updated Workflow for Planar Coil Design
======================================

The coil design pipeline has been refactored into a more modular and Python-native workflow.

Pipeline Overview
-----------------

1. Define geometry and parameters
2. Solve stream function optimization
3. Extract wire paths
4. Generate gradient former geometry
5. Simulate magnetic fields
6. Visualize results

Example Workflow
----------------

.. code-block:: python

    from pyCoilGen.sub_functions.stream_function_optimization import optimize_stream_function
    from pyCoilGen.sub_functions.extract_wire_paths import extract_wire_paths
    from pyCoilGen.sub_functions.gradient_former import generate_gradient_former
    from pyCoilGen.sub_functions.simulate_gradient_coil import simulate_gradient_coil

    # Step 1: Optimize stream function
    sf = optimize_stream_function(...)

    # Step 2: Extract wire paths
    loops = extract_wire_paths(sf, ...)

    # Step 3: Generate printable geometry
    generate_gradient_former(loops, ...)

    # Step 4: Simulate field
    field = simulate_gradient_coil(loops, ...)