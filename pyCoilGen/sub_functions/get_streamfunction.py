import numpy as np
from scipy.sparse.linalg import lsmr


def solve_streamfunction_with_initial_guess(
        reduced_sensitivity_matrix,
        target_field_single,
        mesh_vertices,
        gradient_axis="x",
        prior_strength=0.0,
        atol=1e-9,
        btol=1e-9,
        maxiter=1000):

    """
    Solve for streamfunction coefficients using LSMR with a physics-informed prior.

    Parameters
    ----------
    reduced_sensitivity_matrix : ndarray (M x K)
        Sensitivity matrix in reduced basis.

    target_field_single : ndarray (M,)
        Target field values.

    mesh_vertices : ndarray (N x 3)
        Coil mesh vertices.

    streamfunction_basis : ndarray (N x K)
        Basis matrix mapping coefficients -> streamfunction.

    gradient_axis : str
        'x', 'y', or 'z'.

    prior_strength : float
        Weight of the prior regularization.

    Returns
    -------
    psi : ndarray (N,)
        Streamfunction on mesh vertices.

    coeffs : ndarray (K,)
        Reduced basis coefficients.

    B_pred : ndarray (M,)
        Predicted magnetic field.
    """

    A = reduced_sensitivity_matrix
    b = target_field_single
    verts = mesh_vertices
    n_verts = verts.shape[0]
    streamfunction_basis = np.eye(n_verts)

    # -----------------------------------------
    # 1. Build physics-informed ψ prior
    # -----------------------------------------

    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]



    if gradient_axis == "x":
        psi0 = y
    elif gradient_axis == "y":
        psi0 = -x
    elif gradient_axis == "z":
        psi0 = x * y
    else:
        raise ValueError("gradient_axis must be x, y, or z")

    # normalize prior
    psi0 = psi0 - np.mean(psi0)
    psi0 = psi0 / np.max(np.abs(psi0))

    # -----------------------------------------
    # 2. Project ψ prior into reduced basis
    # -----------------------------------------

    x0, *_ = np.linalg.lstsq(streamfunction_basis, psi0, rcond=None)

    # -----------------------------------------
    # 3. Apply prior regularization
    # -----------------------------------------

    if prior_strength > 0:

        A_aug = np.vstack([
            A,
            prior_strength * np.eye(A.shape[1])
        ])

        b_aug = np.concatenate([
            b,
            prior_strength * x0
        ])

    else:

        A_aug = A
        b_aug = b

    # -----------------------------------------
    # 4. Solve with LSMR
    # -----------------------------------------

    result = lsmr(
        A_aug,
        b_aug,
        x0=x0,
        atol=atol,
        btol=btol,
        maxiter=maxiter
    )

    coeffs = result[0]

    # -----------------------------------------
    # 5. Reconstruct streamfunction
    # -----------------------------------------

    psi = streamfunction_basis @ coeffs

    # -----------------------------------------
    # 6. Compute predicted field
    # -----------------------------------------

    B_pred = A @ coeffs

    return psi, coeffs, B_pred