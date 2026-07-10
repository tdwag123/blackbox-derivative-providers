from __future__ import annotations
import numpy as np
import math

"""
Finite element (FE) residual for 1D steady nonlinear heat diffusion:

    R_i = - int N_i' * phi(T_h', T_h, x; m) dx  -  int N_i * r(T_h, x; m) dx  +  (R_i)_Gamma_q     (*)

where phi is the (possibly nonlinear) constitutive flux law, r is the volumetric
heat source, and (R_i)_Gamma_q is the Neumann boundary contribution (zero if the
boundary condition at that end is Dirichlet instead).

This module:
    (1) Uses Gauss-Legendre quadrature to evaluate the integrals in (*) exactly
        (for the polynomial-degree problems considered here).
    (2) Assembles the residual vector and Newton tangent (Jacobian) matrix
        element by element into tridiagonal storage.
    (3) Solves the resulting nonlinear system for nodal temperatures via
        Newton's method with an optional backtracking line search.
"""

# -------------------------------------------------------------------------------------------------
# GAUSS-LEGENDRE QUADRATURE
#
# An n-point Gauss-Legendre rule integrates any polynomial of degree < 2n exactly.
# Here we build a 2-point rule (exact through degree 3) and extend it to higher
# dimensions via a tensor-product construction.
# -------------------------------------------------------------------------------------------------
def get_gl2_values(dim: int):
    """
    Return quadrature nodes and weights for 2-point Gauss-Legendre quadrature in
    `dim` dimensions, built as a tensor product of the 1D rule.

    On the reference interval [-1, 1], the 1D rule has:
        nodes:   x1 = -1/sqrt(3),  x2 = +1/sqrt(3)
        weights: w1 = 1.0,         w2 = 1.0
    """

    if dim < 1:
        raise ValueError("dim must be positive")

    # 1D nodes/weights on the reference interval [-1, 1]
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(2)
    nodes_1d = nodes_1d.reshape(-1, 1)

    quad_pts = nodes_1d  # column vector of 1D nodes
    weights = weights_1d

    if dim == 1:
        return quad_pts, weights

    # Tensor-product extension to higher dimensions: combine the current set of
    # quadrature points with another copy of the 1D rule, one dimension at a time.
    ones_col = np.ones((2, 1))
    for k in range(2, dim + 1):
        prev_quad_pts = quad_pts
        prev_weights = weights

        rep_block = np.ones((2 ** (k - 1), 1))

        quad_pts = np.hstack((np.kron(nodes_1d, rep_block), np.kron(ones_col, prev_quad_pts)))
        weights = np.kron(prev_weights, weights_1d)

    return quad_pts, weights


def gl2_quadrature_integration(f, a, b, dim: int):
    """
    Integrate f(x) over the box [a_1, b_1] x [a_2, b_2] x ... x [a_dim, b_dim]
    using 2-point Gauss-Legendre quadrature in each coordinate direction. Internally
    this maps the physical box to the reference domain [-1, 1]^dim via an affine
    change of variables.
    """

    quad_pts, weights = get_gl2_values(dim)

    # Scalars are the natural calling convention for a 1D interval.  Promote
    # them to length-one arrays internally so the tensor-product code below can
    # use the same representation in every dimension.
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))

    if a.shape != (dim,):
        raise ValueError("a should have length dim")
    if b.shape != (dim,):
        raise ValueError("b should have length dim")
    if np.any(b <= a):
        raise ValueError("each entry of b must > a")

    # Affine map from reference coordinate xi to physical coordinate x:
    #     x(xi) = midpoint + jacobian * xi
    jacobian = (b - a) / 2.0
    midpoint = (b + a) / 2.0

    integral = None
    for node, weight in zip(quad_pts, weights):
        physical_pt = midpoint + jacobian * node
        # Existing 1D FE callbacks operate on scalar x coordinates; callbacks
        # for higher-dimensional integration receive a coordinate vector.
        f_arg = float(physical_pt[0]) if dim == 1 else physical_pt
        f_val = np.asarray(f(f_arg), dtype=float)

        if integral is None:
            integral = weight * f_val
        else:
            integral = integral + weight * f_val

    result = np.prod(jacobian) * integral

    if result.shape == ():
        return float(result)

    return result


# -------------------------------------------------------------------------------------------------
# GLOBAL TRIDIAGONAL ASSEMBLY
#
# For linear 1D elements, each element only couples its two endpoint nodes, so the
# assembled global tangent matrix is tridiagonal. This helper scatters a 2x2 element
# tangent K_elem into the global (lower, diag, upper) tridiagonal storage.
# -------------------------------------------------------------------------------------------------
def assemble_element_tangent(sub_diag, main_diag, super_diag, elem_idx: int, K_elem):
    main_diag[elem_idx] += K_elem[0, 0]
    super_diag[elem_idx] += K_elem[0, 1]
    sub_diag[elem_idx] += K_elem[1, 0]
    main_diag[elem_idx + 1] += K_elem[1, 1]


# -------------------------------------------------------------------------------------------------
# THOMAS ALGORITHM
#
# Fast direct solver for a tridiagonal linear system A*x = rhs, exploiting the
# banded structure produced by 1D linear elements.
# -------------------------------------------------------------------------------------------------
def thomas_solve(sub_diag, main_diag, super_diag, rhs):
    """Solve a tridiagonal linear system using the Thomas algorithm (forward sweep + back-substitution)."""

    sub_diag = np.asarray(sub_diag, dtype=float).copy()
    main_diag = np.asarray(main_diag, dtype=float).copy()
    super_diag = np.asarray(super_diag, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()

    n = len(main_diag)

    if n == 0:
        return np.array([], dtype=float)

    if abs(main_diag[0]) < 1.0e-14:
        raise ZeroDivisionError("Zero pivot at row 0.")

    if n == 1:
        return np.array([rhs[0] / main_diag[0]], dtype=float)

    # Modified coefficients from the forward elimination sweep
    modified_super = np.zeros(n, dtype=float)
    modified_rhs = np.zeros(n, dtype=float)
    solution = np.zeros(n, dtype=float)

    modified_super[0] = super_diag[0] / main_diag[0]
    modified_rhs[0] = rhs[0] / main_diag[0]

    for i in range(1, n):
        pivot = main_diag[i] - sub_diag[i - 1] * modified_super[i - 1]

        if abs(pivot) < 1.0e-14:
            raise ZeroDivisionError(f"Zero pivot at row {i}.")

        modified_super[i] = super_diag[i] / pivot if i < n - 1 else 0.0
        modified_rhs[i] = (rhs[i] - sub_diag[i - 1] * modified_rhs[i - 1]) / pivot

    # Back-substitution
    solution[-1] = modified_rhs[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = modified_rhs[i] - modified_super[i] * solution[i + 1]

    return solution


def tridiag_block(sub_diag, main_diag, super_diag, start, end):
    """
    Extract the tridiagonal sub-block corresponding to the *free* (non-Dirichlet)
    degrees of freedom, i.e. the rows/columns [start, end) that Newton's method
    actually needs to solve for. Prescribed Dirichlet values (T(0)=TL, T(L)=TR)
    are not part of the unknown vector.
    """
    main_diag_free = main_diag[start:end].copy()
    sub_diag_free = sub_diag[start:end - 1].copy()
    super_diag_free = super_diag[start:end - 1].copy()
    return sub_diag_free, main_diag_free, super_diag_free


# -------------------------------------------------------------------------------------------------
# ELEMENT-BY-ELEMENT RESIDUAL AND TANGENT ASSEMBLY
#
# Governing PDE:
#   q'(x) = r(T, x),      q = phi(s, T, x; m),      s = T'(x)
#
# Weak residual:
#   R_i = - int N_i' * phi(T_h', T_h, x) dx  -  int N_i * r(T_h, x) dx  +  Neumann boundary terms
#
# Newton tangent (Jacobian):
#   J_ij = dR_i/dU_j = - int N_i' * [phi_s * N_j' + phi_T * N_j] dx  -  int N_i * r_T * N_j dx
# -------------------------------------------------------------------------------------------------

def resid_and_tan(node_coords, T_nodal, flux_law, source_fn, dsource_dT=None,
                   neumann_flux_left=None, neumann_flux_right=None):
    """
    Assemble the global residual vector and tridiagonal Newton tangent for the
    current nodal temperature guess.

    Inputs:
        node_coords: 1D array of mesh node positions
        T_nodal: current nodal temperature guess (the Newton unknown)
        flux_law: constitutive flux map phi(T', T, x; m); callable returning
                  (q, dphi_ds, dphi_dT)
        source_fn: heat source r(T, x); callable returning the source value
        dsource_dT: dr/dT; pass None if the source does not depend on T
        neumann_flux_left / neumann_flux_right: prescribed Neumann flux at the
                  left/right boundary; pass None if that boundary is Dirichlet instead

    Note: this function assumes no Dirichlet boundary handling on its own -- the
    caller (NM) is responsible for holding Dirichlet nodes fixed.
    """
    if dsource_dT is None:
        dsource_dT = lambda T, xg: 0.0

    node_coords = np.asarray(node_coords, dtype=float)
    T_nodal = np.asarray(T_nodal, dtype=float)

    n_nodes = len(node_coords)

    R_global = np.zeros(n_nodes)

    # Tridiagonal storage for the global Newton tangent
    sub_diag = np.zeros(n_nodes - 1)
    main_diag = np.zeros(n_nodes)
    super_diag = np.zeros(n_nodes - 1)

    # Linear (2-node) elements give piecewise-constant shape-function derivatives,
    # so each element's local quantities are computed once per element below.
    for e in range(n_nodes - 1):
        x_left = node_coords[e]
        x_right = node_coords[e + 1]
        elem_length = x_right - x_left

        if elem_length <= 0.0:
            raise ValueError("mesh nodes need to be strictly increasing")

        T_elem = np.array([T_nodal[e], T_nodal[e + 1]])  # local nodal temperatures

        dN_dx = np.array([-1.0 / elem_length, 1.0 / elem_length])  # constant on this element

        def shape_fns(xg):
            """Linear hat functions evaluated at xg, local to this element."""
            return np.array([(x_right - xg) / elem_length, (xg - x_left) / elem_length])

        def residual_integrand(xg):
            """Element contribution to the residual integrand at quadrature point xg."""
            N = shape_fns(xg)

            T_g = N @ T_elem            # temperature at the quadrature point
            Tprime_g = dN_dx @ T_elem   # temperature gradient (constant for linear elements)

            q_g, dphi_ds, dphi_dT = flux_law(Tprime_g, T_g, xg)
            r_g = source_fn(T_g, xg)

            return -dN_dx * q_g - N * r_g

        def tangent_integrand(xg):
            """Element contribution (2x2) to the Newton tangent at quadrature point xg."""
            N = shape_fns(xg)

            T_g = N @ T_elem
            Tprime_g = dN_dx @ T_elem

            q_g, dphi_ds, dphi_dT = flux_law(Tprime_g, T_g, xg)
            drdt_g = dsource_dT(T_g, xg)

            # Flux contribution: -N_i' * [phi_s * N_j' + phi_T * N_j]
            K_flux = -np.outer(dN_dx, dphi_ds * dN_dx + dphi_dT * N)

            # Source contribution: -N_i * r_T * N_j
            K_source = -drdt_g * np.outer(N, N)

            return K_flux + K_source

        # 2-point Gauss-Legendre quadrature over this element
        R_elem = gl2_quadrature_integration(residual_integrand, x_left, x_right, 1)
        K_elem = gl2_quadrature_integration(tangent_integrand, x_left, x_right, 1)

        # Scatter local (element) contributions into the global tridiagonal system
        R_global[e:e + 2] += R_elem
        assemble_element_tangent(sub_diag, main_diag, super_diag, e, K_elem)

    # Neumann boundary contributions (prescribed outward flux q . n):
    #   at x=0, outward normal n=-1  ->  contribution is +neumann_flux_left
    #   at x=L, outward normal n=+1  ->  contribution is +neumann_flux_right
    # Skip entirely at any boundary that is Dirichlet instead.
    if neumann_flux_left is not None:
        R_global[0] += neumann_flux_left

    if neumann_flux_right is not None:
        R_global[-1] += neumann_flux_right

    return R_global, sub_diag, main_diag, super_diag


# -------------------------------------------------------------------------------------------------
# NEWTON SOLVER FOR THE NODAL TEMPERATURE VECTOR
# -------------------------------------------------------------------------------------------------
def NM(node_coords, flux_law, source_fn, T_dirichlet_left, T_dirichlet_right,
       dsource_dT=None, T_initial=None, neumann_flux_left=None, neumann_flux_right=None,
       tol=1e-10, maxiter=30, verbose=True, line_search=True):
    """
    Solve the nonlinear steady-state 1D diffusion problem for nodal temperatures
    using Newton's method (with optional backtracking line search).

    Boundary condition convention (per node, left and right independently):
        - Dirichlet: pass the prescribed temperature (T_dirichlet_left / _right);
          leave the matching Neumann flux argument as None.
        - Neumann: pass the prescribed flux (neumann_flux_left / _right); leave
          the matching Dirichlet temperature argument as None.
        - Robin-type conditions can be built by combining both types across the domain.

    Exactly one boundary condition type must be supplied at each end, and at
    least one end must be Dirichlet (otherwise the 1D problem is singular).
    """
    node_coords = np.asarray(node_coords, dtype=float)
    n_nodes = len(node_coords)

    left_is_dirichlet = T_dirichlet_left is not None
    right_is_dirichlet = T_dirichlet_right is not None

    if left_is_dirichlet and neumann_flux_left is not None:
        raise ValueError("Left BC cannot be Dirichlet and Neumann")
    if right_is_dirichlet and neumann_flux_right is not None:
        raise ValueError("Right BC cannot be Dirichlet and Neumann")

    if not left_is_dirichlet and not right_is_dirichlet:
        raise ValueError(
            "1D problem is singular: at least one Dirichlet BC is required "
            "(or add a Robin-type penalty)"
        )

    # Build an initial guess if none was provided. A valid guess can always be
    # constructed from the supplied boundary conditions.
    if T_initial is None:
        if left_is_dirichlet and right_is_dirichlet:
            T_nodal = np.linspace(T_dirichlet_left, T_dirichlet_right, n_nodes)
        elif left_is_dirichlet:
            T_nodal = np.full(n_nodes, T_dirichlet_left, dtype=float)
        elif right_is_dirichlet:
            T_nodal = np.full(n_nodes, T_dirichlet_right, dtype=float)
        else:
            T_nodal = np.zeros(n_nodes, dtype=float)
    else:
        T_nodal = np.asarray(T_initial, dtype=float).copy()

    # Enforce prescribed Dirichlet values on the initial guess
    if left_is_dirichlet:
        T_nodal[0] = T_dirichlet_left
    if right_is_dirichlet:
        T_nodal[-1] = T_dirichlet_right

    # Dirichlet nodes are excluded from the Newton unknowns; [start, end) marks
    # the range of free (solved-for) degrees of freedom.
    start = 1 if left_is_dirichlet else 0
    end = n_nodes - 1 if right_is_dirichlet else n_nodes

    residual_norm_history = []
    newton_iterations = 0

    for iteration in range(maxiter):

        R_global, sub_diag, main_diag, super_diag = resid_and_tan(
            node_coords, T_nodal, flux_law, source_fn,
            dsource_dT=dsource_dT,
            neumann_flux_left=neumann_flux_left,
            neumann_flux_right=neumann_flux_right,
        )

        # Residual restricted to the free (non-Dirichlet) degrees of freedom
        R_free = R_global[start:end]
        residual_norm = np.linalg.norm(R_free, ord=2)
        residual_norm_history.append(residual_norm)

        if verbose:
            print(f"Newton {iteration:2d}: ||R_free||_2 = {residual_norm:.3e}")

        if residual_norm < tol:
            return T_nodal, residual_norm_history, newton_iterations

        sub_diag_free, main_diag_free, super_diag_free = tridiag_block(
            sub_diag, main_diag, super_diag, start, end
        )

        # Newton step: solve tangent * dT_free = -R_free
        dT_free = thomas_solve(sub_diag_free, main_diag_free, super_diag_free, -R_free)

        alpha = 1.0
        if line_search:
            step_accepted = False
            while alpha > 1.0e-12:
                T_trial = T_nodal.copy()
                T_trial[start:end] += alpha * dT_free

                if left_is_dirichlet:
                    T_trial[0] = T_dirichlet_left
                if right_is_dirichlet:
                    T_trial[-1] = T_dirichlet_right

                R_trial, _, _, _ = resid_and_tan(
                    node_coords, T_trial, flux_law, source_fn,
                    dsource_dT=dsource_dT,
                    neumann_flux_left=neumann_flux_left,
                    neumann_flux_right=neumann_flux_right,
                )

                trial_norm = np.linalg.norm(R_trial[start:end], ord=2)

                if trial_norm < residual_norm:
                    step_accepted = True
                    break

                alpha *= 0.5

            if not step_accepted:
                raise RuntimeError("line search failed to minimize residual")

        else:
            T_trial = T_nodal.copy()
            T_trial[start:end] += dT_free

            if left_is_dirichlet:
                T_trial[0] = T_dirichlet_left
            if right_is_dirichlet:
                T_trial[-1] = T_dirichlet_right

        T_nodal = T_trial
        newton_iterations += 1

    raise RuntimeError("NM didn't converge within desired number maxiter")


## ------------------------------------------------------------------------------------------------
## DEMO / SANITY-CHECK CASES
## ------------------------------------------------------------------------------------------------

x = np.linspace(0.0, 1.0, 21)  # uniform mesh on [0, 1]


def source(T, xg):
    """Constant unit heat source."""
    return 1.0


def fluxLinSanityCheck(T_prime, T, xg):
    """Linear Fourier flux with unit conductivity: q = -T'."""
    q = -T_prime
    dphi_ds = -1.0
    dphi_dT = 0.0
    return q, dphi_ds, dphi_dT


def fluxLinSanityCheck_execute():
    # Since q = -T' and q' = r = 1, this reduces to -T'' = 1 with T(0) = T(1) = 0,
    # which has the closed-form solution below -- useful for verifying the solver.

    U_exact = lambda x: 0.5 * x * (1.0 - x)

    U_sanity, log_sanity, num_iterations_sanity = NM(x, fluxLinSanityCheck, source, T_dirichlet_left=0.0, T_dirichlet_right=0.0, verbose=True)
    print("\nLinear Sanity check solution U:\n")
    for i in range(len(U_sanity)):
        print(U_sanity[i])

    print("\nError between NM U and exact U (2-norm of U_sanity - U_true):\n")
    U_true = U_exact(x)
    error_vec = U_sanity - U_true
    error = np.linalg.norm(error_vec, ord=2)
    print(error)

    print("\nNumber of Newton Iterations:\n")
    print(num_iterations_sanity)
    print("\n")


def fluxNonlinExample(T_prime, T, xg):
    """Nonlinear flux law: conductivity grows with T^2, plus a cubic gradient term."""
    alpha = 0.25
    beta = 0.05

    q = -((1.0 + alpha * T * T) + beta * T_prime * T_prime) * T_prime
    dphi_ds = -(1.0 + alpha * T * T) - 3.0 * beta * T_prime * T_prime
    dphi_dT = -2.0 * alpha * T * T_prime

    return q, dphi_ds, dphi_dT

def fluxNonlinExample_execute():
    U_test, log_test, num_iterations_test = NM(
        x, fluxNonlinExample, source, T_dirichlet_left=0.0, T_dirichlet_right=0.0, verbose=True
    )
    print("\nNonlinear solution U:\n")
    for i in range(len(U_test)):
        print(U_test[i])

    print("\nNumber of Newton Iterations:\n")
    print(num_iterations_test)

if __name__ == "__main__":
    fluxLinSanityCheck_execute()
    fluxNonlinExample_execute()
