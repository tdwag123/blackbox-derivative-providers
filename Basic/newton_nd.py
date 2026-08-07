import itertools
import numpy as np
from scipy.interpolate import griddata
from FEM_Structure.fem import gl2_quadrature_integration

'''
ND (1,2,3) FEM solve with rectangular Neumann/Dirichlet boundaries.
'''

# =================================================================================
# shape functions & helpers

def node_point(idx, grid_vars):
    return np.array([grid_vars['coords'][d][idx[d]] for d in range(grid_vars['dim'])])


def global_id(idx, grid_vars):                # idx = (i,j) or (i,j,k)
    flat = 0
    for d in range(grid_vars["dim"]):
        flat = flat * grid_vars["n_nodes_per_axis"][d] + idx[d]
    return flat


def element_bounds_and_ids(base_idx, grid_vars): # base_idx = (i,j[,k]) = "lower" corner
    a = [grid_vars['coords'][d][base_idx[d]]     for d in range(grid_vars['dim'])]     # lower box corner
    b = [grid_vars['coords'][d][base_idx[d] + 1] for d in range(grid_vars['dim'])]     # upper box corner

    local_ids = []
    for offset in grid_vars['local_corner_offsets']:
        node_idx = tuple(base_idx[d] + int(offset[d]) for d in range(grid_vars['dim']))
        local_ids.append(global_id(node_idx, grid_vars))

    return a, b, local_ids


def shape_fns(pt, a, b, grid_vars):
    pt = np.asarray(pt, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    # 1D hat value at each corner, per axis
    N_1d = []   # N_1d[d] = [N_low(pt[d]), N_high(pt[d])]
    for d in range(grid_vars['dim']):
        h = b[d] - a[d]
        N_low  = (b[d] - pt[d]) / h
        N_high = (pt[d] - a[d]) / h
        N_1d.append([N_low, N_high])

    N = []
    for offset in grid_vars['local_corner_offsets']:
        val = np.prod([N_1d[d][offset[d]] for d in range(grid_vars['dim'])])     # tensor product
        N.append(float(val))

    return N                                    # length nen


def shape_grads(pt, a, b, grid_vars):
    pt = np.asarray(pt, dtype=float).reshape(-1)
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    N_1d  = []   # values, as above
    dN_1d = []   # derivatives: dN_1d[d] = [-1/h, +1/h]  (constant per axis)
    for d in range(grid_vars['dim']):
        h = b[d] - a[d]
        N_1d.append([ (b[d]-pt[d])/h, (pt[d]-a[d])/h ])
        dN_1d.append([ -1.0/h, 1.0/h ])

    G = []
    for offset in grid_vars['local_corner_offsets']:
        grad = np.zeros(grid_vars['dim'])
        for d_diff in range(grid_vars['dim']):
            # d/dx_{d_diff} of the tensor product: differentiate that one
            # factor, keep the others as plain shape-function values
            val = dN_1d[d_diff][offset[d_diff]]
            for d_other in range(grid_vars['dim']):
                if d_other != d_diff:
                    val *= N_1d[d_other][offset[d_other]]
            grad[d_diff] = val
        G.append(grad)
    return G                    
            
# =================================================================================

# =================================================================================
# element / global residual & tangent

def boundary_value(value, pt):
    if callable(value):
        return float(value(pt))
    return float(value)


def add_dirichlet_node(dirichlet_nodes, idx, value, grid_vars):
    val = boundary_value(value, node_point(idx, grid_vars))
    if idx in dirichlet_nodes and not np.isclose(dirichlet_nodes[idx], val):
        raise ValueError(f"conflicting Dirichlet values at node {idx}")
    dirichlet_nodes[idx] = val


def boundary_nodes_for_side(side, grid_vars):
    dim = grid_vars["dim"]
    n = grid_vars["n_nodes_per_axis"]

    if dim == 1:
        nx = n[0]
        nodes = {
            "xmin": [(0,)],
            "xmax": [(nx - 1,)],
        }
    elif dim == 2:
        nx, ny = n
        nodes = {
            "xmin": [(0, j) for j in range(ny)],
            "xmax": [(nx - 1, j) for j in range(ny)],
            "ymin": [(i, 0) for i in range(nx)],
            "ymax": [(i, ny - 1) for i in range(nx)],
        }
    elif dim == 3:
        nx, ny, nz = n
        nodes = {
            "xmin": [(0, j, k) for j in range(ny) for k in range(nz)],
            "xmax": [(nx - 1, j, k) for j in range(ny) for k in range(nz)],
            "ymin": [(i, 0, k) for i in range(nx) for k in range(nz)],
            "ymax": [(i, ny - 1, k) for i in range(nx) for k in range(nz)],
            "zmin": [(i, j, 0) for i in range(nx) for j in range(ny)],
            "zmax": [(i, j, nz - 1) for i in range(nx) for j in range(ny)],
        }
    else:
        raise ValueError("only dim 1, 2, 3 supported")

    if side not in nodes:
        raise ValueError(f"boundary {side} is not valid for dim={dim}")

    return nodes[side]


def split_boundary_conditions(boundary_conditions, grid_vars):
    dirichlet_nodes = {}
    neumann_boundaries = {}

    for side, bc in boundary_conditions.items():
        side = side.lower()
        kind, value = bc
        kind = kind.lower()

        if kind == "dirichlet":
            for idx in boundary_nodes_for_side(side, grid_vars):
                add_dirichlet_node(dirichlet_nodes, idx, value, grid_vars)
        elif kind == "neumann":
            neumann_boundaries[side] = value
        else:
            raise ValueError(f"unknown boundary condition type: {kind}")

    return dirichlet_nodes, neumann_boundaries


def apply_neumann_boundaries(R_global, neumann_boundaries, grid_vars):
    dim = grid_vars["dim"]
    coords = grid_vars["coords"]

    def add_boundary_node(idx, qn):
        R_global[global_id(idx, grid_vars)] += qn

    def add_edge(a, b, length, qn):
        midpoint = 0.5 * (node_point(a, grid_vars) + node_point(b, grid_vars))
        flux = boundary_value(qn, midpoint)
        R_global[global_id(a, grid_vars)] += 0.5 * length * flux
        R_global[global_id(b, grid_vars)] += 0.5 * length * flux

    def add_face(nodes, area, qn):
        midpoint = sum(node_point(node, grid_vars) for node in nodes) / 4.0
        flux = boundary_value(qn, midpoint)
        for node in nodes:
            R_global[global_id(node, grid_vars)] += 0.25 * area * flux

    for side, qn in neumann_boundaries.items():
        if dim == 1:
            nx = grid_vars["n_nodes_per_axis"][0]
            if side == "xmin":
                add_boundary_node((0,), boundary_value(qn, node_point((0,), grid_vars)))
            elif side == "xmax":
                add_boundary_node((nx - 1,), boundary_value(qn, node_point((nx - 1,), grid_vars)))
            else:
                raise ValueError(f"boundary {side} is not valid for dim=1")

        elif dim == 2:
            nx, ny = grid_vars["n_nodes_per_axis"]
            x, y = coords
            if side == "xmin":
                for j in range(ny - 1):
                    add_edge((0, j), (0, j + 1), y[j + 1] - y[j], qn)
            elif side == "xmax":
                for j in range(ny - 1):
                    add_edge((nx - 1, j), (nx - 1, j + 1), y[j + 1] - y[j], qn)
            elif side == "ymin":
                for i in range(nx - 1):
                    add_edge((i, 0), (i + 1, 0), x[i + 1] - x[i], qn)
            elif side == "ymax":
                for i in range(nx - 1):
                    add_edge((i, ny - 1), (i + 1, ny - 1), x[i + 1] - x[i], qn)
            else:
                raise ValueError(f"boundary {side} is not valid for dim=2")

        elif dim == 3:
            nx, ny, nz = grid_vars["n_nodes_per_axis"]
            x, y, z = coords
            if side == "xmin":
                for j in range(ny - 1):
                    for k in range(nz - 1):
                        area = (y[j + 1] - y[j]) * (z[k + 1] - z[k])
                        add_face([(0, j, k), (0, j + 1, k), (0, j, k + 1), (0, j + 1, k + 1)], area, qn)
            elif side == "xmax":
                for j in range(ny - 1):
                    for k in range(nz - 1):
                        area = (y[j + 1] - y[j]) * (z[k + 1] - z[k])
                        add_face([(nx - 1, j, k), (nx - 1, j + 1, k), (nx - 1, j, k + 1), (nx - 1, j + 1, k + 1)], area, qn)
            elif side == "ymin":
                for i in range(nx - 1):
                    for k in range(nz - 1):
                        area = (x[i + 1] - x[i]) * (z[k + 1] - z[k])
                        add_face([(i, 0, k), (i + 1, 0, k), (i, 0, k + 1), (i + 1, 0, k + 1)], area, qn)
            elif side == "ymax":
                for i in range(nx - 1):
                    for k in range(nz - 1):
                        area = (x[i + 1] - x[i]) * (z[k + 1] - z[k])
                        add_face([(i, ny - 1, k), (i + 1, ny - 1, k), (i, ny - 1, k + 1), (i + 1, ny - 1, k + 1)], area, qn)
            elif side == "zmin":
                for i in range(nx - 1):
                    for j in range(ny - 1):
                        area = (x[i + 1] - x[i]) * (y[j + 1] - y[j])
                        add_face([(i, j, 0), (i + 1, j, 0), (i, j + 1, 0), (i + 1, j + 1, 0)], area, qn)
            elif side == "zmax":
                for i in range(nx - 1):
                    for j in range(ny - 1):
                        area = (x[i + 1] - x[i]) * (y[j + 1] - y[j])
                        add_face([(i, j, nz - 1), (i + 1, j, nz - 1), (i, j + 1, nz - 1), (i + 1, j + 1, nz - 1)], area, qn)
            else:
                raise ValueError(f"boundary {side} is not valid for dim=3")

        else:
            raise ValueError("only dim 1, 2, 3 supported")

def element_residual_tangent(a, b, T_elem, flux_law, source_fn, dsource_dT, grid_vars):
    '''
    Construct element residual (R_elem) and Newton tangent (K_elem)
    '''

    def residual_integrand(pt):
        N = np.array(shape_fns(pt, a, b, grid_vars))
        G = np.array(shape_grads(pt, a, b, grid_vars))

        T_g     = N @ T_elem
        gradT_g = G.T @ T_elem   

        q_g, _, _ = flux_law(gradT_g, T_g, pt)
        q_g = np.asarray(q_g, dtype=float).reshape(-1)
        r_g = float(source_fn(T_g, pt))

        return np.array([
            float(-np.dot(G[e], q_g) - N[e] * r_g)
            for e in range(grid_vars["nen"])
        ])

    def tangent_integrand(pt):
        N = np.array(shape_fns(pt, a, b, grid_vars))
        G = np.array(shape_grads(pt, a, b, grid_vars))

        T_g     = N @ T_elem
        gradT_g = G.T @ T_elem   

        _, dq_dgrad, dq_dT = flux_law(gradT_g, T_g, pt)
        dq_dgrad = np.asarray(dq_dgrad, dtype=float).reshape(grid_vars["dim"], grid_vars["dim"])
        dq_dT = np.asarray(dq_dT, dtype=float).reshape(-1)
        drdt_g = float(dsource_dT(T_g, pt))

        K = np.zeros((grid_vars['nen'], grid_vars['nen']))
        for i in range(grid_vars['nen']):
            for j in range(grid_vars['nen']):
                flux_term = -np.dot(G[i], dq_dgrad @ G[j] + dq_dT * N[j])
                src_term  = -drdt_g * N[i] * N[j]
                K[i,j] = float(flux_term + src_term)
        return K


    R_elem = gl2_quadrature_integration(residual_integrand, a=a, b=b, dim=grid_vars['dim'])
    K_elem = gl2_quadrature_integration(tangent_integrand,  a=a, b=b, dim=grid_vars['dim'])
    return R_elem, K_elem      # R_elem: [nen], K_elem: [nen, nen]

def assemble(T_nodal, flux_law, source_fn, dsource_dT, grid_vars, neumann_boundaries=None):
    '''
    Construct global residual (R) and Newton tangent (K).
    '''

    R_global = np.zeros(grid_vars['n_nodes'])
    K_global = np.zeros((grid_vars['n_nodes'], grid_vars['n_nodes'])) # NOTE: should use sparse mtx 

    for base_idx in grid_vars['element_base_indices']:
        a, b, local_ids = element_bounds_and_ids(base_idx, grid_vars)
        T_elem = T_nodal[local_ids]

        R_elem, K_elem = element_residual_tangent(a, b, T_elem, flux_law, source_fn, dsource_dT, grid_vars)

        for i in range(grid_vars['nen']):
            R_global[local_ids[i]] += R_elem[i]
            for j in range(grid_vars['nen']):
                K_global[local_ids[i], local_ids[j]] += K_elem[i,j]

    if neumann_boundaries:
        apply_neumann_boundaries(R_global, neumann_boundaries, grid_vars)

    return R_global, K_global

# =================================================================================

# =================================================================================
# main Newton loop

def grid(boundary_points):
    coords = [np.asarray(axis, dtype=float) for axis in boundary_points]
    dim = len(coords)
    n_nodes_per_axis = [len(coords[d]) for d in range(dim)]
    n_nodes = int(np.prod(n_nodes_per_axis))
    n_elements_per_axis = [n - 1 for n in n_nodes_per_axis]
    nen = 2 ** dim
    local_corner_offsets = list(itertools.product([0, 1], repeat=dim))
    element_base_indices = list(itertools.product(*[range(n) for n in n_elements_per_axis]))
    all_node_indices = list(itertools.product(*[range(n) for n in n_nodes_per_axis]))

    grid_vars = {
            'coords': coords,
            'dim': dim, 
            'n_nodes_per_axis': n_nodes_per_axis, 
            'n_nodes': n_nodes, 
            'nen':  nen,
            'local_corner_offsets': local_corner_offsets,
            'element_base_indices': element_base_indices,
            'all_node_indices': all_node_indices,
            }
    
    return grid_vars


def initial_guess(dirichlet_nodes, grid_vars): # NOTE: make this more flexible
    '''
    Linear interpolation between dirichlet boundary points
    '''

    if not dirichlet_nodes:
        raise ValueError("at least one Dirichlet boundary is required")

    # interpolate
    known_pts = np.array([node_point(idx, grid_vars) for idx in dirichlet_nodes.keys()])
    known_vals = np.array(list(dirichlet_nodes.values()))
    all_pts = np.array([node_point(idx, grid_vars) for idx in grid_vars['all_node_indices']])

    if grid_vars['dim'] == 1:
        T_nodal = np.interp(all_pts[:,0], known_pts[:,0], known_vals)
    else:
        T_nodal = griddata(known_pts, known_vals, all_pts, method='linear')
        T_nodal = np.asarray(T_nodal, dtype=float).reshape(-1)
        if np.any(np.isnan(T_nodal)):
            T_nearest = griddata(known_pts, known_vals, all_pts, method='nearest')
            T_nearest = np.asarray(T_nearest, dtype=float).reshape(-1)
            T_nodal[np.isnan(T_nodal)] = T_nearest[np.isnan(T_nodal)]

    # reapply boundary values
    for idx, val in dirichlet_nodes.items():
        T_nodal[global_id(idx, grid_vars)] = val 

    return T_nodal
    

def NM(boundary_points, flux_law, source_fn, dsource_dT, boundary_conditions, tol=1e-10, maxiter=30, verbose=False, line_search=True, return_grid=False):
    '''
    Newton linearization.

    Expected parameters:
        boundary_points = [[x_coords], [y_coords], [(z_coords)]]
        flux_law(g, T, x) -> (q, dq_dg, dq_dT)
        source_fn(T, x) -> r
        dsource_dT(T, x) -> dr_dT
        boundary_conditions : dict {idx_tuple) -> ('dirichlet'/'neumann', prescribed_value})

    Returns:
        T_nodal : nodal temperature array
        residual_norm_history : residual norm at each Newton check
        num_iters : num Newton updates before converge
    '''

    grid_vars = grid(boundary_points)
    dirichlet_nodes, neumann_boundaries = split_boundary_conditions(boundary_conditions, grid_vars)
    T_nodal = initial_guess(dirichlet_nodes, grid_vars)

    residual_norm_history = []
    free_dofs = [global_id(idx, grid_vars) for idx in grid_vars['all_node_indices'] if idx not in dirichlet_nodes]

    for iteration in range(maxiter):
        R_global, K_global = assemble(T_nodal, flux_law, source_fn, dsource_dT, grid_vars, neumann_boundaries)
        R_free = R_global[free_dofs]

        residual_norm = np.linalg.norm(R_free, ord=2)
        residual_norm_history.append(residual_norm)

        if verbose:
            print(f"Newton {iteration:2d}: ||R_free||_2 = {residual_norm:.3e}")

        if residual_norm < tol:
            if return_grid:
                return T_nodal.reshape(grid_vars["n_nodes_per_axis"]), residual_norm_history, iteration
            return T_nodal, residual_norm_history, iteration 

        K_free = K_global[free_dofs][:, free_dofs]
        dT_free = np.linalg.solve(K_free, -R_free)

        alpha = 1.0 
        if line_search: 
            step_accepted = False
            while alpha > 1.0e-12:
                T_trial = T_nodal.copy()
                T_trial[free_dofs] += alpha * dT_free 
                for idx, val in dirichlet_nodes.items(): T_trial[global_id(idx, grid_vars)] = val

                R_trial, _ = assemble(T_trial, flux_law, source_fn, dsource_dT, grid_vars, neumann_boundaries)
                if np.linalg.norm(R_trial[free_dofs], ord=2) < residual_norm:
                    step_accepted = True
                    break 

                alpha *= 0.5
            if not step_accepted:
                raise RuntimeError("line search failed to minimize residual")
        else: 
            T_trial = T_nodal.copy()
            T_trial[free_dofs] += dT_free 

        T_nodal = T_trial

    raise RuntimeError("NM didn't converge within desired number maxiter")


# =================================================================================

def run_tests():
    from newtonND_tests import dim1test, dim2test, mixed_bc_1dtest, mixed_bc_2dtest, mixed_bc_3dtest, darcy2dtest

    print("Running linear 1D FEM solve:")
    dim1test()

    print("\nRunning linear 2D FEM solve:")
    dim2test()

    # darcyvis()

    print("\nRunning mixed Dirichlet/Neumann 1D FEM solve:")
    mixed_bc_1dtest()

    print("\nRunning mixed Dirichlet/Neumann 2D FEM solve:")
    mixed_bc_2dtest()

    print("\nRunning mixed Dirichlet/Neumann 3D FEM solve:")
    mixed_bc_3dtest()

    print(f"\nDarcy law mixed-boundary FEM solve:")
    darcy2dtest()

if __name__ == "__main__":
    run_tests()
