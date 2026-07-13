import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.interpolate import RegularGridInterpolator


Q_SMOOTH_KERNEL = np.array(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
    dtype=float,
)
Q_SMOOTH_KERNEL /= Q_SMOOTH_KERNEL.sum()

DERIVATIVE_KERNEL_WIDTH = 5
DERIVATIVE_LENGTH_SCALE = 0.5
ENDPOINT_FIT_WIDTH = 20
ENDPOINT_POLYORDER = 2
INTEGRATED_KERNEL_WIDTH = 13
INTEGRATED_LENGTH_SCALE = 5.0
INTEGRATED_Q_FIDELITY = 0.1
INTEGRATED_TRICUBE_WIDTH = 17
INTEGRATED_TRICUBE_Q_FIDELITY = 0.1


def matern52_kernel(width=DERIVATIVE_KERNEL_WIDTH, length_scale=DERIVATIVE_LENGTH_SCALE):
    width = max(1, int(width))
    if width % 2 == 0:
        width -= 1

    r = np.abs(np.arange(-(width // 2), width // 2 + 1, dtype=float))
    z = np.sqrt(5.0) * r / length_scale
    weights = (1.0 + z + z**2 / 3.0) * np.exp(-z)
    return weights / weights.sum()


def tricube_kernel(width=INTEGRATED_TRICUBE_WIDTH):
    width = max(1, int(width))
    if width % 2 == 0:
        width -= 1

    half_width = width // 2
    r = np.abs(np.arange(-half_width, half_width + 1, dtype=float))
    r /= max(half_width, 1)
    weights = (1.0 - np.minimum(r, 1.0) ** 3) ** 3
    return weights / weights.sum()


def smooth_q_grid(q_grid):
    padded = np.pad(q_grid, ((1, 1), (1, 1)), mode="edge")
    smoothed = np.zeros_like(q_grid, dtype=float)

    for i in range(q_grid.shape[0]):
        for j in range(q_grid.shape[1]):
            window = padded[i:i + 3, j:j + 3]
            smoothed[i, j] = np.sum(Q_SMOOTH_KERNEL * window)

    return smoothed


def smooth_line(line, kernel):
    half_width = len(kernel) // 2
    smoothed = np.zeros_like(line, dtype=float)

    for i in range(len(line)):
        left = max(0, i - half_width)
        right = min(len(line), i + half_width + 1)
        k_left = half_width - (i - left)
        k_right = half_width + (right - i)
        k = kernel[k_left:k_right]
        smoothed[i] = np.sum(line[left:right] * k / k.sum())

    return smoothed


def smooth_derivative_grid(derivative_grid):
    kernel_s = matern52_kernel(min(derivative_grid.shape[0], DERIVATIVE_KERNEL_WIDTH))
    kernel_T = matern52_kernel(min(derivative_grid.shape[1], DERIVATIVE_KERNEL_WIDTH))
    smoothed = np.apply_along_axis(smooth_line, 0, derivative_grid, kernel_s)
    return np.apply_along_axis(smooth_line, 1, smoothed, kernel_T)


def smooth_derivative_grid_integrated(derivative_grid):
    kernel_s = matern52_kernel(
        min(derivative_grid.shape[0], INTEGRATED_KERNEL_WIDTH),
        INTEGRATED_LENGTH_SCALE,
    )
    kernel_T = matern52_kernel(
        min(derivative_grid.shape[1], INTEGRATED_KERNEL_WIDTH),
        INTEGRATED_LENGTH_SCALE,
    )
    smoothed = np.apply_along_axis(smooth_line, 0, derivative_grid, kernel_s)
    return np.apply_along_axis(smooth_line, 1, smoothed, kernel_T)


def smooth_derivative_grid_tricube(derivative_grid):
    kernel_s = tricube_kernel(min(derivative_grid.shape[0], INTEGRATED_TRICUBE_WIDTH))
    kernel_T = tricube_kernel(min(derivative_grid.shape[1], INTEGRATED_TRICUBE_WIDTH))
    smoothed = np.apply_along_axis(smooth_line, 0, derivative_grid, kernel_s)
    return np.apply_along_axis(smooth_line, 1, smoothed, kernel_T)


def repair_endpoint_line(x_grid, values, edge_width=None):
    if edge_width is None:
        edge_width = DERIVATIVE_KERNEL_WIDTH // 2
    edge_width = min(edge_width, max(0, (len(values) - 1) // 2))
    if edge_width <= 0 or len(values) <= ENDPOINT_POLYORDER + 1:
        return values

    repaired = values.copy()
    fit_width = min(ENDPOINT_FIT_WIDTH, len(values) - edge_width)
    left_fit = slice(edge_width, min(len(values), edge_width + fit_width))
    right_fit = slice(max(0, len(values) - edge_width - fit_width), len(values) - edge_width)

    if left_fit.stop - left_fit.start > ENDPOINT_POLYORDER:
        coeffs = np.polyfit(x_grid[left_fit], values[left_fit], ENDPOINT_POLYORDER)
        repaired[:edge_width] = np.polyval(coeffs, x_grid[:edge_width])

    if right_fit.stop - right_fit.start > ENDPOINT_POLYORDER:
        coeffs = np.polyfit(x_grid[right_fit], values[right_fit], ENDPOINT_POLYORDER)
        repaired[-edge_width:] = np.polyval(coeffs, x_grid[-edge_width:])

    return repaired


def repair_endpoint_axis(derivative_grid, x_grid, axis, edge_width=None):
    return np.apply_along_axis(
        lambda line: repair_endpoint_line(x_grid, line, edge_width=edge_width),
        axis,
        derivative_grid,
    )


class TabularFDMaternSmoothST:
    """
    q(s,T) provider using robust FD-then-smooth derivatives.

    The high-noise Newton tests were unstable when only derivative fields were
    smoothed. This provider first applies a small 3x3 binomial smoother to q,
    then finite-differences that safer table, smooths the derivative fields with
    a compact Matern 5/2 kernel, and repairs endpoint derivative slices.
    """

    def __init__(self, s_grid, T_grid, q_grid):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        raw_q_grid = np.asarray(q_grid, dtype=float)

        expected_shape = (len(self.s_grid), len(self.T_grid))
        if raw_q_grid.shape != expected_shape:
            raise ValueError(f"q_grid must have shape {expected_shape}")

        self.q_grid = smooth_q_grid(raw_q_grid)
        self.ds = float(np.median(np.diff(self.s_grid)))
        self.dT = float(np.median(np.diff(self.T_grid)))

        dq_ds = np.gradient(self.q_grid, self.s_grid, axis=0, edge_order=2)
        dq_dT = np.gradient(self.q_grid, self.T_grid, axis=1, edge_order=2)

        self.dq_ds_grid = repair_endpoint_axis(
            smooth_derivative_grid(dq_ds),
            self.s_grid,
            axis=0,
        )
        self.dq_dT_grid = repair_endpoint_axis(
            smooth_derivative_grid(dq_dT),
            self.T_grid,
            axis=1,
        )

        self.q_interp = self._interpolator(self.q_grid)
        self.dq_ds_interp = self._interpolator(self.dq_ds_grid)
        self.dq_dT_interp = self._interpolator(self.dq_dT_grid)

    def _interpolator(self, values):
        return RegularGridInterpolator(
            (self.s_grid, self.T_grid),
            values,
            bounds_error=False,
            fill_value=None,
        )

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        points = np.column_stack([s_q.ravel(), T_q.ravel()])
        return (
            self.q_interp(points).reshape(s_q.shape),
            self.dq_ds_interp(points).reshape(s_q.shape),
            self.dq_dT_interp(points).reshape(s_q.shape),
        )


def project_to_integrable_gradient(s_grid, T_grid, dq_ds, dq_dT):
    """
    Find q_hat whose grid gradient best matches the smoothed derivative fields.

    This couples dq/ds and dq/dT by enforcing equality of mixed partials in the
    least-squares sense. The returned derivatives are gradients of q_hat.
    """
    n_s, n_T = dq_ds.shape
    ds = float(np.median(np.diff(s_grid)))
    dT = float(np.median(np.diff(T_grid)))
    n_unknowns = n_s * n_T

    rows = []
    rhs = []

    def flat(i, j):
        return i * n_T + j

    for i in range(n_s - 1):
        for j in range(n_T):
            row = {flat(i + 1, j): 1.0 / ds, flat(i, j): -1.0 / ds}
            rows.append(row)
            rhs.append(0.5 * (dq_ds[i, j] + dq_ds[i + 1, j]))

    for i in range(n_s):
        for j in range(n_T - 1):
            row = {flat(i, j + 1): 1.0 / dT, flat(i, j): -1.0 / dT}
            rows.append(row)
            rhs.append(0.5 * (dq_dT[i, j] + dq_dT[i, j + 1]))

    # Fix the arbitrary integration constant.
    rows.append({0: 1.0})
    rhs.append(0.0)

    system = lil_matrix((len(rows), n_unknowns), dtype=float)
    for row_idx, row in enumerate(rows):
        for col_idx, value in row.items():
            system[row_idx, col_idx] = value

    q_hat = lsqr(system.tocsr(), np.asarray(rhs, dtype=float), atol=1e-10, btol=1e-10)[0]
    q_hat = q_hat.reshape((n_s, n_T))
    return (
        np.gradient(q_hat, s_grid, axis=0, edge_order=2),
        np.gradient(q_hat, T_grid, axis=1, edge_order=2),
    )


def reconstruct_q_from_derivatives(
    s_grid,
    T_grid,
    q_reference,
    dq_ds,
    dq_dT,
    fidelity=INTEGRATED_Q_FIDELITY,
):
    """
    Reconstruct q after derivative smoothing.

    This is derivative-first: q_reference is not smoothed before finite
    differencing. It only anchors the integration constant and keeps q_hat near
    the observed table with a light least-squares fidelity term.
    """
    n_s, n_T = q_reference.shape
    ds = float(np.median(np.diff(s_grid)))
    dT = float(np.median(np.diff(T_grid)))
    rows = []
    rhs = []

    def flat(i, j):
        return i * n_T + j

    for i in range(n_s - 1):
        for j in range(n_T):
            rows.append({flat(i + 1, j): 1.0 / ds, flat(i, j): -1.0 / ds})
            rhs.append(0.5 * (dq_ds[i, j] + dq_ds[i + 1, j]))

    for i in range(n_s):
        for j in range(n_T - 1):
            rows.append({flat(i, j + 1): 1.0 / dT, flat(i, j): -1.0 / dT})
            rhs.append(0.5 * (dq_dT[i, j] + dq_dT[i, j + 1]))

    fidelity_weight = np.sqrt(fidelity)
    for i in range(n_s):
        for j in range(n_T):
            rows.append({flat(i, j): fidelity_weight})
            rhs.append(fidelity_weight * q_reference[i, j])

    system = lil_matrix((len(rows), n_s * n_T), dtype=float)
    for row_idx, row in enumerate(rows):
        for col_idx, value in row.items():
            system[row_idx, col_idx] = value

    q_hat = lsqr(system.tocsr(), np.asarray(rhs, dtype=float), atol=1e-10, btol=1e-10)[0]
    return q_hat.reshape(q_reference.shape)


class TabularFDCoupledMaternSmoothST:
    """
    Raw q(s,T), FD derivatives, Matern smoothing, then coupled gradient projection.

    Unlike TabularFDMaternSmoothST, this keeps q itself unsmoothed and only
    regularizes the derivative/tangent fields.
    """

    def __init__(self, s_grid, T_grid, q_grid):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.q_grid = np.asarray(q_grid, dtype=float)
        self.ds = float(np.median(np.diff(self.s_grid)))
        self.dT = float(np.median(np.diff(self.T_grid)))

        expected_shape = (len(self.s_grid), len(self.T_grid))
        if self.q_grid.shape != expected_shape:
            raise ValueError(f"q_grid must have shape {expected_shape}")

        dq_ds = np.gradient(self.q_grid, self.s_grid, axis=0, edge_order=2)
        dq_dT = np.gradient(self.q_grid, self.T_grid, axis=1, edge_order=2)
        dq_ds = repair_endpoint_axis(smooth_derivative_grid(dq_ds), self.s_grid, axis=0)
        dq_dT = repair_endpoint_axis(smooth_derivative_grid(dq_dT), self.T_grid, axis=1)
        self.dq_ds_grid, self.dq_dT_grid = project_to_integrable_gradient(
            self.s_grid,
            self.T_grid,
            dq_ds,
            dq_dT,
        )

        self.q_interp = self._interpolator(self.q_grid)
        self.dq_ds_interp = self._interpolator(self.dq_ds_grid)
        self.dq_dT_interp = self._interpolator(self.dq_dT_grid)

    def _interpolator(self, values):
        return RegularGridInterpolator(
            (self.s_grid, self.T_grid),
            values,
            bounds_error=False,
            fill_value=None,
        )

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        points = np.column_stack([s_q.ravel(), T_q.ravel()])
        return (
            self.q_interp(points).reshape(s_q.shape),
            self.dq_ds_interp(points).reshape(s_q.shape),
            self.dq_dT_interp(points).reshape(s_q.shape),
        )


class TabularFDIntegratedMaternSmoothST:
    """
    FD first, smooth derivatives, then integrate them back to a compatible q.

    This avoids pre-smoothing q. The residual table is reconstructed from the
    smoothed derivative field plus a light fidelity term to the raw q table.
    """

    def __init__(self, s_grid, T_grid, q_grid):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        raw_q_grid = np.asarray(q_grid, dtype=float)
        self.ds = float(np.median(np.diff(self.s_grid)))
        self.dT = float(np.median(np.diff(self.T_grid)))

        expected_shape = (len(self.s_grid), len(self.T_grid))
        if raw_q_grid.shape != expected_shape:
            raise ValueError(f"q_grid must have shape {expected_shape}")

        dq_ds = np.gradient(raw_q_grid, self.s_grid, axis=0, edge_order=2)
        dq_dT = np.gradient(raw_q_grid, self.T_grid, axis=1, edge_order=2)
        dq_ds = repair_endpoint_axis(
            smooth_derivative_grid_integrated(dq_ds),
            self.s_grid,
            axis=0,
        )
        dq_dT = repair_endpoint_axis(
            smooth_derivative_grid_integrated(dq_dT),
            self.T_grid,
            axis=1,
        )

        self.q_grid = reconstruct_q_from_derivatives(
            self.s_grid,
            self.T_grid,
            raw_q_grid,
            dq_ds,
            dq_dT,
        )
        self.dq_ds_grid = np.gradient(self.q_grid, self.s_grid, axis=0, edge_order=2)
        self.dq_dT_grid = np.gradient(self.q_grid, self.T_grid, axis=1, edge_order=2)

        self.q_interp = self._interpolator(self.q_grid)
        self.dq_ds_interp = self._interpolator(self.dq_ds_grid)
        self.dq_dT_interp = self._interpolator(self.dq_dT_grid)

    def _interpolator(self, values):
        return RegularGridInterpolator(
            (self.s_grid, self.T_grid),
            values,
            bounds_error=False,
            fill_value=None,
        )

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        points = np.column_stack([s_q.ravel(), T_q.ravel()])
        return (
            self.q_interp(points).reshape(s_q.shape),
            self.dq_ds_interp(points).reshape(s_q.shape),
            self.dq_dT_interp(points).reshape(s_q.shape),
        )


class TabularFDIntegratedTricubeSmoothST:
    """
    FD first, tricube-smooth derivatives, then integrate back to compatible q.

    This keeps the no-pre-smoothing story while using the best non-Matern
    kernel found in the Newton sweep: width 17, q fidelity 0.1.
    """

    def __init__(self, s_grid, T_grid, q_grid):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        raw_q_grid = np.asarray(q_grid, dtype=float)
        self.ds = float(np.median(np.diff(self.s_grid)))
        self.dT = float(np.median(np.diff(self.T_grid)))

        expected_shape = (len(self.s_grid), len(self.T_grid))
        if raw_q_grid.shape != expected_shape:
            raise ValueError(f"q_grid must have shape {expected_shape}")

        dq_ds = np.gradient(raw_q_grid, self.s_grid, axis=0, edge_order=2)
        dq_dT = np.gradient(raw_q_grid, self.T_grid, axis=1, edge_order=2)
        dq_ds = repair_endpoint_axis(
            smooth_derivative_grid_tricube(dq_ds),
            self.s_grid,
            axis=0,
            edge_width=INTEGRATED_TRICUBE_WIDTH // 2,
        )
        dq_dT = repair_endpoint_axis(
            smooth_derivative_grid_tricube(dq_dT),
            self.T_grid,
            axis=1,
            edge_width=INTEGRATED_TRICUBE_WIDTH // 2,
        )

        self.q_grid = reconstruct_q_from_derivatives(
            self.s_grid,
            self.T_grid,
            raw_q_grid,
            dq_ds,
            dq_dT,
            fidelity=INTEGRATED_TRICUBE_Q_FIDELITY,
        )
        self.dq_ds_grid = np.gradient(self.q_grid, self.s_grid, axis=0, edge_order=2)
        self.dq_dT_grid = np.gradient(self.q_grid, self.T_grid, axis=1, edge_order=2)

        self.q_interp = self._interpolator(self.q_grid)
        self.dq_ds_interp = self._interpolator(self.dq_ds_grid)
        self.dq_dT_interp = self._interpolator(self.dq_dT_grid)

    def _interpolator(self, values):
        return RegularGridInterpolator(
            (self.s_grid, self.T_grid),
            values,
            bounds_error=False,
            fill_value=None,
        )

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        if s_q.shape != T_q.shape:
            raise ValueError("s_q and T_q must have the same shape")

        points = np.column_stack([s_q.ravel(), T_q.ravel()])
        return (
            self.q_interp(points).reshape(s_q.shape),
            self.dq_ds_interp(points).reshape(s_q.shape),
            self.dq_dT_interp(points).reshape(s_q.shape),
        )

