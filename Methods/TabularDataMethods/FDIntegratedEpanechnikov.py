import numpy as np
from scipy.interpolate import RegularGridInterpolator

from Methods.TabularDataMethods.FDMaternSmoothing import (
    reconstruct_q_from_derivatives,
    repair_endpoint_axis,
    smooth_line,
)


KERNEL_WIDTH = 15
Q_FIDELITY = 0.03


def epanechnikov_kernel(width=KERNEL_WIDTH):
    width = max(1, int(width))
    if width % 2 == 0:
        width -= 1

    half_width = width // 2
    r = np.abs(np.arange(-half_width, half_width + 1, dtype=float))
    r /= max(half_width, 1)
    weights = np.maximum(0.0, 1.0 - r**2)
    return weights / weights.sum()


def smooth_derivative_grid(derivative_grid):
    kernel_s = epanechnikov_kernel(min(derivative_grid.shape[0], KERNEL_WIDTH))
    kernel_T = epanechnikov_kernel(min(derivative_grid.shape[1], KERNEL_WIDTH))
    smoothed = np.apply_along_axis(smooth_line, 0, derivative_grid, kernel_s)
    return np.apply_along_axis(smooth_line, 1, smoothed, kernel_T)


class TabularFDIntegratedEpanechnikovSmoothST:
    """
    Best overall FD-then-smooth provider from the three-dataset sweep.

    Raw q is finite-differenced first. The derivative fields are smoothed with
    an Epanechnikov kernel, endpoint-repaired, then integrated back to a
    compatible q table with light fidelity to the raw q values.
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
            smooth_derivative_grid(dq_ds),
            self.s_grid,
            axis=0,
            edge_width=KERNEL_WIDTH // 2,
        )
        dq_dT = repair_endpoint_axis(
            smooth_derivative_grid(dq_dT),
            self.T_grid,
            axis=1,
            edge_width=KERNEL_WIDTH // 2,
        )

        self.q_grid = reconstruct_q_from_derivatives(
            self.s_grid,
            self.T_grid,
            raw_q_grid,
            dq_ds,
            dq_dT,
            fidelity=Q_FIDELITY,
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
