"""Local tabular provider implementations and table transforms."""

from __future__ import annotations

import numpy as np


SAVGOL_NEIGHBORS = 25


def quadratic_basis(eta):
    eta = np.asarray(eta, dtype=float)
    p = len(eta)
    phi = [1.0]
    phi.extend(eta[j] for j in range(p))
    phi.extend(eta[j] ** 2 for j in range(p))
    for j in range(p):
        for k in range(j + 1, p):
            phi.append(eta[j] * eta[k])
    return np.asarray(phi)


class LocalSavGolProvider:
    def __init__(self, z_data, q_data, K=45, h=0.9):
        self.z_data = np.asarray(z_data, dtype=float)
        self.q_data = np.asarray(q_data, dtype=float)
        self.K = K
        self.h = h

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        q = np.zeros_like(s_q, dtype=float)
        a = np.zeros_like(s_q, dtype=float)
        b = np.zeros_like(s_q, dtype=float)

        for idx in np.ndindex(s_q.shape):
            z0 = np.array([s_q[idx], T_q[idx]])
            distances = np.linalg.norm(self.z_data - z0, axis=1)
            nn = np.argpartition(distances, self.K - 1)[: self.K]

            Phi = []
            y = []
            w = []
            for j in nn:
                eta = self.z_data[j] - z0
                Phi.append(quadratic_basis(eta))
                y.append(self.q_data[j])
                w.append(np.exp(-(np.linalg.norm(eta) / self.h) ** 2))

            Phi = np.asarray(Phi)
            y = np.asarray(y)
            sw = np.sqrt(np.asarray(w))
            beta_hat, *_ = np.linalg.lstsq(Phi * sw[:, None], y * sw, rcond=None)
            q[idx] = beta_hat[0]
            a[idx] = beta_hat[1]
            b[idx] = beta_hat[2]

        return q, a, b


class TabularFiniteDifferenceProvider:
    def __init__(self, s_grid, T_grid, q_grid):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.q_grid = np.asarray(q_grid, dtype=float)
        self.ds = float(np.median(np.diff(self.s_grid)))
        self.dT = float(np.median(np.diff(self.T_grid)))

    def _q_single(self, s, T):
        s = float(np.clip(s, self.s_grid[0], self.s_grid[-1]))
        T = float(np.clip(T, self.T_grid[0], self.T_grid[-1]))

        i = np.searchsorted(self.s_grid, s, side="right") - 1
        j = np.searchsorted(self.T_grid, T, side="right") - 1
        i = int(np.clip(i, 0, len(self.s_grid) - 2))
        j = int(np.clip(j, 0, len(self.T_grid) - 2))

        s0 = self.s_grid[i]
        s1 = self.s_grid[i + 1]
        T0 = self.T_grid[j]
        T1 = self.T_grid[j + 1]
        ws = (s - s0) / (s1 - s0)
        wT = (T - T0) / (T1 - T0)

        q00 = self.q_grid[i, j]
        q10 = self.q_grid[i + 1, j]
        q01 = self.q_grid[i, j + 1]
        q11 = self.q_grid[i + 1, j + 1]
        return (
            (1.0 - ws) * (1.0 - wT) * q00
            + ws * (1.0 - wT) * q10
            + (1.0 - ws) * wT * q01
            + ws * wT * q11
        )

    def _finite_difference(self, s, T, axis):
        if axis == "s":
            lo = max(self.s_grid[0], s - self.ds)
            hi = min(self.s_grid[-1], s + self.ds)
            if hi == lo:
                return 0.0
            return (self._q_single(hi, T) - self._q_single(lo, T)) / (hi - lo)

        lo = max(self.T_grid[0], T - self.dT)
        hi = min(self.T_grid[-1], T + self.dT)
        if hi == lo:
            return 0.0
        return (self._q_single(s, hi) - self._q_single(s, lo)) / (hi - lo)

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        q = np.zeros_like(s_q, dtype=float)
        a = np.zeros_like(s_q, dtype=float)
        b = np.zeros_like(s_q, dtype=float)

        for idx in np.ndindex(s_q.shape):
            s = float(s_q[idx])
            T = float(T_q[idx])
            q[idx] = self._q_single(s, T)
            a[idx] = self._finite_difference(s, T, axis="s")
            b[idx] = self._finite_difference(s, T, axis="T")

        return q, a, b


def project_nonincreasing(values):
    values = np.asarray(values, dtype=float)
    projected_neg = project_nondecreasing(-values)
    return -projected_neg


def project_nondecreasing(values):
    levels = []
    weights = []
    counts = []

    for value in values:
        levels.append(float(value))
        weights.append(1.0)
        counts.append(1)

        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_weight = weights[-2] + weights[-1]
            merged_level = (
                levels[-2] * weights[-2] + levels[-1] * weights[-1]
            ) / merged_weight
            merged_count = counts[-2] + counts[-1]
            levels[-2:] = [merged_level]
            weights[-2:] = [merged_weight]
            counts[-2:] = [merged_count]

    return np.repeat(levels, counts)


def project_table_nonincreasing_in_s(q_grid):
    q_projected = np.asarray(q_grid, dtype=float).copy()
    for T_idx in range(q_projected.shape[1]):
        q_projected[:, T_idx] = project_nonincreasing(q_projected[:, T_idx])
    return q_projected


def smooth_table_3x3(q_grid):
    q_grid = np.asarray(q_grid, dtype=float)
    padded = np.pad(q_grid, ((1, 1), (1, 1)), mode="edge")
    smoothed = np.zeros_like(q_grid)
    kernel = np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=float,
    )
    kernel /= np.sum(kernel)

    for i in range(q_grid.shape[0]):
        for j in range(q_grid.shape[1]):
            smoothed[i, j] = np.sum(kernel * padded[i : i + 3, j : j + 3])

    return smoothed
