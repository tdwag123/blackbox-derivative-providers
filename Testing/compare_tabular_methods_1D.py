from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TABULAR_DIR = ROOT / "Methods" / "TabularDataMethods"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TABULAR_DIR))

from Methods.TabularDataMethods.CubicSplines import CubicSplineFluxST
# from Methods.TabularDataMethods.MonotoneInterpolation import PchipFluxST
from Methods.TabularDataMethods.RadialBasisFunctions import RBFDerivativeProviderST
from Methods.TabularDataMethods.GaussianProcessesWrap import KISSGPFluxST
from Methods.TabularDataMethods.RandomFeature.RFF import RFFDerivativeProviderST


warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(precision=4, suppress=True)

CHOP_GRID_S = 17
CHOP_GRID_T = 17
MAX_ROWS_PER_CELL = 1
SAVGOL_NEIGHBORS = 25
CLEAN_EVAL_POINTS = 500
NOISY_TEST_POINTS = 500
PHYSICS_TOL = 1.0e-10


def q_true(s, T, k0, alpha, beta):
    return -(k0 * (1.0 + alpha * T**2) + beta * s**2) * s


def a_true(s, T, k0, alpha, beta):
    return -k0 * (1.0 + alpha * T**2) - 3.0 * beta * s**2


def b_true(s, T, k0, alpha, beta):
    return -2.0 * k0 * alpha * T * s


def GLquadrature_twoPoint(f, a, b):
    nodes = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    weights = [1.0, 1.0]
    jacobian = (b - a) / 2.0
    midpoint = (a + b) / 2.0

    integral = None
    for node, weight in zip(nodes, weights):
        xg = midpoint + jacobian * node
        value = np.asarray(f(xg), dtype=float)
        integral = weight * value if integral is None else integral + weight * value

    result = jacobian * integral
    if result.shape == ():
        return float(result)
    return result


def constructTridiag(lower, diag, upper, e, Ke):
    diag[e] += Ke[0, 0]
    upper[e] += Ke[0, 1]
    lower[e] += Ke[1, 0]
    diag[e + 1] += Ke[1, 1]


def thomas_solve(lower, diagonal, upper, rhs):
    lower = np.asarray(lower, dtype=float).copy()
    diagonal = np.asarray(diagonal, dtype=float).copy()
    upper = np.asarray(upper, dtype=float).copy()
    rhs = np.asarray(rhs, dtype=float).copy()

    n = len(diagonal)
    if n == 0:
        return np.array([], dtype=float)
    if abs(diagonal[0]) < 1.0e-14:
        raise ZeroDivisionError("zero pivot at row 0")
    if n == 1:
        return np.array([rhs[0] / diagonal[0]], dtype=float)

    gamma = np.zeros(n)
    rho = np.zeros(n)
    x = np.zeros(n)

    gamma[0] = upper[0] / diagonal[0]
    rho[0] = rhs[0] / diagonal[0]

    for i in range(1, n):
        denom = diagonal[i] - lower[i - 1] * gamma[i - 1]
        if abs(denom) < 1.0e-14:
            raise ZeroDivisionError(f"zero pivot at row {i}")
        gamma[i] = upper[i] / denom if i < n - 1 else 0.0
        rho[i] = (rhs[i] - lower[i - 1] * rho[i - 1]) / denom

    x[-1] = rho[-1]
    for i in range(n - 2, -1, -1):
        x[i] = rho[i] - gamma[i] * x[i + 1]

    return x


def tridiag_block(lower, diag, upper, start, end):
    return diag[start:end].copy(), lower[start:end - 1].copy(), upper[start:end - 1].copy()


def residAndTan(x, U, fluxLaw, r, rT=None, q_BCL=None, q_BCR=None):
    if rT is None:
        rT = lambda T, xg: 0.0

    x = np.asarray(x, dtype=float)
    U = np.asarray(U, dtype=float)
    n_nodes = len(x)

    R = np.zeros(n_nodes)
    lower = np.zeros(n_nodes - 1)
    diag = np.zeros(n_nodes)
    upper = np.zeros(n_nodes - 1)

    for e in range(n_nodes - 1):
        xl = x[e]
        xr = x[e + 1]
        h = xr - xl
        Ue = np.array([U[e], U[e + 1]])
        dNdx = np.array([-1.0 / h, 1.0 / h])

        def N_j(xg):
            return np.array([(xr - xg) / h, (xg - xl) / h])

        def residual_integrand(xg):
            N = N_j(xg)
            Tg = N @ Ue
            sg = dNdx @ Ue
            qg, phi_s, phi_T = fluxLaw(sg, Tg, xg)
            rg = r(Tg, xg)
            return -dNdx * qg - N * rg

        def tangent_integrand(xg):
            N = N_j(xg)
            Tg = N @ Ue
            sg = dNdx @ Ue
            qg, phi_s, phi_T = fluxLaw(sg, Tg, xg)
            rTg = rT(Tg, xg)
            K_flux = -np.outer(dNdx, phi_s * dNdx + phi_T * N)
            K_source = -rTg * np.outer(N, N)
            return K_flux + K_source

        Re = GLquadrature_twoPoint(residual_integrand, xl, xr)
        Ke = GLquadrature_twoPoint(tangent_integrand, xl, xr)
        R[e:e + 2] += Re
        constructTridiag(lower, diag, upper, e, Ke)

    if q_BCL is not None:
        R[0] += q_BCL
    if q_BCR is not None:
        R[-1] += q_BCR

    return R, lower, diag, upper


def NM(x, fluxLaw, r, TL, TR, rT=None, U0=None, q_BCL=None, q_BCR=None, tol=1e-10, maxiter=30, line_search=True):
    x = np.asarray(x, dtype=float)
    n_nodes = len(x)
    left_dirich = TL is not None
    right_dirich = TR is not None

    if U0 is None:
        if left_dirich and right_dirich:
            U = np.linspace(TL, TR, n_nodes)
        elif left_dirich:
            U = np.full(n_nodes, TL, dtype=float)
        elif right_dirich:
            U = np.full(n_nodes, TR, dtype=float)
        else:
            raise ValueError("need at least one Dirichlet condition")
    else:
        U = np.asarray(U0, dtype=float).copy()

    if left_dirich:
        U[0] = TL
    if right_dirich:
        U[-1] = TR

    start = 1 if left_dirich else 0
    end = n_nodes - 1 if right_dirich else n_nodes
    log = []

    for _ in range(maxiter):
        R, lower, diag, upper = residAndTan(x, U, fluxLaw, r, rT=rT, q_BCL=q_BCL, q_BCR=q_BCR)
        R_eff = R[start:end]
        norm_R = np.linalg.norm(R_eff, ord=2)
        log.append(norm_R)

        if norm_R < tol:
            return U, log, True

        diag_eff, lower_eff, upper_eff = tridiag_block(lower, diag, upper, start, end)
        dU_eff = thomas_solve(lower_eff, diag_eff, upper_eff, -R_eff)

        alpha_ls = 1.0
        if line_search:
            accepted = False
            while alpha_ls > 1.0e-12:
                U_trial = U.copy()
                U_trial[start:end] += alpha_ls * dU_eff
                if left_dirich:
                    U_trial[0] = TL
                if right_dirich:
                    U_trial[-1] = TR

                R_trial, _, _, _ = residAndTan(x, U_trial, fluxLaw, r, rT=rT, q_BCL=q_BCL, q_BCR=q_BCR)
                if np.linalg.norm(R_trial[start:end], ord=2) < norm_R:
                    accepted = True
                    break
                alpha_ls *= 0.5

            if not accepted:
                return U, log, False
        else:
            U_trial = U.copy()
            U_trial[start:end] += dU_eff

        U = U_trial

    return U, log, False


def grid_chop_dataframe(df, n_s=CHOP_GRID_S, n_T=CHOP_GRID_T, max_rows_per_cell=MAX_ROWS_PER_CELL, seed=11):
    s_edges = np.linspace(float(df["s"].min()), float(df["s"].max()), n_s + 1)
    T_edges = np.linspace(float(df["T"].min()), float(df["T"].max()), n_T + 1)

    chopped_parts = []
    for s_idx in range(n_s):
        s_left = s_edges[s_idx]
        s_right = s_edges[s_idx + 1]
        s_mask = (df["s"] >= s_left) & ((df["s"] < s_right) if s_idx < n_s - 1 else (df["s"] <= s_right))

        for T_idx in range(n_T):
            T_left = T_edges[T_idx]
            T_right = T_edges[T_idx + 1]
            T_mask = (df["T"] >= T_left) & ((df["T"] < T_right) if T_idx < n_T - 1 else (df["T"] <= T_right))
            cell = df[s_mask & T_mask]
            if cell.empty:
                continue
            chopped_parts.append(cell.sample(n=min(max_rows_per_cell, len(cell)), random_state=seed + s_idx * n_T + T_idx))

    if not chopped_parts:
        raise ValueError("grid chop did not keep any data")

    return pd.concat(chopped_parts, ignore_index=True)


def structured_table_from_chopped_data(df, n_s=CHOP_GRID_S, n_T=CHOP_GRID_T):
    s_grid = np.linspace(float(df["s"].min()), float(df["s"].max()), n_s)
    T_grid = np.linspace(float(df["T"].min()), float(df["T"].max()), n_T)
    s_edges = np.linspace(float(df["s"].min()), float(df["s"].max()), n_s + 1)
    T_edges = np.linspace(float(df["T"].min()), float(df["T"].max()), n_T + 1)
    q_grid = np.full((n_s, n_T), np.nan, dtype=float)

    for s_idx in range(n_s):
        s_left = s_edges[s_idx]
        s_right = s_edges[s_idx + 1]
        s_mask = (df["s"] >= s_left) & ((df["s"] < s_right) if s_idx < n_s - 1 else (df["s"] <= s_right))

        for T_idx in range(n_T):
            T_left = T_edges[T_idx]
            T_right = T_edges[T_idx + 1]
            T_mask = (df["T"] >= T_left) & ((df["T"] < T_right) if T_idx < n_T - 1 else (df["T"] <= T_right))
            cell = df[s_mask & T_mask]
            if not cell.empty:
                q_grid[s_idx, T_idx] = float(cell["q_noisy"].mean())

    missing = np.argwhere(~np.isfinite(q_grid))
    if len(missing) > 0:
        filled = np.argwhere(np.isfinite(q_grid))
        for s_idx, T_idx in missing:
            distances = (filled[:, 0] - s_idx) ** 2 + (filled[:, 1] - T_idx) ** 2
            nearest_s, nearest_T = filled[np.argmin(distances)]
            q_grid[s_idx, T_idx] = q_grid[nearest_s, nearest_T]

    return s_grid, T_grid, q_grid


class ScaledProvider:
    def __init__(self, provider, s_mean, s_std, T_mean, T_std):
        self.provider = provider
        self.s_mean = s_mean
        self.s_std = s_std
        self.T_mean = T_mean
        self.T_std = T_std

    def evaluate(self, s_q, T_q):
        s_q = np.asarray(s_q, dtype=float)
        T_q = np.asarray(T_q, dtype=float)
        s_hat = (s_q - self.s_mean) / self.s_std
        T_hat = (T_q - self.T_mean) / self.T_std
        q, a_hat, b_hat = self.provider.evaluate(s_hat, T_hat)
        a = a_hat / self.s_std
        b = b_hat / self.T_std
        return q, a, b


def wrap_provider(provider):
    def fluxLaw(s, T, xg):
        q, a, b = provider.evaluate(np.array([s]), np.array([T]))
        return float(q[0]), float(a[0]), float(b[0])

    return fluxLaw


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
            nn = np.argpartition(distances, self.K - 1)[:self.K]

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


class TimedFluxLaw:
    def __init__(self, flux_law):
        self.flux_law = flux_law
        self.calls = 0
        self.elapsed_s = 0.0

    def __call__(self, s, T, xg):
        start = time.perf_counter()
        result = self.flux_law(s, T, xg)
        self.elapsed_s += time.perf_counter() - start
        self.calls += 1
        return result


def scaled_structured_table_from_physical(s_grid, T_grid, q_grid, s_mean, s_std, T_mean, T_std):
    s_hat_grid = (s_grid - s_mean) / s_std
    T_hat_grid = (T_grid - T_mean) / T_std
    return s_hat_grid, T_hat_grid, q_grid


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
            merged_level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / merged_weight
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
        [
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    kernel /= np.sum(kernel)

    for i in range(q_grid.shape[0]):
        for j in range(q_grid.shape[1]):
            smoothed[i, j] = np.sum(kernel * padded[i:i + 3, j:j + 3])

    return smoothed


def rmse(predicted, reference):
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))


def noise_normalized_rmse(predicted, reference, noise_std):
    return rmse(predicted / noise_std, reference / noise_std)


def build_methods(df, training_df=None, use_gp=True):
    if training_df is None:
        training_df = df

    k0 = float(df["k_0"].iloc[0])
    alpha = float(df["alpha"].iloc[0])
    beta = float(df["beta"].iloc[0])

    def analytic_fluxLaw(s, T, xg):
        return (
            float(q_true(s, T, k0, alpha, beta)),
            float(a_true(s, T, k0, alpha, beta)),
            float(b_true(s, T, k0, alpha, beta)),
        )

    providers = {}
    build_times = {}

    

    def build_provider(name, factory):
        start = time.perf_counter()
        providers[name] = factory()
        build_times[name] = time.perf_counter() - start
    s_mean = float(training_df["s"].mean())
    s_std = float(training_df["s"].std())
    T_mean = float(training_df["T"].mean())
    T_std = float(training_df["T"].std())

    if s_std == 0.0:
        s_std = 1.0
    if T_std == 0.0:
        T_std = 1.0

    s_grid, T_grid, q_grid = structured_table_from_chopped_data(training_df)
    s_hat_grid, T_hat_grid, q_grid = scaled_structured_table_from_physical(
        s_grid, T_grid, q_grid, s_mean, s_std, T_mean, T_std
    )
    q_grid_pchip = project_table_nonincreasing_in_s(q_grid)
    q_grid_smooth_pchip = project_table_nonincreasing_in_s(smooth_table_3x3(q_grid))

    build_provider(
        "CubicSpline",
        lambda: ScaledProvider(
            CubicSplineFluxST(s_hat_grid, T_hat_grid, q_grid),
            s_mean,
            s_std,
            T_mean,
            T_std,
        ),
    )
    # build_provider(
        # "PCHIP",
        # lambda: ScaledProvider(
        #    PchipFluxST(s_hat_grid, T_hat_grid, q_grid_pchip, extrapolate=False, clip=True),
        #    s_mean,
        #    s_std,
        #    T_mean,
        #    T_std,
        #),
    #)
    # build_provider(
        # "Smooth+PCHIP",
        # lambda: ScaledProvider(
            # PchipFluxST(s_hat_grid, T_hat_grid, q_grid_smooth_pchip, extrapolate=False, clip=True),
            # s_mean,
            # s_std,
            # T_mean,
            # T_std,
        # ),
    # )

    s_hat_data = (training_df["s"].to_numpy() - s_mean) / s_std
    T_hat_data = (training_df["T"].to_numpy() - T_mean) / T_std
    q_noisy_data = training_df["q_noisy"].to_numpy()

    build_provider(
        "RFF",
        lambda: ScaledProvider(
            RFFDerivativeProviderST(
                s_hat_data,
                T_hat_data,
                q_noisy_data,
                n_components=2000,
                gamma=1.0,
                alpha=1e-6,
                random_state=0,
            ),
            s_mean,
            s_std,
            T_mean,
            T_std,
        ),
    )

    build_provider(
        "RBF",
        lambda: ScaledProvider(
            RBFDerivativeProviderST(
                s_hat_data,
                T_hat_data,
                q_noisy_data,
                function="gaussian",
                epsilon=1.1,
                smooth=1.0,
            ),
            s_mean,
            s_std,
            T_mean,
            T_std,
        ),
    )
    build_provider(
        "SavGol",
        lambda: ScaledProvider(
            LocalSavGolProvider(
                np.column_stack([s_hat_data, T_hat_data]),
                q_noisy_data,
                K=min(SAVGOL_NEIGHBORS, len(training_df)),
                h=0.9,
            ),
            s_mean,
            s_std,
            T_mean,
            T_std,
        ),
    )

    if use_gp:
        gp_subset = training_df.sample(n=min(250, len(training_df)), random_state=4)
        build_provider(
            "KISS-GP",
            lambda: KISSGPFluxST(
                gp_subset["s"].to_numpy(),
                gp_subset["T"].to_numpy(),
                gp_subset["q_noisy"].to_numpy(),
                grid_size=16,
                training_iter=20,
                learning_rate=0.08,
            ),
        )

    flux_laws = {name: wrap_provider(provider) for name, provider in providers.items()}
    method_info = {}
    start = time.perf_counter()
    finite_diff_provider = TabularFiniteDifferenceProvider(s_grid, T_grid, q_grid)
    flux_laws["FiniteDiff"] = wrap_provider(finite_diff_provider)
    build_times["FiniteDiff"] = time.perf_counter() - start
    method_info["FiniteDiff"] = {"h_s": finite_diff_provider.ds, "h_T": finite_diff_provider.dT}
    method_info["derivative_noise_h"] = {"h_s": finite_diff_provider.ds, "h_T": finite_diff_provider.dT}
    build_times["Analytic"] = 0.0
    flux_laws["Analytic"] = analytic_fluxLaw

    return flux_laws, build_times, method_info


def evaluate_flux_law_on_points(flux_law, s_values, T_values):
    q_pred = np.zeros_like(s_values, dtype=float)
    a_pred = np.zeros_like(s_values, dtype=float)
    b_pred = np.zeros_like(s_values, dtype=float)

    for i, (s, T) in enumerate(zip(s_values, T_values)):
        try:
            q_pred[i], a_pred[i], b_pred[i] = flux_law(s, T, 0.0)
        except Exception:
            q_pred[i] = np.nan
            a_pred[i] = np.nan
            b_pred[i] = np.nan

    return q_pred, a_pred, b_pred


def make_clean_eval_set(df, n_points=CLEAN_EVAL_POINTS, seed=23):
    rng = np.random.default_rng(seed)
    s_values = rng.uniform(float(df["s"].min()), float(df["s"].max()), n_points)
    T_values = rng.uniform(float(df["T"].min()), float(df["T"].max()), n_points)
    return s_values, T_values


def accuracy_comparison(df, training_df, flux_laws, method_info, sigma):
    noisy_test = df[~df["_row_id"].isin(training_df["_row_id"])]
    noisy_test = noisy_test.sample(n=min(NOISY_TEST_POINTS, len(noisy_test)), random_state=12)

    s_noisy = noisy_test["s"].to_numpy()
    T_noisy = noisy_test["T"].to_numpy()
    q_obs_noisy = noisy_test["q_noisy"].to_numpy()
    q_true_noisy = noisy_test["q_true"].to_numpy()
    noise_std_noisy = sigma * np.maximum(1.0, np.abs(q_true_noisy))

    s_clean, T_clean = make_clean_eval_set(df)
    k0 = float(df["k_0"].iloc[0])
    alpha = float(df["alpha"].iloc[0])
    beta = float(df["beta"].iloc[0])
    q_clean = q_true(s_clean, T_clean, k0, alpha, beta)
    a_clean = a_true(s_clean, T_clean, k0, alpha, beta)
    b_clean = b_true(s_clean, T_clean, k0, alpha, beta)
    noise_std_clean = sigma * np.maximum(1.0, np.abs(q_clean))
    h_s = method_info["derivative_noise_h"]["h_s"]
    h_T = method_info["derivative_noise_h"]["h_T"]
    dq_ds_noise_floor = np.sqrt(np.mean((noise_std_clean / (np.sqrt(2.0) * h_s)) ** 2))
    dq_dT_noise_floor = np.sqrt(np.mean((noise_std_clean / (np.sqrt(2.0) * h_T)) ** 2))

    rows = []
    for name, fluxLaw in flux_laws.items():
        q_pred_clean, a_pred_clean, b_pred_clean = evaluate_flux_law_on_points(fluxLaw, s_clean, T_clean)
        q_pred_noisy, _, _ = evaluate_flux_law_on_points(fluxLaw, s_noisy, T_noisy)

        noisy_q_noise_units = noise_normalized_rmse(q_pred_noisy, q_obs_noisy, noise_std_noisy)
        row = {
            "method": name,
            "test_obs_q_RMSE/noise": noisy_q_noise_units,
            "clean_dq_ds_RMSE/noise": rmse(a_pred_clean, a_clean) / dq_ds_noise_floor,
            "clean_dq_dT_RMSE/noise": rmse(b_pred_clean, b_clean) / dq_dT_noise_floor,
        }

        rows.append(row)

    return pd.DataFrame(rows).sort_values("test_obs_q_RMSE/noise")


def finite_difference_dq_ds(flux_law, s_values, T_values, h, s_min, s_max):
    a_fd = np.zeros_like(s_values, dtype=float)
    for i, (s, T) in enumerate(zip(s_values, T_values)):
        s_left = max(s_min, s - h)
        s_right = min(s_max, s + h)
        if s_right == s_left:
            a_fd[i] = np.nan
            continue
        try:
            q_right, _, _ = flux_law(s_right, T, 0.0)
            q_left, _, _ = flux_law(s_left, T, 0.0)
        except Exception:
            a_fd[i] = np.nan
            continue
        a_fd[i] = (q_right - q_left) / (s_right - s_left)
    return a_fd


def physical_consistency_comparison(df, flux_laws, method_info, tol=PHYSICS_TOL):
    s_eval, T_eval = make_clean_eval_set(df)
    h_s = method_info["derivative_noise_h"]["h_s"]
    s_min = float(df["s"].min())
    s_max = float(df["s"].max())

    rows = []
    for name, fluxLaw in flux_laws.items():
        q_pred, a_pred, _ = evaluate_flux_law_on_points(fluxLaw, s_eval, T_eval)
        if not np.all(np.isfinite(a_pred)):
            a_pred = finite_difference_dq_ds(fluxLaw, s_eval, T_eval, h_s, s_min, s_max)

        entropy_values = np.maximum(0.0, q_pred * s_eval)
        deriv_values = np.maximum(0.0, a_pred)

        rows.append(
            {
                "method": name,
                "entropy_violation_%": 100.0 * np.mean(q_pred * s_eval > tol),
                "worst_entropy_violation": np.nanmax(entropy_values),
                "deriv_violation_%": 100.0 * np.mean(a_pred > tol),
                "worst_deriv_violation": np.nanmax(deriv_values),
            }
        )

    return pd.DataFrame(rows).sort_values(["entropy_violation_%", "deriv_violation_%", "method"])


def fem_comparison(flux_laws, build_times):
    def source(T, xg):
        return 1.0

    x_mesh = np.linspace(0.0, 1.0, 21)
    TL = 0.0
    TR = 1.5

    U_ref, log_ref, ok_ref = NM(x_mesh, flux_laws["Analytic"], source, TL=TL, TR=TR, tol=1e-10, maxiter=40)
    rows = []
    solutions = {"Analytic": U_ref}

    for name, fluxLaw in flux_laws.items():
        timed_fluxLaw = TimedFluxLaw(fluxLaw)
        t0 = time.perf_counter()
        try:
            U, log, ok = NM(x_mesh, timed_fluxLaw, source, TL=TL, TR=TR, tol=1e-8, maxiter=40)
            err = np.linalg.norm(U - U_ref) / np.linalg.norm(U_ref)
            status = "ok" if ok else "not converged"
        except Exception as exc:
            U = np.full_like(U_ref, np.nan)
            log = [np.nan]
            err = np.nan
            status = f"failed: {type(exc).__name__}"

        elapsed = time.perf_counter() - t0
        solutions[name] = U
        rows.append(
            {
                "method": name,
                "status": status,
                "build_s": build_times.get(name, np.nan),
                "newton_steps": len(log),
                "flux_calls": timed_fluxLaw.calls,
                "final_residual": log[-1],
                "rel_solution_err": err,
                "solve_total_s": elapsed,
                "flux_eval_s": timed_fluxLaw.elapsed_s,
                "nonflux_s": elapsed - timed_fluxLaw.elapsed_s,
                "avg_flux_eval_us": 1.0e6 * timed_fluxLaw.elapsed_s / timed_fluxLaw.calls if timed_fluxLaw.calls else np.nan,
            }
        )

    return x_mesh, solutions, pd.DataFrame(rows).sort_values(["status", "rel_solution_err"], na_position="last")


def plot_solutions(x_mesh, solutions, dataset_name):
    image_dir = ROOT / "Data" / "Images"
    image_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    for name, U in solutions.items():
        if np.all(np.isfinite(U)):
            style = "k--" if name == "Analytic" else "-"
            plt.plot(x_mesh, U, style, label=name)

    plt.xlabel("x")
    plt.ylabel("T(x)")
    plt.title(f"FEM solutions using different derivative providers: {dataset_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = image_dir / f"tabular_methods_fem_solutions_{dataset_name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def dataframe_to_markdown(df):
    headers = list(df.columns)
    rows = [headers]
    rows.append(["---"] * len(headers))

    for _, row in df.iterrows():
        formatted = []
        for value in row:
            if isinstance(value, float):
                formatted.append(f"{value:.4e}")
            else:
                formatted.append(str(value))
        rows.append(formatted)

    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def run_dataset(dataset_path):
    dataset_name = dataset_path.stem
    df = pd.read_csv(dataset_path)
    df["_row_id"] = np.arange(len(df))
    training_df = grid_chop_dataframe(df)

    k0 = float(df["k_0"].iloc[0])
    alpha = float(df["alpha"].iloc[0])
    beta = float(df["beta"].iloc[0])
    sigma = float(df["sigma"].iloc[0])

    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"k0={k0}, alpha={alpha}, beta={beta}, sigma={sigma}")
    print(
        f"training rows after {CHOP_GRID_S}x{CHOP_GRID_T} chop: "
        f"{len(training_df)} / {len(df)}"
    )

    start = time.perf_counter()
    flux_laws, build_times, method_info = build_methods(df, training_df=training_df, use_gp=True)
    build_time = time.perf_counter() - start

    accuracy_results = accuracy_comparison(df, training_df, flux_laws, method_info, sigma)
    physics_results = physical_consistency_comparison(df, flux_laws, method_info)
    x_mesh, solutions, fem_results = fem_comparison(flux_laws, build_times)
    plot_path = plot_solutions(x_mesh, solutions, dataset_name)

    print("\nAccuracy comparison:")
    print(accuracy_results.to_string(index=False))
    print("\nPhysics checks:")
    print(physics_results.to_string(index=False))
    print("\nFEM comparison:")
    print(fem_results.to_string(index=False))
    print(f"\nSaved plot: {plot_path}")

    metadata = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "k0": k0,
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "build_time": build_time,
        "plot_path": plot_path,
        "original_rows": len(df),
        "training_rows": len(training_df),
        "noisy_test_rows": min(NOISY_TEST_POINTS, len(df) - len(training_df)),
        "clean_eval_points": CLEAN_EVAL_POINTS,
        "chop_grid_s": CHOP_GRID_S,
        "chop_grid_T": CHOP_GRID_T,
        "max_rows_per_cell": MAX_ROWS_PER_CELL,
    }

    return metadata, accuracy_results, physics_results, fem_results


def write_markdown_report(results):
    report_path = ROOT / "Testing" / "tabular_methods_1D_results.md"

    lines = [
        "# 1D tabular method comparison",
        "",
        "This compares the derivative-provider methods inside the same 1D nonlinear FEM solve.",
        "",
        "Each method supplies `q`, `dq/ds`, and `dq/dT` to the Newton residual/tangent assembly.",
        "",
        (
            f"Training data are reduced by chopping each dataset into a "
            f"{CHOP_GRID_S}x{CHOP_GRID_T} `(s,T)` grid and keeping at most "
            f"{MAX_ROWS_PER_CELL} row per occupied cell."
        ),
        "",
        (
            "Only Analytic evaluates the exact constitutive law. FiniteDiff now finite-differences "
            "a noisy tabular interpolant built from the same chopped training data."
        ),
        "",
        (
            "Accuracy is measured using held-out noisy CSV rows for `q_obs` residuals and random "
            "clean points for derivative errors against the known clean derivative functions."
        ),
        "",
        (
            "The main flux accuracy column is `test_obs_q_RMSE/noise`: RMSE to held-out noisy "
            "observations divided by the pointwise noise scale "
            "`sigma_eff = sigma * max(1, abs(q_true))`. Values near 1 mean the residual is about "
            "the noise floor."
        ),
        "",
        (
            "Derivative accuracy columns compare to clean true derivatives. They use "
            "`sigma_eff / (sqrt(2) h)` with the chopped-grid spacing in the relevant direction."
        ),
        "",
        (
            "Physics checks are evaluated on the same clean random points. Entropy violations count "
            "`q*s > tol`, which indicates anti-diffusion. Derivative violations count `dq/ds > tol`, "
            "equivalently negative tangent conductivity. Violation columns are percentages of "
            "evaluation points."
        ),
        "",
        (
            "FEM cost columns include method build time, Newton steps, total solve wall-clock time, "
            "flux-provider call count, total flux-provider evaluation time, non-flux solve time, "
            "and average flux-provider call time."
        ),
        "",
        "Methods tested: CubicSpline, PCHIP, Smooth+PCHIP, RBF, SavGol, KISS-GP, FiniteDiff, and Analytic.",
        "",
    ]

    for metadata, accuracy_results, physics_results, fem_results in results:
        lines.extend(
            [
                f"## {metadata['dataset_name']}",
                "",
                f"Dataset: `{metadata['dataset_path'].relative_to(ROOT)}`",
                "",
                (
                    f"Parameters: `k0={metadata['k0']}`, `alpha={metadata['alpha']}`, "
                    f"`beta={metadata['beta']}`, `sigma={metadata['sigma']}`"
                ),
                "",
                (
                    f"Training rows after grid chop: `{metadata['training_rows']}` "
                    f"of `{metadata['original_rows']}` "
                    f"(`{metadata['chop_grid_s']}x{metadata['chop_grid_T']}`, "
                    f"max `{metadata['max_rows_per_cell']}` per cell)"
                ),
                "",
                (
                    f"Noisy held-out test rows: `{metadata['noisy_test_rows']}`; "
                    f"clean random evaluation points: `{metadata['clean_eval_points']}`"
                ),
                "",
                f"Provider build time, including GP training: `{metadata['build_time']:.2f} s`",
                "",
                f"FEM plot: `{metadata['plot_path'].relative_to(ROOT)}`",
                "",
                "### Accuracy metrics",
                "",
                dataframe_to_markdown(accuracy_results),
                "",
                "### Physics checks",
                "",
                dataframe_to_markdown(physics_results),
                "",
                "### FEM results",
                "",
                dataframe_to_markdown(fem_results),
                "",
                "### Quick read",
                "",
            ]
        )

        converged = fem_results[fem_results["status"] == "ok"].copy()
        failed = fem_results[fem_results["status"] != "ok"].copy()

        if not converged.empty:
            best = converged.sort_values("rel_solution_err").iloc[0]
            lines.append(
                f"- Best converged method by solution error: `{best['method']}` "
                f"with relative solution error `{best['rel_solution_err']:.3e}`."
            )
        if not failed.empty:
            bad = ", ".join(f"`{name}`" for name in failed["method"].tolist())
            lines.append(f"- Non-converged or failed methods: {bad}.")
        lines.extend(["", ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    dataset_dir = ROOT / "Data" / "NoisyDeterministicOracles" / "datasets"
    dataset_paths = [
        dataset_dir / "nonlinear_high_noise.csv",
        dataset_dir / "nonlinear_low_noise.csv",
        dataset_dir / "linear_medium_noise.csv",
    ]

    results = []
    for dataset_path in dataset_paths:
        results.append(run_dataset(dataset_path))

    report_path = write_markdown_report(results)
    print(f"\nSaved Markdown report: {report_path}")


if __name__ == "__main__":
    main()
